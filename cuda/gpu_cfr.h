#pragma once
// GPU-CFR: multi-street (flop/turn/river) range-based solver compiled to
// static dataflow (Li & Huang, arXiv:2609.11923; see
// docs/solver-algorithms.md).
//
// The bet tree is built once on the host with the shared bet-abstraction
// logic (pf::betActions — the same one the CPU river solver uses), then
// flattened into arrays: node kinds/players/children, per-node compact
// combo lists, per-node regret and strategy-sum rows, chance-branch
// maps, and per-river-board strength tables. Each DCFR half-step then
// runs as depth-level batched kernel passes — a forward pass computing
// per-node opponent reach vectors through decision and chance nodes, a
// backward pass computing per-combo counterfactual values — and one full
// iteration (both alternating half-steps) is captured into a CUDA graph
// and replayed per iteration. The discount schedule lives in device
// memory and is advanced by the graph itself.
//
// Streets: the board argument is 3, 4 or 5 cards (flop / turn / river
// spot). Chance nodes deal the next street uniformly over cards not on
// the board; hole-card blocking is handled by masking each branch's
// combo lists. The update rule mirrors the CPU river solver exactly
// (alternating DCFR / HS-DCFR with regret matching+), so 5-card spots
// must match the CPU solver bit-for-bit up to summation order
// (gated by `make gpu-parity`), and 3/4-card spots are checked against
// the postflop-solver oracle (tools/pfs-verify).
#include <string>
#include <vector>

#include "postflop_cfr.h"
#include "range.h"

namespace pps {
namespace gpu {

struct PostflopSpot {
  Card board[5] = {0, 0, 0, 0, 0};
  int nBoard = 0;  // 3, 4 or 5
  Range oop;       // player 0, out of position
  Range ip;        // player 1, in position
  int64_t pot = 0;
  int64_t stack = 0;
};

// One action of a decide node, labeled with chips: kind is the
// abstraction action and amount is the street-level total contribution
// for Bet/Raise/AllIn (0 for Fold/Check/Call). Amounts are in chips, so
// the same abstraction yields the same labels across callers.
struct ActionLabel {
  pf::ActionKind kind = pf::ActionKind::Check;
  int64_t amount = 0;
};

// A decide node's static structure (tree introspection).
struct DecideNodeInfo {
  int nodeId = 0;   // global tree index
  int player = 0;   // 0 = OOP, 1 = IP
  int depth = 0;
  int nBoard = 0;   // cards on the board at this node
  int nCombos = 0;  // the deciding player's combos here
  std::vector<ActionLabel> actions;
  std::vector<int> children;  // global node id per action (aligned with actions)
};

// Per-combo strategy of a decide node: the average-strategy frequency
// of each action for each of the node's combos. `combos` indexes the
// deciding player's base list (the same list comboStrategy always
// returns; `playerCards` maps it to cards); `freqs` is action-major
// (freqs[a * combos.size() + c]).
struct ComboStrategy {
  std::vector<ActionLabel> actions;
  std::vector<int> combos;
  std::vector<double> freqs;
};

// Per-combo root values against the average strategy, in chips: the EV
// of holding each combo vs the opponent's full (board-filtered) range.
// combos[p] indexes player p's base list, ev[p] aligns with it, and
// mass[p] is the opponent's valid range weight against that combo (the
// per-combo EV normalizer; Σ w[p]*mass[p]*ev[p] / Σ w[p]*mass[p] is the
// aggregate root EV of player p).
struct ComboEv {
  std::vector<int> combos[2];
  std::vector<double> ev[2];
  std::vector<double> mass[2];
};

// Raw tree structure for traversal/UIs: per-node kind (0 decide,
// 1 showdown, 2 fold, 3 chance) and the CSR child table (node u's
// children are children[childBase[u] .. ] for decide/chance nodes;
// fold/showdown nodes have none).
struct TreeStructure {
  std::vector<uint8_t> kinds;
  std::vector<int> childBase;
  std::vector<int> children;
};

// Per-node values against the average strategy, in chips (the
// whole-hand EV of holding each combo at this node, conditional on
// reaching it). mass is the opponent's reach (range x action
// frequencies) valid against the combo — the row normalizer, so
// Σ w*mass*ev / Σ w*mass is the range-weighted node EV (equal to
// stats() at the root). Action values are exposed for the deciding
// player only, aligned with side[decider].combos.
struct NodeEv {
  struct Side {
    std::vector<int> combos;   // the player's combos at the node (base slots)
    std::vector<double> ev;    // per combo, chips (0 where mass == 0)
    std::vector<double> mass;  // per combo normalizer
    double aggEv = 0.0;        // range-weighted node EV (chips)
  };
  struct ActionEv {
    std::vector<double> perCombo;  // decider's combos, chips
    double aggEv = 0.0;            // range- and frequency-weighted EV
  };
  int decider = 0;
  std::vector<ActionLabel> actions;
  std::vector<ActionEv> actionEv;
  Side side[2];
};

class GpuPostflopSolver {
 public:
  // Compiles the spot to device dataflow (tree build + upload).
  GpuPostflopSolver(const PostflopSpot& spot, const pf::BetConfig& cfg);
  // Loads a saved solution file (see save()): recompiles the spot
  // recorded in the file and restores the trained rows and the discount
  // schedule state. Throws std::runtime_error on IO/format errors or if
  // the file's compiled tree no longer matches this engine version.
  explicit GpuPostflopSolver(const std::string& solutionPath);
  ~GpuPostflopSolver();
  GpuPostflopSolver(const GpuPostflopSolver&) = delete;
  GpuPostflopSolver& operator=(const GpuPostflopSolver&) = delete;

  // Runs up to `iterations` DCFR iterations via CUDA graph replay.
  // algo: "dcfr" or "hs30". If target > 0, the solve stops early once
  // the exploitability (checked every 128 iterations via two BR walks)
  // drops to or below it; iterationsRun() reports how many ran.
  // A fresh solve: zeroes any previously accumulated rows and restarts
  // the discount schedule (see reset() / continueSolve()).
  void solve(int iterations, const std::string& algo, double target = -1.0);

  // Continues the current solution for up to `iterations` more
  // iterations (DCFR's discount staircase depends only on the
  // cumulative iteration index, so the schedule continues exactly where
  // solve() left off). Throws if the last solve used "hs30" (its
  // schedule depends on the planned iteration count, so continuation
  // is not defined). Requires a prior solve() call.
  void continueSolve(int iterations, double target = -1.0);

  // Zeroes the regret/strategy rows and the schedule: back to an
  // untrained solver with the same compiled tree.
  void reset();

  // Iterations executed by the last solve()/continueSolve() call
  // (early-stopped counts included).
  int iterationsRun() const { return iterationsRun_; }

  // Cumulative iterations across solve()/continueSolve() since the
  // last reset() (or construction, or a solution load).
  int64_t totalIterations() const { return totalIters_; }

  // Final walks (EV + best response against the average strategy);
  // call after solve().
  pf::NodeStats stats();

  // Range-weighted average frequencies of the root (OOP) actions.
  std::vector<double> rootStrategy();

  // Range-weighted average frequencies of decide node `decideIdx` (the
  // decideMeta order: 0 = root, 1 = IP's first decision after OOP
  // checks, ...). Returns {} for anything but a first-street decision:
  // deeper (chance-compacted) nodes would need reach-weighted averaging,
  // and the range-weighted aggregate is exact only where the deciding
  // player has no earlier action on the path (the root, and each
  // player's first decision of the street).
  std::vector<double> nodeStrategy(int decideIdx);

  // Tree introspection. decideIdx orders the decide nodes in global
  // node order (same order as nodeStrategy / comboStrategy).
  int numDecideNodes() const;
  DecideNodeInfo decideNode(int decideIdx) const;

  // Per-combo average-strategy frequencies of any decide node —
  // including chance-compacted ones (per-combo frequencies need no
  // range weighting, unlike nodeStrategy's first-street-only
  // aggregate). Rows are normalized per combo; combos with a zero
  // strategy sum (unreached, e.g. vs a pure fold) return uniform.
  ComboStrategy comboStrategy(int decideIdx);

  // Per-combo root EVs vs the average strategy (two value walks).
  ComboEv rootEvPerCombo();

  // Per-combo EVs of any decide node vs the average strategy (one
  // value walk per player), plus the deciding player's per-action
  // EVs. Runs two full-tree EV walks.
  NodeEv nodeEv(int decideIdx);

  // The base combo lists the solver was compiled with: cards[2*i],
  // cards[2*i+1] is combo slot i of player p's range (board-filtered,
  // positive-weight only — the same slots combos[] refers to).
  std::vector<uint8_t> playerCards(int player) const;

  // The number of base combos per player (playerCards(p).size()/2).
  int numBaseCombos(int player) const;

  // The compiled range weights per player, aligned with playerCards(p)
  // (board-filtered, positive-weight only).
  std::vector<double> playerWeights(int player) const;

  // Per-node kind (decide/showdown/fold/chance) plus the CSR children
  // table — enough to walk the whole tree from the root.
  TreeStructure treeStructure() const;

  int numNodes() const { return numNodes_; }
  int maxDepth() const { return maxDepth_; }

  // Serializes the compiled spot, the bet config and the trained
  // regret/strategy-sum rows plus the schedule state. The file is
  // self-describing: the load constructor rebuilds the tree from it.
  void save(const std::string& path) const;

  // Debug: host-side sums of the device buffers (GPU_CFR_DEBUG=1).
  void debugDump();

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  int numNodes_ = 0;
  int maxDepth_ = 0;
  int iterationsRun_ = 0;
  int64_t totalIters_ = 0;

  // Shared by both constructors: compile + upload, then optional rows.
  void init(const PostflopSpot& spot, const pf::BetConfig& cfg);
  // Graph replay in 128-iteration chunks from the current schedule
  // state; updates iterationsRun_ / totalIters_.
  void replayChunks(int iterations, double target);
};

}  // namespace gpu
}  // namespace pps
