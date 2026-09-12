#pragma once
// Range-based (abstraction-free) CFR solver for heads-up river spots,
// built for parity with b-inary/postflop-solver (docs/solver-algorithms.md).
//
// - Bet tree construction mirrors postflop-solver's action_tree.rs exactly:
//   pot-relative bet sizes, previous-bet-relative raise sizes, call-inclusive
//   pot for threshold checks, effective all-in capping, min-raise floor,
//   add/force all-in thresholds, and the bet-amount merging pass.
// - Solving: vanilla full-tree CFR with alternating updates and
//   regret-matching+, with DCFR discounting. Two schemes:
//     "dcfr": postflop-solver's exact parameters (alpha = t^1.5/(t^1.5+1),
//             beta = 0.5, gamma = (t'/(t'+1))^3 with t' counted from the
//             last power of 4, i.e. the average-strategy reset).
//     "hs30": HS-DCFR(30) — the 2026 state of the art
//             (Zhang, McAleer & Sandholm, arXiv:2404.09097):
//             alpha(t) = 1 + 3t/n, beta(t) = -1 - 2t/n, gamma(t) = 30 - 5t/n
//             plugged into the same DCFR update equations.
// - Values are in chips; root EVs sum to the starting pot, matching the
//   reference engine's convention.
// - Exploitability via best response against the average strategy:
//   (BR_0 + BR_1 - pot) / 2.
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "cards.h"
#include "range.h"

namespace pps {
namespace pf {

enum class ActionKind : uint8_t { Fold, Check, Call, Bet, Raise, AllIn };

struct TreeAction {
  ActionKind kind;
  int64_t amount = 0;  // street-level total contribution (Bet/Raise/AllIn)
};

struct BetConfig {
  // Bet sizes as fractions of the (call-inclusive) pot; raises as
  // multiples of the previous bet.
  std::vector<double> betFracs = {0.5, 0.75, 1.0};
  std::vector<double> raiseMults = {2.5, 3.0};
  double addAllinThreshold = 1.5;
  double forceAllinThreshold = 0.15;
  double mergingThreshold = 0.1;
};

struct Spot {
  Card board[5];
  Range oop;   // player 0, out of position: acts first on the river
  Range ip;    // player 1, in position: acts second
  int64_t pot = 0;    // starting pot (already includes prior-street chips)
  int64_t stack = 0;   // effective stack behind for each player
};

struct NodeStats {
  double ev0 = 0.0;      // average-strategy value for OOP (chips)
  double ev1 = 0.0;      // average-strategy value for IP (chips)
  double expl = 0.0;     // exploitability (chips)
  int64_t pairMass = 0;  // number of disjoint weighted combo pairs (x2^40)
};

class RiverSolver {
 public:
  RiverSolver(const Spot& spot, const BetConfig& cfg);

  // Runs `iterations` full DCFR iterations (two alternating half-steps).
  void solve(int iterations, const std::string& algo);

  NodeStats stats() const { return stats_; }
  // Range-weighted average frequencies of the root (OOP) actions.
  std::vector<double> rootStrategy() const;
  int numNodes() const { return numNodes_; }
  int nCombos(int player) const { return sides_[player].n; }
  void dumpTree(std::FILE* f) const;

 private:
  struct Side {
    int n = 0;
    std::vector<Card> cards;   // 2 per combo
    std::vector<double> w;     // range weights
    std::vector<uint64_t> strength;
    std::vector<int> sorted;   // combo indices sorted by strength ascending
    // sameOther[i]: index into the OTHER side's combo list of the identical
    // combo (same two cards), or -1.
    std::vector<int> sameOther;
  };

  struct Node {
    enum Kind : uint8_t { DECIDE, SHOWDOWN, FOLD } kind = DECIDE;
    int player = 0;             // deciding player (for DECIDE)
    int foldBy = 0;             // for FOLD: the player who folded
    int64_t sc[2] = {0, 0};     // street contributions (for terminals)
    std::vector<TreeAction> actions;
    std::vector<int> children;
    // CFR state (DECIDE only): per action x per combo of `player`
    std::vector<double> regret;
    std::vector<double> stratSum;
    // Scratch: counterfactual action values (actions x combos), node value
    std::vector<double> cfv;
    std::vector<double> val;
  };

  void buildTree();
  int buildNode(int64_t sc0, int64_t sc1, int actor, bool afterAllin, int depth);
  std::vector<TreeAction> possibleActions(int64_t sc0, int64_t sc1,
                                          int actor, bool afterAllin) const;

  // DCFR parameters per iteration.
  struct Discount {
    double posCoef, negCoef, avgCoef;
  };
  Discount makeDiscount(int t, int iters) const;

  // One alternating half-step for `tr`: fills `outVal` with the traverser's
  // counterfactual values given `reachOpp` (opponent combo weights).
  void passRec(int nodeIdx, int tr, const double* reachOpp, const Discount& d,
               double* outVal, int depth);

  // Showdown values for the traverser given opponent reach.
  void showdownValues(int nodeIdx, int tr, const double* reachOpp,
                      double winV, double tieV, double loseV, double* out) const;
  // Fast path: disjoint reach mass per traverser combo, O(52 + n).
  void disjointMass(int tr, const double* reachOpp, double* out) const;

  void regretMatching(const Node& nd, int n, double* sigma) const;

  // Final walks.
  void computeStats();

  void avgStrategy(const Node& nd, int n, double* sigma) const;
  // Value walk for one traverser with the average strategy (mirrors
  // passRec's structure; values carry the opponent's reach mass and the
  // traverser's own per-combo strategy applied at their own nodes).
  void evOne(int nodeIdx, int tr, const double* reachOpp, double* outVal) const;
  // Best-response walk for player `br` against the opponent's average.
  void brRec(int nodeIdx, int br, const double* reachOpp, double* v) const;

  Spot spot_;
  BetConfig cfg_;
  Side sides_[2];
  std::deque<Node> nodes_;  // stable references while building
  int numNodes_ = 0;
  NodeStats stats_;
  std::string algo_ = "dcfr";
  // Depth-indexed scratch buffers (avoids per-recursion heap allocation).
  // [depth][k]: 0 = sigma (na*n), 1 = cfv (na*ntr), 2 = child reach (n),
  // 3 = per-action child value (ntr).
  static constexpr size_t kScratchStride = 8 * 1326;
  static constexpr int kMaxDepth = 64;
  std::vector<double> scratch_[4][kMaxDepth];
  void initScratch();
};

}  // namespace pf
}  // namespace pps
