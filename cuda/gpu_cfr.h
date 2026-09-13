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

class GpuPostflopSolver {
 public:
  // Compiles the spot to device dataflow (tree build + upload).
  GpuPostflopSolver(const PostflopSpot& spot, const pf::BetConfig& cfg);
  ~GpuPostflopSolver();
  GpuPostflopSolver(const GpuPostflopSolver&) = delete;
  GpuPostflopSolver& operator=(const GpuPostflopSolver&) = delete;

  // Runs `iterations` DCFR iterations via CUDA graph replay.
  // algo: "dcfr" or "hs30" (same schedules as the CPU river solver).
  void solve(int iterations, const std::string& algo);

  // Final walks (EV + best response against the average strategy);
  // call after solve().
  pf::NodeStats stats();

  // Range-weighted average frequencies of the root (OOP) actions.
  std::vector<double> rootStrategy();

  int numNodes() const { return numNodes_; }
  int maxDepth() const { return maxDepth_; }

  // Debug: host-side sums of the device buffers (GPU_CFR_DEBUG=1).
  void debugDump();

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  int numNodes_ = 0;
  int maxDepth_ = 0;
};

}  // namespace gpu
}  // namespace pps
