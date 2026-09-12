#pragma once
// Exploitability measurement for the heads-up poker game (chip-EV mode).
//
// For each sampled deal we compute, with a full tree DP against the
// opponent's fixed average strategy:
//   V_avg(p): expected value to p of avg-vs-avg play on that deal
//   V_br(p):  best-response value to p (maximizing at p's nodes)
// Exploitability per deal = (V_br(0) - V_avg(0)) + (V_br(1) - V_avg(1));
// averaged over deals it estimates how far the average strategy pair is
// from equilibrium, in chips per hand. The game must be chip-EV
// (payoffs are exactly zero-sum).
//
// Board chance and continuation (FGS stub) terminals are evaluated
// analytically from a per-deal all-in equity estimated by Monte Carlo over
// `boardsPerEval` boards. Because every chance-dependent value is an
// expectation (not a per-branch sample), no best-response max is taken
// over noisy estimates — the classic winner's-curse bias that would
// otherwise dominate the bound.
//
// The per-deal greedy BR conditions on the concrete hole cards, which is
// more information than a bucket infoset permits, so the reported number
// is an upper bound on the true exploitability of the abstracted game.
// With 169 buckets (canonical hands) the gap is only suit-level slack.
// A standard error of the mean is reported alongside.
#include <cstdint>
#include <unordered_map>

#include "poker.h"
#include "solver.h"

namespace pps {

struct ExploitabilityResult {
  double exploitability = 0.0;  // chips per hand (upper bound), both players
  double vAvg0 = 0.0;          // value of avg-vs-avg to player 0, chips/hand
  double sem = 0.0;            // standard error of the exploitability mean
  int deals = 0;
};

ExploitabilityResult huExploitability(
    const PokerGame& game,
    const std::unordered_map<uint64_t, InfosetStrategy>& avg,
    int deals, int boardsPerEval, uint64_t seed);

}  // namespace pps
