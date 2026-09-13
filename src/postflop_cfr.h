#pragma once
// Bet-tree abstraction for the range-based postflop solver, shared by the
// GPU engine (cuda/gpu_cfr.cu). betActions mirrors b-inary/postflop-solver's
// action_tree.rs exactly: pot-relative bet sizes, previous-bet-relative
// raise sizes, call-inclusive pot for threshold checks, effective all-in
// capping, min-raise floor, add/force all-in thresholds, and the
// bet-amount merging pass. The multi-street stack semantics (the caller
// passes the remaining behind-stack, which shrinks by the prior streets'
// matched contributions) are handled by the caller's tree builder.
#include <cstdint>
#include <string>
#include <vector>

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

// The bet-abstraction action list for a betting state: contributions
// (sc0, sc1) for the current street, `potBase` the pot before this
// street's chips, `stack` the behind-stack at the start of this street.
std::vector<TreeAction> betActions(int64_t potBase, int64_t stack,
                                   const BetConfig& cfg, int64_t sc0,
                                   int64_t sc1, int actor, bool afterAllin);

// Result of a solve (GPU engine's final walks). Values are in chips;
// root EVs sum to the starting pot.
struct NodeStats {
  double ev0 = 0.0;      // average-strategy value for OOP (chips)
  double ev1 = 0.0;      // average-strategy value for IP (chips)
  double expl = 0.0;     // exploitability (chips)
  int64_t pairMass = 0;  // number of disjoint weighted combo pairs (x2^20)
};

}  // namespace pf
}  // namespace pps
