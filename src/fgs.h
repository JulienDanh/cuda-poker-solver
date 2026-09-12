#pragma once
// FGS (Future Game Simulation) continuation interface.
//
// In a preflop solver, a betting sequence can end with two or more players
// still holding chips (a "called pot going to the flop"). The value of that
// state is not just the current pot: the hand continues postflop. FGS
// models that continuation without building the full postflop tree here:
// the terminal node hands off to a ContinuationModel that returns a
// distribution over per-seat winnings outcomes, which the game evaluates
// through ICM (ICM is nonlinear, so we need the distribution, not the
// expected stack).
//
// The default ShowdownContinuation is a documented approximation: it
// resolves the state as if all remaining players were all-in (sample
// boards, distribute the pot by showdown with side pots and uncalled-bet
// returns). It ignores position, realization and postflop play; it exists
// so the preflop solver runs end-to-end today. Plugging in a real postflop
// solver later requires no changes elsewhere in the codebase.
#include <array>
#include <cstdint>
#include <vector>

#include "cards.h"

namespace pps {

constexpr int kMaxPlayersCont = 8;

struct ContinuationContext {
  int numPlayers = 0;
  // Seats still in the hand (not folded), sorted ascending.
  std::vector<int> inHand;
  // holeCards[player] = {c0, c1}; valid for players in `inHand`.
  std::vector<std::array<Card, 2>> holeCards;
  // Total chips contributed by every seat this hand (folded seats included:
  // dead money belongs to the pot).
  std::vector<int64_t> contributed;
  // Boards to sample for the default model.
  int samples = 16;
};

struct WinningsOutcome {
  std::vector<int64_t> winnings;  // per seat, chips added to behind-stack
  double probability = 0.0;
};

class ContinuationModel {
 public:
  virtual ~ContinuationModel() = default;
  virtual std::vector<WinningsOutcome> continuationEV(const ContinuationContext& ctx,
                                                      uint64_t rngSeed) = 0;
};

// Default model: sample `ctx.samples` boards; for each board distribute the
// pot by showdown (side-pot layering, uncalled-bet return); aggregate the
// sampled outcomes into a probability distribution.
class ShowdownContinuation : public ContinuationModel {
 public:
  std::vector<WinningsOutcome> continuationEV(const ContinuationContext& ctx,
                                              uint64_t rngSeed) override;
};

}  // namespace pps
