#pragma once
// 8-max tournament NLHE preflop game for the MCCFR solver.
//
// One hand from a fixed tournament state: all seats play, antes and blinds
// are posted, and a preflop betting sequence runs until one of:
//   - everyone folds to one player           -> fold-win terminal
//   - all remaining players are all-in       -> board chance, showdown
//   - betting ends with 2+ chip holders      -> FGS continuation terminal
// Terminal payoffs are ICM (Malmuth-Harville) equity deltas vs. the
// start-of-hand stack vector.
//
// Seats are fixed for the solve (the solver computes a single-hand
// equilibrium, not a sequential game across hands): seat 0 = BTN,
// then SB, BB, UTG, ..., CO. HU: seat 0 = BTN/SB, seat 1 = BB.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cards.h"
#include "common.h"
#include "fgs.h"
#include "hand169.h"
#include "icm.h"

namespace pps {

constexpr int kMaxPlayers = 8;

struct BetAbstraction {
  // --- preflop ---
  // Open sizes (in bb) for the first raise of the hand.
  std::vector<double> openSizes = {2.2, 2.5, 3.0};
  // Later preflop raises, as multiples of the current bet.
  std::vector<double> raiseMultipliers = {2.5, 3.0};
  // Max raises (including the open) per preflop.
  int maxBets = 4;
  // If a candidate raise exceeds this fraction of the effective stack,
  // only the all-in remains.
  double shoveThreshold = 0.5;

  // --- postflop streets ---
  // Bet sizes as fractions of the current pot.
  std::vector<double> streetBetSizes = {0.5, 0.75, 1.0};
  // Raises as multiples of the current street bet.
  std::vector<double> streetRaiseMultipliers = {2.5, 3.0};
  // Max bets+raises per street.
  int streetMaxBets = 3;
};

class PokerGame {
 public:
  struct Config {
    int numPlayers = 8;
    int64_t startStack = 1000000;  // chips per player (uniform default)
    // Optional per-player starting stacks (size numPlayers, all > 0).
    // Overrides startStack when non-empty.
    std::vector<int64_t> stacks;
    int64_t sb = 5000;
    int64_t bb = 10000;
    int64_t ante = 1250;  // 0 for no ante
    std::vector<double> payouts = {0.5, 0.3, 0.2};
    int numBuckets = 169;  // 169 (exact hands) or 15 (coarse)
    BetAbstraction bets;
    ContinuationModel* continuation = nullptr;  // FGS model (owned by caller)
    // Payoff model: true = ICM equity deltas (tournament), false = chip
    // deltas (chip-EV). Chip-EV is the correctness-validation mode: the
    // game is exactly zero-sum in chips and no ICM approximation is in
    // play.
    bool icm = true;
    // Boards sampled per continuation (FGS stub) resolution. More boards
    // = less per-history noise in the stub's outcome distribution.
    int continuationSamples = 16;
    // Multi-street mode: instead of resolving flop-bound pots through the
    // FGS continuation model, deal the flop/turn/river and continue
    // betting on each street to showdown. The continuation model is
    // unused in this mode.
    bool postflop = false;
    // Postflop infoset abstraction: 1 = made-hand category x rank tier
    // (per street); 0 = exact board in the infoset key (no abstraction,
    // very large infoset space).
    int postflopAbstraction = 1;
  };

  struct Action {
    enum Type : uint8_t { FOLD = 0, MATCH = 1, RAISE = 2 };
    Type type = FOLD;
    int64_t raiseTo = 0;  // total contributed by raiser (valid for RAISE)
  };

  struct State {
    int numPlayers = 0;
    std::array<int64_t, kMaxPlayers> stack{};        // chips behind
    std::array<int64_t, kMaxPlayers> contributed{}; // total put in this hand
    std::array<int64_t, kMaxPlayers> streetContrib{}; // put in current street
    std::array<uint8_t, kMaxPlayers> folded{};
    std::array<uint8_t, kMaxPlayers> allin{};
    int current = 0;
    int64_t streetBet = 0;  // current street's highest contribution
    int64_t minRaiseIncrement = 0;
    int pendingActors = 0;
    int raisesSoFar = 0;    // bets+raises this street (incl. blinds)
    int street = 0;         // 0 preflop, 1 flop, 2 turn, 3 river
    // History words: 0=fold 1=check/call, raises 0x80000000|street target.
    std::array<uint32_t, 64> hist{};
    int nhist = 0;
    // Cards: hole cards dealt at root chance; board at street chances.
    std::array<Card, kMaxPlayers * 2> hole{};
    int cardsDealt = 0;  // 0 = hole not dealt, 2n = dealt
    std::array<Card, 5> board{};
    int boardDealt = 0;
    // Stage machine: 0 = hole-card chance, 1 = betting, 2 = street chance
    // (deal the next street), 3 = runout chance (deal the remaining board
    // because no more betting is possible), >=4 = resolved (terminalKind
    // set).
    uint8_t stage = 0;
    // Terminal resolution
    uint8_t terminalKind = 0;  // 0 none, 1 fold-win, 2 showdown, 3 continuation
    std::array<int64_t, kMaxPlayers> finalStack{};  // kinds 1 and 2
    std::vector<WinningsOutcome> contOutcomes;      // kind 3 only
  };

  explicit PokerGame(Config cfg);

  int numPlayers() const { return cfg_.numPlayers; }
  State rootState() const;
  bool isTerminal(const State& s) const { return s.terminalKind != 0; }
  bool isChance(const State& s) const;
  State sampleChance(const State& s, RNG& rng) const;
  int currentPlayer(const State& s) const { return s.current; }
  std::vector<Action> legalActions(const State& s) const;
  State apply(const State& s, const Action& a) const;
  double payoff(const State& s, int player) const;
  uint64_t infosetKey(const State& s, int player) const;

  // Position label for a seat ("BTN", "SB", "BB", "UTG", "UTG+1", "MP",
  // "HJ", "CO"; HU: "BTN/SB", "BB").
  std::string positionName(int seat) const;
  int smallBlindSeat() const { return cfg_.numPlayers == 2 ? 0 : 1; }
  int bigBlindSeat() const { return cfg_.numPlayers == 2 ? 1 : 2; }
  int64_t startStackOf(int seat) const { return startStacks_[seat]; }
  const Config& config() const { return cfg_; }

  // Representative hole cards for a canonical hand index (used to build
  // infoset keys for strategy extraction without a sampled deal).
  int bucketOfPostflop(const State& s, int player) const;  // exposed for tests
  static std::array<Card, 2> representativeCards(int handIdx169);

 private:
  Config cfg_;
  std::vector<double> icm0_;          // start-of-hand ICM equity per seat
  std::vector<int64_t> startStacks_;  // per-seat start-of-hand stacks
  uint64_t gameId_ = 0;               // separates ICM caches across instances
  void resolveFoldWin(State& s) const;
  void resolveShowdown(State& s) const;
  void resolveContinuation(State& s) const;
  void streetEnd(State& s) const;
  void startStreet(State& s) const;  // reset per-street betting state
  // Shared payoff core: expected ICM delta given a set of outcomes.
  double icmDelta(const std::vector<std::array<int64_t, kMaxPlayers>>& finalStacks,
                  const std::vector<double>& probs, int player) const;
  bool anyActiveChips(const State& s) const;
};

}  // namespace pps
