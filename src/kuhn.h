#pragma once
// Kuhn poker, used to validate the MCCFR engine against a known
// Nash equilibrium. Three cards: J(0) < Q(1) < K(2). Both players ante 1.
// P1 acts first: check or bet 1. Equilibrium family: P1 bets K always and
// bluffs J with prob 1/3; P2 calls a bet with K always and with Q prob 1/3.
#include <cstdint>
#include <vector>

#include "common.h"

namespace pps {

class KuhnGame {
 public:
  enum : uint8_t { A_CHECK = 0, A_BET = 1, A_FOLD = 2, A_CALL = 3 };
  using Action = uint8_t;
  struct State {
    int card0 = -1, card1 = -1;  // -1 = undealt
    int actor = 0;
    uint8_t hist[4] = {0, 0, 0, 0};
    int nhist = 0;
    bool terminal = false;
    int foldeder = -1;  // seat that folded, or -1
  };

  int numPlayers() const { return 2; }
  State rootState() const { return State{}; }
  bool isTerminal(const State& s) const { return s.terminal; }
  bool isChance(const State& s) const { return s.card0 < 0; }

  // Uniform over the 6 ordered deals of 2 distinct cards from 3.
  State sampleChance(const State& s, RNG& rng) const {
    State n = s;
    uint32_t r = rng.nextBelow(6);
    n.card0 = r / 2;
    n.card1 = r % 2;
    if (n.card1 >= n.card0) ++n.card1;
    return n;
  }

  int currentPlayer(const State& s) const { return s.actor; }

  std::vector<Action> legalActions(const State& s) const {
    if (s.nhist == 0) return {A_CHECK, A_BET};                          // P1 first
    if (s.nhist == 1 && s.hist[0] == A_CHECK) return {A_CHECK, A_BET};  // P2 after check
    if (s.nhist == 1) return {A_FOLD, A_CALL};                          // P2 facing bet
    if (s.nhist == 2 && s.hist[1] == A_CHECK) return {A_CHECK, A_BET};  // P1 after x-x
    return {A_FOLD, A_CALL};                                            // P1 facing bet
  }

  State apply(const State& s, Action a) const {
    State n = s;
    n.hist[n.nhist++] = a;
    if (a == A_FOLD) {
      n.foldeder = s.actor;
      n.terminal = true;
      return n;
    }
    n.actor = 1 - n.actor;
    // Terminal on a call, or on a check that closes a check-back.
    if (a == A_CALL || (a == A_CHECK && s.nhist >= 1)) n.terminal = true;
    return n;
  }

  double payoff(const State& s, int player) const {
    if (s.foldeder >= 0) {
      int winner = 1 - s.foldeder;
      return (player == winner) ? 1.0 : -1.0;
    }
    bool betCalled = false;
    for (int i = 0; i < s.nhist; ++i) {
      if (s.hist[i] == A_BET) betCalled = true;
    }
    double amt = betCalled ? 2.0 : 1.0;  // net over the ante
    int winner = s.card0 > s.card1 ? 0 : 1;
    return (player == winner) ? amt : -amt;
  }

  uint64_t infosetKey(const State& s, int player) const {
    // The player id is part of the key: identical (card, history) inputs
    // would otherwise collide across players.
    int card = player == 0 ? s.card0 : s.card1;
    uint64_t h = 0xCBF29CE484222325ULL;
    h = hashCombine(h, static_cast<uint64_t>(player));
    h = hashCombine(h, static_cast<uint64_t>(card));
    h = fnv1a(h, s.hist, s.nhist * sizeof(uint8_t));
    return h;
  }
};

}  // namespace pps
