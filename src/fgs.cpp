#include "fgs.h"

#include <map>

#include "common.h"
#include "showdown.h"

namespace pps {

std::vector<WinningsOutcome> ShowdownContinuation::continuationEV(
    const ContinuationContext& ctx, uint64_t rngSeed) {
  std::vector<WinningsOutcome> out;
  if (ctx.inHand.empty()) return out;
  RNG rng(rngSeed);

  // Available deck: 52 cards minus the in-hand hole cards.
  std::vector<Card> deck;
  deck.reserve(52);
  for (int c = 0; c < 52; ++c) {
    bool used = false;
    for (int s : ctx.inHand) {
      if (ctx.holeCards[s][0] == c || ctx.holeCards[s][1] == c) used = true;
    }
    if (!used) deck.push_back(static_cast<Card>(c));
  }

  std::array<Card, kMaxSeats * 2> hole{};
  for (int s = 0; s < ctx.numPlayers; ++s) {
    if (s < static_cast<int>(ctx.holeCards.size())) {
      hole[2 * s] = ctx.holeCards[s][0];
      hole[2 * s + 1] = ctx.holeCards[s][1];
    }
  }
  std::vector<int64_t> contrib = ctx.contributed;

  std::map<std::vector<int64_t>, int> counts;
  int total = std::max(1, ctx.samples);
  for (int t = 0; t < total; ++t) {
    // Partial Fisher-Yates over 5 board cards.
    for (int i = 0; i < 5; ++i) {
      int j = i + static_cast<int>(rng.nextBelow(static_cast<uint32_t>(deck.size() - i)));
      std::swap(deck[i], deck[j]);
    }
    Card board[5] = {deck[0], deck[1], deck[2], deck[3], deck[4]};
    auto w = distributeShowdown(ctx.numPlayers, hole, board, contrib.data(), ctx.inHand);
    std::vector<int64_t> key(w.begin(), w.end());
    ++counts[key];
  }

  out.reserve(counts.size());
  for (auto& [k, c] : counts) {
    WinningsOutcome o;
    o.winnings = k;
    o.probability = static_cast<double>(c) / total;
    out.push_back(o);
  }
  return out;
}

}  // namespace pps
