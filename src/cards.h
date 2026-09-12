#pragma once
// Card representation and 7-card hand evaluation.
// Card is uint8_t in [0, 52): suit = card / 13 (0..3), rank = card % 13
// (0 = deuce, ..., 12 = ace).
#include <array>
#include <cstdint>
#include <string>

#include "common.h"

namespace pps {

using Card = uint8_t;
constexpr int kNumCards = 52;

inline int rankOf(Card c) { return c % 13; }
inline int suitOf(Card c) { return c / 13; }

std::string cardName(Card c);

// Best-5-of-7 hand value. Comparison is a total order; the struct is
// directly comparable: higher is better. category: 8 straight flush,
// 7 quads, 6 full house, 5 flush, 4 straight, 3 trips, 2 two pair,
// 1 pair, 0 high card. tiebreak holds 5 kickers/ranks in order.
struct HandValue {
  int category = 0;
  std::array<int, 5> tiebreak{};
  bool operator<(const HandValue& o) const;
  bool operator==(const HandValue& o) const;
};

HandValue evaluate7(const Card* cards /* exactly 7 */);

}  // namespace pps
