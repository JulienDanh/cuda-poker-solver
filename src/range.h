#pragma once
// Weighted hand ranges for range-based (abstraction-free) postflop
// solving, in the same conceptual format as postflop-solver.
//
// A Range holds a weight per unordered hole-card pair (1326 combos).
// Combos conflicting with a known board are filtered to weight 0.
#include <cstdint>
#include <string>

#include "cards.h"

namespace pps {

constexpr int kCombos = 1326;

// Combo i is the pair (comboCards[i][0], comboCards[i][1]) with
// cards in ascending index order.
extern const uint8_t kComboCards[kCombos][2];

// Index of the combo {a, b} (a != b), order-independent.
int comboIndex(uint8_t a, uint8_t b);

struct Range {
  double w[kCombos];
  Range();               // all zeros
  static Range ones();   // all combos weight 1 (filter vs board after)
  double total() const;
  // Normalize so weights sum to 1.
  void normalize();
  // Zero out combos containing any board card.
  void filterBoard(const Card* board, int n);
};

// Parses a range string in postflop-solver's syntax subset:
//   - hand classes: pairs ("TT"), suited ("AKs"), offsuit ("AKo")
//   - "+" ranges: "TT+" (pairs), "A9s+" (suited with second card up),
//     "ATo+" (offsuit)
//   - dash ranges: "55-22", "A5s-A2s", "KTo-K7o"
//   - optional weights: "AA:0.5"
// Weight applies to the whole token. Unsupported syntax throws.
Range parseRange(const std::string& s);

// Rank helpers shared with the parser ("A" -> 12, "T" -> 8, ...).
int rankFromChar(char c);

}  // namespace pps
