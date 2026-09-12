#pragma once
// Pot distribution at showdown: side-pot layering and uncalled-bet returns.
#include <array>
#include <cstdint>
#include <vector>

#include "cards.h"

namespace pps {

constexpr int kMaxSeats = 8;

// Distribute the pot among `inHand` players given a 5-card `board`.
// `contributed[seat]` = total chips the seat put in this hand (all seats).
// Returns per-seat winnings: chips added back to each seat's behind-stack,
// including uncalled-bet returns. Dead money from seats not in `inHand`
// is layered into the first pot. Ties split evenly (remainder to the
// lowest seat).
std::array<int64_t, kMaxSeats> distributeShowdown(
    int numPlayers, const std::array<Card, kMaxSeats * 2>& hole, const Card* board,
    const int64_t* contributed, const std::vector<int>& inHand);

}  // namespace pps
