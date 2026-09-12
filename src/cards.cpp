#include "cards.h"

#include <algorithm>

namespace pps {

std::string cardName(Card c) {
  static const char* kRanks = "23456789TJQKA";
  static const char* kSuits = "cdhs";
  return {kRanks[rankOf(c)], kSuits[suitOf(c)]};
}

bool HandValue::operator<(const HandValue& o) const {
  if (category != o.category) return category < o.category;
  for (int i = 0; i < 5; ++i) {
    if (tiebreak[i] != o.tiebreak[i]) return tiebreak[i] < o.tiebreak[i];
  }
  return false;
}

bool HandValue::operator==(const HandValue& o) const {
  if (category != o.category) return false;
  for (int i = 0; i < 5; ++i) {
    if (tiebreak[i] != o.tiebreak[i]) return false;
  }
  return true;
}

HandValue evaluate7(const Card* cards) {
  int rankCount[13] = {0};
  int suitCount[4] = {0};
  int suitRankMask[4] = {0};
  int rankMask = 0;
  for (int i = 0; i < 7; ++i) {
    int r = rankOf(cards[i]);
    int s = suitOf(cards[i]);
    rankCount[r]++;
    suitCount[s]++;
    suitRankMask[s] |= 1 << r;
    rankMask |= 1 << r;
  }

  // Straight helper: highest straight in a rank mask (ranks 0..12, ace = 12;
  // wheel uses ace as low = bit -1 handled by appending ace below deuce).
  auto straightHigh = [](int mask) -> int {
    // Allow ace-low: duplicate ace as rank -1.
    int m = (mask << 1) | ((mask >> 12) & 1);  // bit i+1 = rank i; ace low at 0
    for (int hi = 13; hi >= 4; --hi) {  // hi = lowRank + 4 in shifted space
      bool ok = true;
      for (int j = 0; j < 5; ++j) {
        if (!((m >> (hi - j)) & 1)) {
          ok = false;
          break;
        }
      }
      if (ok) return hi - 1;  // convert back to unshifted high rank
    }
    return -1;
  };

  // Flush?
  for (int s = 0; s < 4; ++s) {
    if (suitCount[s] >= 5) {
      int sh = straightHigh(suitRankMask[s]);
      if (sh >= 0) {
        HandValue v;
        v.category = 8;
        v.tiebreak[0] = sh;
        return v;
      }
      HandValue v;
      v.category = 5;
      int placed = 0;
      for (int r = 12; r >= 0 && placed < 5; --r) {
        if (suitRankMask[s] & (1 << r)) v.tiebreak[placed++] = r;
      }
      return v;
    }
  }

  // Quads / full house / trips / two pair / pair / high card from rank counts.
  // Note: 7 cards can contain two trip ranks (e.g. 222+444+x), which is a
  // full house (higher trips over lower trips), and quads+trips is quads.
  int quads = -1, tripsHi = -1, tripsLo = -1;
  int pairs[13];
  int numPairs = 0;
  for (int r = 12; r >= 0; --r) {
    if (rankCount[r] == 4) quads = r;
    else if (rankCount[r] == 3) {
      if (tripsHi < 0) tripsHi = r;
      tripsLo = r;
    } else if (rankCount[r] == 2) pairs[numPairs++] = r;
  }

  HandValue v;
  if (quads >= 0) {
    v.category = 7;
    v.tiebreak[0] = quads;
    for (int r = 12; r >= 0; --r) {
      if (r != quads && rankCount[r] > 0) {
        v.tiebreak[1] = r;
        break;
      }
    }
    return v;
  }
  if (tripsHi >= 0 && (numPairs >= 1 || tripsLo != tripsHi)) {
    v.category = 6;
    v.tiebreak[0] = tripsHi;
    // The "pair" side of the full house is the best available pair or
    // the lower trips.
    int pairSide = -1;
    if (numPairs >= 1) pairSide = pairs[0];
    if (tripsLo >= 0 && tripsLo != tripsHi) pairSide = std::max(pairSide, tripsLo);
    v.tiebreak[1] = pairSide;
    return v;
  }
  int sh = straightHigh(rankMask);
  if (sh >= 0) {
    v.category = 4;
    v.tiebreak[0] = sh;
    return v;
  }
  if (tripsHi >= 0) {
    v.category = 3;
    v.tiebreak[0] = tripsHi;
    int placed = 1;
    for (int r = 12; r >= 0 && placed < 3; --r) {
      if (r != tripsHi && rankCount[r] > 0) v.tiebreak[placed++] = r;
    }
    return v;
  }
  if (numPairs >= 2) {
    v.category = 2;
    v.tiebreak[0] = pairs[0];
    v.tiebreak[1] = pairs[1];
    for (int r = 12; r >= 0; --r) {
      if (r != pairs[0] && r != pairs[1] && rankCount[r] > 0) {
        v.tiebreak[2] = r;
        break;
      }
    }
    return v;
  }
  if (numPairs == 1) {
    v.category = 1;
    v.tiebreak[0] = pairs[0];
    int placed = 1;
    for (int r = 12; r >= 0 && placed < 5; --r) {
      if (r != pairs[0] && rankCount[r] > 0) v.tiebreak[placed++] = r;
    }
    return v;
  }
  v.category = 0;
  for (int i = 0; i < 5; ++i) v.tiebreak[i] = -1;
  int placed = 0;
  for (int r = 12; r >= 0 && placed < 5; --r) {
    if (rankCount[r] > 0) v.tiebreak[placed++] = r;
  }
  return v;
}

}  // namespace pps
