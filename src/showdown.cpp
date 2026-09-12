#include "showdown.h"

#include <algorithm>
#include <map>

#include "cards.h"

namespace pps {

std::array<int64_t, kMaxSeats> distributeShowdown(
    int numPlayers, const std::array<Card, kMaxSeats * 2>& hole, const Card* board,
    const int64_t* contributed, const std::vector<int>& inHand) {
  std::array<int64_t, kMaxSeats> winnings{};
  if (inHand.empty()) return winnings;

  // Uncalled-bet return: excess of the top in-hand contribution over the
  // second-highest goes back to the top contributor, and is excluded from
  // the layering below (otherwise it would be re-distributed).
  std::vector<int64_t> contribs;
  contribs.reserve(inHand.size());
  for (int s : inHand) contribs.push_back(contributed[s]);
  std::sort(contribs.begin(), contribs.end(), std::greater<int64_t>());
  int64_t uncalled = contribs.size() >= 2 ? contribs[0] - contribs[1] : 0;
  if (contribs.size() == 1) uncalled = 0;  // sole survivor: no bet is "called"
  int64_t eff[kMaxSeats] = {0};
  for (int s = 0; s < kMaxSeats; ++s) eff[s] = contributed[s];
  if (uncalled > 0) {
    for (int s : inHand) {
      if (contributed[s] == contribs[0]) {
        winnings[s] += uncalled;
        eff[s] -= uncalled;
        break;
      }
    }
  }

  // Dead money from folded seats joins the first layer.
  int64_t dead = 0;
  for (int s = 0; s < numPlayers; ++s) {
    bool in = false;
    for (int h : inHand) in |= (h == s);
    if (!in) dead += contributed[s];
  }

  // Layered side pots over the distinct effective contribution levels.
  std::vector<int64_t> levels;
  for (int s : inHand) levels.push_back(eff[s]);
  std::sort(levels.begin(), levels.end());
  levels.erase(std::unique(levels.begin(), levels.end()), levels.end());

  int64_t prev = 0;
  for (size_t li = 0; li < levels.size(); ++li) {
    int64_t L = levels[li];
    int64_t layer = (li == 0 ? dead : 0);
    std::vector<int> eligible;
    for (int s : inHand) {
      if (eff[s] >= L) {
        layer += std::min(eff[s], L) - prev;
        eligible.push_back(s);
      }
    }
    if (layer <= 0 || eligible.empty()) {
      prev = L;
      continue;
    }
    // Best hand among eligible.
    std::vector<Card> seven;
    HandValue bestV;
    bool first = true;
    std::vector<int> winners;
    std::map<int, HandValue> vals;
    for (int s : eligible) {
      seven.clear();
      seven.push_back(hole[2 * s]);
      seven.push_back(hole[2 * s + 1]);
      for (int b = 0; b < 5; ++b) seven.push_back(board[b]);
      HandValue v = evaluate7(seven.data());
      vals[s] = v;
      if (first || bestV < v) {
        bestV = v;
        first = false;
      }
    }
    for (int s : eligible) {
      if (!(vals[s] < bestV) && !(bestV < vals[s])) winners.push_back(s);
    }
    int64_t share = layer / static_cast<int64_t>(winners.size());
    int64_t rem = layer - share * static_cast<int64_t>(winners.size());
    for (int s : winners) {
      winnings[s] += share;
      if (rem > 0) {
        winnings[s] += rem;
        rem = 0;
      }
    }
    prev = L;
  }
  return winnings;
}

}  // namespace pps
