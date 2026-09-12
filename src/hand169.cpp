#include "hand169.h"

#include <cassert>

namespace pps {

namespace {
constexpr char kRanks[14] = "23456789TJQKA";

// Dense layout inside suited/offsuit group for hi > lo:
//   slot = hi*(hi-1)/2 + lo,  hi in [1,12], lo in [0, hi-1], slot in [0,77]
inline int slotOf(int hi, int lo) { return hi * (hi - 1) / 2 + lo; }
inline void slotDecode(int slot, int& hi, int& lo) {
  hi = 1;
  while ((hi + 1) * hi / 2 <= slot) ++hi;
  lo = slot - hi * (hi - 1) / 2;
  assert(lo < hi);
}
}  // namespace

int handIndex169(Card a, Card b) {
  int r1 = rankOf(a), r2 = rankOf(b);
  if (r1 == r2) return r1;
  int hi = r1 > r2 ? r1 : r2;
  int lo = r1 > r2 ? r2 : r1;
  int slot = slotOf(hi, lo);
  bool suited = suitOf(a) == suitOf(b);
  return suited ? 13 + slot : 91 + slot;
}

std::string handName169(int idx) {
  std::string s;
  if (idx < 13) {
    s += kRanks[idx];
    s += kRanks[idx];
    return s;
  }
  int hi, lo;
  bool suited = idx < 91;
  slotDecode(suited ? idx - 13 : idx - 91, hi, lo);
  s += kRanks[hi];
  s += kRanks[lo];
  s += suited ? 's' : 'o';
  return s;
}

std::vector<int> allHands169() {
  std::vector<int> v(kNumHands169);
  for (int i = 0; i < kNumHands169; ++i) v[i] = i;
  return v;
}

int bucketOf(int idx, int numBuckets) {
  if (numBuckets >= kNumHands169) return idx;
  if (numBuckets != kNumCoarseBuckets) return idx;
  // Coarse 15-bucket scheme:
  //   0-2: pairs, split hi (TT+)/mid (55-99)/lo (22-44)
  //   3-8: suited, by hi tier (broadway/mid/low) x connected (gap<=1)/wide
  //   9-14: offsuit, same grid
  if (idx < 13) {
    if (idx >= 8) return 0;
    if (idx >= 4) return 1;
    return 2;
  }
  int hi, lo;
  bool suited = idx < 91;
  slotDecode(suited ? idx - 13 : idx - 91, hi, lo);
  int hiTier = hi >= 10 ? 2 : (hi >= 5 ? 1 : 0);
  int connected = (hi - lo) <= 1 ? 1 : 0;
  int base = suited ? 3 : 9;
  return base + hiTier * 2 + connected;
}

std::string bucketName(int bucket, int numBuckets) {
  if (numBuckets >= kNumHands169) return handName169(bucket);
  if (numBuckets != kNumCoarseBuckets) return handName169(bucket);
  // Order matches bucketOf: base + hiTier*2 + connected, with
  // hiTier 0 = low ranks, 1 = mid ranks, 2 = broadway, and the connected
  // flag (gap <= 1) at offset +1 within each tier pair.
  static const char* kNames[kNumCoarseBuckets] = {
      "PPhi", "PPmid", "PPlo",
      "S-lo-w", "S-lo-c", "S-mid-w", "S-mid-c", "S-hi-w", "S-hi-c",
      "O-lo-w", "O-lo-c", "O-mid-w", "O-mid-c", "O-hi-w", "O-hi-c",
  };
  return kNames[bucket];
}

}  // namespace pps
