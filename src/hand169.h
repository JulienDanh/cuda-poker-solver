#pragma once
// Preflop hand canonicalization: 169 canonical starting hands
// (13 pairs, 78 suited, 78 offsuit). Optional coarse clustering into
// 15 buckets for fast low-fidelity runs.
#include <cstdint>
#include <string>
#include <vector>

#include "cards.h"

namespace pps {

constexpr int kNumHands169 = 169;
constexpr int kNumCoarseBuckets = 15;

// Canonical hand index in [0, 169):
//   pair r            -> r                    (0..12)
//   hi>lo, suited     -> 13  + hi*(hi-1)/2 + lo   (13..90)
//   hi>lo, offsuit    -> 91  + hi*(hi-1)/2 + lo   (91..168)
int handIndex169(Card a, Card b);

// Human-readable name, e.g. "77", "AKs", "Q9o".
std::string handName169(int idx);

// Bucket id in [0, numBuckets). numBuckets == 169 -> exact hands;
// numBuckets == 15 -> coarse preset clustering; otherwise 169 is used.
// Cluster layout: 0-2 pairs by rank tier; then suited (base 3) and
// offsuit (base 9) groups, bucket = base + hiTier*2 + connected, where
// hiTier is 0 (low ranks) / 1 (mid) / 2 (broadway+) and connected means
// gap <= 1. E.g. bucket 8 = "S-hi-c" (broadway suited connectors).
int bucketOf(int handIdx169, int numBuckets);
std::string bucketName(int bucket, int numBuckets);

// Full list of 169 hand indices (row-major: pairs, suited, offsuit).
std::vector<int> allHands169();

}  // namespace pps
