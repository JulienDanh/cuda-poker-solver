#include "icm.h"

#include <algorithm>
#include <cmath>

namespace pps {

namespace {

// A(S) = probability that the set S of players occupies the top |S|
// finishing spots (in any order), under the sequential Harville model.
// Recursion over subsets: consider the player in S who finishes last
// among S (i.e., in spot |S|); all players below them are outside S.
//   A(S) = sum_{i in S} s_i / (T_all - (T(S) - s_i)) * A(S \ {i})
// where T_all = total chips. Base: A({i}) = s_i / T_all.
// Implemented with bitmask DP.
double subsetProb(uint32_t S, const std::vector<int64_t>& stacks,
                  double total, std::vector<double>& memo,
                  const std::vector<double>& subsetSum) {
  if (memo[S] >= 0.0) return memo[S];
  double v = 0.0;
  for (int i = 0; i < static_cast<int>(stacks.size()); ++i) {
    if (!(S & (1u << i))) continue;
    uint32_t rest = S & ~(1u << i);
    double aRest = (rest == 0) ? 1.0 : subsetProb(rest, stacks, total, memo, subsetSum);
    double denom = total - (subsetSum[S] - static_cast<double>(stacks[i]));
    v += static_cast<double>(stacks[i]) / denom * aRest;
  }
  memo[S] = v;
  return v;
}

}  // namespace

std::vector<double> icmEquities(const std::vector<int64_t>& stacks,
                                const std::vector<double>& payouts) {
  const int n = static_cast<int>(stacks.size());
  // Degenerate stacks: a player with 0 chips can never finish above a
  // player with chips, so positive-stack players occupy the top m places
  // (Harville among themselves) and zero-stack players split the tail
  // payouts evenly. Without this, Harville divides by zero.
  std::vector<int> posIdx;
  posIdx.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (stacks[i] > 0) posIdx.push_back(i);
  }
  const int m = static_cast<int>(posIdx.size());
  if (m == 0) {
    double sum = 0.0;
    for (double p : payouts) sum += p;
    return std::vector<double>(n, sum / n);
  }
  if (m < n) {
    std::vector<int64_t> pos;
    pos.reserve(m);
    for (int i : posIdx) pos.push_back(stacks[i]);
    std::vector<double> top(payouts.begin(),
                             payouts.begin() + std::min<int>(m, payouts.size()));
    auto posEq = icmEquities(pos, top);
    double tail = 0.0;
    for (int j = m; j < static_cast<int>(payouts.size()); ++j) tail += payouts[j];
    std::vector<double> eq(n, 0.0);
    for (int k = 0; k < m; ++k) eq[posIdx[k]] = posEq[k];
    for (int i = 0; i < n; ++i) {
      if (stacks[i] == 0) eq[i] = tail / (n - m);
    }
    return eq;
  }

  const int full = 1 << n;
  std::vector<double> memo(full, -1.0);
  std::vector<double> subsetSum(full, 0.0);
  double total = 0.0;
  for (auto s : stacks) total += static_cast<double>(s);
  for (uint32_t S = 1; S < static_cast<uint32_t>(full); ++S) {
    int low = __builtin_ctz(S);
    subsetSum[S] = subsetSum[S & (S - 1)] + static_cast<double>(stacks[low]);
  }
  std::vector<double> eq(n, 0.0);
  for (int place = 1; place <= n; ++place) {
    double pay = place <= static_cast<int>(payouts.size()) ? payouts[place - 1] : 0.0;
    if (pay == 0.0) continue;
    for (int i = 0; i < n; ++i) {
      // P(i finishes exactly `place`) = sum over S of size place-1, i notin S:
      //   A(S) * s_i / (T_all - T(S))
      double p = 0.0;
      for (uint32_t S = 0; S < static_cast<uint32_t>(full); ++S) {
        if (S & (1u << i)) continue;
        if (__builtin_popcount(S) != place - 1) continue;
        if (place == 1) {
          p += 1.0 * static_cast<double>(stacks[i]) / total;
        } else {
          double a = subsetProb(S, stacks, total, memo, subsetSum);
          p += a * static_cast<double>(stacks[i]) / (total - subsetSum[S]);
        }
      }
      eq[i] += pay * p;
    }
  }
  return eq;
}

std::vector<double> icmBruteForce(const std::vector<int64_t>& stacks,
                                 const std::vector<double>& payouts) {
  const int n = static_cast<int>(stacks.size());
  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i) perm[i] = i;
  std::vector<double> eq(n, 0.0);
  double total = 0.0;
  for (auto s : stacks) total += static_cast<double>(s);
  do {
    double prob = 1.0;
    double remaining = total;
    for (int k = 0; k < n; ++k) {
      prob *= static_cast<double>(stacks[perm[k]]) / remaining;
      remaining -= static_cast<double>(stacks[perm[k]]);
    }
    for (int k = 0; k < n; ++k) {
      double pay = k < static_cast<int>(payouts.size()) ? payouts[k] : 0.0;
      eq[perm[k]] += pay * prob;
    }
  } while (std::next_permutation(perm.begin(), perm.end()));
  return eq;
}

}  // namespace pps
