#pragma once
// Malmuth-Harville ICM (Independent Chip Model).
// Maps a stack vector to each player's expected share of the prize pool.
#include <cstdint>
#include <vector>

namespace pps {

// Fast ICM via subset dynamic programming. stacks.size() must equal
// payouts.size(); payouts must sum to 1. Returns per-player equity
// (expected fraction of the prize pool). Valid for up to ~13 players.
std::vector<double> icmEquities(const std::vector<int64_t>& stacks,
                                const std::vector<double>& payouts);

// Reference implementation: exact sum over all finishing permutations.
// O(n!) — use only for validation and small n.
std::vector<double> icmBruteForce(const std::vector<int64_t>& stacks,
                                 const std::vector<double>& payouts);

}  // namespace pps
