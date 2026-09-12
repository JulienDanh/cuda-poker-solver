#pragma once
// Common small utilities: deterministic RNG and hashing.
#include <cstdint>
#include <string>
#include <vector>

namespace pps {

// xorshift64* deterministic RNG, cheap enough for Monte Carlo CFR.
struct RNG {
  uint64_t s;
  explicit RNG(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  uint64_t nextU64() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
  }
  uint32_t nextU32() { return static_cast<uint32_t>(nextU64() >> 32); }
  double nextDouble() { return static_cast<double>(nextU64() >> 11) * 0x1.0p-53; }
  // Uniform integer in [0, n)
  uint32_t nextBelow(uint32_t n) { return static_cast<uint32_t>(nextDouble() * n); }
};

// FNV-1a hash for infoset keys and history encoding.
inline uint64_t fnv1a(uint64_t h, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) {
    h ^= p[i];
    h *= 0x100000001B3ULL;
  }
  return h;
}

inline uint64_t hashCombine(uint64_t h, uint64_t v) {
  h = fnv1a(h, &v, sizeof(v));
  return h;
}

std::vector<std::string> splitString(const std::string& s, char sep);

}  // namespace pps
