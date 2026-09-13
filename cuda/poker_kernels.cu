// GPU kernels for the hot loops of the preflop ICM solver.
//
// The two dominant costs during MCCFR training are:
//   1. ICM evaluation of terminal stack vectors (Malmuth-Harville subset DP)
//   2. Board runout sampling for all-in showdowns / continuation stubs
//
// This file provides batched versions of both for NVIDIA GPUs. It is only
// compiled when ENABLE_CUDA=ON (see CMakeLists.txt); the CPU solver never
// includes it. Requires CUDA compute capability >= 6.0.
#include <cstdint>
#include <cuda_runtime.h>

namespace pps::cuda {

// ---------------------------------------------------------------------------
// Batched ICM (Malmuth-Harville, subset DP) — one thread per
// (batch item, player, payout place triple is computed in-thread).
// Mirrors src/icm.cpp including the zero-stack normalization.
// ---------------------------------------------------------------------------

constexpr int kMaxPlayersGpu = 8;
constexpr int kMaxPayouts = 8;

__device__ __forceinline__ double subsetProbGpu(uint32_t S, const double* stacks,
                                                 double total, double* memo,
                                                 const double* subsetSum) {
  if (memo[S] >= 0.0) return memo[S];
  double v = 0.0;
  for (int i = 0; i < kMaxPlayersGpu; ++i) {
    if (!(S & (1u << i))) continue;
    uint32_t rest = S & ~(1u << i);
    double aRest = (rest == 0) ? 1.0 : subsetProbGpu(rest, stacks, total, memo, subsetSum);
    double denom = total - (subsetSum[S] - stacks[i]);
    v += stacks[i] / denom * aRest;
  }
  memo[S] = v;
  return v;
}

// stacks: [batch][n] int64; payouts: [numPayouts] double (sum == 1).
// out: [batch][n] equity. Mirrors src/icm.cpp including the zero-stack
// normalization: players with 0 chips can never finish above a player with
// chips, so positive stacks compete Harville-style for the top m payouts
// and zero stacks split the tail evenly.
__global__ void icmBatchKernel(const int64_t* __restrict__ stacks, int n,
                               int batch, const double* __restrict__ payouts,
                               int numPayouts, double* __restrict__ out) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= batch) return;
  // Thread-private scratch (NOT shared memory, which is per block and
  // would be clobbered by the other threads of this block).
  double sStacks[kMaxPlayersGpu];
  double memo[1 << kMaxPlayersGpu];
  double subsetSum[1 << kMaxPlayersGpu];

  // Compact the positive stacks to the front; remember the source seats.
  int origIdx[kMaxPlayersGpu];
  int m = 0;
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    double s = static_cast<double>(stacks[item * n + i]);
    if (s > 0.0) {
      sStacks[m] = s;
      origIdx[m] = i;
      total += s;
      ++m;
    }
  }
  if (m == 0) {
    double sum = 0.0;
    for (int p = 0; p < numPayouts; ++p) sum += payouts[p];
    for (int i = 0; i < n; ++i) out[item * n + i] = sum / n;
    return;
  }

  const int full = 1 << m;
  // subsetSum[0] must be set before the fill loop: S=1 reads S&(S-1)=0.
  subsetSum[0] = 0.0;
  memo[0] = 1.0;
  for (int S = 1; S < full; ++S) {
    int low = __ffs(S) - 1;
    subsetSum[S] = subsetSum[S & (S - 1)] + sStacks[low];
    memo[S] = -1.0;
  }

  const int topPayouts = m < numPayouts ? m : numPayouts;
  double eq[kMaxPlayersGpu];
  for (int k = 0; k < m; ++k) eq[k] = 0.0;
  for (int place = 1; place <= m; ++place) {
    double pay = place <= topPayouts ? payouts[place - 1] : 0.0;
    if (pay == 0.0) continue;
    for (int k = 0; k < m; ++k) {
      double p = 0.0;
      for (uint32_t S = 0; S < static_cast<uint32_t>(full); ++S) {
        if (S & (1u << k)) continue;
        if (__popc(S) != place - 1) continue;
        if (place == 1) {
          p += sStacks[k] / total;
        } else {
          double a = subsetProbGpu(S, sStacks, total, memo, subsetSum);
          p += a * sStacks[k] / (total - subsetSum[S]);
        }
      }
      eq[k] += pay * p;
    }
  }

  if (m == n) {
    for (int k = 0; k < m; ++k) out[item * n + k] = eq[k];
  } else {
    // Zero-stack players split the remaining tail payouts evenly.
    double tail = 0.0;
    for (int j = m; j < numPayouts; ++j) tail += payouts[j];
    for (int k = 0; k < m; ++k) out[item * n + origIdx[k]] = eq[k];
    for (int i = 0; i < n; ++i) {
      if (stacks[item * n + i] <= 0) out[item * n + i] = tail / (n - m);
    }
  }
}

// ---------------------------------------------------------------------------
// Board sampling: each thread samples one 5-card board from the remaining
// deck given the dealt hole cards, and writes the 5 cards out. Used to
// parallelize showdown / continuation resolution.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t xorshift64(uint64_t& s) {
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  return s * 0x2545F4914F6CDD1DULL;
}

// holes: [batch][16] (2 cards per seat, up to 8 seats); boards out:
// [batch][5]. Cards are 0..51; undealt hole slots should repeat a card that
// is already present (duplicate slots are deduplicated against the seen
// set, so fills must be valid existing cards).
__global__ void boardBatchKernel(const uint8_t* __restrict__ holes, int batch,
                                 uint64_t seed, uint8_t* __restrict__ boards) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= batch) return;
  uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (item + 1));
  bool dealt[52];
  for (int c = 0; c < 52; ++c) dealt[c] = false;
  for (int i = 0; i < 16; ++i) {
    uint8_t c = holes[item * 16 + i];
    if (c < 52) dealt[c] = true;
  }
  uint8_t avail[52];
  int nAvail = 0;
  for (int c = 0; c < 52; ++c) {
    if (!dealt[c]) avail[nAvail++] = static_cast<uint8_t>(c);
  }
  // Partial Fisher-Yates over 5 cards.
  for (int i = 0; i < 5 && i < nAvail; ++i) {
    uint64_t r = xorshift64(s);
    int j = i + static_cast<int>(r % (nAvail - i));
    uint8_t tmp = avail[i];
    avail[i] = avail[j];
    avail[j] = tmp;
  }
  for (int i = 0; i < 5; ++i) {
    boards[item * 5 + i] = (i < nAvail) ? avail[i] : 0;
  }
}

// Host-side launch helpers --------------------------------------------------

inline cudaError_t launchIcmBatch(const int64_t* dStacks, int n, int batch,
                                  const double* dPayouts, int numPayouts,
                                  double* dOut, cudaStream_t stream = 0) {
  // The kernel keeps its per-thread DP scratch (memo/subsetSum, up to
  // (1<<n) doubles each) on the device stack; the 1KB default limit is
  // too small and causes illegal memory accesses. Set once per process.
  static bool stackLimitSet = [] {
    cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024);
    return true;
  }();
  (void)stackLimitSet;
  int threads = 128;
  int blocks = (batch + threads - 1) / threads;
  icmBatchKernel<<<blocks, threads, 0, stream>>>(dStacks, n, batch, dPayouts,
                                                numPayouts, dOut);
  return cudaGetLastError();
}

inline cudaError_t launchBoardBatch(const uint8_t* dHoles, int batch,
                                    uint64_t seed, uint8_t* dBoards,
                                    cudaStream_t stream = 0) {
  int threads = 128;
  int blocks = (batch + threads - 1) / threads;
  boardBatchKernel<<<blocks, threads, 0, stream>>>(dHoles, batch, seed, dBoards);
  return cudaGetLastError();
}

}  // namespace pps::cuda
