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
// out: [batch][n] equity.
__global__ void icmBatchKernel(const int64_t* __restrict__ stacks, int n,
                               int batch, const double* __restrict__ payouts,
                               int numPayouts, double* __restrict__ out) {
  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= batch) return;
  extern __shared__ double smem[];  // see dynamic size computation on the host
  double* sStacks = smem;                            // n
  double* memo = smem + kMaxPlayersGpu;              // 1 << n
  double* subsetSum = memo + (1 << n);               // 1 << n

  const int full = 1 << n;
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    sStacks[i] = static_cast<double>(stacks[item * n + i]);
    total += sStacks[i];
  }
  for (int S = 1; S < full; ++S) {
    int low = __ffs(S) - 1;
    subsetSum[S] = subsetSum[S & (S - 1)] + sStacks[low];
    memo[S] = -1.0;
  }
  subsetSum[0] = 0.0;
  memo[0] = 1.0;

  for (int i = 0; i < n; ++i) {
    double eq = 0.0;
    for (int place = 1; place <= n; ++place) {
      double pay = place <= numPayouts ? payouts[place - 1] : 0.0;
      if (pay == 0.0) continue;
      double p = 0.0;
      for (uint32_t S = 0; S < static_cast<uint32_t>(full); ++S) {
        if (S & (1u << i)) continue;
        if (__popc(S) != place - 1) continue;
        if (place == 1) {
          p += sStacks[i] / total;
        } else {
          double a = subsetProbGpu(S, sStacks, total, memo, subsetSum);
          p += a * sStacks[i] / (total - subsetSum[S]);
        }
      }
      eq += pay * p;
    }
    out[item * n + i] = eq;
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
  int threads = 128;
  int blocks = (batch + threads - 1) / threads;
  size_t smem = (kMaxPlayersGpu + 2 * (1u << n)) * sizeof(double);
  icmBatchKernel<<<blocks, threads, smem, stream>>>(dStacks, n, batch, dPayouts,
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
