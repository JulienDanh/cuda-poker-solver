// On-hardware validation for cuda/poker_kernels.cu.
// Compares the batched ICM kernel against src/icm.cpp and the board
// sampling kernel against its output contract. Not wired into CMakeLists;
// build manually with nvcc (see docs/STATUS.md "CUDA" or README):
//   nvcc -std=c++17 -O2 -arch=sm_89 -Isrc -I. -o build/cuda_validate \
//       tools/cuda_validate.cu src/icm.cpp
#include <cuda/poker_kernels.cu>

#include "icm.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using pps::cuda::launchBoardBatch;
using pps::cuda::launchIcmBatch;

static int failures = 0;
#define CHECK(cond, msg)                                                \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("FAIL: %s\n", msg);                                  \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

#define CUDA_CHECK(call)                                                    \
  do {                                                                     \
    cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) {                                               \
      std::printf("FAIL: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
                  __FILE__, __LINE__);                                    \
      ++failures;                                                          \
      return failures;                                                     \
    }                                                                      \
  } while (0)

static uint64_t rngState = 0x123456789ABCDEFULL;
static uint64_t hostRand() {
  rngState ^= rngState >> 12;
  rngState ^= rngState << 25;
  rngState ^= rngState >> 27;
  return rngState * 0x2545F4914F6CDD1DULL;
}

int main() {
  // ---- device sanity -----------------------------------------------------
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int runtimeVer = 0, driverVer = 0;
  cudaRuntimeGetVersion(&runtimeVer);
  cudaDriverGetVersion(&driverVer);
  std::printf("device %d: %s (sm_%d%d, %.0f MiB, cuda runtime %d.%d, "
              "driver %d.%d)\n",
              dev, prop.name, prop.major, prop.minor,
              prop.totalGlobalMem / (1024.0 * 1024.0), runtimeVer / 1000,
              runtimeVer % 1000 / 10, driverVer / 1000, driverVer % 1000 / 10);

  // ---- ICM cross-check vs CPU reference -----------------------------------
  const double payoutsArr[3] = {0.5, 0.3, 0.2};
  const int numPayouts = 3;
  double* dPayouts = nullptr;
  CUDA_CHECK(cudaMalloc(&dPayouts, numPayouts * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dPayouts, payoutsArr, numPayouts * sizeof(double),
                        cudaMemcpyHostToDevice));

  for (int n : {2, 3, 4, 6, 8}) {
    const int batch = 4096;
    std::vector<int64_t> hStacks(batch * n);
    std::mt19937_64 mt(1234 + n);
    for (int i = 0; i < batch * n; ++i) {
      hStacks[i] = 1 + static_cast<int64_t>(mt() % 5000000);
      if (mt() % 10 == 0) hStacks[i] = 0;  // busted seats (ICM tail case)
    }

    int64_t* dStacks = nullptr;
    double* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dStacks, hStacks.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dOut, batch * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dStacks, hStacks.data(),
                          hStacks.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    cudaError_t le = launchIcmBatch(dStacks, n, batch, dPayouts, numPayouts, dOut);
    cudaEventRecord(t1);
    if (le != cudaSuccess) {
      std::printf("FAIL: icmBatchKernel launch (n=%d): %s\n", n,
                  cudaGetErrorString(le));
      ++failures;
      continue;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    std::vector<double> gpu(batch * n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, gpu.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double maxDiff = 0.0;
    double maxSumErr = 0.0;
    for (int item = 0; item < batch; ++item) {
      std::vector<int64_t> sv(hStacks.begin() + item * n,
                              hStacks.begin() + (item + 1) * n);
      auto cpu = pps::icmEquities(sv, {payoutsArr, payoutsArr + numPayouts});
      double sum = 0.0;
      for (int i = 0; i < n; ++i) {
        double d = std::fabs(cpu[i] - gpu[item * n + i]);
        if (d > maxDiff) maxDiff = d;
        sum += gpu[item * n + i];
      }
      maxSumErr = std::max(maxSumErr, std::fabs(sum - 1.0));
    }
    bool ok = maxDiff < 1e-9 &&
              (maxSumErr < 1e-9 || n < numPayouts);
    std::printf("ICM n=%d batch=%d: max |gpu-cpu| = %.3e, max |sum-1| = %.3e, "
                "kernel %.3f ms -> %s\n",
                n, batch, maxDiff, maxSumErr, ms, ok ? "OK" : "FAIL");
    CHECK(maxDiff < 1e-9, "ICM kernel mismatch vs CPU reference");
    // With more payouts than players (n < numPayouts) the un-awardable tail
    // payouts are dropped, so equities sum to < 1 — same as the CPU reference.
    if (n >= numPayouts)
      CHECK(maxSumErr < 1e-9, "ICM kernel equities do not sum to 1");

    cudaFree(dStacks);
    cudaFree(dOut);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }

  // ---- ICM zero-stack case (busted players split the tail) ---------------
  {
    const int n = 4, batch = 1;
    std::vector<int64_t> hStacks = {0, 1000, 5000, 0};
    int64_t* dStacks = nullptr;
    double* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dStacks, n * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dStacks, hStacks.data(), n * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    launchIcmBatch(dStacks, n, batch, dPayouts, numPayouts, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> gpu(n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    auto cpu = pps::icmEquities(hStacks, {payoutsArr, payoutsArr + numPayouts});
    std::printf("zero-stack probe n=4 {0,1000,5000,0}:\n");
    double maxDiff = 0.0;
    for (int i = 0; i < n; ++i) {
      std::printf("  seat %d: gpu %.6f  cpu %.6f\n", i, gpu[i], cpu[i]);
      maxDiff = std::max(maxDiff, std::fabs(gpu[i] - cpu[i]));
    }
    CHECK(maxDiff < 1e-12, "zero-stack ICM mismatch vs CPU normalization");
    cudaFree(dStacks);
    cudaFree(dOut);
  }

  // ---- board sampling: contract + chi-square ------------------------------
  {
    const int batch = 65536;
    std::vector<uint8_t> holes(batch * 16, 255);
    // 2 to 6 seats with distinct cards; unused slots repeat seat 0's cards.
    for (int item = 0; item < batch; ++item) {
      int seats = 2 + static_cast<int>(hostRand() % 5);
      int seen[52] = {0};
      int placed = 0;
      for (int s = 0; s < seats; ++s) {
        for (int k = 0; k < 2; ++k) {
          int c;
          do {
            c = static_cast<int>(hostRand() % 52);
          } while (seen[c]);
          seen[c] = 1;
          holes[item * 16 + 2 * s + k] = static_cast<uint8_t>(c);
          ++placed;
        }
      }
      for (int i = placed; i < 16; ++i) holes[item * 16 + i] = holes[item * 16];
    }
    uint8_t* dHoles = nullptr;
    uint8_t* dBoards = nullptr;
    CUDA_CHECK(cudaMalloc(&dHoles, holes.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dBoards, batch * 5 * sizeof(uint8_t)));
    CUDA_CHECK(
        cudaMemcpy(dHoles, holes.data(), holes.size(), cudaMemcpyHostToDevice));
    cudaError_t le = launchBoardBatch(dHoles, batch, 0xBEEFCAFEULL, dBoards);
    if (le != cudaSuccess) {
      std::printf("FAIL: boardBatchKernel launch: %s\n",
                  cudaGetErrorString(le));
      return ++failures;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> boards(batch * 5);
    CUDA_CHECK(cudaMemcpy(boards.data(), dBoards, boards.size(),
                          cudaMemcpyDeviceToHost));

    // Contract: 5 distinct cards per board, none colliding with dealt cards.
    long long contractViolations = 0;
    for (int item = 0; item < batch; ++item) {
      bool dealt[52] = {false};
      for (int i = 0; i < 16; ++i) {
        uint8_t c = holes[item * 16 + i];
        if (c < 52) dealt[c] = true;
      }
      bool seenBoard[52] = {false};
      for (int i = 0; i < 5; ++i) {
        uint8_t c = boards[item * 5 + i];
        if (c >= 52 || seenBoard[c] || dealt[c]) {
          ++contractViolations;
          break;
        }
        seenBoard[c] = true;
      }
    }
    std::printf("board kernel batch=%d: contract violations = %lld\n", batch,
                contractViolations);
    CHECK(contractViolations == 0, "board sampling violated its contract");

    // Uniformity of single-card marginals over the 48 unseen cards.
    // Fixed deal: 2 seats, cards 0..3; every board card must come from 4..51.
    const int fixedBatch = 65536;
    std::vector<uint8_t> fixedHoles(fixedBatch * 16, 0);
    for (int item = 0; item < fixedBatch; ++item) {
      fixedHoles[item * 16 + 0] = 0;
      fixedHoles[item * 16 + 1] = 1;
      fixedHoles[item * 16 + 2] = 2;
      fixedHoles[item * 16 + 3] = 3;
      for (int i = 4; i < 16; ++i) fixedHoles[item * 16 + i] = 0;
    }
    CUDA_CHECK(cudaMemcpy(dHoles, fixedHoles.data(), fixedHoles.size(),
                          cudaMemcpyHostToDevice));
    uint8_t* dBoards2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dBoards2, fixedBatch * 5));
    launchBoardBatch(dHoles, fixedBatch, 0xDEADBEEFULL, dBoards2);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> boards2(fixedBatch * 5);
    CUDA_CHECK(cudaMemcpy(boards2.data(), dBoards2, boards2.size(),
                          cudaMemcpyDeviceToHost));
    int obs[52] = {0};
    for (int i = 0; i < fixedBatch * 5; ++i) ++obs[boards2[i]];
    // 52 - 4 dealt = 48 unseen cards; E[chi2] ~ 48 * (1 - 5/48) ~ 43.
    double expected = 5.0 * fixedBatch / 48.0;
    double chi2 = 0.0;
    for (int c = 4; c < 52; ++c)
      chi2 += (obs[c] - expected) * (obs[c] - expected) / expected;
    double massOutside = 0;
    for (int c = 0; c < 4; ++c) massOutside += obs[c];
    std::printf("board uniformity: chi2=%.2f (df~47, E~43), expected count "
                "~%.1f; samples on dealt cards: %d\n",
                chi2, expected, static_cast<int>(massOutside));
    CHECK(chi2 < 120.0, "board card marginals fail chi-square (p<1e-6)");
    CHECK(massOutside == 0, "board sampled a dealt card");

    cudaFree(dHoles);
    cudaFree(dBoards);
    cudaFree(dBoards2);
  }

  // ---- throughput demo -----------------------------------------------------
  {
    const int n = 8, batch = 1 << 20;
    std::vector<int64_t> hStacks(batch * n);
    for (auto& s : hStacks)
      s = 1 + static_cast<int64_t>(hostRand() % 5000000);
    int64_t* dStacks = nullptr;
    double* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dStacks, hStacks.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&dOut, batch * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dStacks, hStacks.data(),
                          hStacks.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    launchIcmBatch(dStacks, n, batch, dPayouts, numPayouts, dOut);
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gms = 0.0f;
    cudaEventElapsedTime(&gms, t0, t1);

    auto c0 = std::chrono::steady_clock::now();
    const int cpuBatch = 4096;  // CPU reference is far slower; extrapolate
    for (int item = 0; item < cpuBatch; ++item) {
      std::vector<int64_t> sv(hStacks.begin() + item * n,
                              hStacks.begin() + (item + 1) * n);
      auto eq = pps::icmEquities(sv, {payoutsArr, payoutsArr + numPayouts});
      (void)eq;
    }
    auto c1 = std::chrono::steady_clock::now();
    double cpuPerItem =
        std::chrono::duration<double, std::milli>(c1 - c0).count() / cpuBatch;
    std::printf("throughput: GPU %d ICM items in %.2f ms (%.1f M items/s); "
                "CPU %.3f ms/item (extrapolated %.0fx speedup)\n",
                batch, gms, batch / (gms / 1000.0) / 1e6, cpuPerItem,
                cpuPerItem * batch / gms);
    cudaFree(dStacks);
    cudaFree(dOut);
  }

  cudaFree(dPayouts);
  if (failures) {
    std::printf("%d FAILURES\n", failures);
    return 1;
  }
  std::printf("ALL GPU VALIDATIONS PASSED\n");
  return 0;
}
