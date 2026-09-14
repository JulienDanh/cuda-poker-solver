// GPU-CFR engine: multi-street range-based postflop solver compiled to
// static dataflow + CUDA graph replay. See cuda/gpu_cfr.h.
//
// Structure mirrors the river-only engine it replaces: depth-level
// batched passes (forward reach, fused backward cfv per depth), one CUDA
// graph per iteration, device-resident discount schedule. New for
// multi-street:
//   - per-node compact combo lists (chance children drop blocked combos)
//   - CHANCE nodes: forward masks reach through each board branch,
//     backward expands per-combo via a CSR over (branch, child slot)
//   - per-river-board strength tables (sorted orders, less/tie
//     boundaries, per-card position lists for the conflict corrections)
//   - terminal kernels read the compact combo lists directly through
//     per-board reverse maps (showdown) and a per-node identical-combo
//     map (fold); no base-space staging array
#include "gpu_cfr.h"

#include "cards.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

namespace pps {
namespace gpu {

namespace {

constexpr int kThreads = 128;
constexpr int kMaxNa = 32;

#define GPU_CHECK(call)                                                    \
  do {                                                                     \
    cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                   cudaGetErrorString(e_), __FILE__, __LINE__);            \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

enum Mode : int { kModeCFR = 0, kModeEV = 1, kModeBR = 2 };
enum Kind : uint8_t { kDecide = 0, kShowdown = 1, kFold = 2, kChance = 3 };

// ---------------------------------------------------------------------------
// Block primitives (f32; one node per block — see bwdDepthK).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warpInclScan(float v, int lane) {
  for (int off = 1; off < 32; off <<= 1) {
    float u = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += u;
  }
  return v;
}

// Out-of-place inclusive block scan: dst[k] = sum_{i<=k} src[i]. The
// showdown keeps src (raw sorted reach) for the card-correction reads,
// so scanning into a second array saves a separate copy pass + sync.
__device__ void blockScanIncTo(const float* src, float* dst, int n) {
  const int tid = threadIdx.x;
  const int nthr = blockDim.x;
  const int chunk = (n + nthr - 1) / nthr;
  const int lo = tid * chunk;
  const int hi = lo + chunk < n ? lo + chunk : n;
  float local = 0.0f;
  for (int k = lo; k < hi; ++k) local += src[k];
  __shared__ float warpSum[32];
  const int warp = tid >> 5;
  const int lane = tid & 31;
  float incl = warpInclScan(local, lane);
  if (lane == 31) warpSum[warp] = incl;
  __syncthreads();
  const int nWarps = (nthr + 31) >> 5;
  if (warp == 0) {
    float w = lane < nWarps ? warpSum[lane] : 0.0f;
    w = warpInclScan(w, lane);
    if (lane < nWarps) warpSum[lane] = w;
  }
  __syncthreads();
  float pre = warp > 0 ? warpSum[warp - 1] : 0.0f;
  float run = pre + incl - local;
  for (int k = lo; k < hi; ++k) {
    run += src[k];
    dst[k] = run;
  }
}

__device__ __forceinline__ float blockReduceSum(float v) {
  const int lane = threadIdx.x & 31;
  for (int off = 16; off > 0; off >>= 1) {
    float u = __shfl_down_sync(0xffffffffu, v, off);
    v += u;
  }
  __shared__ float warpSum[32];
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warpSum[warp] = v;
  __syncthreads();
  const int nWarps = (blockDim.x + 31) >> 5;
  if (threadIdx.x < nWarps) {
    const unsigned mask = (nWarps == 32) ? 0xffffffffu : ((1u << nWarps) - 1u);
    v = warpSum[threadIdx.x];
    for (int off = (nWarps >> 1); off > 0; off >>= 1) {
      float u = __shfl_down_sync(mask, v, off);
      v += u;
    }
    if (threadIdx.x == 0) warpSum[0] = v;
  }
  __syncthreads();
  return warpSum[0];
}

// ---------------------------------------------------------------------------
// Device tree state (fixed after compile).
// ---------------------------------------------------------------------------

// All strategy/regret/reach/cfv buffers are f32: consumer GPUs (RTX
// 4070) execute fp64 at 1/64 the fp32 rate, and the parity oracle
// (postflop-solver) is itself an f32 solver, so this matches its
// precision class. Host-side walks keep double accumulation.
struct TreeBuf {
  const float* regret;
  float* regretRW;
  const float* strat;
  float* stratRW;
  const int* regOff;
  const int* strOff;
  const int* na;
  const int* player;
  const int* foldBy;
  const int* childFlat;
  const int* childBase;
  const int* kind;
  const int* boardId;   // showdown: river-board table index
  const int* nC0;       // node combo counts (player 0)
  const int* nC1;
  const int* comboOff0;  // offsets into comboId
  const int* comboOff1;
  const int* comboId;    // base slots (per side) per node combo
  const int* reachOff;  // per node slot offset in the traverser's
                         // reach layout (aliased; see the compile)
  const int* cfvOff;    // per node dense slot offset in cfv (never
                         // aliased: children of a pass-through node
                         // still produce distinct cfv rows)
  const int64_t* sc0;
  const int64_t* sc1;
  const int64_t* potBase;
  const float* reach;
  float* reachRW;
  const float* cfv;
  float* cfvRW;
  // chance
  const int* nBranch;
  const int* chanceQ;  // per pair: 52 - nBoard - 4 valid branches
  const int64_t* pc0;  // inherited prior-street contributions; all values
  const int64_t* pc1;  // are measured from the root (hand-start) baseline
  const int* runOffOff;  // per node: offset into runOff
  const int* runOff;     // per (node, p, branch) childToParent run starts
  const int* ctap;       // child slot -> parent slot
  const int* ctapBase;    // per node: this node's ctap entries' start
  const int* expOffOff;  // per node: offset into expOff
  const int* expOff;     // per (node, p): CSR offsets over parent slots
  const int* expIdx;     // packed (branch * 4096 + childSlot)
  const int* expIdxBase;  // per node: this node's entries' start in expIdx
  // base cards / identity map (base space)
  const uint8_t* cards0;
  const uint8_t* cards1;
  const int* sameOther0;  // base slot side0 -> base slot side1 (or -1)
  const int* sameOther1;
  // per-river-board strength tables
  const int* brOppSortedOff;
  const int* brOppSorted;
  const int* brLessOff;  // per table: start in brLessEnd
  const int* brLessEnd;
  const int* brTieEnd;
  const int* brCardPosOff;  // per table: 53 offsets into brCardPosIdx
  const int* brCardPosIdx;
  // per (board, traverser): sorted position -> node-local combo index of
  // that base slot, or -1 when the runout blocked the combo. Lets the
  // showdown gather raw reach straight from the compact list.
  const int* brRevSortedOff;
  const int* brRevSorted;
  // per fold node: for each traverser combo, the node-local index of
  // the identical opponent combo in this node's opponent list (-1 if
  // the side has no such combo at all).
  const int* foldSonOff;
  const int* foldSon;
  int n0Base, n1Base;
  int64_t pot;
};

__device__ __forceinline__ int childOf(const TreeBuf& b, int node, int a) {
  return b.childFlat[b.childBase[node] + a];
}

__device__ __forceinline__ int nOfP(const TreeBuf& b, int node, int p) {
  return p == 0 ? b.nC0[node] : b.nC1[node];
}

__device__ __forceinline__ int comboOffP(const TreeBuf& b, int node, int p) {
  return p == 0 ? b.comboOff0[node] : b.comboOff1[node];
}

// Advances the iteration counter at the end of a graph replay; the CFR
// half-steps read their discount coefficients from a pre-uploaded
// schedule via this counter, so replays never touch host memory.
__global__ void advanceCounterK(int* counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) ++(*counter);
}

// ---------------------------------------------------------------------------
// Forward passes.
// ---------------------------------------------------------------------------

// DECIDE nodes at one depth (children share the parent's combo lists).
__device__ __forceinline__ void fwdDecideBlock(int node, int c, int tr,
                                               int sigSrc, TreeBuf b) {
  const int opp = 1 - tr;
  const int nOppNode = nOfP(b, node, opp);
  if (c >= nOppNode) return;
  const int p = b.player[node];
  // Pass-through (traverser acts): every child's reach row is identical
  // to this node's row, so the layout ALIASES it (the children's
  // reachOff[tr] points here) — nothing to write. The root's row is
  // initialized at compile from the range weights (w_opp into layout tr),
  // so the plain row read covers node 0 too.
  if (p == tr) return;
  const int na = b.na[node];
  const float r = b.reach[(size_t)b.reachOff[node] + c];
  const float* src = sigSrc == 0 ? b.regret : b.strat;
  const int* off = sigSrc == 0 ? b.regOff : b.strOff;
  const size_t base = (size_t)off[node];
  float acc = 0.0f;
  for (int a = 0; a < na; ++a) {
    float x = src[base + (size_t)a * nOppNode + c];
    if (sigSrc == 0) {
      if (x > 0.0f) acc += x;
    } else {
      acc += x;
    }
  }
  const float uniform = 1.0f / na;
  for (int a = 0; a < na; ++a) {
    float x = src[base + (size_t)a * nOppNode + c];
    float s;
    if (sigSrc == 0)
      s = acc > 0.0f ? (x > 0.0f ? x / acc : 0.0f) : uniform;
    else
      s = acc > 0.0f ? x / acc : uniform;
    const int ch = childOf(b, node, a);
    b.reachRW[(size_t)b.reachOff[ch] + c] = r * s;
  }
}

// One branch of a CHANCE node: reach masked and compacted through the
// board branch (opponent combo space).
__device__ __forceinline__ void fwdChanceBlock(int node, int br, int tr,
                                               TreeBuf b) {
  const int nb = b.nBranch[node];
  if (br >= nb) return;
  const int p = 1 - tr;
  const int runBase = b.runOffOff[node] + p * (nb + 1);
  const int start = b.runOff[runBase + br];
  const int end = b.runOff[runBase + br + 1];
  const int child = childOf(b, node, br);
  // The chance weight conditions on each pair's own hole cards: a pair
  // cannot see the 4 cards it holds, so the branch probability is
  // 1 / (52 - nBoard - 4), not 1 / nBranch (mirrors postflop-solver's
  // chance_factor).
  const float inv = 1.0f / b.chanceQ[node];
  // The node's ctap section is player-major (p=0's runs, then p=1's); the
  // run offsets are relative within the player's section. Add the p=0
  // section size when reading p=1.
  const int p0Total = b.runOff[b.runOffOff[node] + nb];
  const int ctapIdxBase =
      b.ctapBase[node] + (p == 0 ? 0 : p0Total);
  const int nChild = end - start;
  for (int j = threadIdx.x; j < nChild; j += blockDim.x) {
    const int parentSlot = b.ctap[ctapIdxBase + start + j];
    b.reachRW[(size_t)b.reachOff[child] + j] =
        b.reach[(size_t)b.reachOff[node] + parentSlot] * inv;
  }
}

// Whole forward level: decide nodes and chance nodes at one depth in a
// single flat launch (they are independent — both read depth-d reach
// and write disjoint children at d+1), halving the forward graph
// nodes. Blocks below nD*gy are decide (node = flat/gy, one combo per
// thread); the rest are chance (node, branch) pairs located by binary
// search over the per-depth branch prefix table.
__global__ void fwdLevelK(const int* __restrict__ dlist, int nD,
                          const int* __restrict__ clist, int nC, int gy,
                          const int* __restrict__ cBrOff, int tr, int sigSrc,
                          TreeBuf b) {
  const int flat = blockIdx.x;
  if (flat < nD * gy) {
    const int node = dlist[flat / gy];
    const int c = (flat % gy) * kThreads + threadIdx.x;
    fwdDecideBlock(node, c, tr, sigSrc, b);
    return;
  }
  const int f2 = flat - nD * gy;
  int lo = 0, hi = nC;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (cBrOff[mid] <= f2)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo <= 0 || lo > nC) return;
  const int ci = lo - 1;
  fwdChanceBlock(clist[ci], f2 - cBrOff[ci], tr, b);
}

// ---------------------------------------------------------------------------
// Backward node handlers (one block per node, dispatched on kind).
// ---------------------------------------------------------------------------

__device__ void showdownNode(int node, int tr, TreeBuf b,
                             float* shRaw, float* shScan) {
  const int opp = 1 - tr;
  const int nOppBase = opp == 0 ? b.n0Base : b.n1Base;
  const int tIdx = b.boardId[node] * 2 + tr;
  const int* oppSorted = b.brOppSorted + b.brOppSortedOff[tIdx];
  const int* rev = b.brRevSorted + b.brRevSortedOff[tIdx];
  const int lessOff = b.brLessOff[tIdx];
  const int* lessEnd = b.brLessEnd + lessOff;
  const int* tieEnd = b.brTieEnd + lessOff;
  const int* cardPosOff = b.brCardPosOff + (size_t)tIdx * 53;
  const int* cardPosIdx = b.brCardPosIdx;
  const float* r = b.reach + (size_t)b.reachOff[node];
  // Fused zero/scatter/gather: raw opponent reach in strength-sorted
  // order, one pass straight from the compact list. Combos blocked by
  // the runout read 0 through the per-board reverse map; no base-space
  // staging array, no separate zero pass.
  for (int k = threadIdx.x; k < nOppBase; k += blockDim.x) {
    const int j = rev[k];
    shRaw[k] = j >= 0 ? r[j] : 0.0f;
  }
  __syncthreads();
  blockScanIncTo(shRaw, shScan, nOppBase);
  __syncthreads();
  const float total = nOppBase > 0 ? shScan[nOppBase - 1] : 0.0f;
  const int64_t sc0 = b.sc0[node];
  const int64_t sc1 = b.sc1[node];
  const int64_t u = sc0 > sc1 ? sc0 - sc1 : sc1 - sc0;
  const float totalPot = (float)(b.potBase[node] + sc0 + sc1 - u);
  const float mySc =
      (float)((tr == 0 ? sc0 : sc1) + (tr == 0 ? b.pc0[node] : b.pc1[node]));
  const float winV = totalPot - mySc;
  const float tieV = totalPot * 0.5f - mySc;
  const float loseV = -mySc;
  const uint8_t* cardsTr = tr == 0 ? b.cards0 : b.cards1;
  const uint8_t* cardsOpp = opp == 0 ? b.cards0 : b.cards1;
  const int nTrNode = nOfP(b, node, tr);
  const int trComboOff = comboOffP(b, node, tr);
  float* out = b.cfvRW + (size_t)b.cfvOff[node];
  for (int i = threadIdx.x; i < nTrNode; i += blockDim.x) {
    const int baseId = b.comboId[trComboOff + i];
    const int le = lessEnd[baseId];
    const int te = tieEnd[baseId];
    const uint8_t c1 = cardsTr[2 * baseId];
    const uint8_t c2 = cardsTr[2 * baseId + 1];
    // Lose mass: opponents sharing a card, sorted position < le.
    float lm = le > 0 ? shScan[le - 1] : 0.0f;
    for (int ci = 0; ci < 2; ++ci) {
      const uint8_t c = ci == 0 ? c1 : c2;
      for (int q = cardPosOff[c]; q < cardPosOff[c + 1]; ++q) {
        const int pos = cardPosIdx[q];
        if (pos < le) lm -= shRaw[pos];
      }
    }
    if (lm < 0.0f) lm = 0.0f;
    // Win mass: opponents sharing a card, sorted position >= te.
    float gm = te > 0 ? total - shScan[te - 1] : total;
    for (int ci = 0; ci < 2; ++ci) {
      const uint8_t c = ci == 0 ? c1 : c2;
      for (int q = cardPosOff[c]; q < cardPosOff[c + 1]; ++q) {
        const int pos = cardPosIdx[q];
        if (pos >= te) gm -= shRaw[pos];
      }
    }
    if (gm < 0.0f) gm = 0.0f;
    // Tie band minus shared-card ties.
    float tm = te > le ? shScan[te - 1] - (le > 0 ? shScan[le - 1] : 0.0f)
                       : 0.0f;
    for (int pos = le; pos < te; ++pos) {
      const int o = oppSorted[pos];
      const uint8_t oc1 = cardsOpp[2 * o];
      const uint8_t oc2 = cardsOpp[2 * o + 1];
      if (oc1 == c1 || oc1 == c2 || oc2 == c1 || oc2 == c2) tm -= shRaw[pos];
    }
    out[i] = lm * winV + tm * tieV + gm * loseV;
  }
}

__device__ void foldNode(int node, int tr, TreeBuf b) {
  const int opp = 1 - tr;
  const int nOppNode = nOfP(b, node, opp);
  const int comboOff = comboOffP(b, node, opp);
  const uint8_t* cardsOpp = opp == 0 ? b.cards0 : b.cards1;
  const float* r = b.reach + (size_t)b.reachOff[node];
  __shared__ float cardSum[52];
  // Per-card sums straight from the compact list: each combo scatters
  // its reach to its two cards (shared atomics). No base-space staging:
  // blocked combos never enter, and the identical-combo correction
  // reads the same reach row through the per-node foldSon map.
  if (threadIdx.x < 52) cardSum[threadIdx.x] = 0.0f;
  __syncthreads();
  float tot = 0.0f;
  for (int j = threadIdx.x; j < nOppNode; j += blockDim.x) {
    const float rj = r[j];
    tot += rj;
    const int baseId = b.comboId[comboOff + j];
    atomicAdd(&cardSum[cardsOpp[2 * baseId]], rj);
    atomicAdd(&cardSum[cardsOpp[2 * baseId + 1]], rj);
  }
  __syncthreads();
  tot = blockReduceSum(tot);
  __syncthreads();
  const int fb = b.foldBy[node];
  const int64_t sc = fb == 0 ? b.sc0[node] : b.sc1[node];
  const float v =
      tr == fb
          ? -((float)sc + (float)(fb == 0 ? b.pc0[node] : b.pc1[node]))
          : ((float)(b.potBase[node] + sc) -
             (float)(tr == 0 ? b.pc0[node] : b.pc1[node]));
  const uint8_t* cardsTr = tr == 0 ? b.cards0 : b.cards1;
  const int nTrNode = nOfP(b, node, tr);
  const int* son =
      b.foldSon + b.foldSonOff[node] + (tr == 0 ? 0 : nOfP(b, node, 0));
  const int trComboOff = comboOffP(b, node, tr);
  float* out = b.cfvRW + (size_t)b.cfvOff[node];
  for (int i = threadIdx.x; i < nTrNode; i += blockDim.x) {
    const int baseId = b.comboId[trComboOff + i];
    float w = tot - cardSum[cardsTr[2 * baseId]] -
              cardSum[cardsTr[2 * baseId + 1]];
    const int j2 = son[i];
    if (j2 >= 0) w += r[j2];  // identical combo subtracted twice
    if (w < 0.0f) w = 0.0f;
    out[i] = w * v;
  }
}

__device__ void decideNode(int node, int tr, int mode,
                          const float* __restrict__ coefTab,
                          const int* __restrict__ counter, TreeBuf b) {
  const int nTr = nOfP(b, node, tr);
  const int p = b.player[node];
  const int na = b.na[node];
  const float* src = mode == kModeCFR ? b.regret : b.strat;
  const int* off = mode == kModeCFR ? b.regOff : b.strOff;
  const size_t base = (size_t)off[node];
  const float uniform = 1.0f / na;
  for (int c = threadIdx.x; c < nTr; c += blockDim.x) {
    if (p != tr) {
      float v = 0.0f;
      for (int a = 0; a < na; ++a) {
        const int ch = childOf(b, node, a);
        v += b.cfv[(size_t)b.cfvOff[ch] + c];
      }
      b.cfvRW[(size_t)b.cfvOff[node] + c] = v;
      continue;
    }
    float val = 0.0f;
    float sig[kMaxNa];
    if (mode != kModeBR) {
      float acc = 0.0f;
      for (int a = 0; a < na; ++a) {
        float x = src[base + (size_t)a * nTr + c];
        sig[a] = x;
        if (mode == kModeCFR) {
          if (x > 0.0f) acc += x;
        } else {
          acc += x;
        }
      }
      for (int a = 0; a < na; ++a) {
        float x = sig[a];
        float s;
        if (mode == kModeCFR)
          s = acc > 0.0f ? (x > 0.0f ? x / acc : 0.0f) : uniform;
        else
          s = acc > 0.0f ? x / acc : uniform;
        sig[a] = s;
        const int ch = childOf(b, node, a);
        val += s * b.cfv[(size_t)b.cfvOff[ch] + c];
      }
    } else {
      float best = -1e30f;
      for (int a = 0; a < na; ++a) {
        const int ch = childOf(b, node, a);
        float v = b.cfv[(size_t)b.cfvOff[ch] + c];
        if (v > best) best = v;
      }
      val = best;
    }
    b.cfvRW[(size_t)b.cfvOff[node] + c] = val;
    if (mode != kModeCFR) continue;
    const float* dCoef = coefTab + (size_t)(*counter) * 3;
    float* rreg = b.regretRW + base;
    float* sst = b.stratRW + base;
    const float posC = dCoef[0], negC = dCoef[1], avgC = dCoef[2];
    for (int a = 0; a < na; ++a) {
      const int ch = childOf(b, node, a);
      const float av = b.cfv[(size_t)b.cfvOff[ch] + c];
      const float r0 = rreg[(size_t)a * nTr + c];
      sst[(size_t)a * nTr + c] = sst[(size_t)a * nTr + c] * avgC + sig[a];
      rreg[(size_t)a * nTr + c] =
          rreg[(size_t)a * nTr + c] * (r0 >= 0.0f ? posC : negC) + (av - val);
    }
  }
}

// Backward through a CHANCE node: per parent (traverser) combo, sum the
// child cfvs over the branches that combo survives, scaled 1/nBranch.
__device__ void chanceNodeBwd(int node, int tr, TreeBuf b) {
  const int nb = b.nBranch[node];
  const int nTr = nOfP(b, node, tr);
  // The two players' CSR sections have different lengths (nC0+1 and
  // nC1+1); player 1's starts after player 0's, not after nTr+1.
  const int expBase =
      b.expOffOff[node] + (tr == 0 ? 0 : (nOfP(b, node, 0) + 1));
  // expOff holds per-player relative offsets; expIdx is one global array,
  // so add this node's (and, for player 1, player 0's) absolute base.
  const int p0Total = b.expOff[b.expOffOff[node] + nOfP(b, node, 0)];
  const int idxBase =
      b.expIdxBase[node] + (tr == 0 ? 0 : p0Total);
  float* out = b.cfvRW + (size_t)b.cfvOff[node];
  for (int i = threadIdx.x; i < nTr; i += blockDim.x) {
    const int start = b.expOff[expBase + i];
    const int end = b.expOff[expBase + i + 1];
    float v = 0.0f;
    for (int q = idxBase + start; q < idxBase + end; ++q) {
      const int packed = b.expIdx[q];
      const int br = packed / 4096;
      const int js = packed % 4096;
      const int child = childOf(b, node, br);
      v += b.cfv[(size_t)b.cfvOff[child] + js];
    }
    out[i] = v;
  }
}

// kindMask: profiling attribution only — blocks whose node kind is not
// in the mask return early (-1 = all kinds). Production launches pass
// -1; the graph never contains masked launches.
__global__ void bwdDepthK(const int* __restrict__ nodes, int count, int tr,
                         int mode, const float* __restrict__ coefTab,
                         const int* __restrict__ counter, int kindMask,
                         TreeBuf b) {
  if (blockIdx.x >= count) return;
  const int node = nodes[blockIdx.x];
  const int kind = b.kind[node];
  if (kindMask >= 0 && ((1 << kind) & kindMask) == 0) return;
  const int nOppBase = (1 - tr) == 0 ? b.n0Base : b.n1Base;
  extern __shared__ float sh[];
  if (kind == kShowdown) {
    showdownNode(node, tr, b, sh, sh + nOppBase);
  } else if (kind == kFold) {
    foldNode(node, tr, b);
  } else if (kind == kChance) {
    chanceNodeBwd(node, tr, b);
  } else {
    decideNode(node, tr, mode, coefTab, counter, b);
  }
}

// ---------------------------------------------------------------------------
// Host side: schedule + hand keys.
// ---------------------------------------------------------------------------

void discountCoefs(int t, int iters, const std::string& algo, double out[3]) {
  double posCoef, negCoef, avgCoef;
  if (algo == "hs30") {
    double n = static_cast<double>(iters);
    double tt = static_cast<double>(t + 1);
    double alpha = 1.0 + 3.0 * tt / n;
    double beta = -1.0 - 2.0 * tt / n;
    double gamma = 30.0 - 5.0 * tt / n;
    double pa = std::pow(tt, alpha);
    double pb = std::pow(tt, beta);
    posCoef = pa / (pa + 1.0);
    negCoef = pb / (pb + 1.0);
    avgCoef = std::pow(tt / (tt + 1.0), gamma);
  } else {
    double ta = std::max(t - 1, 0);
    double pa = ta * std::sqrt(ta);
    uint32_t k = 0;
    if (t > 0) {
      uint32_t x = static_cast<uint32_t>(t);
      k = 1u << ((31 - __builtin_clz(x)) & ~1u);
    }
    double tg = static_cast<double>(t - static_cast<int>(k));
    posCoef = pa / (pa + 1.0);
    negCoef = 0.5;
    avgCoef = std::pow(tg / (tg + 1.0), 3.0);
  }
  out[0] = posCoef;
  out[1] = negCoef;
  out[2] = avgCoef;
}

uint64_t handKey(const HandValue& v) {
  uint64_t key = v.category;
  for (int i = 0; i < 5; ++i)
    key = key * 13 + static_cast<uint64_t>(v.tiebreak[i]);
  return key;
}

}  // namespace

struct GpuPostflopSolver::Impl {
  int numNodes = 0, maxDepth = 0;
  int n0Base = 0, n1Base = 0, maxNa = 1;
  int spotNBoard = 0;
  size_t reachTotal = 0, regTotal = 0;  // cfv / regret row totals
  size_t reachTotal0 = 0, reachTotal1 = 0;  // reach layouts
  int64_t pot = 0;
  bool empty = false;

  int* dNa = nullptr;
  int* dPlayer = nullptr;
  int* dFoldBy = nullptr;
  int* dChildFlat = nullptr;
  int* dChildBase = nullptr;
  int* dKind = nullptr;
  int* dBoardId = nullptr;
  int* dNC0 = nullptr;
  int* dNC1 = nullptr;
  int* dComboOff0 = nullptr;
  int* dComboOff1 = nullptr;
  int* dComboId = nullptr;
  int* dReachOff0 = nullptr;
  int* dReachOff1 = nullptr;
  int* dCfvOff = nullptr;
  int64_t* dSc0 = nullptr;
  int64_t* dSc1 = nullptr;
  int64_t* dPotBase = nullptr;
  float* dReach0 = nullptr;
  float* dReach1 = nullptr;
  float* dCfv = nullptr;
  float* dRegret = nullptr;
  float* dStrat = nullptr;
  int* dRegOff = nullptr;
  int* dStrOff = nullptr;
  float* dW0 = nullptr;
  float* dW1 = nullptr;
  uint8_t* dCards0 = nullptr;
  uint8_t* dCards1 = nullptr;
  int* dSameOther0 = nullptr;
  int* dSameOther1 = nullptr;
  int* dNBranch = nullptr;
  int* dChanceQ = nullptr;
  int64_t* dPc0 = nullptr;
  int64_t* dPc1 = nullptr;
  int* dRunOffOff = nullptr;
  int* dRunOff = nullptr;
  int* dCtapBase = nullptr;
  int* dCtap = nullptr;
  int* dExpOffOff = nullptr;
  int* dExpIdxBase = nullptr;
  int* dExpOff = nullptr;
  int* dExpIdx = nullptr;
  int* dBrOppSortedOff = nullptr;
  int* dBrOppSorted = nullptr;
  int* dBrLessOff = nullptr;
  int* dBrLessEnd = nullptr;
  int* dBrTieEnd = nullptr;
  int* dBrCardPosOff = nullptr;
  int* dBrCardPosIdx = nullptr;
  int* dBrRevSortedOff = nullptr;
  int* dBrRevSorted = nullptr;
  int* dFoldSonOff = nullptr;
  int* dFoldSon = nullptr;

  float* dCoefTab = nullptr;
  float* dCoefDummy = nullptr;
  int* dCounter = nullptr;
  int* dCounterDummy = nullptr;
  bool capturing = false;

  std::vector<int*> dDecideList, dChanceList, dBwdList, dChanceBrOff;
  std::vector<int> decideCount, chanceCount, bwdCount, chanceBrTotal;
  // per depth: [decide, showdown, fold, chance] node counts (profiling)
  std::vector<std::array<int, 4>> depthKinds;

  // host copies for stats
  std::vector<double> w0, w1;
  // decide-node table for the strategy seeder and nodeStrategy():
  // (regOff, na, player, nC, depth, nBoard)
  struct NodeMeta {
    int regOff, na, player, nC, depth, nBoard;
  };
  std::vector<NodeMeta> decideMeta;
  std::vector<uint8_t> baseCards0, baseCards1;
  std::vector<int> sameOther0;
  std::vector<int> rootCombos0;
  int rootNa = 0, rootRegOff = 0;
  std::vector<float> rootStratRows;

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  TreeBuf buf() const {
    TreeBuf b{};
    b.regret = dRegret;
    b.regretRW = dRegret;
    b.strat = dStrat;
    b.stratRW = dStrat;
    b.regOff = dRegOff;
    b.strOff = dStrOff;
    b.na = dNa;
    b.player = dPlayer;
    b.foldBy = dFoldBy;
    b.childFlat = dChildFlat;
    b.childBase = dChildBase;
    b.kind = dKind;
    b.boardId = dBoardId;
    b.nC0 = dNC0;
    b.nC1 = dNC1;
    b.comboOff0 = dComboOff0;
    b.comboOff1 = dComboOff1;
    b.comboId = dComboId;
    b.sc0 = dSc0;
    b.sc1 = dSc1;
    b.potBase = dPotBase;
    // Defaults (layout 0); halfStep re-points reach/reachRW/reachOff at
    // the traverser's layout before launching.
    b.reachOff = dReachOff0;
    b.cfvOff = dCfvOff;
    b.reach = dReach0;
    b.reachRW = dReach0;
    b.cfv = dCfv;
    b.cfvRW = dCfv;
    b.nBranch = dNBranch;
    b.chanceQ = dChanceQ;
    b.pc0 = dPc0;
    b.pc1 = dPc1;
    b.runOffOff = dRunOffOff;
    b.runOff = dRunOff;
    b.ctapBase = dCtapBase;
    b.ctap = dCtap;
    b.expOffOff = dExpOffOff;
    b.expIdxBase = dExpIdxBase;
    b.expOff = dExpOff;
    b.expIdx = dExpIdx;
    b.cards0 = dCards0;
    b.cards1 = dCards1;
    b.sameOther0 = dSameOther0;
    b.sameOther1 = dSameOther1;
    b.brOppSortedOff = dBrOppSortedOff;
    b.brOppSorted = dBrOppSorted;
    b.brLessOff = dBrLessOff;
    b.brLessEnd = dBrLessEnd;
    b.brTieEnd = dBrTieEnd;
    b.brCardPosOff = dBrCardPosOff;
    b.brCardPosIdx = dBrCardPosIdx;
    b.brRevSortedOff = dBrRevSortedOff;
    b.brRevSorted = dBrRevSorted;
    b.foldSonOff = dFoldSonOff;
    b.foldSon = dFoldSon;
    b.n0Base = n0Base;
    b.n1Base = n1Base;
    b.pot = pot;
    return b;
  }

  void halfStep(int tr, int mode, int sigSrc, cudaStream_t s) {
    TreeBuf b = buf();
    // Reach rows live in per-traverser layouts (pass-through aliasing).
    b.reachOff = tr == 0 ? dReachOff0 : dReachOff1;
    b.reach = tr == 0 ? dReach0 : dReach1;
    b.reachRW = tr == 0 ? dReach0 : dReach1;
    const float* coefTab = mode == kModeCFR ? dCoefTab : dCoefDummy;
    const int* counter = mode == kModeCFR ? dCounter : dCounterDummy;
    const int maxBase = std::max(n0Base, n1Base);
    const size_t smem = (size_t)2 * maxBase * sizeof(float);
    const int gy = (maxBase + kThreads - 1) / kThreads;
    // Forward: one flat launch per depth covering both node kinds (they
    // are independent; both read depth-d reach and write disjoint
    // children at d+1). Depth levels stay ordered: a depth d+1 decide
    // node reads the chance writes from depth d.
    for (int d = 0; d <= maxDepth; ++d) {
      const int nD = decideCount[d], nC = chanceCount[d];
      if (nD == 0 && nC == 0) continue;
      const dim3 g(nD * gy + chanceBrTotal[d]);
      fwdLevelK<<<g, kThreads, 0, s>>>(dDecideList[d], nD, dChanceList[d],
                                      nC, gy, dChanceBrOff[d], tr, sigSrc,
                                      b);
      GPU_CHECK(cudaGetLastError());
    }
    for (int d = maxDepth; d >= 0; --d) {
      if (bwdCount[d] > 0) {
        bwdDepthK<<<bwdCount[d], kThreads, smem, s>>>(
            dBwdList[d], bwdCount[d], tr, mode, coefTab, counter, -1, b);
        GPU_CHECK(cudaGetLastError());
      }
    }
  }

  // Disjoint weighted pair mass over the base lists (the EV normalizer).
  double pairMass() const {
    double Z = 0.0;
    double total1 = 0.0;
    double cardSum[52] = {0.0};
    for (int j = 0; j < n1Base; ++j) {
      total1 += w1[j];
      cardSum[baseCards1[2 * j]] += w1[j];
      cardSum[baseCards1[2 * j + 1]] += w1[j];
    }
    for (int i = 0; i < n0Base; ++i) {
      double w = total1 - cardSum[baseCards0[2 * i]] -
                 cardSum[baseCards0[2 * i + 1]];
      int so = sameOther0[i];
      if (so >= 0) w += w1[so];
      if (w > 0.0) Z += w0[i] * w;
    }
    return Z;
  }

  // Range-weighted root value for traverser `tr` in the given walk mode
  // (kModeEV: average strategy; kModeBR: best response), divided by the
  // disjoint pair mass. Walks only read the regret/strategy rows and
  // scratch, so they are safe to interleave with CFR graph replays.
  double rootValue(int tr, int mode) {
    cudaStream_t s = stream;
    halfStep(tr, mode, 1, s);
    GPU_CHECK(cudaStreamSynchronize(s));
    const int nBase = tr == 0 ? n0Base : n1Base;
    std::vector<float> cfv(nBase);
    GPU_CHECK(cudaMemcpy(cfv.data(), dCfv, (size_t)nBase * sizeof(float),
                         cudaMemcpyDeviceToHost));
    // Host accumulation in f64 over the f32 rows.
    double v = 0.0;
    for (int i = 0; i < nBase; ++i)
      v += (double)(tr == 0 ? w0[i] : w1[i]) * (double)cfv[i];
    const double Z = pairMass();
    if (Z <= 0.0) return 0.0;
    if (std::getenv("GPU_CFR_DEBUG") && std::getenv("GPU_CFR_SEED_STRAT") &&
        tr == 0) {
      std::fprintf(stderr, "SEEDROOTCFV0:");
      for (int i = 0; i < n0Base; ++i)
        std::fprintf(stderr, " %.2f", cfv[i]);
      std::fprintf(stderr, "\n");
    }
    return v / Z;
  }

  // Current exploitability in chips (two BR walks).
  double explNow() {
    const double V0 = rootValue(0, kModeBR);
    const double V1 = rootValue(1, kModeBR);
    return (V0 + V1 - (double)pot) / 2.0;
  }

  void free() {
    auto kill = [](auto& p) {
      if (p) {
        cudaFree(p);
        p = nullptr;
      }
    };
    kill(dNa);
    kill(dPlayer);
    kill(dFoldBy);
    kill(dChildFlat);
    kill(dChildBase);
    kill(dKind);
    kill(dBoardId);
    kill(dNC0);
    kill(dNC1);
    kill(dComboOff0);
    kill(dComboOff1);
    kill(dComboId);
    kill(dReachOff0);
    kill(dReachOff1);
    kill(dCfvOff);
    kill(dSc0);
    kill(dSc1);
    kill(dPotBase);
    kill(dReach0);
    kill(dReach1);
    kill(dCfv);
    kill(dRegret);
    kill(dStrat);
    kill(dRegOff);
    kill(dStrOff);
    kill(dW0);
    kill(dW1);
    kill(dCards0);
    kill(dCards1);
    kill(dSameOther0);
    kill(dSameOther1);
    kill(dNBranch);
    kill(dChanceQ);
    kill(dPc0);
    kill(dPc1);
    kill(dRunOffOff);
    kill(dRunOff);
    kill(dCtapBase);
    kill(dCtap);
    kill(dExpOffOff);
    kill(dExpIdxBase);
    kill(dExpOff);
    kill(dExpIdx);
    kill(dBrOppSortedOff);
    kill(dBrOppSorted);
    kill(dBrLessOff);
    kill(dBrLessEnd);
    kill(dBrTieEnd);
    kill(dBrCardPosOff);
    kill(dBrCardPosIdx);
    kill(dBrRevSortedOff);
    kill(dBrRevSorted);
    kill(dFoldSonOff);
    kill(dFoldSon);
    kill(dCoefTab);
    kill(dCoefDummy);
    kill(dCounter);
    kill(dCounterDummy);
    for (auto* p : dDecideList) cudaFree(p);
    for (auto* p : dChanceList) cudaFree(p);
    for (auto* p : dBwdList) cudaFree(p);
    for (auto* p : dChanceBrOff) cudaFree(p);
    dDecideList.clear();
    dChanceList.clear();
    dBwdList.clear();
    dChanceBrOff.clear();
    if (exec) {
      cudaGraphExecDestroy(exec);
      exec = nullptr;
    }
    if (graph) {
      cudaGraphDestroy(graph);
      graph = nullptr;
    }
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

// ---------------------------------------------------------------------------
// Host tree compilation.
// ---------------------------------------------------------------------------

namespace {

struct NodeH {
  uint8_t kind = kDecide;
  int player = 0, foldBy = 0, na = 0, childBase = 0;
  int64_t sc0 = 0, sc1 = 0, potBase = 0;
  int depth = 0;
  int boardId = 0;
  uint8_t board[5] = {0, 0, 0, 0, 0};
  int nBoard = 0;
  int comboOff0 = 0, nC0 = 0;
  int comboOff1 = 0, nC1 = 0;
  int runOffOff = 0, expOffOff = 0, expIdxBase = 0;
  int ctapBase = 0;  // this node's entries' start in the global ctap
  int64_t pc0 = 0, pc1 = 0;  // inherited (prior-street) contributions
  int nBranch = 0;
  std::vector<int> children;
};

struct SideH {
  int n = 0;
  std::vector<uint8_t> cards;
  std::vector<double> w;
  std::vector<int> sameOther;
};

struct Compiler {
  const PostflopSpot& spot;
  const pf::BetConfig& cfg;
  std::deque<NodeH> nodes;  // stable references across recursion
  SideH sides[2];
  std::vector<int> comboId;
  std::vector<int> childFlat;
  std::vector<int> runOff;
  std::vector<int> ctap;
  std::vector<int> expOff;
  std::vector<int> expIdx;
  std::map<std::array<uint8_t, 5>, int> boardIds;
  int maxNa = 1;

  Compiler(const PostflopSpot& s, const pf::BetConfig& c) : spot(s), cfg(c) {
    for (int p = 0; p < 2; ++p) {
      const Range& r = p == 0 ? spot.oop : spot.ip;
      SideH& sd = sides[p];
      for (int i = 0; i < kCombos; ++i) {
        uint8_t a, b2;
        comboCards(i, a, b2);
        bool blocked = false;
        for (int c = 0; c < spot.nBoard; ++c) {
          if (a == spot.board[c] || b2 == spot.board[c]) blocked = true;
        }
        if (blocked || r.w[i] <= 0.0) continue;
        sd.cards.push_back(a);
        sd.cards.push_back(b2);
        sd.w.push_back(r.w[i]);
        ++sd.n;
      }
    }
    for (int p = 0; p < 2; ++p) {
      const SideH& s0 = sides[p];
      const SideH& s1 = sides[1 - p];
      std::map<uint16_t, int> idx;
      for (int j = 0; j < s1.n; ++j) {
        idx[static_cast<uint16_t>((uint16_t)s1.cards[2 * j] << 8) |
            s1.cards[2 * j + 1]] = j;
      }
      sides[p].sameOther.assign(s0.n, -1);
      for (int i = 0; i < s0.n; ++i) {
        auto it = idx.find(static_cast<uint16_t>(
            (uint16_t)s0.cards[2 * i] << 8) | s0.cards[2 * i + 1]);
        if (it != idx.end()) sides[p].sameOther[i] = it->second;
      }
    }
  }

  bool sharesCardP(int p, int baseId, uint8_t c) const {
    const SideH& s = sides[p];
    return s.cards[2 * baseId] == c || s.cards[2 * baseId + 1] == c;
  }

  int newNode() {
    nodes.emplace_back();
    return (int)nodes.size() - 1;
  }

  int internBoard(const uint8_t board[5]) {
    std::array<uint8_t, 5> key = {board[0], board[1], board[2], board[3],
                                  board[4]};
    auto it = boardIds.find(key);
    if (it != boardIds.end()) return it->second;
    int id = (int)boardIds.size();
    boardIds[key] = id;
    return id;
  }

  // Builds a node. streetClosed: betting for the current street is done
  // (call or check-through); all-in runouts stay closed through chance.
  int build(const uint8_t board[5], int nBoard, int64_t potBase, int64_t sc0,
            int64_t sc1, int actor, bool afterAllin, bool streetClosed,
            int depth, int comboOff0, int nC0, int comboOff1, int nC1,
            int64_t pc0 = 0, int64_t pc1 = 0, int numBets = 0) {
    const int idx = newNode();
    NodeH& nd = nodes[idx];
    nd.depth = depth;
    nd.potBase = potBase;
    nd.sc0 = sc0;
    nd.sc1 = sc1;
    nd.nBoard = nBoard;
    for (int i = 0; i < nBoard; ++i) nd.board[i] = board[i];
    nd.comboOff0 = comboOff0;
    nd.nC0 = nC0;
    nd.comboOff1 = comboOff1;
    nd.nC1 = nC1;
    nd.pc0 = pc0;
    nd.pc1 = pc1;

    // Nothing behind (prior streets put in the full stack): no betting is
    // possible; the street is dead and the runout deals through chance to
    // showdown. Without this, betActions would emit AllIn(0) actions that
    // never equalize contributions and recurse forever.
    if (!streetClosed && spot.stack - pc0 <= 0) streetClosed = true;

    if (streetClosed) {
      if (nBoard == 5) {
        nd.kind = kShowdown;
        nd.boardId = internBoard(board);
        return idx;
      }
      nd.kind = kChance;
      const bool allin = (sc0 == spot.stack && sc1 == spot.stack);
      // Branch cards: everything not on the board.
      std::vector<int> branchCards;
      for (int c = 0; c < 52; ++c) {
        bool onBoard = false;
        for (int i = 0; i < nBoard; ++i)
          if (board[i] == c) onBoard = true;
        if (!onBoard) branchCards.push_back(c);
      }
      const int nb = (int)branchCards.size();
      nd.nBranch = nb;
      nd.runOffOff = (int)runOff.size();
      runOff.resize(runOff.size() + 2 * (nb + 1), 0);
      // Build children first (they append their own map sections; ours are
      // reserved above and filled below).
      std::vector<std::vector<int>> childLists[2];
      uint8_t nboard[5];
      for (int i = 0; i < nBoard; ++i) nboard[i] = board[i];
      const int64_t childPot = potBase + sc0 + sc1;
      (void)allin;
      for (int br = 0; br < nb; ++br) {
        const int c = branchCards[br];
        nboard[nBoard] = (uint8_t)c;
        // Child combo lists: parent lists minus combos containing c.
        std::vector<int> l0, l1;
        for (int s = 0; s < nC0; ++s) {
          const int base = comboId[comboOff0 + s];
          if (!sharesCardP(0, base, (uint8_t)c)) l0.push_back(base);
        }
        for (int s = 0; s < nC1; ++s) {
          const int base = comboId[comboOff1 + s];
          if (!sharesCardP(1, base, (uint8_t)c)) l1.push_back(base);
        }
        const int off0 = (int)comboId.size();
        for (int v : l0) comboId.push_back(v);
        const int off1 = (int)comboId.size();
        for (int v : l1) comboId.push_back(v);
        childLists[0].push_back(l0);
        childLists[1].push_back(l1);
        nd.children.push_back(build(nboard, nBoard + 1, childPot, 0, 0,
                                     0, false, false, depth + 1, off0,
                                     (int)l0.size(), off1, (int)l1.size(),
                                     pc0 + sc0, pc1 + sc1));
      }
      // childToParent runs (forward reach; per player). The node's ctap
      // section is contiguous (children do not push ctap), so record the
      // base right before the fill.
      nd.ctapBase = (int)ctap.size();
      for (int p = 0; p < 2; ++p) {
        const int parentOff = p == 0 ? comboOff0 : comboOff1;
        const int parentN = p == 0 ? nC0 : nC1;
        const int runBase = nd.runOffOff + p * (nb + 1);
        int total = 0;
        for (int br = 0; br < nb; ++br) {
          runOff[runBase + br] = total;
          const std::vector<int>& cl = childLists[p][br];
          for (int j = 0; j < (int)cl.size(); ++j) {
            int pslot = -1;
            for (int s = 0; s < parentN; ++s) {
              if (comboId[parentOff + s] == cl[j]) {
                pslot = s;
                break;
              }
            }
            ctap.push_back(pslot);
            ++total;
          }
        }
        runOff[runBase + nb] = total;
      }
      // Expand CSR (backward cfv; per player): parent slot -> entries.
      // The section offset can only be captured now, after the children's
      // own sections have been appended (they interleave with ours).
      nd.expOffOff = (int)expOff.size();
      nd.expIdxBase = (int)expIdx.size();
      for (int p = 0; p < 2; ++p) {
        const int parentN = p == 0 ? nC0 : nC1;
        const int parentOff = p == 0 ? comboOff0 : comboOff1;
        std::vector<std::vector<std::pair<int, int>>> perSlot(parentN);
        for (int br = 0; br < nb; ++br) {
          const std::vector<int>& cl = childLists[p][br];
          for (int j = 0; j < (int)cl.size(); ++j) {
            int pslot = -1;
            for (int s = 0; s < parentN; ++s) {
              if (comboId[parentOff + s] == cl[j]) {
                pslot = s;
                break;
              }
            }
            perSlot[pslot].push_back({br, j});
          }
        }
        int cnt = 0;
        for (int s = 0; s < parentN; ++s) {
          expOff.push_back(cnt);
          for (auto& pr : perSlot[s]) {
            expIdx.push_back(pr.first * 4096 + pr.second);
            ++cnt;
          }
        }
        expOff.push_back(cnt);
      }
      return idx;
    }

    // DECIDE node: betting on the current street. The behind-stack
    // shrinks by the prior streets' matched contributions (pc0 == pc1 at
    // every decide node; they advance only at street close). The total
    // bet per player is capped at spot.stack across all streets,
    // matching the oracle's per-player BuildTreeInfo stacks.
    auto acts = pf::betActions(potBase, spot.stack - pc0, cfg, sc0, sc1,
                               actor, afterAllin, numBets,
                               6 - nBoard);
    nd.na = (int)acts.size();
    maxNa = std::max(maxNa, nd.na);
    nd.kind = kDecide;
    nd.player = actor;
    for (int a = 0; a < nd.na; ++a) {
      const auto& act = acts[a];
      int child = -1;
      int64_t nsc0 = sc0, nsc1 = sc1;
      bool nAfterAllin = afterAllin;
      bool closed = false;
      switch (act.kind) {
        case pf::ActionKind::Check:
          if (actor == 0) {
            child = build(board, nBoard, potBase, sc0, sc1, 1, afterAllin,
                          false, depth + 1, comboOff0, nC0, comboOff1, nC1,
                          pc0, pc1, numBets);
          } else {
            closed = true;  // IP checks behind: street over
          }
          break;
        case pf::ActionKind::Fold: {
          child = newNode();
          NodeH& f = nodes[child];
          f.kind = kFold;
          f.foldBy = actor;
          f.potBase = potBase;
          f.sc0 = sc0;
          f.sc1 = sc1;
          f.pc0 = pc0;
          f.pc1 = pc1;
          f.depth = depth + 1;
          f.nBoard = nBoard;
          for (int i = 0; i < nBoard; ++i) f.board[i] = board[i];
          f.comboOff0 = comboOff0;
          f.nC0 = nC0;
          f.comboOff1 = comboOff1;
          f.nC1 = nC1;
          break;
        }
        case pf::ActionKind::Call: {
          const int64_t target = std::max(sc0, sc1);
          if (actor == 0) nsc0 = target;
          else nsc1 = target;
          closed = true;
          break;
        }
        case pf::ActionKind::Bet:
        case pf::ActionKind::Raise:
        case pf::ActionKind::AllIn:
          if (act.kind == pf::ActionKind::AllIn) nAfterAllin = true;
          if (actor == 0) nsc0 = act.amount;
          else nsc1 = act.amount;
          break;
      }
      if (closed) {
        child = build(board, nBoard, potBase, nsc0, nsc1, 0, false, true,
                      depth + 1, comboOff0, nC0, comboOff1, nC1, pc0, pc1);
      } else if (child == -1) {
        const int nNumBets =
            act.kind == pf::ActionKind::Check ? numBets : numBets + 1;
        child = build(board, nBoard, potBase, nsc0, nsc1, actor ^ 1,
                      nAfterAllin, false, depth + 1, comboOff0, nC0,
                      comboOff1, nC1, pc0, pc1, nNumBets);
      }
      nd.children.push_back(child);
    }
    return idx;
  }
};

}  // namespace

GpuPostflopSolver::GpuPostflopSolver(const PostflopSpot& spot,
                                     const pf::BetConfig& cfg)
    : impl_(new Impl()) {
  Impl* I = impl_;
  I->pot = spot.pot;
  Compiler cc(spot, cfg);
  // Root lists: the base lists themselves.
  const int off0 = 0;
  for (int i = 0; i < cc.sides[0].n; ++i) cc.comboId.push_back(i);
  const int off1 = (int)cc.comboId.size();
  for (int i = 0; i < cc.sides[1].n; ++i) cc.comboId.push_back(i);
  const int n0 = cc.sides[0].n, n1 = cc.sides[1].n;
  I->n0Base = n0;
  I->n1Base = n1;
  I->spotNBoard = spot.nBoard;
  if (n0 == 0 || n1 == 0) {
    I->empty = true;
    return;
  }
  cc.build(spot.board, spot.nBoard, spot.pot, 0, 0, 0, false, false, 0, off0,
           n0, off1, n1);
  I->numNodes = (int)cc.nodes.size();
  numNodes_ = I->numNodes;
  I->maxNa = cc.maxNa;
  if (I->maxNa > kMaxNa) {
    std::fprintf(stderr,
                 "gpu_cfr: bet config too wide (max actions %d > %d)\n",
                 I->maxNa, kMaxNa);
    std::exit(1);
  }

  // Flatten the per-node children now that the tree is complete:
  // childBase must index each node's OWN slice, which is only possible
  // once no further interleaved pushes can happen.
  std::vector<int> parentOf(cc.nodes.size(), -1);
  for (int u = 0; u < (int)cc.nodes.size(); ++u) {
    NodeH& nd = cc.nodes[u];
    nd.childBase = (int)cc.childFlat.size();
    for (int ch : nd.children) {
      cc.childFlat.push_back(ch);
      parentOf[ch] = u;
    }
  }


  // Depths and depth lists.
  int D = 1;
  for (auto& nd : cc.nodes) D = std::max(D, nd.depth + 1);
  I->maxDepth = D - 1;
  maxDepth_ = I->maxDepth;
  std::vector<std::vector<int>> decideList(D), chanceList(D), bwdList(D);
  I->depthKinds.assign(D, {0, 0, 0, 0});
  for (int u = 0; u < I->numNodes; ++u) {
    const NodeH& nd = cc.nodes[u];
    bwdList[nd.depth].push_back(u);
    ++I->depthKinds[nd.depth][nd.kind];
    if (nd.kind == kDecide)
      decideList[nd.depth].push_back(u);
    else if (nd.kind == kChance)
      chanceList[nd.depth].push_back(u);
  }
  auto upload = [&](int*& dst, const std::vector<int>& v) {
    if (v.empty()) return;
    GPU_CHECK(cudaMalloc(&dst, v.size() * sizeof(int)));
    GPU_CHECK(cudaMemcpy(dst, v.data(), v.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
  };
  I->dDecideList.resize(D, nullptr);
  I->dChanceList.resize(D, nullptr);
  I->dBwdList.resize(D, nullptr);
  I->dChanceBrOff.resize(D, nullptr);
  I->decideCount.resize(D);
  I->chanceCount.resize(D);
  I->bwdCount.resize(D);
  I->chanceBrTotal.assign(D, 0);
  for (int d = 0; d < D; ++d) {
    I->decideCount[d] = (int)decideList[d].size();
    I->chanceCount[d] = (int)chanceList[d].size();
    I->bwdCount[d] = (int)bwdList[d].size();
    // Per-depth (chance node, branch) prefix table for the fused
    // forward launch: blocks past the decide section binary-search it.
    std::vector<int> brOff(1, 0);
    for (int u : chanceList[d]) brOff.push_back(brOff.back() +
                                                cc.nodes[u].nBranch);
    I->chanceBrTotal[d] = brOff.back();
    upload(I->dDecideList[d], decideList[d]);
    upload(I->dChanceList[d], chanceList[d]);
    upload(I->dBwdList[d], bwdList[d]);
    upload(I->dChanceBrOff[d], brOff);
  }

  // Flat node arrays.
  std::vector<int> vkind, vplayer, vfoldBy, vna, vchildBase, vnc0, vnc1,
      vcomboOff0, vcomboOff1, vreachOff, vregOff, vnBranch, vrunOffOff,
      vexpOffOff, vboardId, vchanceQ, vexpIdxBase, vctapBase;
  std::vector<int64_t> vsc0, vsc1, vpotBase, vpc0, vpc1;
  size_t reachTotal = 0;
  size_t regTotal = 0;
  int rootRegOff = 0;
  for (int u = 0; u < I->numNodes; ++u) {
    const NodeH& nd = cc.nodes[u];
    vkind.push_back(nd.kind);
    vplayer.push_back(nd.player);
    vfoldBy.push_back(nd.foldBy);
    vna.push_back(nd.na);
    vchildBase.push_back(nd.childBase);
    vboardId.push_back(nd.boardId);
    vnc0.push_back(nd.nC0);
    vnc1.push_back(nd.nC1);
    vcomboOff0.push_back(nd.comboOff0);
    vcomboOff1.push_back(nd.comboOff1);
    vsc0.push_back(nd.sc0);
    vsc1.push_back(nd.sc1);
    vpotBase.push_back(nd.potBase);
    vnBranch.push_back(nd.nBranch);
    vchanceQ.push_back(52 - nd.nBoard - 4);
    vpc0.push_back(nd.pc0);
    vpc1.push_back(nd.pc1);
    vrunOffOff.push_back(nd.runOffOff);
    vexpOffOff.push_back(nd.expOffOff);
    vexpIdxBase.push_back(nd.expIdxBase);
    vctapBase.push_back(nd.ctapBase);
    vreachOff.push_back((int)reachTotal);
    reachTotal += (size_t)std::max(nd.nC0, nd.nC1);  // cfv rows stay dense
    if (nd.kind == kDecide) {
      const int nTr = nd.player == 0 ? nd.nC0 : nd.nC1;
      vregOff.push_back((int)regTotal);
      regTotal += (size_t)nd.na * nTr;
      if (u == 0) rootRegOff = vregOff.back();
    } else {
      vregOff.push_back(0);
    }
  }
  // Per-traverser reach row layouts with pass-through aliasing: when a
  // decide node's player == tr, its forward pass would copy the row
  // unchanged to every child — so the children's rows ALIAS the node's
  // row and the copy disappears. Rows only change where the opponent
  // acts (sigma-masked) or a chance node compacts. The root row of
  // layout tr is initialized with the OPPONENT side's range weights.
  std::vector<int> vreachOff0(I->numNodes), vreachOff1(I->numNodes);
  for (int u = 0; u < I->numNodes; ++u) {
    const NodeH& nd = cc.nodes[u];
    const int par = parentOf[u];
    const size_t row = (size_t)std::max(nd.nC0, nd.nC1);
    const bool alias0 =
        par >= 0 && cc.nodes[par].kind == kDecide &&
        cc.nodes[par].player == 0;
    const bool alias1 =
        par >= 0 && cc.nodes[par].kind == kDecide &&
        cc.nodes[par].player == 1;
    if (alias0)
      vreachOff0[u] = vreachOff0[par];
    else {
      vreachOff0[u] = (int)I->reachTotal0;
      I->reachTotal0 += row;
    }
    if (alias1)
      vreachOff1[u] = vreachOff1[par];
    else {
      vreachOff1[u] = (int)I->reachTotal1;
      I->reachTotal1 += row;
    }
  }
  upload(I->dReachOff0, vreachOff0);
  upload(I->dReachOff1, vreachOff1);
  upload(I->dKind, vkind);
  upload(I->dPlayer, vplayer);
  upload(I->dFoldBy, vfoldBy);
  upload(I->dNa, vna);
  upload(I->dChildBase, vchildBase);
  upload(I->dBoardId, vboardId);
  upload(I->dNC0, vnc0);
  upload(I->dNC1, vnc1);
  upload(I->dComboOff0, vcomboOff0);
  upload(I->dComboOff1, vcomboOff1);
  upload(I->dCfvOff, vreachOff);
  upload(I->dRegOff, vregOff);
  upload(I->dStrOff, vregOff);  // identical row offsets
  upload(I->dNBranch, vnBranch);
  upload(I->dChanceQ, vchanceQ);
  GPU_CHECK(cudaMalloc(&I->dPc0, vpc0.size() * sizeof(int64_t)));
  GPU_CHECK(cudaMemcpy(I->dPc0, vpc0.data(), vpc0.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMalloc(&I->dPc1, vpc1.size() * sizeof(int64_t)));
  GPU_CHECK(cudaMemcpy(I->dPc1, vpc1.data(), vpc1.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  upload(I->dRunOffOff, vrunOffOff);
  upload(I->dExpOffOff, vexpOffOff);
  upload(I->dExpIdxBase, vexpIdxBase);
  upload(I->dCtapBase, vctapBase);
  GPU_CHECK(cudaMalloc(&I->dSc0, vsc0.size() * sizeof(int64_t)));
  GPU_CHECK(cudaMemcpy(I->dSc0, vsc0.data(), vsc0.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMalloc(&I->dSc1, vsc1.size() * sizeof(int64_t)));
  GPU_CHECK(cudaMemcpy(I->dSc1, vsc1.data(), vsc1.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMalloc(&I->dPotBase, vpotBase.size() * sizeof(int64_t)));
  GPU_CHECK(cudaMemcpy(I->dPotBase, vpotBase.data(),
                       vpotBase.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
  upload(I->dChildFlat, cc.childFlat);
  upload(I->dComboId, cc.comboId);
  upload(I->dRunOff, cc.runOff);
  upload(I->dCtap, cc.ctap);
  upload(I->dExpOff, cc.expOff);
  upload(I->dExpIdx, cc.expIdx);
  I->reachTotal = reachTotal;
  I->regTotal = regTotal;

  GPU_CHECK(cudaMalloc(&I->dRegret, regTotal * sizeof(float)));
  GPU_CHECK(cudaMalloc(&I->dStrat, regTotal * sizeof(float)));
  GPU_CHECK(cudaMemset(I->dRegret, 0, regTotal * sizeof(float)));
  GPU_CHECK(cudaMemset(I->dStrat, 0, regTotal * sizeof(float)));
  GPU_CHECK(cudaMalloc(&I->dCfv, reachTotal * sizeof(float)));
  GPU_CHECK(cudaMalloc(&I->dReach0, I->reachTotal0 * sizeof(float)));
  GPU_CHECK(cudaMalloc(&I->dReach1, I->reachTotal1 * sizeof(float)));

  // Range weights / cards / identity map (base space).
  auto uploadD = [&](float*& dst, const std::vector<double>& v) {
    GPU_CHECK(cudaMalloc(&dst, v.size() * sizeof(float)));
    std::vector<float> f(v.begin(), v.end());
    GPU_CHECK(cudaMemcpy(dst, f.data(), f.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
  };
  uploadD(I->dW0, cc.sides[0].w);
  uploadD(I->dW1, cc.sides[1].w);
  // Root rows of the reach layouts hold the OPPONENT's initial weights
  // (the forward pass reads reach, not w, even at node 0). Row 0 of
  // each layout is the root's (offset 0).
  GPU_CHECK(cudaMemcpy(I->dReach0, I->dW1, (size_t)n1 * sizeof(float),
                       cudaMemcpyDeviceToDevice));
  GPU_CHECK(cudaMemcpy(I->dReach1, I->dW0, (size_t)n0 * sizeof(float),
                       cudaMemcpyDeviceToDevice));
  GPU_CHECK(cudaMalloc(&I->dCards0, 2 * n0));
  GPU_CHECK(cudaMemcpy(I->dCards0, cc.sides[0].cards.data(), 2 * n0,
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMalloc(&I->dCards1, 2 * n1));
  GPU_CHECK(cudaMemcpy(I->dCards1, cc.sides[1].cards.data(), 2 * n1,
                       cudaMemcpyHostToDevice));
  upload(I->dSameOther0, cc.sides[0].sameOther);
  upload(I->dSameOther1, cc.sides[1].sameOther);

  // Per-river-board strength tables (one table per (board, traverser)).
  {
    const int nTables = (int)cc.boardIds.size() * 2;
    std::vector<int> tOppSortedOff(nTables + 1, 0);
    std::vector<int> tOppSorted;
    std::vector<int> tRevSortedOff(nTables + 1, 0);
    std::vector<int> tRevSorted;
    std::vector<int> tLessOff(nTables + 1, 0);
    std::vector<int> tLessEnd, tTieEnd;
    std::vector<int> tCardPosOff(nTables * 53 + 1, 0);
    std::vector<int> tCardPosIdx;
    // Iterate boards in ID order (the tables' offsets are stored by id;
    // the std::map iterates in key order, which differs once there is
    // more than one river board).
    std::vector<std::array<uint8_t, 5>> boardsById(cc.boardIds.size());
    for (const auto& kv : cc.boardIds) boardsById[kv.second] = kv.first;
    int tIdx = 0;
    for (const auto& key : boardsById) {
      const int boardId = cc.boardIds.at(key);
      uint8_t b5[5];
      for (int i = 0; i < 5; ++i) b5[i] = key[i];
      std::vector<uint64_t> st[2];
      for (int p = 0; p < 2; ++p) {
        st[p].assign(cc.sides[p].n, 0);
        for (int i = 0; i < cc.sides[p].n; ++i) {
          Card seven[7] = {cc.sides[p].cards[2 * i],
                           cc.sides[p].cards[2 * i + 1],
                           b5[0], b5[1], b5[2], b5[3], b5[4]};
          st[p][i] = handKey(evaluateN(seven, 7));
        }
      }
      for (int tr = 0; tr < 2; ++tr) {
        const int tIdx = boardId * 2 + tr;
        const int opp = 1 - tr;
        const int nTr = cc.sides[tr].n, nOpp = cc.sides[opp].n;
        std::vector<int> os(nOpp);
        for (int i = 0; i < nOpp; ++i) os[i] = i;
        std::stable_sort(os.begin(), os.end(), [&](int x, int y) {
          return st[opp][x] < st[opp][y];
        });
        tOppSortedOff[tIdx] = (int)tOppSorted.size();
        for (int i = 0; i < nOpp; ++i) tOppSorted.push_back(os[i]);
        // Reverse map (sorted position -> node-local combo index, or -1
        // for combos the runout blocked). Every showdown node with this
        // board carries exactly the canonical list — base minus combos
        // containing a card dealt after the spot board; verified after
        // the tree build — so one table per (board, traverser) serves
        // all showdown nodes and the kernel can gather raw reach
        // straight from each node's compact list.
        std::vector<int> jOf(nOpp, -1);
        {
          int j = 0;
          for (int i = 0; i < nOpp; ++i) {
            bool blocked = false;
            for (int c = spot.nBoard; c < 5; ++c)
              if (cc.sides[opp].cards[2 * i] == b5[c] ||
                  cc.sides[opp].cards[2 * i + 1] == b5[c])
                blocked = true;
            if (!blocked) jOf[i] = j++;
          }
        }
        tRevSortedOff[tIdx] = (int)tRevSorted.size();
        for (int k = 0; k < nOpp; ++k) tRevSorted.push_back(jOf[os[k]]);
        tLessOff[tIdx] = (int)tLessEnd.size();
        for (int i = 0; i < nTr; ++i) {
          uint64_t s = st[tr][i];
          int lo = 0, hi = nOpp;
          while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (st[opp][os[mid]] < s) lo = mid + 1;
            else hi = mid;
          }
          tLessEnd.push_back(lo);
          int lo2 = lo, hi2 = nOpp;
          while (lo2 < hi2) {
            int mid = (lo2 + hi2) / 2;
            if (st[opp][os[mid]] <= s) lo2 = mid + 1;
            else hi2 = mid;
          }
          tTieEnd.push_back(lo2);
        }
        const int cpo = tIdx * 53;
        // Per-table cardPos offsets are absolute into tCardPosIdx: seed
        // the table's first entry with the current global size (later
        // tables must not start at 0, or they read the earlier table's
        // positions).
        tCardPosOff[cpo] = (int)tCardPosIdx.size();
        for (int c = 0; c < 52; ++c) {
          tCardPosOff[cpo + c + 1] = tCardPosOff[cpo + c];
          for (int pos = 0; pos < nOpp; ++pos) {
            const int o = os[pos];
            if (cc.sides[opp].cards[2 * o] == c ||
                cc.sides[opp].cards[2 * o + 1] == c) {
              tCardPosIdx.push_back(pos);
              ++tCardPosOff[cpo + c + 1];
            }
          }
        }
      }
    }
    upload(I->dBrOppSortedOff, tOppSortedOff);
    upload(I->dBrOppSorted, tOppSorted);
    upload(I->dBrLessOff, tLessOff);
    upload(I->dBrLessEnd, tLessEnd);
    upload(I->dBrTieEnd, tTieEnd);
    upload(I->dBrCardPosOff, tCardPosOff);
    upload(I->dBrCardPosIdx, tCardPosIdx);
    upload(I->dBrRevSortedOff, tRevSortedOff);
    upload(I->dBrRevSorted, tRevSorted);
  }

  // The showdown gather routes reads through the per-board reverse map,
  // which assumes every showdown node's combo lists equal the canonical
  // lists for its interned board. Chance compaction is deterministic —
  // each deal only removes combos containing the dealt card, in base
  // order — so this always holds; verify once so a future tree change
  // cannot silently corrupt the showdown values.
  for (int u = 0; u < I->numNodes; ++u) {
    const NodeH& nd = cc.nodes[u];
    if (nd.kind != kShowdown) continue;
    for (int p = 0; p < 2; ++p) {
      const SideH& sd = cc.sides[p];
      const int off = p == 0 ? nd.comboOff0 : nd.comboOff1;
      const int n = p == 0 ? nd.nC0 : nd.nC1;
      int j = 0;
      bool ok = true;
      for (int i = 0; i < sd.n && ok; ++i) {
        bool blocked = false;
        for (int c = spot.nBoard; c < 5; ++c)
          if (sd.cards[2 * i] == nd.board[c] ||
              sd.cards[2 * i + 1] == nd.board[c])
            blocked = true;
        if (!blocked) {
          ok = j < n && cc.comboId[off + j] == i;
          ++j;
        }
      }
      if (!ok || j != n) {
        std::fprintf(stderr,
                     "gpu_cfr: showdown node %d combo list deviates from "
                     "its board's canonical list\n",
                     u);
        std::exit(1);
      }
    }
  }

  // Decide-node table for the strategy seeder.
  for (int u = 0; u < I->numNodes; ++u) {
    const NodeH& nd = cc.nodes[u];
    if (nd.kind != kDecide) continue;
    I->decideMeta.push_back({vregOff[u], nd.na, nd.player,
                             nd.player == 0 ? nd.nC0 : nd.nC1, nd.depth,
                             nd.nBoard});
  }

  // Per-fold-node identical-combo maps for foldNode: for each traverser
  // combo, the node-local index of the same two cards in this node's
  // opponent list (or -1). The lists are strictly increasing in base
  // slot, so the lookup is a binary search at compile time.
  {
    std::vector<int> vFoldSonOff(I->numNodes, 0), vFoldSon;
    for (int u = 0; u < I->numNodes; ++u) {
      const NodeH& nd = cc.nodes[u];
      if (nd.kind != kFold) continue;
      vFoldSonOff[u] = (int)vFoldSon.size();
      for (int p = 0; p < 2; ++p) {
        const int off = p == 0 ? nd.comboOff0 : nd.comboOff1;
        const int n = p == 0 ? nd.nC0 : nd.nC1;
        const int offOpp = p == 0 ? nd.comboOff1 : nd.comboOff0;
        const int nOppL = p == 0 ? nd.nC1 : nd.nC0;
        for (int i = 0; i < n; ++i) {
          const int baseId = cc.comboId[off + i];
          const int so = cc.sides[p].sameOther[baseId];
          int son = -1;
          if (so >= 0) {
            int lo = 0, hi = nOppL;
            while (lo < hi) {
              const int mid = (lo + hi) / 2;
              if (cc.comboId[offOpp + mid] < so)
                lo = mid + 1;
              else
                hi = mid;
            }
            if (lo < nOppL && cc.comboId[offOpp + lo] == so) son = lo;
          }
          vFoldSon.push_back(son);
        }
      }
    }
    upload(I->dFoldSonOff, vFoldSonOff);
    upload(I->dFoldSon, vFoldSon);
  }

  // Host copies for stats.
  I->w0 = cc.sides[0].w;
  I->w1 = cc.sides[1].w;
  I->baseCards0 = cc.sides[0].cards;
  I->baseCards1 = cc.sides[1].cards;
  I->sameOther0 = cc.sides[0].sameOther;
  const NodeH& root = cc.nodes[0];
  I->rootNa = root.na;
  I->rootRegOff = rootRegOff;
  I->rootCombos0.resize(root.nC0);
  for (int i = 0; i < root.nC0; ++i)
    I->rootCombos0[i] = cc.comboId[root.comboOff0 + i];
  I->rootStratRows.assign((size_t)I->rootNa * root.nC0, 0.0);

  GPU_CHECK(cudaMalloc(&I->dCoefDummy, 3 * sizeof(float)));
  GPU_CHECK(cudaMemset(I->dCoefDummy, 0, 3 * sizeof(float)));
  GPU_CHECK(cudaMalloc(&I->dCounterDummy, sizeof(int)));
  GPU_CHECK(cudaMemset(I->dCounterDummy, 0, sizeof(int)));
  GPU_CHECK(cudaStreamCreate(&I->stream));
}

GpuPostflopSolver::~GpuPostflopSolver() {
  if (impl_) {
    impl_->free();
    delete impl_;
  }
}

void GpuPostflopSolver::solve(int iterations, const std::string& algo,
                              double target) {
  Impl* I = impl_;
  iterationsRun_ = 0;
  if (I->empty) return;
  cudaStream_t s = I->stream;
  // The schedule (powers of t) is computed in double and stored f32;
  // coefficients are O(1), so the cast is lossless to ~1e-7.
  std::vector<double> tab((size_t)iterations * 3);
  for (int t = 0; t < iterations; ++t)
    discountCoefs(t, iterations, algo, &tab[(size_t)t * 3]);
  std::vector<float> tabF(tab.begin(), tab.end());
  GPU_CHECK(cudaMalloc(&I->dCoefTab, tabF.size() * sizeof(float)));
  GPU_CHECK(cudaMemcpy(I->dCoefTab, tabF.data(), tabF.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMalloc(&I->dCounter, sizeof(int)));
  GPU_CHECK(cudaMemset(I->dCounter, 0, sizeof(int)));

  if (iterations > 0 && std::getenv("GPU_CFR_PROFILE")) {
    // Manual timed iterations instead of the graph: per-depth CUDA
    // event timings for the forward and backward launches, capped so
    // the run stays quick. The per-launch sync serializes work, so
    // absolute times are inflated on launch-bound trees; kernel
    // durations and their ratios are what to read. Shares the launch
    // sequence with halfStep.
    const int pit = std::min(iterations, 64);
    cudaEvent_t ev0, ev1;
    GPU_CHECK(cudaEventCreate(&ev0));
    GPU_CHECK(cudaEventCreate(&ev1));
    const int D = I->maxDepth + 1;
    std::vector<double> fwdT(2 * D, 0.0), bwdT(2 * D, 0.0);
    // per (tr, depth, kind) bwd attribution, kind in {show, fold,
    // decide+chance}
    std::vector<double> bwdK(2 * D * 3, 0.0);
    const int maxBase = std::max(I->n0Base, I->n1Base);
    const size_t smem = (size_t)2 * maxBase * sizeof(float);
    const int gy = (maxBase + kThreads - 1) / kThreads;
    auto timed = [&](bool fwd, int tr, int d, int kmIdx, int mask) {
      TreeBuf b = I->buf();
      b.reachOff = tr == 0 ? I->dReachOff0 : I->dReachOff1;
      b.reach = tr == 0 ? I->dReach0 : I->dReach1;
      b.reachRW = tr == 0 ? I->dReach0 : I->dReach1;
      GPU_CHECK(cudaEventRecord(ev0, s));
      if (fwd) {
        const int nD = I->decideCount[d], nC = I->chanceCount[d];
        const dim3 g(nD * gy + I->chanceBrTotal[d]);
        fwdLevelK<<<g, kThreads, 0, s>>>(I->dDecideList[d], nD,
                                         I->dChanceList[d], nC, gy,
                                         I->dChanceBrOff[d], tr, 0, b);
      } else {
        bwdDepthK<<<I->bwdCount[d], kThreads, smem, s>>>(
            I->dBwdList[d], I->bwdCount[d], tr, kModeCFR, I->dCoefTab,
            I->dCounter, mask, b);
      }
      GPU_CHECK(cudaEventRecord(ev1, s));
      GPU_CHECK(cudaEventSynchronize(ev1));
      float ms;
      GPU_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
      if (kmIdx < 0)
        (fwd ? fwdT : bwdT)[tr * D + d] += ms * 1000.0;
      else
        bwdK[tr * D * 3 + d * 3 + kmIdx] += ms * 1000.0;
    };
    // kind masks: bit per Kind (kDecide=0, kShowdown=1, kFold=2,
    // kChance=3); the attribution classes are show, fold, decide+chance.
    const int kKindMask[3] = {1 << kShowdown, 1 << kFold,
                              (1 << kDecide) | (1 << kChance)};
    for (int t = 0; t < pit; ++t) {
      for (int tr = 0; tr < 2; ++tr) {
        for (int d = 0; d < D; ++d)
          if (I->decideCount[d] > 0 || I->chanceCount[d] > 0)
            timed(true, tr, d, -1, -1);
        for (int d = I->maxDepth; d >= 0; --d) {
          if (I->bwdCount[d] > 0) {
            timed(false, tr, d, -1, -1);
            for (int km = 0; km < 3; ++km)
              timed(false, tr, d, km, kKindMask[km]);
          }
        }
      }
      advanceCounterK<<<1, 1, 0, s>>>(I->dCounter);
    }
    GPU_CHECK(cudaStreamSynchronize(s));
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    for (int d = 0; d < D; ++d) {
      const auto& k = I->depthKinds[d];
      std::fprintf(stderr,
                   "gpu_cfr: d%-2d decide=%-5d show=%-5d fold=%-5d "
                   "chance=%-3d  fwd %6.1f/%6.1f us  bwd %6.1f/%6.1f us"
                   "  [show %5.1f  fold %5.1f  decide %5.1f]\n",
                   d, k[0], k[1], k[2], k[3], fwdT[0 * D + d] / pit,
                   fwdT[1 * D + d] / pit, bwdT[0 * D + d] / pit,
                   bwdT[1 * D + d] / pit, bwdK[0 * D * 3 + d * 3 + 0] / pit,
                   bwdK[0 * D * 3 + d * 3 + 1] / pit,
                   bwdK[0 * D * 3 + d * 3 + 2] / pit);
    }
    double ft = 0.0, bt = 0.0;
    for (int i = 0; i < 2 * D; ++i) {
      ft += fwdT[i];
      bt += bwdT[i];
    }
    std::fprintf(stderr,
                 "gpu_cfr: per-iter (serialized) fwd %.1f us  bwd %.1f us "
                 "over %d iters\n",
                 ft / pit, bt / pit, pit);
    iterationsRun_ = pit;
    return;
  }

  if (iterations > 0) {
    I->capturing = true;
    GPU_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    I->halfStep(0, kModeCFR, 0, s);
    I->halfStep(1, kModeCFR, 0, s);
    advanceCounterK<<<1, 1, 0, s>>>(I->dCounter);
    GPU_CHECK(cudaStreamEndCapture(s, &I->graph));
    I->capturing = false;
    GPU_CHECK(cudaGraphInstantiate(&I->exec, I->graph, nullptr, nullptr, 0));
    {
      size_t nn = 0;
      cudaGraphGetNodes(I->graph, nullptr, &nn);
      std::fprintf(stderr, "gpu_cfr: graph nodes=%zu\n", nn);
    }
    // Replay in chunks; with a target set, check the exploitability
    // (two BR walks, ~4 iteration-equivalents) after each chunk and
    // stop early once it crosses the target. The walks never touch the
    // CFR rows, so resuming replays is sound.
    const int chunk = 128;
    int done = 0;
    while (done < iterations) {
      const int n = std::min(chunk, iterations - done);
      for (int t = 0; t < n; ++t) GPU_CHECK(cudaGraphLaunch(I->exec, s));
      done += n;
      iterationsRun_ = done;
      if (target > 0.0 && done < iterations && I->explNow() <= target) {
        iterationsRun_ = done;
        break;
      }
      iterationsRun_ = done;
    }
    GPU_CHECK(cudaStreamSynchronize(s));
  } else {
    iterationsRun_ = 0;
  }
}

pf::NodeStats GpuPostflopSolver::stats() {
  Impl* I = impl_;
  pf::NodeStats st{};
  if (I->empty) return st;
  cudaStream_t s = I->stream;

  // Disjoint pair mass Z over the base lists.
  double Z = 0.0;
  {
    double total1 = 0.0;
    double cardSum[52] = {0.0};
    for (int j = 0; j < I->n1Base; ++j) {
      total1 += I->w1[j];
      cardSum[I->baseCards1[2 * j]] += I->w1[j];
      cardSum[I->baseCards1[2 * j + 1]] += I->w1[j];
    }
    for (int i = 0; i < I->n0Base; ++i) {
      double w = total1 - cardSum[I->baseCards0[2 * i]] -
                 cardSum[I->baseCards0[2 * i + 1]];
      int so = I->sameOther0[i];
      if (so >= 0) w += I->w1[so];
      if (w > 0.0) Z += I->w0[i] * w;
    }
  }
  if (Z <= 0.0) return st;

  // Debug: seed the average-strategy rows with a deterministic non-
  // uniform profile to test the value-walk invariant independent of
  // training. GPU_CFR_SEED_STRAT: 1 = all decide nodes; 2 = turn-level
  // (depth <= 2) only; 3 = river-level (depth > 2) only.
  if (const char* sm = std::getenv("GPU_CFR_SEED_STRAT")) {
    const int seedMode = std::atoi(sm);
    std::vector<float> rows(I->regTotal, 0.0f);
    for (size_t mi = 0; mi < I->decideMeta.size(); ++mi) {
      const auto& m = I->decideMeta[mi];
      (void)mi;
      const bool turnNode = m.depth <= 2;
      if (seedMode == 2 && !turnNode) continue;
      if (seedMode == 3 && turnNode) continue;
      // Fine-grained: 4 = root only (d==0); 5 = d==1 nodes; 6 = d==2 nodes.
      if (seedMode == 4 && m.depth != 0) continue;
      if (seedMode == 5 && m.depth != 1) continue;
      if (seedMode == 6 && m.depth != 2) continue;
      // 7/8: single-node probes (decideMeta index 1 = ipTop, 2 = ipFace).
      if (seedMode == 7 && mi != 1) continue;
      if (seedMode == 8 && mi != 2) continue;
      // 9: ipFace = d1 decide nodes other than node 1.
      if (seedMode == 9 && !(m.depth == 1 && mi != 1)) continue;
      if (m.na < 2) continue;
      for (int c = 0; c < m.nC; ++c) {
        float p0 = (float)(0.15 + 0.7 * ((c % 8) / 7.0));
        rows[m.regOff + (size_t)0 * m.nC + c] = p0;
        rows[m.regOff + (size_t)1 * m.nC + c] = 1.0f - p0;
        for (int a = 2; a < m.na; ++a)
          rows[m.regOff + (size_t)a * m.nC + c] = 0.0f;
      }
    }
    GPU_CHECK(cudaMemcpy(I->dStrat, rows.data(),
                         I->regTotal * sizeof(float),
                         cudaMemcpyHostToDevice));
    std::fprintf(stderr, "seeded strat rows (mode %d)\n", seedMode);
  }
  double ev0 = I->rootValue(0, kModeEV);
  double ev1 = I->rootValue(1, kModeEV);
  double V0 = I->rootValue(0, kModeBR);
  double V1 = I->rootValue(1, kModeBR);
  st.ev0 = ev0;
  st.ev1 = ev1;
  st.expl = (V0 + V1 - (double)I->pot) / 2.0;
  st.pairMass = (int64_t)(Z * (1LL << 20));
  return st;
}

std::vector<double> GpuPostflopSolver::nodeStrategy(int decideIdx) {
  Impl* I = impl_;
  if (I->empty || decideIdx < 0 ||
      decideIdx >= (int)I->decideMeta.size())
    return {};
  const auto& m = I->decideMeta[decideIdx];
  // Only first-street nodes share the root's combo lists, whose comboId
  // entries are the identity base slots of the deciding player. Deeper
  // (chance-compacted) nodes would need reach-weighted averaging instead.
  if (m.nBoard != I->spotNBoard) return {};
  const int nC = m.nC;
  const int na = m.na;
  // First-street decide nodes share the root's combo lists, whose
  // comboId entries are the identity base slots of the deciding player
  // (each side's slots are 0-based into its own w0/w1 vector).
  std::vector<float> rows((size_t)na * nC);
  GPU_CHECK(cudaMemcpy(rows.data(), I->dStrat + (size_t)m.regOff,
                       (size_t)na * nC * sizeof(float),
                       cudaMemcpyDeviceToHost));
  const std::vector<double>& w = m.player == 0 ? I->w0 : I->w1;
  std::vector<double> freq(na, 0.0), comboTot(nC, 0.0);
  double z = 0.0;
  for (int c = 0; c < nC; ++c) {
    double s = 0.0;
    for (int a = 0; a < na; ++a)
      s += rows[(size_t)a * nC + c];
    comboTot[c] = s;
    z += w[c];
  }
  if (z <= 0.0) return freq;
  const double uniform = 1.0 / na;
  for (int a = 0; a < na; ++a) {
    double acc = 0.0;
    for (int c = 0; c < nC; ++c) {
      const double sigma =
          comboTot[c] > 0.0 ? rows[(size_t)a * nC + c] / comboTot[c]
                            : uniform;
      acc += w[c] * sigma;
    }
    freq[a] = acc / z;
  }
  return freq;
}

std::vector<double> GpuPostflopSolver::rootStrategy() {
  Impl* I = impl_;
  if (I->empty) return {};
  const int n0 = (int)I->rootCombos0.size();
  GPU_CHECK(cudaMemcpy(I->rootStratRows.data(),
                       I->dStrat + (size_t)I->rootRegOff,
                       (size_t)I->rootNa * n0 * sizeof(float),
                       cudaMemcpyDeviceToHost));
  std::vector<double> freq(I->rootNa, 0.0);
  double z = 0.0;
  for (int c = 0; c < n0; ++c) z += I->w0[I->rootCombos0[c]];
  if (z <= 0.0) return freq;
  const double uniform = 1.0 / I->rootNa;
  std::vector<double> comboTot(n0, 0.0);
  for (int a = 0; a < I->rootNa; ++a)
    for (int c = 0; c < n0; ++c)
      comboTot[c] += I->rootStratRows[(size_t)a * n0 + c];
  for (int a = 0; a < I->rootNa; ++a) {
    double s = 0.0;
    for (int c = 0; c < n0; ++c) {
      double sigma = comboTot[c] > 0.0
                         ? I->rootStratRows[(size_t)a * n0 + c] / comboTot[c]
                         : uniform;
      s += I->w0[I->rootCombos0[c]] * sigma;
    }
    freq[a] = s / z;
  }
  return freq;
}

void GpuPostflopSolver::debugDump() {
  Impl* I = impl_;
  if (I->empty) return;
  GPU_CHECK(cudaStreamSynchronize(I->stream));
  auto sum = [&](const double* d, size_t n) {
    if (n == 0) return 0.0;
    std::vector<double> h(n);
    GPU_CHECK(cudaMemcpy(h.data(), d, n * sizeof(double),
                         cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (double v : h) s += v;
    return s;
  };
  {
    std::vector<double> h(I->regTotal);
    GPU_CHECK(cudaMemcpy(h.data(), I->dStrat, I->regTotal * sizeof(double),
                         cudaMemcpyDeviceToHost));
    long long bad = 0;
    for (double v : h)
      if (!std::isfinite(v)) ++bad;
    std::fprintf(stderr, "strat nonfinite=%lld of %zu\n", bad, I->regTotal);
    GPU_CHECK(cudaMemcpy(h.data(), I->dRegret,
                         I->regTotal * sizeof(double),
                         cudaMemcpyDeviceToHost));
    bad = 0;
    for (double v : h)
      if (!std::isfinite(v)) ++bad;
    std::fprintf(stderr, "regret nonfinite=%lld of %zu\n", bad, I->regTotal);
  }
}

}  // namespace gpu
}  // namespace pps
