# Project status

Where the solver stands, what is validated, and what is missing.
Last updated: perf-phase kickoff (feedback loop, gate noise calibration).

## 0. Scope

The active project is the **GPU postflop solver** (heads-up, range-based,
no abstraction): `cuda/gpu_cfr.*` behind the `gpu_pfflop` CLI. Multi
-street boards (flop/turn/river) are supported by the engine; the current
focus is **turn spots and performance**. Flop spots are deferred until
the turn performance work is done. **Preflop solver work (the 8-max
MCCFR/ICM engine, FGS) is out of scope** — the engine still builds and
its tests still pass (`ppsolve`), but no further work is planned on it.

The CPU range-based river solver (`pfflop`) has been removed: with the
GPU engine validated directly against the postflop-solver oracle, it was
a redundant middle rung. Its bet-tree construction lives on as
`pf::betActions` (src/postflop_cfr.*), which the GPU tree builder calls.

## 1. What exists and is validated

### GPU postflop solver (the focus)

- **Compile-to-dataflow** (GPU-CFR, arXiv:2609.11923): the bet tree is
  built once on the host (`pf::betActions` — postflop-solver's exact bet
  abstraction) and flattened into device arrays: node kinds/children,
  per-node compact combo lists (chance branches drop blocked combos),
  per-node regret/strategy rows, chance-branch CSR maps, per-river-board
  strength tables with card-conflict lists.
- **Solving**: vanilla full-tree DCFR (or HS-DCFR(30)) with alternating
  updates and regret-matching+, run as depth-level batched kernel passes
  (forward reach, fused backward cfv+update), one iteration captured into
  a CUDA graph and replayed per iteration; the discount schedule is
  device-resident and advanced by the graph itself.
- **Multi-street chance semantics** (all ground-truth-validated,
  tools/tiny_check.cpp): chance weight 1/(52 − nBoard − 4) per valid
  branch (per-pair hole-card masking), prior-street contributions
  tracked for value baselines, the behind-stack shrinks by prior streets'
  matched contributions, and a dead street (nothing behind) deals through
  chance to showdown.
- **Precision**: all hot device buffers and kernels are f32 — the RTX
  4070 executes fp64 at 1/64 the fp32 rate, and the parity oracle is
  itself an f32 solver. The discount schedule is computed in f64 on the
  host and uploaded f32; the final EV/exploitability walks accumulate in
  f64 on the host over f32 rows.

### Correctness measurements that pass today

| Check | Result |
|---|---|
| 32 unit tests | all pass |
| Evaluator vs postflop-solver (`make verify-pfs`) | 8M hands, 0 violations |
| Kuhn exploitability (MCCFR engine) | < 0.02 after 100k iters |
| Turn spots vs postflop-solver oracle (`make gpu-parity`) | 6 geometries x 500 iters: EV within 0.03 chips (measured max 0.017), expl ratio in [0.30, 2] (measured [0.41, 0.91]), root strategy within 0.10 |
| Node strategy (IP post-check decide node) vs oracle (`gpu-quick`/`gpu-parity`) | same spots: node1 maxdiff < 0.10 (measured ≤ 0.025) |
| Tiny-turn ground truth (`tools/tiny_check.cpp`) | uniform + 2 seeded profiles match the f64 hand computation to 1e-4 (measured 2-5e-6; f32 walk) |
| Quick gate (`make gpu-quick`) | same spots at 200 iters with loose gates, ~20 s |
| Chip payoffs (engine invariants) | zero-sum for fold/all-in/continuation/multi-street |
| 8-max ICM preflop ranges (engine) | directionally sane (kept, out of scope) |

### Performance (turn spots, RTX 4070, `make turn-bench`, 2000 iters)

| spot | nodes | compile | solve | iters/s |
|---|---|---|---|---|
| shortstack (all-in runout) | 537 | ~195 ms | 89 us/iter | ~11,240 |
| standard (200 pot, 500 stack) | 5,943 | ~205 ms | 215 us/iter | ~4,650 |
| wide/deep (1500 stack, 4+ sizes) | 51,903 | ~230 ms | ~1,240-1,340 us/iter | ~747-807 |

### Flop spots (measured, not yet gated)

| spot | nodes | depth | GPU compile | GPU solve | oracle solve | speedup |
|---|---|---|---|---|---|---|
| standard (200 pot, 500 stack, 2 sizes) | 615,234 | 11 | 2.7 s | 18 ms/iter (55/s) | ~86 ms/iter (~11.6/s) | ~4.7x |
| wide (1500 stack, 4 sizes) | 11,716,550 | 14 | 49.6 s | 1747 ms/iter (0.57/s) | ~150 ms/iter (~0.65/s) | ~0.9x |

Parity holds on both (EV diff 0.0008 at 200 iters standard; 0.0018 at
100 iters wide). The practical abstraction (one size + all-in,
e.g. "0.75,a", raises 2.5x/3x) is much smaller and the GPU keeps its
edge everywhere: turn 500 stack 2,085 nodes at 6,697 iters/s vs oracle
~1,563 (4.3x); turn 1500 stack 5,943 nodes at 4,283 vs ~657 (6.5x);
flop 500 stack 164k nodes at 201 vs ~40 (5.0x); flop 1500 stack 615k
nodes at 56 vs ~11.8 (4.7x). End-to-end at low iteration counts the
GPU's compile floor (0.2 s turn, 1.6-2.7 s flop) eats into it: 1.1-2.4x
turn, 2.4-3.1x flop. The wide-flop weakness below only bites with
multi-size abstractions. The wide flop fits in ~1.5 GB VRAM. Two flop-specific
findings: the GPU's node throughput drops ~6x vs the turn trees (39M
-> 6.7M nodes/s) because chance-dealt flop trees are showdown-heavy
(~600k showdown nodes per iteration over ~1,900 runout boards), and
the per-showdown block cost is the bottleneck — so on the widest flop
spot the 12-thread CPU oracle matches the GPU per iteration. The
single-threaded host tree build also becomes a real cost (50 s for
the 11.7M-node tree). Making the GPU win on wide flops is the
strength-sorted row relayout / per-board showdown amortization work,
not more micro-tuning.

Per-iteration cost is dominated by the solve replay; compile is a fixed
~200 ms (mostly CUDA context init on WSL); the stats walk is single-digit
ms. Work log for the turn perf phase (all gated by `make gpu-quick`
after each step):

1. **f32 hot path** (rows + kernels; f64 only on the host walks): 1.8x.
   The wide tree is not fp64-ALU-bound but per-node work + traffic.
2. **Fold-terminal card sums via shared atomics** (each combo scatters
   its reach to its two cards) replacing 52 threads each scanning the
   whole base range: 1.5-1.6x, and numerically cleaner (EV diffs vs the
   oracle improved).
3. **kThreads 256 -> 128** (more resident blocks per SM): 1.25x on the
   wide tree. 64 regresses small/launch-bound trees.
4. **Tried and reverted**: batching 8 nodes per block — 2-4x *slower*;
   the serial node loop with block-wide syncs destroyed cross-node
   concurrency. Block launch/scheduling is not the bottleneck; the
   remaining cost is the per-node base-space work itself.
5. **Showdown/fold reverse-map fusion**: the terminal kernels no longer
   stage reach in base space. The showdown gathers raw reach in
   strength-sorted order straight from the node's compact list through
   a per-(board, traverser) reverse map (sorted position -> combo
   index, -1 for runout-blocked slots; combo lists are board-canonical,
   verified at compile), fusing the zero/scatter/gather passes into
   one and the scan into an out-of-place variant (2 fewer passes and
   syncs). Fold scatters per-card sums directly from the compact list
   (the atomics loop now runs over nOppNode, not nOppBase) and reads
   the identical-combo correction through a per-node foldSon map.
   +12% standard, +5% wide/deep/shortstack. The maps are tiny: the
   showdown table is per interned river board (~66 KB), foldSon is
   per fold node (~12 MB on the 52k-node tree).
6. **Forward kernel fusion** (GPU_CFR_PROFILE attribution showed the
   graph replay cost flooring small trees): decide and chance nodes at
   one depth run in a single flat launch — they are independent (both
   read depth-d reach, write disjoint children at d+1) — with the
   chance (node, branch) blocks located by binary search over a
   per-depth branch prefix table. Halves the forward graph nodes
   (shortstack tree: 61 -> 27 total). +6.6% shortstack, +1-1.5% on the
   wide trees. Also: fold accumulates the reach total in the same pass
   as its per-card scatter (barrier kept after the atomics — dropping
   it exposed a run-to-run nondeterminism on one gate spot, so it
   stays), and the graph now prints its node count.
10. **Tested and rejected: HS-DCFR(30) for solve-to-target**: hs30's
    discount schedule assumes a large fixed budget — against a 0.01
    target on the 1500-stack flop it needed 6,912 iterations (120 s)
    vs DCFR's ~2,560 (~50 s), and 2,048 vs 512 against 0.1. Vanilla
    DCFR stays the default for target-based stopping. The full-gate
    expl-ratio floor also moved 0.35 -> 0.30 after a second observed
    noise-tail flake (a broken BR walk collapses to ~0.001, so the
    guard is unaffected).
9. **Compile: quadratic chance-table scan removed**: the ctap and
   expIdx fills rescanned the parent combo list for every (branch,
   child combo) pair — O(branches x combos x parentN) per chance node,
   billions of comparisons on flop trees. The parent slot is captured
   inline while filtering the child list (it IS the filter index), so
   both fills are O(entries). Instrumented phases: tree-build
   1,644 -> 739 ms on the 615k flop tree (still chrono-inflated);
   compile 2.67 s -> 1.76 s (1500-stack flop), 1.60 -> 0.73 s
   (500-stack), 49.6 -> 35.2 s (the 11.7M wide flop). End-to-end to the
   0.1-chip target: 500-stack flop 3.2 -> 2.1 s, 1500-stack 13.8 ->
   11.1 s.
8. **Reach-row aliasing (traffic, not compute)**: profiling the
   practical flop tree showed the iteration is DRAM-bandwidth-bound
   (~7 GB of row traffic per iteration vs the 4070's ~500 GB/s). The
   forward pass used to copy the reach row unchanged to every child of
   a pass-through decide node (player == traverser) — now the row
   LAYOUTS are per traverser and such children alias the parent's row,
   so the copy disappears. cfv rows stay dense per node (children of a
   pass-through node still produce distinct cfv), which is why TreeBuf
   carries a separate cfvOff table. The root rows are initialized from
   the opponent's range weights and the fwd kernel reads reach, not w.
   Turn spots +2-6%, practical flop +3.5-5%.
7. **Tried and reverted: warp-per-node backward** (motivated by the
   per-kind profile: showdown ~45-50% of the backward, decide ~20%,
   fold ~15%): one node per warp, per-warp shared slices, no block
   barriers. 1.5-2x SLOWER across the board — small trees lose block
   parallelism (count/4 blocks on a 46-SM GPU) and the wide tree's
   per-lane serial chains got 2.5x longer. The block-level version
   hides gather/scatter latency with 4x more threads per node;
   same lesson as the block-batching experiment (item 4).

Cumulative turn throughput: ~4.0x (wide 193 -> ~785, standard 1401 ->
~4,730, shortstack 96 -> ~87 us/iter); practical flop 56 -> ~58
iters/s (1500 stack) and 201 -> ~211 (500 stack).

### Bugs found by validation so far (all fixed)

1. Showdown winner comparison inverted (kept the *worst* hand).
2. ICM NaN on zero-chip stacks silently froze whole infosets at uniform.
3. Double-trips hands misclassified as trips instead of full house.
4. Preflop calls charged blind-minus-ante (antes absorbed into calls).
5. Bucket label arrays out of order relative to the encoding.
6. Exploitability evaluator winner's-curse (BR max over noisy samples).
7. **Multi-street behind-stack bug** (found by the turn-oracle gate):
   the tree builder passed the original stack to every street's bet
   abstraction, allowing over-betting after prior-street chips went in —
   a persistent ~0.38-chip EV gap vs the oracle on turn spots (both
   engines converged, to different games). Fix: the behind-stack shrinks
   by prior-street matched contributions; the dead street (all-in runout)
   deals through chance to showdown instead of recursing on AllIn(0).

## 2. What is missing (turn focus, highest impact first)

1. **Launch-bound small trees** — the forward fusion cut the graph to
   27 nodes on the shortstack tree; the remaining floor is the
   per-depth backward launches (one per depth per traverser). The
   per-kind profile attribution (GPU_CFR_PROFILE=1) shows the wide
   tree is backward-compute-bound (showdown ~45-50% of the backward),
   so further small-tree gains now trade against big-tree throughput.
2. **Flop spots** — engine support exists (3-card boards), but gates
   (`make gpu-quick`/`gpu-parity`) and `turn-bench` cover turn only, by
   decision, until the turn perf work lands. The oracle flop solves are
   also ~50x the turn cost per iteration, which would slow the loop.
3. **Strategy output** — the root aggregate (`--root`) and the
   range-weighted average strategy of any first-street decide node
   (`--strat <idx>`, 0 = root, 1 = IP after OOP checks) are exposed;
   deeper (chance-compacted) nodes would need reach-weighted averaging
   and return nothing. Still missing: save/load of trained solutions,
   a spot-query interface.
4. **Exploitability** is measured per solve (EV + BR walk) but there is
   no multiway (3+ player) support at all in the GPU engine.
5. **No continuous integration**; the gates are run manually.

## 3. Quality harness (the gates)

The gates live in `tools/quality/`, wired into the Makefile. The
workflow for any change to the GPU engine:

    make perf-loop            # the perf feedback loop (~13 s, see below)
    make gpu-parity           # full commit gate (~15 s)
    make turn-bench           # before/after numbers

- `make perf-loop` — the performance feedback loop, one command: rebuild
  `build-cuda`, run the quick gate, run `turn-bench`, then compare
  per-geometry iters/s against the previous run (history in
  `build-cuda/.bench_last`, gitignored) and flag anything under 95% of
  it as a REGRESSION. Correctness failure, bench regression, or build
  error all exit nonzero. Run-to-run variance is ~1%, so 95% leaves
  headroom. Total ~13 s (build 5 s, gate 6 s, bench 7 s).
- `make gpu-quick` — turn spots vs the postflop-solver oracle at 200
  iterations with loose gates (EV 0.10, expl ratio [0.25, 4], strategy
  0.15) plus the tiny-turn ground truth (1e-4). Catches tree/convention
  regressions fast: the bugs this harness found (the 0.38-chip
  behind-stack gap, the all-in-runout recursion) show up at any
  iteration count.
- `make gpu-parity` — six turn geometries at 500 iterations with tight
  gates (EV 0.03, expl ratio [0.35, 2], strategy 0.10) plus the ground
  truth. The EV gate is meaningful because the oracle is f32 like the
  GPU path. The gates are sized to the measured run-to-run
  nondeterminism, both directions: our fold-terminal shared atomics
  reorder f32 adds (EV wobbles ~0.01 chips on the deep spot), and the
  oracle's rayon-parallel exploitability wobbles ~2x (pinning
  RAYON_NUM_THREADS=1 makes it deterministic but ~7x slower).
- `make turn-bench` — four turn geometries with a phase breakdown
  (compile / solve / stats).
- `make test` — 32 unit tests (engine invariants, MCCFR convergence,
  range parser, evaluator, poker state machine).
- `make verify-pfs` — evaluator cross-check vs the oracle (8M hands).

The quick/full split exists because the gates serve two different jobs:
the quick gate answers "did I break the conventions/tree" in seconds
while iterating; the full gate answers "is the solver still at oracle
parity" before a commit. The tiny ground truth is engine-independent
(a hand-computed f64 brute force) and runs in both.

## 4. Recommended next steps

1. **Forward kernel fusion for the graph-node count** (item 1), if
   small trees matter: fuse the same-depth forward decide+chance
   kernels and the per-depth backward dispatch (~30% fewer graph
   nodes; the shortstack spot is launch-bound).
2. Then flop spots: add 3-card boards back to the gates with
   `pfs-verify solve` (already exposed), at reduced iteration counts
   (the oracle flop solve is ~30 s at 300 iterations).
3. Then: save-load of trained solutions / spot query (item 3; the
   per-node first-street strategy query `--strat` landed).
