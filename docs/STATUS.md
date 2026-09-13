# Project status

Where the solver stands, what is validated, and what is missing.
Last updated: after the multi-street GPU turn milestone (f32 hot path,
turn-oracle parity, turn-only focus).

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
| Turn spots vs postflop-solver oracle (`make gpu-parity`) | 6 geometries x 500 iters: EV within 0.01 chips (measured max 0.0034), expl ratio in [0.5, 2] (measured [0.64, 1.13]), root strategy within 0.10 |
| Tiny-turn ground truth (`tools/tiny_check.cpp`) | uniform + 2 seeded profiles match the f64 hand computation to 1e-4 (measured 2-5e-6; f32 walk) |
| Quick gate (`make gpu-quick`) | same spots at 200 iters with loose gates, ~20 s |
| Chip payoffs (engine invariants) | zero-sum for fold/all-in/continuation/multi-street |
| 8-max ICM preflop ranges (engine) | directionally sane (kept, out of scope) |

### Performance (turn spots, RTX 4070, `make turn-bench`, 2000 iters)

| spot | nodes | compile | solve | iters/s |
|---|---|---|---|---|
| shortstack (all-in runout) | 537 | ~195 ms | 101 us/iter | ~9,900 |
| standard (200 pot, 500 stack) | 5,943 | ~205 ms | 237 us/iter | ~4,200 |
| wide/deep (1500 stack, 4+ sizes) | 51,903 | ~230 ms | ~1,400 us/iter | ~700 |

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

Cumulative turn throughput: ~3.5x (wide 193 -> ~700, standard 1401 ->
~4,200 iters/s).

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

1. **Showdown base-space walk** — the identified next perf lever. Every
   showdown node zeroes/loads its opponent reach over the full base range
   and block-scans it in strength-sorted order (plus a separate gather
   pass). A precomputed per-node reverse map (base slot -> node-local
   index) would fuse the zero/gather passes and drop two syncs; the
   card-correction loops can read through the same map.
2. **Launch-bound small trees** — the ~61-graph-node replay costs
   ~100 us on WSL regardless of tree size (the shortstack spot is pure
   floor). Fusing the same-depth forward decide+chance kernels and the
   per-depth backward dispatch would cut graph nodes ~30%.
3. **Flop spots** — engine support exists (3-card boards), but gates
   (`make gpu-quick`/`gpu-parity`) and `turn-bench` cover turn only, by
   decision, until the turn perf work lands. The oracle flop solves are
   also ~50x the turn cost per iteration, which would slow the loop.
4. **Strategy output** — only the root aggregate is exposed (`--root`).
   No per-node strategy dump, no save/load of trained solutions, no
   spot-query interface.
5. **Exploitability** is measured per solve (EV + BR walk) but there is
   no multiway (3+ player) support at all in the GPU engine.
6. **No continuous integration**; the gates are run manually.

## 3. Quality harness (the gates)

The gates live in `tools/quality/`, wired into the Makefile. The
workflow for any change to the GPU engine:

    make gpu-quick            # fast debug loop (~20 s)
    make gpu-parity           # full commit gate (~20 s)
    make turn-bench           # before/after numbers

- `make gpu-quick` — turn spots vs the postflop-solver oracle at 200
  iterations with loose gates (EV 0.10, expl ratio [0.25, 4], strategy
  0.15) plus the tiny-turn ground truth (1e-4). Catches tree/convention
  regressions fast: the bugs this harness found (the 0.38-chip
  behind-stack gap, the all-in-runout recursion) show up at any
  iteration count.
- `make gpu-parity` — six turn geometries at 500 iterations with tight
  gates (EV 0.01, expl ratio [0.5, 2], strategy 0.10) plus the ground
  truth. The tight EV gate is meaningful because the oracle is f32 like
  the GPU path.
- `make turn-bench` — four turn geometries with a phase breakdown
  (compile / solve / stats).
- `make test` — 32 unit tests (engine invariants, MCCFR convergence,
  range parser, evaluator, poker state machine).
- `make verify-pfs` — evaluator cross-check vs the oracle (8M hands).

The quick/full split exists because the gates serve two different jobs:
the quick gate answers "did I break the conventions/tree" in 20 s while
iterating; the full gate answers "is the solver still at oracle parity"
before a commit. The tiny ground truth is engine-independent (a hand
-computed f64 brute force) and runs in both.

## 4. Recommended next steps

1. **Showdown reverse-map fusion** (item 1 above): precompute per
   showdown node a base-slot -> local-index map (~13 MB for a 52k-node
   tree), fuse the zero+fill+gather passes, route the card-correction
   reads through it. Measure with `make turn-bench`, gate with
   `make gpu-quick`.
2. Forward kernel fusion for the graph-node count (item 2), if small
   trees matter.
3. Then flop spots: add 3-card boards back to the gates with
   `pfs-verify solve` (already exposed), at reduced iteration counts
   (the oracle flop solve is ~30 s at 300 iterations).
4. Then: per-node strategy dump / save-load / spot query (item 4).
