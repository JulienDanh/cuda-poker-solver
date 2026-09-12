# Project status

Where the solver stands, what is validated, and what is missing.
Last updated: after commit aab9688 (multi-street play).

## 1. What exists and is validated

### Equilibrium engine
- **External-sampling MCCFR** (`src/solver.h`): game-agnostic (template
  over the game), multithreaded via per-thread tables merged at the end,
  linear-weighted average strategy.
- Validated on **Kuhn poker**: exact exploitability (pure-strategy
  enumeration) < 0.02 after 100k iterations, with the value/BR harness
  itself pinned against hand-computed degenerate strategies.

### Hand evaluation
- 5–7 card evaluator (`evaluateN`) with complete category and kicker
  ordering.
- Cross-validated against **b-inary/postflop-solver's** evaluator
  (`make verify-pfs`): 8 million random 7-card hands, zero ordering
  violations (equal hands equal, stronger hands stronger). This check
  found a real bug (double-trips full houses, now fixed and regression
  tested).

### Game domain (heads-up through 8-max)
- Full multi-street NLHE: preflop, flop, turn, river; blinds, antes,
  min-raise rules, all-in runouts, short all-ins (betting continues
  between covered players), uncalled-bet returns, side-pot layering,
  tie splits.
- Two payoff modes: **chip-EV** (exactly zero-sum, the correctness
  mode) and **ICM** (Malmuth-Harville subset DP, zero-stack-safe,
  brute-force tested).
- Preflop abstraction: exact 169 canonical hands, or 15 coarse buckets.
- Postflop abstraction: made-hand category x rank tier (27 buckets per
  street) or exact-board keys (`--postflop-exact`).
- Optional FGS continuation mode (`--stub`): flop-bound pots resolve
  through a `ContinuationModel` (default: seeded showdown stub).

### Correctness measurements that pass today
| Check | Result |
|---|---|
| 29 unit tests | all pass |
| Evaluator vs postflop-solver | 8M hands, 0 violations |
| Kuhn exploitability | < 0.02 after 100k iters |
| HU preflop exploitability (chip-EV) | 7.3 bb/hand @ 60k, 3.7 @ 240k — halves with 4x iterations (ES-MCCFR O(1/sqrt(T)) confirmed) |
| Chip payoffs | zero-sum invariants for fold, all-in, continuation, multi-street |
| 8-max ICM preflop ranges | directionally sane (UTG folds low offsuit ~100%, opens broadway ~98%) |
| Mode comparison (HU chip-EV, 150k iters) | BTN/SB: -0.065 bb with the position-blind stub, **+0.154 bb** with postflop play — matches the measured stub bias |
| FGS stub bias vs real postflop solves | OOP overvalued by ~4% of pot on average (-3.4%, -11.2%, +2.3% across three flops) |

### Tooling
- CLI (`ppsolve`): seats 2–8, stacks, payouts, bet abstraction (preflop
  + per-street), `--chip-ev`, `--postflop`/`--stub`, `--postflop-exact`,
  `--mc-value N` (Monte-Carlo rollout value per seat),
  `--exploitability N` (HU, preflop-only game), CSV preflop strategy
  dump, UTG range print.
- `tools/pfs-verify`: harness around vendored postflop-solver (AGPL;
  patched to expose its evaluator): `make verify-pfs` (evaluator
  cross-check), `make stub-bias` (stub bias vs full flop solves).
- Build: Makefile (CPU), CMakeLists with optional CUDA target.

### Bugs found by validation so far (all fixed)
1. Showdown winner comparison inverted (kept the *worst* hand).
2. ICM NaN on zero-chip stacks silently froze whole infosets at uniform.
3. Double-trips hands misclassified as trips instead of full house.
4. Preflop calls charged blind-minus-ante (antes absorbed into calls).
5. Bucket label arrays out of order relative to the encoding.
6. Exploitability evaluator winner's-curse: BR max over noisy
   single-sample boards inflated the bound from ~4 bb to ~50 bb.

## 2. What is missing

### Solver quality (highest impact first)
1. **Postflop abstraction is crude.** Category x rank tier ignores
   draws, board texture, and opponent-range interaction. Needed: equity
   clustering (OCHS / k-means over expected-hand-strength histograms),
   likely also opponent-clustering for multiway. This is the single
   biggest lever on postflop strategy quality.
2. **Convergence speed.** External sampling converges O(1/sqrt(T));
   HU preflop-only is still 3.7 bb/hand exploitable at 240k iterations,
   and the multi-street game needs vastly more. Missing: DCFR/CFR+
   regret discounting, vanilla (full-tree) CFR for HU where the tree is
   small enough, better parallel aggregation than merge-at-end.
3. **Multi-street exploitability.** The analytic best-response evaluator
   covers the preflop-only game only (street chances cannot be collapsed
   analytically once betting follows them). Postflop convergence is
   currently only observable via `--mc-value`. Needed: a proper
   multi-street BR (average over sampled boards at each chance node) or
   a per-street abstraction-consistent BR.
4. **Multiway best response.** No exploitability for 3–8 players (BR
   with card removal across many hands is a project of its own).
5. **Tournament FGS (across hands).** The solver computes a single-hand
   equilibrium from one tournament state. Nothing models future hands,
   blind levels, or stack dynamics (the original "FGS" idea in the
   tournament sense). All ICM decisions are single-hand.
6. **The FGS stub is not replaced by a real continuation solver.** In
   `--stub` mode it still carries the measured ~4%-of-pot OOP bias. If
   8-max preflop runs matter, a smarter continuation model (e.g. a
   small postflop solve per terminal class) is wanted.

### Performance
7. **CUDA is written but dead code.** `cuda/poker_kernels.cu` (batched
   ICM, board sampling) has never been compiled or run — no NVIDIA GPU
   on the dev machine, and the MCCFR hot loop does not call them at all.
   This is the repo's founding goal and is entirely undone. Needs:
   kernel validation on hardware, GPU infoset table / regret updates,
   batched-equity pipeline, and a CPU/GPU benchmark.
8. **CPU hot loop is unoptimized.** State copies per action,
   `unordered_map` infoset storage, per-node allocations. Feasible:
   flat arrays with interned infoset ids, incremental state updates,
   SIMD evaluator, prefetched ICM tables. Expect 5–20x before touching
   CUDA.
9. **8-max postflop is not tractable on CPU** at current abstraction
   quality and iteration counts (1.5M infosets after only 1500
   iterations). Requires items 1–2 plus 7.

### Output and tooling
10. No postflop strategy output: the CSV dump covers preflop decision
    nodes only; postflop infosets are trained but invisible.
11. No save/load of trained solutions (results are recomputed each
    run).
12. No spot-query interface (given a hand + history, print the
    strategy/EV), no subtree locking / exploitative re-solve.

### Validation gaps
13. Postflop strategies are not yet compared against postflop-solver
    spot solves (only the stub-bias EV measurement exists).
14. No continuous integration; tests are run manually (`make test`,
    `make verify-pfs`).
15. ICM multiway equilibria are validated structurally, not against an
    external reference (none exists in the toolchain).

## 3. Quality harness (the gates for performance work)

The performance phase must not regress quality. Three gates, all wired
into the Makefile:

- `make test` — 33 unit tests: engine invariants (EV-sum-to-pot at every
  iteration, hand-derived uniform-strategy values, tie-board equilibrium),
  range parser, evaluator cross-check, poker state machine, Kuhn/MCCFR
  convergence.
- `make river-parity` — head-to-head vs postflop-solver on 8 spots x 3
  geometries: EV within 0.01 chips, exploitability ratio within [0.5, 2.0]
  (measured cross-arithmetic noise band: 0.88-1.48), root strategy within
  0.10 per action. Requires cargo.
- `make river-quality` — oracle-free golden baseline (12 spot/iteration
  points) + monotone-convergence ladder. The solver is deterministic, so
  EV diffs are exactly 0 today: any drift trips immediately. After an
  INTENTIONAL quality change: `make river-baseline` regenerates.
- `make river-bench` — performance tracker (currently ~41k iters/s small
  spot, ~3k iters/s wide-config spot).

Workflow for any performance change:
    make test river-parity river-quality   # must pass
    make river-bench                       # before/after numbers
The exploitability gates are deliberately loose (2x) because they
compare at fixed iteration counts where f64-vs-f32 convergence wobble
is real; the EV gates (0.02 chips absolute) are the tight ones.

## 4. Recommended next steps

First goal: **parity with postflop-solver as a range-based postflop
solver**. Research in docs/solver-algorithms.md concluded. Status: the
river milestone is DONE — src/postflop_cfr.{h,cpp} (pfflop CLI) solves
river spots with vanilla DCFR using the oracle's exact discounting, and
matches postflop-solver's EVs to ~1e-4 chips and exploitability along
the whole iteration trajectory (make river-parity). Two algorithms are
implemented: parity DCFR and HS-DCFR(30), the 2026 state of the art
(HS is 2-4x faster early; parity DCFR with its power-of-4 average reset
polishes tighter at 1000+ iterations). Remaining for full parity:
turn and flop chances (the tree/chance machinery is designed for it),
16-bit compression, suit isomorphism, and the performance engineering
(SIMD/flat arrays). Next: 

1. Build the range-based postflop engine with vanilla **DCFR** using
   postflop-solver's exact update rules (alternating updates, RM+,
   alpha_t = t^1.5/(t^1.5+1), beta_t = 0.5, gamma_t with power-of-4
   reset) so A/B comparisons against the oracle isolate implementation
   bugs, not algorithm differences. River-only first (no chance nodes),
   then turn, then flop.
2. Implement **HS-DCFR(30)** schedules (the 2026 SOTA, ~15 lines on top
   of DCFR) behind a flag and measure both against the oracle.
3. Keep sampled MCCFR for 8-max preflop; vanilla DCFR is for spot
   solving where accuracy is the goal.
4. When targeting CUDA: compile the game to static dataflow with
   depth-level batched passes + CUDA Graph replay (the GPU-CFR
   approach), not per-node kernels; the flat layout is worth adopting
   on CPU first. A range module (src/range.h/.cpp, postflop-solver
   syntax subset) is already in place.
5. Then: equity-clustering abstraction for multi-street MCCFR, postflop
   strategy CSV + save/load, multiway BR, FGS-across-hands.
