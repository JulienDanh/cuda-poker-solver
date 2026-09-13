# cuda-poker-solver

> Current status and what is missing: [docs/STATUS.md](docs/STATUS.md)
> Algorithm research (modern CFR variants, GPU approaches): [docs/solver-algorithms.md](docs/solver-algorithms.md)

A GPU-oriented poker solver. The active focus is the **GPU postflop
solver** (heads-up, range-based, no abstraction): the bet tree is
compiled once into flat device arrays and solved with vanilla DCFR as
depth-level batched kernel passes replayed as a CUDA graph — the GPU-CFR
recipe (arXiv:2609.11923). Multi-street boards are supported (flop/turn/
river); the current focus is **turn spots** and **performance**.

The repo also carries an 8-max tournament NLHE engine (ICM payoffs,
external-sampling MCCFR, validated on Kuhn poker) reachable through
`ppsolve`. It is not the active development focus.

## Build

CPU build (no GPU required):

```
make            # builds build/ppsolve
make test       # builds and runs the unit tests
```

Or with CMake:

```
cmake -B build-cmake -DENABLE_CUDA=ON   # or OFF for CPU-only
cmake --build build-cmake
./build-cmake/ppsolver_tests
```

The CUDA kernels (`cuda/poker_kernels.cu`: batched ICM subset-DP and board
sampling) compile only with an NVIDIA toolkit. They are **validated on
hardware** (RTX 4070, sm_89, CUDA 13.x) by `build-cmake/cuda_validate`
(`tools/cuda_validate.cu`): batched ICM matches `src/icm.cpp` to 1e-16
across 2-8 players including zero-stack terminals, and the board sampler
satisfies its contract with uniform card marginals. The kernels are not
yet wired into the MCCFR hot loop.

**GPU-CFR postflop solver** (`cuda/gpu_cfr.cu`, binary `gpu_pfflop`):
the exported bet tree is compiled to static dataflow — flat node/edge
arrays, per-node compact combo lists (chance branches drop blocked
combos), per-node regret/strategy rows, precomputed per-river-board
strength orders and card-conflict lists — and each DCFR iteration runs
as depth-level batched kernel passes (fused per-depth backward kernel,
root reach read in-kernel), with one iteration captured into a CUDA
graph and replayed per iteration. The discount schedule is device
-resident and advanced by the graph itself. All hot buffers are f32
(the RTX 4070 runs fp64 at 1/64 the fp32 rate, and the parity oracle is
itself an f32 solver); host-side final walks accumulate in f64.
Turn-spot throughput on the RTX 4070: ~4,200 iters/s on a 6k-node tree,
~700 iters/s on a 52k-node tree (see `make turn-bench`).

Two payoff modes: `--chip-ev` (default off) evaluates terminals as chip
deltas — the correctness-validation mode (exactly zero-sum, no ICM
approximation); the default ICM mode evaluates Malmuth-Harville equity
deltas.

## Usage

```
./build/ppsolve --seats 8 --stack 1000000 --sb 5000 --bb 10000 --ante 1250 \
    --payouts 0.5,0.3,0.2 --iters 20000 --threads 4 --buckets 15 \
    --out strategy.csv --dump-nodes 20000
```

Options: `--seats` (2..8), `--stack` (uniform chips), `--stacks a,b,...`
(per-player stacks for uneven tournament states), `--sb/--bb/--ante`,
`--payouts` (fractions summing to 1), `--iters`, `--threads`,
`--buckets 169|15` (169 exact canonical hands, or a coarse 15-bucket
abstraction), `--out` (CSV path), `--dump-nodes` (cap on the strategy
dump enumeration; the full 8-max public tree is combinatorially large),
`--chip-ev` (chip-delta payoffs), `--exploitability N` (heads-up only:
measure the exploitability bound over N deals after training),
`--open-sizes/--raise-mult/--max-bets` (bet abstraction), `--cont-samples`
(boards per FGS stub resolution).

Exploitability example (heads-up chip-EV, full abstraction):

```
./build/ppsolve --seats 2 --stack 50000 --sb 500 --bb 1000 --ante 0 \
    --chip-ev --buckets 169 --iters 240000 --threads 1 \
    --exploitability 200 --expl-boards 128 --dump-nodes 0 --out /dev/null
```

External-sampling MCCFR converges as O(1/sqrt(T)): measured exploitability
bound halves with 4x iterations (7.3 bb/hand at 60k, 3.7 at 240k).

The solver prints a UTG first-decision range and writes the visited
preflop strategy to CSV: `position, history, bucket, action frequencies`
(the CSV dump enumerates preflop decision nodes; postflop infosets are
trained but not dumped yet). `--mc-value N` estimates the achieved value
per seat by Monte-Carlo rollout of the average strategy.

Example comparing modes (heads-up, chip-EV, 150k iterations):

```
# multi-street (default)
ppsolve --seats 2 --chip-ev --iters 150000 --mc-value 100000 ...
#   BTN/SB +0.154 bb, BB -0.154 bb

# FGS-stub preflop-only mode (position-blind showdown continuation)
ppsolve ... --stub --mc-value 100000
#   BTN/SB -0.065 bb, BB +0.065 bb
```

The BTN value flips negative without postflop play, matching the ~4%-of-
pot IP bias measured against postflop-solver (see below).

## Verification against b-inary/postflop-solver

`tools/pfs-verify` wraps the (archived)
[b-inary/postflop-solver](https://github.com/b-inary/postflop-solver)
Rust crate — an independent, production-used postflop solver — and uses it
two ways (requires `cargo`; the crate is vendored under
`tools/pfs-verify/third_party/` with a patch that exposes its otherwise
private hand evaluator):

```
make verify-pfs   # evaluator ordering cross-check (N random 7-card hands)
make stub-bias    # FGS stub EV-bias measurement vs full postflop solves
```

For the GPU postflop solver, the quality gates are:

- `make gpu-quick` — the fast debug loop (~20 s): turn spots vs the
  oracle at low iteration counts with loose gates, plus the
  hand-computed tiny-turn ground truth (`tools/tiny_check.cpp`).
- `make gpu-parity` — the full commit gate (~20 s): six turn geometries
  at 500 iterations with tight gates (EV within 0.01 chips of the
  oracle, exploitability ratio within [0.5, 2], root strategy within
  0.10) plus the ground truth.
- `make turn-bench` — performance tracker with a per-phase breakdown
  (compile / solve / stats walk).

See docs/STATUS.md for the harness design and tolerances.

**Evaluator cross-check** (`tools/verify_eval7.cpp` + `pfs-verify eval7`):
for millions of random 7-card hands, our `evaluate7` total order and their
`Hand::evaluate()` total order must agree exactly (equal hands equal,
stronger hands stronger). This check caught a real bug in our evaluator:
hands with two trip ranks in 7 cards (e.g. `222+444+K`) were misclassified
as trips instead of a full house — every paired runout with double trips
was ranked wrong, and no unit test covered it. After the fix: **8 million
hands, zero ordering violations**.

**FGS stub bias** (`tools/pfs-verify/stub_bias.sh`): compares the
ShowdownContinuation stub (pure equity split of the flop pot) against a
full postflop solve of the same spot (pot 500, SPR 4, 60%/all-in + 2.5x
betting, exploitability ~0.4% of pot). BB is OOP:

| flop | stub EV (BB) | real EV (BB) | bias |
|---|---|---|---|
| Qs9h2d | 251.6 | 234.4 | -3.4% of pot |
| 7h8h9c | 267.4 | 211.4 | -11.2% of pot |
| KdKc4c | 247.7 | 259.4 | +2.3% of pot |

The stub overvalues the out-of-position player by ~4% of the pot on
average (it strips position), which is the main reason the preflop
solver's BTN/SB values come out slightly negative. Replacing the stub
with a real postflop solve is the intended fix (the `ContinuationModel`
interface already matches what this engine provides).

## Architecture

```
src/
├── common.h/.cpp     xorshift64* RNG, FNV-1a hashing
├── cards.h/.cpp      52-card layout, 7-card hand evaluator
├── icm.h/.cpp        Malmuth-Harville ICM: subset DP + O(n!) reference
├── hand169.h/.cpp    169 canonical preflop hands, 15-bucket coarse mode
├── showdown.h/.cpp   side-pot layering, uncalled-bet returns, tie splits
├── fgs.h/.cpp        ContinuationModel interface + ShowdownContinuation
├── poker.h/.cpp      8-max tournament NLHE multi-street game (blinds,
│                     antes, min-raise, all-in, street chances, side
│                     pots, ICM-delta or chip-delta payoffs, postflop
│                     hand bucketing)
├── solver.h          game-agnostic external-sampling MCCFR (templates)
├── eval.h/.cpp       HU chip-EV exploitability bound (best-response DP)
├── range.h/.cpp      postflop-solver-syntax range parser
├── postflop_cfr.h/.cpp  bet-tree abstraction (pf::betActions), shared by
│                     the GPU tree builder; mirrors postflop-solver's
│                     action_tree.rs (pot-relative bets, prev-bet-relative
│                     raises, all-in capping/merging)

tools/
├── verify_eval7.cpp     evaluator cross-check driver (see make verify-pfs)
├── tiny_check.cpp       hand-computed tiny-turn ground truth (f64)
├── gpu_pfflop.cu        GPU postflop solver CLI (flop/turn/river boards)
├── cuda_validate.cu     on-hardware kernel validation vs the CPU reference
├── quality/            the gate scripts (gpu-quick/gpu-parity/turn-bench)
└── pfs-verify/          Rust harness around b-inary/postflop-solver
    ├── src/main.rs      eval7 / equity / solve / solve-turn subcommands
    └── third_party/     vendored postflop-solver (AGPL, local patch)
cuda/
├── gpu_cfr.cu/.h    GPU-CFR postflop engine (compile-to-dataflow + CUDA
│                     graph replay, f32 rows, multi-street chance nodes)
└── poker_kernels.cu batched ICM + board sampling kernels (optional)
tests/
└── test_main.cpp     32 unit tests
```

### The MCCFR engine

`solver.h` runs external-sampling MCCFR: per iteration, one chance deal is
sampled (concrete cards — card removal is exact), then for each player a
traversal enumerates that player's actions while sampling opponents and
chance. Regret matching with linear-weighted average strategies. Threads
run independent tables that merge at the end.

The engine is validated two ways in tests:
- **Kuhn poker**: after 100k iterations, exact exploitability (by pure
  strategy enumeration) drops below 0.02.
- **Harness sanity**: value/BR/exploitability functions are pinned against
  hand-computed degenerate strategies.

### Correctness validation without ICM

The engine is validated in `--chip-ev` mode, where payoffs are exact chip
deltas and the game is exactly zero-sum:

- **Kuhn poker** (analytic equilibrium): exploitability < 0.02 after 100k
  iterations.
- **Real poker exploitability** (`src/eval.h/.cpp`): for heads-up, a full
  tree DP computes the best-response value against the trained average
  strategy per sampled deal. Board chance and continuation terminals are
  evaluated analytically from a per-deal all-in equity (MC over boards),
  so no best-response decision can condition on a board it is about to
  receive, and no maximum is taken over noisy per-branch estimates
  (a winner's-curse bias that otherwise dominates the measurement — the
  naive sampled evaluator reports ~50 bb/hand where the true bound is
  ~4 bb). The result is an upper bound (per-deal greedy BR conditions on
  hole cards; with 169 buckets the gap is suit-level slack). The
  convergence test asserts the bound halves with 4x iterations, matching
  ES-MCCFR theory.

The chip-EV BTN value under the showdown stub is slightly negative
(~-0.1 bb): the stub strips position, so the SB/BTN seat loses its main
structural advantage. This is a modeling artifact, not an engine issue.

### Payoff model

At any terminal the game computes final stacks and evaluates
`ICM(final) − ICM(initial)` per player. ICM is constant-sum, so payoffs
sum to zero — a proper constant-sum game in equity space. Zero-chip
outcomes (all-ins) are handled deterministically: zero stacks occupy the
bottom places.

All-in confrontations are exact chance nodes: the 5-card board is
sampled and side pots distribute by showdown layering (uncalled portions
returned first).

### FGS: continuation states

A betting sequence that ends with two or more players holding chips (a
flop-bound pot) cannot be valued by the pot alone. The game calls
`ContinuationModel::continuationEV` with the in-hand players' hole cards
and contributions and receives a **distribution over pot-winnings
vectors**, which is folded through ICM (ICM is nonlinear; averaging
expected stacks would be wrong).

The default `ShowdownContinuation` resolves the state as if everyone were
all-in (sampled boards, layered showdown). This is a documented
approximation: it ignores position and realization — measured at ~4% of
the pot favoring the out-of-position player (see the postflop-solver
verification section). The default mode no longer uses it: hands play out
flop/turn/river instead. Pass `--stub` to use it (cheap 8-max runs).

### Postflop play

Streets: preflop betting, flop/turn/river chances with per-street betting
(`--street-bets`, `--street-mult`, `--street-max-bets`). All-in pots run
out to showdown exactly; short all-ins keep betting between the covered
players; side pots layer at showdown as before.

Postflop infosets are bucketed by made-hand category (high card through
straight flush) x rank tier, per street, on the player's own cards plus
the current board — 27 buckets. This ignores draws and board texture
(`--postflop-exact` keys on the exact board instead: no abstraction, but
a very large infoset space). Preflop infosets keep the 169-hand canonical
abstraction.

### Abstraction

Default is exact 169-hand canonical preflop: each infoset key is
(seat, hand bucket, action history). `--buckets 15` coarsens hands to a
preset grid (pairs by tier; suited/offsuit by rank tier and connectedness)
for fast low-fidelity runs. Raise sizes are an abstraction config
(`BetAbstraction`): opens {2.2, 2.5, 3}bb, raises {2.5, 3}x, max 4 bets,
with an all-in collapse threshold at 50% of the effective stack.

## Current limitations

- Continuation (FGS) stub only — no postflop solver yet.
- External sampling converges as O(1/sqrt(T)); 8-max runs need many more
  iterations than the defaults for tight ranges. Exploitability
  measurement exists for heads-up chip-EV only (multi-way best response
  with card removal is future work).
- The strategy dump enumerates the public tree up to a node budget; full
  8-max trees with all raise branches are combinatorially large.
- CUDA kernels are validated on hardware (RTX 4070) but not yet called
  from the MCCFR hot loop.
- Single-hand equilibrium from one tournament state: no FGS-across-hands
  (future blind-level/stack-dynamics modeling).

## Roadmap

- [x] Game engine (external-sampling MCCFR, Kuhn-validated)
- [x] Chip-EV mode + HU exploitability evaluator (convergence-validated)
- [x] Multi-street preflop+postflop play (flop/turn/river betting, street
  chances, side pots; BTN EV flips from -0.065 bb to +0.154 bb vs the
  position-blind stub, matching the measured stub bias)
- [x] ICM payoffs (Harville subset DP, brute-force-tested)
- [x] Evaluator cross-validated against postflop-solver (found and fixed
  a double-trips full-house bug; 8M hands, zero violations)
- [x] Range-based river solver at parity with postflop-solver: identical
  bet trees, vanilla DCFR with the oracle's exact discounting; HS-DCFR(30)
  also implemented. (The CPU river solver has since been removed; the
  GPU engine is validated directly against the oracle.)
- [x] 8-max preflop state machine (blinds, antes, min-raises, side pots)
- [x] FGS continuation interface + showdown stub
- [x] CLI + CSV strategy dump (ppsolve)
- [x] CUDA kernels validated on hardware (RTX 4070): batched ICM matches
  the CPU reference to 1e-16 (2-8 players, zero stacks included); board
  sampling contract + chi-square uniformity (`tools/cuda_validate.cu`)
- [x] GPU-CFR postflop engine: compile-to-dataflow + CUDA graph replay,
  multi-street (flop/turn/river) with per-branch combo sub-ranges
  (`cuda/gpu_cfr.*`, `gpu_pfflop`)
- [x] Turn-spot oracle parity: EV within 0.01 chips of postflop-solver
  across six geometries, plus the hand-computed tiny-turn ground truth
  (`make gpu-parity`)
- [x] f32 hot path (rows + kernels; host walks accumulate f64) with the
  fold-terminal card sums scattered via shared atomics
- [ ] Turn-solver performance: identified next lever is the showdown
  base-space walk (fused zero/gather/scan via precomputed reverse maps)
- [ ] Flop spots: supported by the engine, gated and benchmarked only
  after the turn performance work
- [ ] Preflop solver work is out of current scope (dropped)

## License

[MIT](LICENSE)
