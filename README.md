# cuda-poker-solver

A GPU-oriented poker solver for **8-max tournament NLHE**, built around
three ideas:

1. **ICM payoffs** — terminal states are evaluated as Malmuth-Harville
   tournament-equity deltas, not chip deltas.
2. **External-sampling MCCFR** — the equilibrium engine, game-agnostic and
   validated on Kuhn poker against its known equilibrium.
3. **FGS (Future Game Simulation)** — called pots that reach postflop
   resolve through a `ContinuationModel` interface, so the preflop solver
   stays exact while postflop play is modeled rather than fully expanded.

The first milestone targets **8-max preflop solving**; the postflop solver
that plugs into the FGS interface is future work.

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
sampling) compile only with an NVIDIA toolkit and are currently
**unvalidated — there is no NVIDIA GPU on the development machine**.

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

The solver prints a UTG first-decision range and writes the full visited
strategy to CSV: `position, history, bucket, action frequencies`.

## Architecture

```
src/
├── common.h/.cpp     xorshift64* RNG, FNV-1a hashing
├── cards.h/.cpp      52-card layout, 7-card hand evaluator
├── icm.h/.cpp        Malmuth-Harville ICM: subset DP + O(n!) reference
├── hand169.h/.cpp    169 canonical preflop hands, 15-bucket coarse mode
├── showdown.h/.cpp   side-pot layering, uncalled-bet returns, tie splits
├── fgs.h/.cpp        ContinuationModel interface + ShowdownContinuation
├── poker.h/.cpp      8-max tournament NLHE preflop game (blinds, antes,
│                     min-raise, all-in, board chance, ICM-delta payoffs)
├── solver.h          game-agnostic external-sampling MCCFR (templates)
├── eval.h/.cpp       HU chip-EV exploitability bound (best-response DP)
├── kuhn.h            Kuhn poker (engine validation)
└── main.cpp          CLI, strategy extraction, CSV dump
cuda/
└── poker_kernels.cu  batched ICM + board sampling kernels (optional)
tests/
└── test_main.cpp     20 unit tests
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
approximation: it ignores position, realization, and postflop play, and
it overvalues limping/flatting slightly. Plugging in a real postflop
solver later changes nothing else.

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
- CUDA kernels are not validated (no NVIDIA GPU available).
- Single-hand equilibrium from one tournament state: no FGS-across-hands
  (future blind-level/stack-dynamics modeling).

## Roadmap

- [x] Game engine (external-sampling MCCFR, Kuhn-validated)
- [x] Chip-EV mode + HU exploitability evaluator (convergence-validated)
- [x] ICM payoffs (Harville subset DP, brute-force-tested)
- [x] 8-max preflop state machine (blinds, antes, min-raises, side pots)
- [x] FGS continuation interface + showdown stub
- [x] CLI + CSV strategy dump
- [ ] Convergence: deeper iterations, weighted regret variants (DCFR)
- [ ] Postflop solver to replace the continuation stub (real FGS)
- [ ] CUDA integration into the MCCFR hot loop + validation on hardware
- [ ] Exploitability for multiway (3-8 players)
- [ ] Multi-street FGS-across-hands for tournament dynamics

## License

[MIT](LICENSE)
