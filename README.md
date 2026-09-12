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
dump enumeration; the full 8-max public tree is combinatorially large).

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
  iterations than the defaults for tight ranges. No exploitability
  measurement for the poker game yet (Kuhn only).
- The strategy dump enumerates the public tree up to a node budget; full
  8-max trees with all raise branches are combinatorially large.
- CUDA kernels are not validated (no NVIDIA GPU available).
- Single-hand equilibrium from one tournament state: no FGS-across-hands
  (future blind-level/stack-dynamics modeling).

## Roadmap

- [x] Game engine (external-sampling MCCFR, Kuhn-validated)
- [x] ICM payoffs (Harville subset DP, brute-force-tested)
- [x] 8-max preflop state machine (blinds, antes, min-raises, side pots)
- [x] FGS continuation interface + showdown stub
- [x] CLI + CSV strategy dump
- [ ] Convergence: deeper iterations, weighted regret variants (DCFR)
- [ ] Postflop solver to replace the continuation stub (real FGS)
- [ ] CUDA integration into the MCCFR hot loop + validation on hardware
- [ ] Exploitability / EV estimation for the poker game
- [ ] Multi-street FGS-across-hands for tournament dynamics

## License

[MIT](LICENSE)
