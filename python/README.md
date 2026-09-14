# pps — programmatic interface

Python bindings for the GPU postflop solver (compile-to-dataflow DCFR,
CUDA graph replay; see `cuda/gpu_cfr.h`). Heads-up, range-based, no
abstraction; flop/turn/river boards.

## Build

```
make python          # builds python/pps/pps_native*.so via the CUDA build
make python-test     # build + run the API test suite (python/test_pps.py)
```

Requires `pybind11` and `numpy` in the active Python environment
(`python -m pip install pybind11 numpy`) plus the CUDA toolkit. The
build reconfigures `build-cuda` with `-DPython3_EXECUTABLE=$(PYTHON)`
(`PYTHON` is overridable). Import from the checkout:

```python
import sys; sys.path.insert(0, "python")
import pps
```

## API

```python
import pps

s = pps.Solver(
    board="Qs9h2d",            # flop/turn/river: 3-5 cards, str or list
    oop="TT+,AKo",             # player 0 range (str / dict / 1326 floats)
    ip="AQ+,KQs",              # player 1 range
    pot=200,                    # chips in the middle before betting
    stack=500,                  # effective stacks behind
    bets="0.75,a",             # pot fractions; "a" = all-in, "e" = geometric
    raises="2.5,3",             # raise multiples of the previous bet
    max_raises=0,               # 0 = unlimited
)

s.solve(2000)                        # iterations; or target-based:
s.solve(max_iters=100000, target=0.1)   # stops early (exploitability <= target)
s.stats()
# {"ev_oop": 140.4, "ev_ip": 59.6, "exploitability": 0.065, "pair_mass": ...}

s.strategy(0)          # labeled per-combo strategy of decide node 0 (root)
s.strategy(1)          # IP's first decision (after OOP checks)
# {
#   "actions":  [{"kind": "check", "amount": 0},
#               {"kind": "bet", "amount": 150},       # chips, street total
#               {"kind": "allin", "amount": 500}],
#   "cards":    ["TcTd", "TcTh", ...],   # the player's combos at this node
#   "freqs":    ndarray (n_actions, n_combos), rows sum to 1 per combo
#   "aggregate": [0.78, 0.22, 0.0],     # range-weighted (first-street
#                                        # nodes only)
# }

s.decide_nodes()       # tree introspection: player, depth, board size,
                       # labeled actions, child node ids
s.root_ev()            # per-combo EVs vs the average strategy (chips),
                       # plus the per-combo opponent mass (normalizer)
s.save("spot.sol")     # solution file: spot + config + rows + schedule

# warm-start / re-solve:
s.continue_solve(2000)         # continues the DCFR schedule exactly where
                               # solve() stopped (cumulative discount t)
s.reset()                      # back to untrained, same compiled tree
t = pps.Solver.load("spot.sol")  # recompiles the spot, restores the rows
t.continue_solve(1000)          # ... and continues training
```

Range syntax is postflop-solver's: hand classes (`"TT"`, `"AKs"`,
`"AKo"`), `"+"`/dash ranges (`"TT+"`, `"A5s-A2s"`), weights
(`"AA:0.5"`), or a dict `{"AA": 0.5, "AKs": 1.0}`, or a raw list of
1326 combo weights.

## Semantics worth knowing

- **Labeled actions**: `kind` is `fold|check|call|bet|raise|allin`;
  `amount` is the street-level total contribution in chips (0 for
  check/call/fold).
- **Per-combo strategy** works at ANY decide node (including past
  chance nodes) — per-combo frequencies need no reach weighting.
  The range-weighted `aggregate` is only exact on first-street nodes
  (the root and each player's first decision), where the base range
  weights are the true reach weights.
- **Warm start is exact**: DCFR's discount staircase depends only on
  the cumulative iteration index, so `solve(n)` and
  `solve(k) + continue_solve(n-k)` follow the same trajectory (up to
  the engine's replay nondeterminism, ~1e-5 chips of EV from the
  fold-terminal shared atomics).
- **Replay nondeterminism**: fold-terminal atomics reorder f32 adds, so
  EVs wobble ~0.01 chips between runs and exploitability (a difference
  of small numbers) wobbles more. Size comparisons accordingly.
- **Solution files** are self-describing: spot, bet config, iteration
  count and both row tables. `Solver.load` recompiles the tree (layout
  is deterministic) and validates it against the file.

## Tests

`python/test_pps.py` (via `make python-test`) checks the internal
invariants: labeled action structure, per-combo normalization, EV
aggregation against `stats()`, warm-start trajectory equivalence,
save/load roundtrip (rows identical, continuation), and error paths.
The oracle parity gates (`make gpu-parity`) remain the correctness
gate for the engine conventions.
