# Solver algorithms: state of the art (research note)

What modern range-based postflop solvers actually run, as of September 2026.
Scope: two-player zero-sum, abstraction-free (range-based) solving of
flop/turn/river spots — the "be on par with postflop-solver" target.

## TL;DR

- Production postflop solvers (postflop-solver, opensolver, DCFR-SOLVER)
  all use **vanilla full-tree CFR with DCFR discounting** — not sampled
  MCCFR. Sampling is for huge preflop games; abstraction-free full
  traversal is the standard for range solving.
- The current research frontier is **Hyperparameter Schedule-powered
  DCFR (HS-DCFR)** — Zhang, McAleer & Sandholm, AAAI 2026 — a
  training-free dynamic discounting scheme that beats DCFR, PCFR+ and
  DDCFR across 17 games including HUNL endgames. It is ~15 lines of code
  on top of DCFR.
- For GPU acceleration, the modern approach is **compile the game to
  static dataflow + CUDA Graph replay** (GPU-CFR, 2026) or matrix-form
  CFR (2024) — not per-node kernels.

## The algorithm ladder

| Algorithm | Year | Key idea | Status |
|---|---|---|---|
| CFR | 2007 | counterfactual regret decomposition | baseline |
| CFR+ | 2014 | alternating updates, RM+, linear averaging; solved limit HU hold'em (Cepheus) | classic |
| DCFR | 2019 (Brown & Sandholm, AAAI) | discount early iterations; hyperparameters (alpha, beta, gamma) = (1.5, 0, 2) | **what production postflop solvers run** |
| PCFR+ | 2021 (Farina et al., AAAI) | predictive RM+ (regret matching on extrapolated regret), quadratic averaging | SoTA for non-poker EFGs; on poker slightly worse than DCFR |
| DDCFR | 2024 (ICLR, Spotlight) | RL-learned dynamic discounting (alpha, beta, gamma in [-5,5]) | beats DCFR; costs a trained agent (24h on 200 cores) + runtime inference |
| **HS-DCFR / HS-PCFR+** | 2024→2026 (Zhang, McAleer, Sandholm, AAAI 2026) | training-free *schedules* for (alpha, beta, gamma) | **new SOTA**; generalizes across games without tuning |
| PDCFR+ / APCFR+ | 2024/2025 (IJCAI 2024; arXiv 2503.12770) | principled PCFR+ x DCFR blend; robustness fixes for prediction | comparable on HUNL subgames, wins after ~10k iterations on Leduc |

## Exact update rules (verified against production source)

postflop-solver's `src/solver.rs` (vendored under `tools/pfs-verify/`)
implements DCFR as follows, per iteration t, with **alternating updates**
(one half-step per player) and **regret-matching+**:

```
alpha_t = t^1.5 / (t^1.5 + 1)                    # positive-regret discount
beta_t  = 0.5                                    # negative-regret discount
k       = 4^floor(log4(t))                        # 0, 1, 4, 16, 64, ...
t'      = t - k
gamma_t = (t' / (t' + 1))^3                      # average-strategy discount
                                                  # (gamma_t = 0 at powers of 4:
                                                  #  resets the cumulative
                                                  #  strategy there)
# per decision node, per combo, per action:
cum_regret[c,a]    = cum_regret[c,a] * (cum_regret[c,a] >= 0 ? alpha_t : beta_t)
                     + (cfv[c,a] - node_value[c])
cum_strategy[c,a]   = cum_strategy[c,a] * gamma_t + sigma[c,a]
```

The beta = 0.5 constant corresponds to beta = 0 in the paper's
parametrization (negative regret weight t^0/(t^0+1) = 1/2); gamma = 3 is
their choice over the paper's best fixed 2; the power-of-4 reset is their
own twist (README: "resets the cumulative strategy when the number of
iterations is a power of 4").

## HS-DCFR: the current SOTA schedules

From arXiv 2404.09097 ("Faster Game Solving via Hyperparameter
Schedules"), Equation 4 — plug into the DCFR update equations above,
with t = current iteration, n = total iterations:

```
alpha(t) = 1 + (3/n) * t
beta(t)  = -1 - (2/n) * t
gamma(t) = 30 - (5/n) * t        # HS-DCFR(30); or 15 - (5/n)*t for HS-DCFR(15)
```

Rationale: with fixed gamma = 2, the average-strategy weight reaches 0.9
by iteration ~19 — early junk still matters. HS starts gamma at 30
(weight stays low for hundreds of iterations: a short memory), then
relaxes. Verified on 17 games including HUNL endgames; no per-game
tuning; ~15 lines of code. HS-PCFR+ = same gamma schedules applied to
PCFR+.

## GPU acceleration (for our CUDA roadmap)

- **GPU-CFR** (arXiv 2609.11923, 2026): the key observation is that for
  a fixed game, everything about a CFR iteration except the numbers is
  known ahead of time. Compile once into static dataflow (flat edge and
  infoset arrays, precomputed indices, depth-level batched passes),
  then replay each iteration as a single CUDA graph launch. 29.8-80.4x
  vs prior GPU CFR on one A100; 14-258x vs LiteEFG CPU on the largest
  games. Notably, the compiled flat representation alone is 2.2-51.1x
  faster on CPU — i.e., do this even before targeting a GPU.
  **Implemented, multi-street** (`cuda/gpu_cfr.*`, `gpu_pfflop`): tree
  exported flat with per-branch combo sub-ranges and per-river-board
  strength tables, depth-level forward/backward kernels, CUDA graph
  replay with a device-resident discount schedule, f32 hot path
  (consumer GPUs run fp64 at 1/64 rate; the f32 oracle makes this also
  the right precision class for parity). Turn-spot parity vs
  postflop-solver is gated by `make gpu-parity` (EV within 0.01 chips);
  the hand-computed tiny-turn ground truth (`tools/tiny_check.cpp`)
  pins the chance conventions exactly. The per-graph-node replay floor
  is ~100us on WSL (~1.5us/graph node), so small trees are
  launch-bound; big trees are per-node-work bound (see docs/STATUS.md
  for the perf log).
- **Matrix-form CFR** (arXiv 2408.14778): CFR as dense/sparse
  matrix/vector products; 203x vs OpenSpiel C++ on larger games; higher
  memory use.
- Practical consequence: per-node CUDA kernels (what our
  `cuda/poker_kernels.cu` sketches) are the wrong shape; the right shape
  is flat arrays + batched depth-level passes + graph replay.

## Production engineering tricks (postflop-solver internals)

- Per-combo flat arrays at every node (not per-infoset maps)
- Alternating updates; RM+ (clipped at zero in matching)
- f32 storage, f64 accumulation, optional 16-bit compressed regrets
  with per-node scaling factors
- Suit isomorphism: fold isomorphic chance cards into one branch
- Multithreading across private hands with fine-grained locks
- Node locking (exploitative re-solving with part of the tree frozen)
- Traversal is plain recursion over the public tree with counterfactual
  reach vectors; showdown values via sorted strength sweeps with
  per-card conflict correction (the same sweep technique we reverse-
  engineered in their equity function)

## What this means for our solver

1. **The parity target runs vanilla DCFR.** Our range-based river/
   postflop engine should implement exactly the update rules above
   (identical alpha_t/beta_t/gamma_t, alternating updates, RM+) so that
   A/B comparisons against pfs-verify isolate implementation bugs, not
   algorithm differences.
2. **Implement HS-DCFR(30) as the modern accelerator** behind a flag:
   same code path, schedules instead of constants. Measure both against
   the oracle (EV agreement + exploitability per iteration budget).
3. **Keep MCCFR for 8-max preflop** (sampling handles the huge
   multi-street space); vanilla DCFR is for spot solving where accuracy
   is the goal. The two engines share ranges/cards/ICM/showdown code.
4. **When we do CUDA**: compile-to-dataflow + graph replay, mirroring
   GPU-CFR — and its flat-array layout is worth adopting on CPU first.

## References

- Zinkevich et al., *Regret Minimization in Games with Incomplete
  Information*, NIPS 2007 (CFR)
- Tammelin, *CFR+*, 2014; Bowling et al., *Cepheus*, Science 2015
- Brown & Sandholm, *Solving Imperfect-Information Games via Discounted
  Regret Minimization*, AAAI 2019, arXiv:1809.04040 (DCFR)
- Farina, Kroer, Sandholm, *Faster Game Solving via Predictive
  Blackwell Approachability*, AAAI 2021 (PCFR+)
- Xu et al., *Dynamic Discounted CFR*, ICLR 2024 (DDCFR)
- Zhang, McAleer, Sandholm, *Faster Game Solving via Hyperparameter
  Schedules*, AAAI 2026, arXiv:2404.09097 (HS-DCFR, HS-PCFR+)
- Kim, *Minimizing Weighted CFR with Optimistic OMD*, IJCAI 2024
  (PDCFR+); *Asymmetry of Step Sizes*, arXiv:2503.12770 (APCFR+)
- Kim, *GPU-Accelerated CFR*, arXiv:2408.14778
- *GPU-CFR*, arXiv:2609.11923 (compile to static dataflow + CUDA graphs)
- b-inary/postflop-solver (vendored; production DCFR reference)
- JoakimMich/opensolver, exinori/DCFR-SOLVER (open DCFR postflop
  solvers; the latter reports 0.016% of pot exploitability at 10k
  iterations)
