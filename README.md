# cuda-poker-solver

A GPU-accelerated poker hand range solver using CUDA.

## Overview

`cuda-poker-solver` aims to compute approximate Nash-equilibrium strategies
for heads-up no-limit hold'em by running counterfactual regret minimization
(CFR) on NVIDIA GPUs.

## Status

Early stage — project skeleton only. See the roadmap below.

## Roadmap

- [ ] Card and hand-evaluation primitives
- [ ] Bet abstraction and game tree
- [ ] CUDA kernels for CFR updates
- [ ] Solver CLI
- [ ] Benchmarks vs. CPU baseline

## License

[MIT](LICENSE)
