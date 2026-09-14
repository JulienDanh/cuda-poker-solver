# pps: GPU postflop solver — programmatic interface.
#
#   import pps
#   s = pps.Solver(board="Qs9h2d", oop="TT+,AKo", ip="AQ+,KQs",
#                 pot=200, stack=500, bets="0.75,a", raises="2.5,3")
#   s.solve(2000)               # or s.solve(max_iters=100000, target=0.1)
#   s.stats()                   # {"ev_oop", "ev_ip", "exploitability", ...}
#   s.strategy()                # labeled per-combo root strategy
#   s.decide_nodes()            # tree introspection
#   s.save("spot.sol")
#   t = pps.Solver.load("spot.sol"); t.continue_solve(1000)
#
# Cards/ranges/bet sizes use the gpu_pfflop CLI syntax. The native
# extension (pps.pps_native) is built into this directory by
# `make python`; numpy is required for the array results.
"""GPU postflop solver (DCFR, compile-to-dataflow + CUDA graph replay)."""

__all__ = ["Solver", "__version__"]

__version__ = "0.1.0"

try:
    from . import pps_native as _native  # noqa: F401
except ImportError as e:  # pragma: no cover - build not run
    raise ImportError(
        "the pps native module is not built; run `make python` in the "
        "cuda-poker-solver checkout (requires pybind11 + numpy in the "
        "active Python env)"
    ) from e

Solver = _native.Solver
