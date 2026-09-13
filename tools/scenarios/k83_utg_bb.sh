#!/bin/sh
# Reference scenario: UTG opens 2.5bb, BB defends ~60%, flop Kc8h3s,
# 40bb effective (5bb pot, 37.5bb behind each).
#
#   OOP = BB (the preflop caller acts first; ~786 combos)
#   IP  = UTG (the preflop raiser, ~16% opening range)
#
# Sizing: geometric ('e') + all-in ('a') bets, 2.5x raises — the
# geometric ladder converges into the all-in, so no re-raise sprawl at
# any stack depth. Solved to exploitability 0.5 chips (0.1% of pot).
#
# usage: k83_utg_bb.sh [--oracle]
#   --oracle: also run the postflop-solver oracle on the same spot and
#   print the comparison (takes ~1 min; saturates the CPU).
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
GPU="${GPU_PFFLOP:-$ROOTDIR/build-cuda/gpu_pfflop}"
PFS="$ROOTDIR/tools/pfs-verify/target/release/pfs-verify"

# BB defense vs an UTG open (~60%): all pairs, all suited, all aces and
# kings offsuit, the better offsuit remainder.
BB="22+,A2s+,K2s+,Q2s+,J2s+,T2s+,98s-92s,87s-82s,76s-72s,65s-62s,54s-52s,43s-42s,32s,A2o+,K2o+,Q8o+,J9o+,T9o,98o,97o,87o"
# UTG 6-max opening range (~16%).
UTG="55+,A5s+,K9s+,Q9s+,J9s+,T9s,98s,AJo+,KQo"

BOARD=Kc8h3s
POT=500     # 5bb (units of 0.01bb)
STACK=3750  # 37.5bb behind on the flop
BETS=e,a
RAISES=2.5
TARGET=0.5

echo "== GPU solve (target expl $TARGET)"
"$GPU" postflop "$BOARD" "$BB" "$UTG" "$POT" "$STACK" "$BETS" "$RAISES" \
      100000 --algo dcfr --target "$TARGET" --root

if [ "$1" = "--oracle" ]; then
  if [ ! -x "$PFS" ]; then
    (cd "$ROOTDIR/tools/pfs-verify" && cargo build --release)
  fi
  echo "== postflop-solver oracle (same spot, same target)"
  "$PFS" solve "$BOARD" "$BB" "$UTG" "$POT" "$STACK" e,a 2.5x 100000 "$TARGET"
fi

# Reference results (RTX 4070, f32, 2026-09-13):
#   GPU:    41.3s total (20.7s compile + 20.4s solve, 256 iters),
#           expl 0.364, EV 169.90/330.10, BB checks 99.9995% at the root.
#   Oracle: 54.0s, expl 0.494, EV 169.91/330.09, checks 99.9994%.
#   EVs agree within the pre-convergence band (~expl); the compile phase
#   dominates the GPU time at wide ranges (the known next optimization).
