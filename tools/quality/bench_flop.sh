#!/bin/sh
# Flop-solver performance tracker. Same phase breakdown as bench_turn.sh
# (compile = tree build + upload + graph capture; solve = CUDA graph
# replay; stats = final EV + best-response walks), on flop boards (two
# chance streets: 45 turn cards x 44 river cards per betting line, so
# trees are ~40x the turn equivalents and compile time is a much larger
# share of a one-shot solve).
#
# Abstraction: bet 40% of pot or all-in ("a", the oracle's explicit
# all-in size); raises 2.5x of the previous bet. One bet size + all-in
# keeps the trees small enough for fast benchmark loops while still
# exercising raise lines, the all-in runout and both chance streets.
#
# usage: bench_flop.sh [iters]
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
GPU="${GPU_PFFLOP:-$ROOTDIR/build-cuda/gpu_pfflop}"
ITERS="${1:-500}"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"

if [ ! -x "$GPU" ]; then
  echo "gpu_pfflop not found at $GPU (build with ENABLE_CUDA=ON)" >&2
  exit 2
fi

# <label> <board> <pot> <stack> <bets> <raises>
run() {
  label="$1"; board="$2"; pot="$3"; stack="$4"; bets="$5"; raises="$6"
  err=$("$GPU" postflop "$board" "$OOP" "$IP" "$pot" "$stack" "$bets" \
        "$raises" "$ITERS" --algo dcfr 2>&1 >/dev/null)
  nodes=$(echo "$err" | sed -n 's/gpu_cfr: nodes=\([0-9]*\).*/\1/p')
  depth=$(echo "$err" | sed -n 's/gpu_cfr: nodes=[0-9]* depth=\([0-9]*\).*/\1/p')
  compile=$(echo "$err" | sed -n 's/gpu_cfr: compile \([0-9.]*\) ms/\1/p')
  solve=$(echo "$err" | sed -n 's/gpu_cfr: [0-9]* iters in \([0-9.]*\) ms.*/\1/p')
  rate=$(echo "$err" | sed -n 's/.*(\([0-9]*\) iters\/s.*/\1/p')
  stats=$(echo "$err" | sed -n 's/gpu_cfr: stats walk \([0-9.]*\) ms/\1/p')
  awk -v l="$label" -v n="$nodes" -v d="$depth" -v c="$compile" \
      -v s="$solve" -v r="$rate" -v t="$stats" -v i="$ITERS" 'BEGIN {
    printf "%-10s nodes=%-8d depth=%d  compile %8.1f ms  solve %8.1f ms  stats %6.1f ms  %7.0f iters/s (%.1f ms/iter)\n", l, n, d, c, s, t, r, s / i
  }'
}

echo "flop bench (iters=$ITERS)"
run standard Qs9h2d 200 500 "0.4,a" "2.5"
run deep AhTd9s 300 1500 "0.4,a" "2.5"
run shortstack Qs9h2d 500 150 "0.4,a" "2.5"
