#!/bin/sh
# Turn-solver performance tracker. Runs the gate turn geometries at a
# fixed iteration count and prints the phase breakdown (compile = tree
# build + upload + graph capture; solve = CUDA graph replay; stats =
# final EV + best-response walks) plus iterations/s. Compare before and
# after any performance change; correctness is gated separately
# (`make gpu-quick` / `make gpu-parity`).
#
# usage: bench_turn.sh [iters]
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
GPU="${GPU_PFFLOP:-$ROOTDIR/build-cuda/gpu_pfflop}"
ITERS="${1:-2000}"
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
    printf "%-10s nodes=%-6d depth=%d  compile %8.1f ms  solve %8.1f ms  stats %6.1f ms  %7.0f iters/s (%.1f us/iter)\n", l, n, d, c, s, t, r, 1000 * s / i
  }'
}

echo "turn bench (iters=$ITERS)"
run standard Qs9h2d7c 200 500 "0.5,0.75" "2.5,3.0"
run wide Qs9h2d7c 300 1500 "0.25,0.5,0.75,1.25" "2.5,3.0"
run deep AhTd9s6h 300 1500 "0.25,0.5,0.75,1.25" "2.5,3.0"
run shortstack Qs9h2d7c 500 150 "0.5,0.75" "2.5,3.0"
