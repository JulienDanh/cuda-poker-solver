#!/bin/sh
# One-command feedback loop for GPU performance work.
#
#   1. rebuild build-cuda (incremental nvcc, ~5 s)
#   2. correctness gate: gpu-quick (4 turn spots vs the oracle + the
#      tiny-turn hand-computed ground truth, ~6 s)
#   3. benchmark: turn-bench (4 geometries, fixed iteration count)
#   4. compare iters/s per geometry against the previous run and flag
#      anything slower than 95% of it
#
# The bench history lives in build-cuda/.bench_last (gitignored); each
# run overwrites it, so the comparison is always against the immediately
# preceding run. Iteration count is fixed at the default (2000) unless
# overridden, so runs stay comparable.
#
# usage: perf_loop.sh [bench_iters]
# exit: nonzero if the build, the gate, or the perf comparison fails.
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOTDIR/build-cuda"
BENCH="$ROOTDIR/tools/quality/bench_turn.sh"
GATE="$ROOTDIR/tools/quality/gpu_postflop_parity.sh"
BENCHHIST="$BUILD/.bench_last"
ITERS="${1:-2000}"

if [ ! -f "$BUILD/Makefile" ]; then
  echo "no cmake build at $BUILD (see Makefile header for the setup)" >&2
  exit 2
fi

echo "== rebuild"
if ! cmake --build "$BUILD" -j8 > "$BUILD/.build.log" 2>&1; then
  cat "$BUILD/.build.log"
  echo "PERF LOOP: BUILD FAILED" >&2
  exit 1
fi

echo "== gpu-quick gate"
"$GATE" quick

echo "== turn bench (iters=$ITERS)"
"$BENCH" "$ITERS" | tee "$BENCHHIST.new"
echo ""

if [ -f "$BENCHHIST" ]; then
  echo "== vs previous run"
  if awk '
        function getrate(i) {
          for (i = NF; i >= 1; i--)
            if ($i == "iters/s") return $(i - 1)
          return -1
        }
        NR == FNR { if (NF > 1) oldr[$1] = getrate(); next }
        NF > 1 && $1 in oldr {
          r = getrate(); o = oldr[$1]
          if (r > 0 && o > 0) {
            ratio = r / o
            printf "  %-10s %7.0f -> %7.0f iters/s  x%.3f%s\n", \
              $1, o, r, ratio, (ratio < 0.95) ? "  REGRESSION" : ""
            if (ratio < 0.95) bad = 1
          }
        }
        END { exit bad ? 1 : 0 }
      ' "$BENCHHIST" "$BENCHHIST.new"; then
    :
  else
    mv "$BENCHHIST.new" "$BENCHHIST"
    echo "PERF LOOP: correctness OK, bench REGRESSION vs previous run" >&2
    exit 1
  fi
else
  echo "== no previous bench; baseline saved"
fi
mv "$BENCHHIST.new" "$BENCHHIST"
echo "PERF LOOP OK"
