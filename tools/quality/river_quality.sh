#!/bin/sh
# Golden-baseline quality gate for the range-based river solver (part 2 of
# the harness). Unlike river-parity (which needs the oracle + cargo), this
# runs standalone and catches quality drift from performance work:
#   - EV0 within 0.02 chips of the recorded baseline
#   - exploitability within [0.5x, 2.0x] of the baseline at the same
#     iteration count
#   - exploitability strictly decreasing across the 300 -> 3000 iteration
#     ladder (monotone convergence, independent of the oracle)
#
# Regenerate the baseline after an intentional algorithm change:
#   make river-baseline
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
PFF="$ROOTDIR/build/pfflop"
BASE="$ROOTDIR/tools/quality/river_baseline.txt"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"

# Fixed spot list: board|pot|stack|bets|raises|algo|iters
SPOTS="Qs9h2d7c8d|1000|500|0.5,0.75,1.0|2.5,3.0|dcfr|300
Qs9h2d7c8d|1000|500|0.5,0.75,1.0|2.5,3.0|dcfr|3000
7h8h9c2c2s|1000|500|0.5,0.75,1.0|2.5,3.0|dcfr|300
7h8h9c2c2s|1000|500|0.5,0.75,1.0|2.5,3.0|dcfr|3000
KdKc4cQh3s|1000|500|0.5,0.75,1.0|2.5,3.0|dcfr|300
KdKc4cQh3s|1000|500|0.5,0.75,1.0|2.5,3.0|dcfr|3000
Qs9h2d7c8d|200|900|0.33,0.6,1.0|2.5,3.0|dcfr|300
Qs9h2d7c8d|200|900|0.33,0.6,1.0|2.5,3.0|dcfr|3000
KdKc4cQh3s|200|900|0.33,0.6,1.0|2.5,3.0|dcfr|300
KdKc4cQh3s|200|900|0.33,0.6,1.0|2.5,3.0|dcfr|3000
7h8h9c2c2s|300|1500|0.25,0.5,0.75,1.25|2.5,3.0|dcfr|300
7h8h9c2c2s|300|1500|0.25,0.5,0.75,1.25|2.5,3.0|dcfr|3000"

if [ ! -x "$PFF" ]; then
  make -C "$ROOTDIR" build/pfflop
fi

run_spot() {
  board=$1; pot=$2; stack=$3; bets=$4; raises=$5; algo=$6; iters=$7
  "$PFF" river "$board" "$OOP" "$IP" "$pot" "$stack" "$bets" "$raises" "$iters" --algo "$algo"
}

if [ "${1:-}" = "--generate" ]; then
  {
    echo "# Golden baseline for the range-based river solver (quality harness)."
    echo "# Regenerate intentionally with: make river-baseline"
    echo "for spot in $SPOTS" | tr '\n' '\n' > /dev/null  # no-op guard
    echo "$SPOTS" | while IFS='|' read -r board pot stack bets raises algo iters; do
      out=$(run_spot "$board" "$pot" "$stack" "$bets" "$raises" "$algo" "$iters")
      ev0=$(echo "$out" | grep "^EV 0" | awk '{print $3}')
      ev1=$(echo "$out" | grep "^EV 1" | awk '{print $3}')
      ex=$(echo "$out" | grep EXPLOIT | awk '{print $2}')
      echo "$board|$pot|$stack|$bets|$raises|$algo|$iters|$ev0|$ev1|$ex"
    done
  } > "$BASE"
  echo "baseline regenerated: $BASE"
  exit 0
fi

if [ ! -f "$BASE" ]; then
  echo "no baseline; run: make river-baseline" >&2
  exit 2
fi

fail=0
echo "$SPOTS" | while IFS='|' read -r board pot stack bets raises algo iters; do
  out=$(run_spot "$board" "$pot" "$stack" "$bets" "$raises" "$algo" "$iters")
  ev0=$(echo "$out" | grep "^EV 0" | awk '{print $3}')
  ex=$(echo "$out" | grep EXPLOIT | awk '{print $2}')
  line=$(grep -v '^#' "$BASE" | grep "^$board|$pot|$stack|$bets|$raises|$algo|$iters|")
  if [ -z "$line" ]; then
    echo "board $board pot=$pot iters=$iters: MISSING FROM BASELINE [FAIL]"
    continue
  fi
  ev0b=$(echo "$line" | awk -F'|' '{print $8}')
  exb=$(echo "$line" | awk -F'|' '{print $10}')
  res=$(awk -v e="$ev0" -v eb="$ev0b" -v x="$ex" -v xb="$exb" 'BEGIN {
    d = e - eb; if (d < 0) d = -d
    evok = (d < 0.02) ? 1 : 0
    if (xb <= 0) r = 1; else r = x / xb
    exok = (r >= 0.5 && r <= 2.0) ? 1 : 0
    status = "OK"
    if (!evok) status = "FAIL(ev)"
    if (!exok) status = "FAIL(expl)"
    printf "%.4f %.3f %s", d, r, status
  }')
  status=$(echo "$res" | awk '{print $3}')
  echo "board $board pot=$pot iters=$iters: EV0 diff=$(echo "$res" | awk '{print $1}')  expl ratio=$(echo "$res" | awk '{print $2}')  [$status]"
  if [ "$status" != "OK" ]; then echo "$board|$pot|$iters" >> /tmp/river_quality_fails; fi
done

if [ -f /tmp/river_quality_fails ]; then fail=1; rm -f /tmp/river_quality_fails; fi

# Monotone convergence across the iteration ladder.
echo "$SPOTS" | while IFS='|' read -r board pot stack bets raises algo iters; do
  case "$iters" in 3000) ;; *) continue ;; esac
  x1=$(run_spot "$board" "$pot" "$stack" "$bets" "$raises" "$algo" 300 | grep EXPLOIT | awk '{print $2}')
  x2=$(run_spot "$board" "$pot" "$stack" "$bets" "$raises" "$algo" 3000 | grep EXPLOIT | awk '{print $2}')
  ok=$(awk -v a="$x1" -v b="$x2" 'BEGIN{print (b < a) ? 1 : 0}')
  echo "monotone convergence $board pot=$pot: expl 300=$x1 -> 3000=$x2  [$([ $ok = 1 ] && echo OK || echo FAIL)]"
  if [ "$ok" != "1" ]; then echo "$board" >> /tmp/river_quality_fails; fi
done

if [ -f /tmp/river_quality_fails ]; then
  rm -f /tmp/river_quality_fails
  echo "QUALITY FAILED"
  exit 1
fi
echo "QUALITY OK"
