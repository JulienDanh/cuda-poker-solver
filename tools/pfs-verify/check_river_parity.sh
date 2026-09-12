#!/bin/sh
# Head-to-head parity check: our range-based river solver (build/pfflop)
# against b-inary/postflop-solver (pfs-verify solve-river) on identical
# spots. EVs must agree within solver noise (f32 vs f64) and
# exploitabilities within a few percent at the same iteration count.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PFF="$ROOT/build/pfflop"
PFS="$ROOT/tools/pfs-verify/target/release/pfs-verify"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"
ITERS="${1:-1000}"

if [ ! -x "$PFS" ]; then
  (cd "$ROOT/tools/pfs-verify" && cargo build --release)
fi

fail=0
for spot in "Qs9h2d7c8d" "7h8h9c2c2s" "KdKc4cQh3s" "2d7sJh3c3s" "AhTd9s6h6d"; do
  ours=$("$PFF" river "$spot" "$OOP" "$IP" 1000 500 "0.5,0.75,1.0" "2.5,3.0" "$ITERS" --algo dcfr)
  theirs=$("$PFS" solve-river "$spot" "$OOP" "$IP" 1000 500 "50%,75%,100%" "2.5x,3x" "$ITERS")
  ev0o=$(echo "$ours" | grep "^EV 0" | awk '{print $3}')
  ev0t=$(echo "$theirs" | grep "^EV 0" | awk '{print $3}')
  exo=$(echo "$ours" | grep EXPLOIT | awk '{print $2}')
  ext=$(echo "$theirs" | grep EXPLOIT | awk '{print $2}')
  evdiff=$(awk -v a="$ev0o" -v b="$ev0t" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.6f", d}')
  exratio=$(awk -v a="$exo" -v b="$ext" 'BEGIN{if(b<=0){print 0} else {r=a/b; printf "%.3f", r}}')
  ok=$(awk -v d="$evdiff" 'BEGIN{print (d < 0.01) ? "OK" : "FAIL"}')
  echo "board $spot: EV0 ours=$ev0o theirs=$ev0t (diff $evdiff)  expl ours=$exo theirs=$ext (ratio $exratio)  [$ok]"
  [ "$ok" = "FAIL" ] && fail=1
done
[ "$fail" = "0" ] && echo "PARITY OK" || { echo "PARITY FAILED"; exit 1; }
