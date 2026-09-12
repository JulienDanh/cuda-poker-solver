#!/bin/sh
# Head-to-head parity gate vs b-inary/postflop-solver (quality harness, part 1).
# Gates (same spot, same iterations, both engines):
#   - EV0 must agree within 0.01 chips (f32-vs-f64 noise is ~1e-4)
#   - exploitability ratio must be within [0.5, 2.0] (measured noise band
#     across spots/counts is 0.88-1.48 from f64-vs-f32 convergence wobble;
#     a broken best-response walk is off by 10x+, not 2x)
#   - root strategy (OOP aggregate frequencies) within 0.10 per action
#     (exact strategy equality is NOT required: different equilibria or
#     slow mixing on indifferent actions are legitimate; EV/expl are the
#     real gates)
# usage: check_river_parity.sh [iters]
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
PFF="$ROOTDIR/build/pfflop"
PFS="$ROOTDIR/tools/pfs-verify/target/release/pfs-verify"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"
ITERS="${1:-1000}"

if [ ! -x "$PFS" ]; then
  (cd "$ROOTDIR/tools/pfs-verify" && cargo build --release)
fi

fail=0
run_spot() {
  board="$1"; pot="$2"; stack="$3"; bets="$4"; raises="$5"; pbets="$6"; praises="$7"
  ours=$("$PFF" river "$board" "$OOP" "$IP" "$pot" "$stack" "$bets" "$raises" "$ITERS" --algo dcfr --root)
  theirs=$("$PFS" solve-river "$board" "$OOP" "$IP" "$pot" "$stack" "$pbets" "$praises" "$ITERS")
  ev0o=$(echo "$ours" | grep "^EV 0" | awk '{print $3}')
  ev0t=$(echo "$theirs" | grep "^EV 0" | awk '{print $3}')
  exo=$(echo "$ours" | grep EXPLOIT | awk '{print $2}')
  ext=$(echo "$theirs" | grep EXPLOIT | awk '{print $2}')
  rso=$(echo "$ours" | grep ROOTSTRAT | cut -d' ' -f2-)
  rst=$(echo "$theirs" | grep ROOTSTRAT | cut -d' ' -f2-)
  result=$(awk -v a="$ev0o" -v b="$ev0t" -v x="$exo" -v y="$ext" \
              -v ro="$rso" -v rt="$rst" 'BEGIN {
    d = a - b; if (d < 0) d = -d
    if (y <= 0) r = 1; else r = x / y
    m = 0; n = split(ro, A, " "); split(rt, B, " ")
    for (i = 1; i <= n; i++) { t = A[i] - B[i]; if (t < 0) t = -t; if (t > m) m = t }
    evok = (d < 0.01) ? 1 : 0
    exok = (r >= 0.50 && r <= 2.00) ? 1 : 0
    rsok = (m < 0.10) ? 1 : 0
    status = "OK"
    if (!evok) status = "FAIL(ev)"
    if (!exok) status = "FAIL(expl)"
    if (!rsok) status = "FAIL(strat)"
    printf "%.6f %.4f %.4f %s", d, r, m, status
  }')
  evdiff=$(echo "$result" | awk '{print $1}')
  exratio=$(echo "$result" | awk '{print $2}')
  rsmax=$(echo "$result" | awk '{print $3}')
  status=$(echo "$result" | awk '{print $4}')
  echo "board $board pot=$pot stack=$stack: EV diff=$evdiff  expl ratio=$exratio  strat maxdiff=$rsmax  [$status]"
  if [ "$status" != "OK" ]; then fail=1; fi
}

run_spot "Qs9h2d7c8d" 1000 500 "0.5,0.75,1.0" "2.5,3.0" "50%,75%,100%" "2.5x,3x"
run_spot "7h8h9c2c2s" 1000 500 "0.5,0.75,1.0" "2.5,3.0" "50%,75%,100%" "2.5x,3x"
run_spot "KdKc4cQh3s" 1000 500 "0.5,0.75,1.0" "2.5,3.0" "50%,75%,100%" "2.5x,3x"
run_spot "2d7sJh3c3s" 1000 500 "0.5,0.75,1.0" "2.5,3.0" "50%,75%,100%" "2.5x,3x"
run_spot "AhTd9s6h6d" 1000 500 "0.5,0.75,1.0" "2.5,3.0" "50%,75%,100%" "2.5x,3x"
# different geometry: small pot / big stack
run_spot "Qs9h2d7c8d" 200 900 "0.33,0.6,1.0" "2.5,3.0" "33%,60%,100%" "2.5x,3x"
run_spot "KdKc4cQh3s" 200 900 "0.33,0.6,1.0" "2.5,3.0" "33%,60%,100%" "2.5x,3x"
# deep SPR with wide ranges
run_spot "7h8h9c2c2s" 300 1500 "0.25,0.5,0.75,1.25" "2.5,3.0" "25%,50%,75%,125%" "2.5x,3x"

if [ "$fail" = "0" ]; then
  echo "PARITY OK"
else
  echo "PARITY FAILED"
  exit 1
fi
