#!/bin/zsh
# Quick serial-vs-parallel timing comparison (scratch, not part of the harness).
set -e
PFF=build/pfflop
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"
run() {
  local th=$1 b=$2 pot=$3 st=$4 bets=$5 iters=$6
  local s=$(python3 -c 'import time; print(time.time())')
  "$PFF" river "$b" "$OOP" "$IP" "$pot" "$st" "$bets" "2.5,3.0" "$iters" --algo dcfr --threads "$th" > /dev/null
  local e=$(python3 -c 'import time; print(time.time())')
  python3 -c "print(f'th=$th $b pot=$pot: {$iters/($e-$s)/1000:.0f}k iters/s')"
}
for th in 1 0 8; do
  run $th Qs9h2d7c8d 1000 500 0.5,0.75,1.0 30000
  run $th 7h8h9c2c2s 300 1500 0.25,0.5,0.75,1.25 30000
done
