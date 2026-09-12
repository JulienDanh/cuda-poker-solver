#!/bin/sh
# River solver benchmark for performance tracking (harness, part 3).
# Reports wall time and iterations/sec on fixed spots. Run before/after
# any performance change; quality gates must still pass:
#   make test river-parity river-quality
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
PFF="$ROOTDIR/build/pfflop"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"
echo "river solver benchmark ($(date '+%Y-%m-%d %H:%M:%S'))"
for spot in "Qs9h2d7c8d 1000 500 0.5,0.75,1.0" "7h8h9c2c2s 300 1500 0.25,0.5,0.75,1.25"; do
  set -- $spot
  b=$1; pot=$2; st=$3; bets=$4
  start=$(python3 -c 'import time; print(time.time())')
  "$PFF" river "$b" "$OOP" "$IP" "$pot" "$st" "$bets" "2.5,3.0" 30000 --algo dcfr > /dev/null
  end=$(python3 -c 'import time; print(time.time())')
  python3 -c "print(f'board $b pot=$pot stack=$st: 30000 iters in {$end-$start:.3f}s = {30000/($end-$start)/1000:.0f}k iters/s')"
done
