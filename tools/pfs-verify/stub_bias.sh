#!/bin/sh
# Measures the FGS continuation stub's EV bias: for a heads-up flop spot
# (pot 500, BB=OOP vs SB=IP), compares the pure equity split of the pot
# (what our ShowdownContinuation stub computes; approximated by a forced
# all-in solve with effective_stack=1) against a full postflop solve
# with the same ranges and a realistic stack.
#
# usage: tools/pfs-verify/stub_bias.sh
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PFS="$ROOT/tools/pfs-verify/target/release/pfs-verify"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"

if [ ! -x "$PFS" ]; then
  (cd "$ROOT/tools/pfs-verify" && cargo build --release)
fi

for flop in "Qs9h2d" "7h8h9c" "KdKc4c"; do
  echo "== flop $flop (pot 500; EV 0 = BB/OOP, EV 1 = SB/IP)"
  echo "  equity reference (forced all-in; what our stub approximates):"
  "$PFS" solve "$flop" "$OOP" "$IP" 500 1 "60%,a" "2.5x" 20 | sed 's/^/    /'
  echo "  full postflop solve (SPR 4; reality):"
  "$PFS" solve "$flop" "$OOP" "$IP" 500 2000 "60%,a" "2.5x" 400 2.0 | sed 's/^/    /'
done
