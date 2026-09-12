#!/bin/sh
# Cross-validates our 7-card hand evaluator against
# b-inary/postflop-solver's Hand::evaluate(): the two total orders must
# agree exactly (equal hands equal, stronger hands stronger) across N
# random 7-card hands.
#
# usage: tools/pfs-verify/check_eval7.sh [num_hands]
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PFS="$ROOT/tools/pfs-verify/target/release/pfs-verify"
VEQ="$ROOT/build/verify_eval7"
N="${1:-4000000}"

if [ ! -x "$PFS" ]; then
  echo "building pfs-verify (cargo)..." >&2
  (cd "$ROOT/tools/pfs-verify" && cargo build --release)
fi
if [ ! -x "$VEQ" ]; then
  echo "building verify_eval7..." >&2
  make -C "$ROOT" build/verify_eval7
fi

"$VEQ" "$N" "$PFS"
