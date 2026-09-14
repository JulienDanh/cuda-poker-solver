#!/bin/sh
# GPU-CFR turn solver vs postflop-solver oracle (4-card boards, river
# to come), plus the hand-computed tiny-turn ground truth. The solver
# also accepts flop (3-card) and river (5-card) boards; the current
# focus is turn performance, so the gates cover turn spots only.
#
# Modes (first argument, default "quick"):
#   quick  - the debug loop: low iteration counts and loose EV gates
#            (~20 s total). Catches tree/convention regressions fast:
#            the kind of bugs this gate found (a 0.38-chip EV gap from a
#            stack-geometry error, an all-in-runout recursion) show up
#            at any iteration count. Use this while iterating.
#   full   - the commit gate: all spots at higher iteration counts with
#            the tight gates (EV within 0.01 chips of the f32 oracle,
#            exploitability ratio within [0.5, 2.0], root strategy
#            within 0.10). The oracle flop solves dominate (~30 s
#            each); expect a few minutes.
#
# Both modes also run the tiny turn ground truth (tools/tiny_check.cpp):
# the spot is solved analytically (uniform strategies + two seeded
# profiles) in f64 and the GPU value walk (f32) must reproduce it to
# 1e-4. This checks the chance-node conventions (1/44 branch weight,
# per-pair masking, prior-street contributions) independent of any
# other engine; convention bugs produce errors >= 0.01, two orders of
# magnitude above the gate.
#
# Needs the cmake CUDA build (cmake -B build-cuda -DENABLE_CUDA=ON) and
# cargo for the oracle (tools/pfs-verify).
set -e
ROOTDIR="$(cd "$(dirname "$0")/../.." && pwd)"
GPU="${GPU_PFFLOP:-$ROOTDIR/build-cuda/gpu_pfflop}"
PFS="$ROOTDIR/tools/pfs-verify/target/release/pfs-verify"
TINY="$ROOTDIR/build/tiny_check"
OOP="22+,A9s+,KTs+,QJs,JTs,T9s"
IP="22+,A9s+,KTs+,QJs,JTs,T9s,AQo+"
MODE="${1:-quick}"
case "$MODE" in
  quick)
    TURN_ITERS=200
    EVGATE=0.10; EXLO=0.25; EXHI=4.0; STRGATE=0.15 ;;
  full)
    TURN_ITERS=500
    EVGATE=0.01; EXLO=0.50; EXHI=2.0; STRGATE=0.10 ;;
  *) echo "usage: $0 [quick|full]" >&2; exit 2 ;;
esac

if [ ! -x "$GPU" ]; then
  echo "gpu_pfflop not found at $GPU (build with ENABLE_CUDA=ON)" >&2
  exit 2
fi
if [ ! -x "$TINY" ]; then
  make -C "$ROOTDIR" build/tiny_check
fi
if [ ! -x "$PFS" ]; then
  (cd "$ROOTDIR/tools/pfs-verify" && cargo build --release)
fi

fail=0

# <board> <pot> <stack> <bets> <raises> <iters> [oop] [ip]
# The oracle's per-iteration cost is ~10x a river solve (one chance
# street), so the full mode runs at a moderate iteration count.
run_spot() {
  board="$1"; pot="$2"; stack="$3"; bets="$4"; raises="$5"; iters="$6"
  oopr="${7:-$OOP}"; ipr="${8:-$IP}"
  # our bet format "0.5,0.75" -> oracle "50%,75%"; raises "2.5,3.0" -> "2.5x,3x"
  pbets=$(echo "$bets" | awk -F',' '{s=""; for(i=1;i<=NF;i++){if(i>1)s=s","; s=s sprintf("%g%%", $i*100)} print s}')
  praises=$(echo "$raises" | awk -F',' '{s=""; for(i=1;i<=NF;i++){if(i>1)s=s","; s=s sprintf("%gx", $i)} print s}')
  case $((${#board} / 2)) in
    3) sub=solve ;;
    4) sub=solve-turn ;;
    5) sub=solve-river ;;
    *) echo "bad board $board" >&2; exit 2 ;;
  esac
  ours=$("$GPU" postflop "$board" "$oopr" "$ipr" "$pot" "$stack" "$bets" \
         "$raises" "$iters" --algo dcfr --root --strat 1 2>/dev/null)
  theirs=$("$PFS" "$sub" "$board" "$oopr" "$ipr" "$pot" "$stack" "$pbets" \
           "$praises" "$iters" 2>/dev/null)
  ev0o=$(echo "$ours" | grep "^EV 0" | awk '{print $3}')
  ev0t=$(echo "$theirs" | grep "^EV 0" | awk '{print $3}')
  exo=$(echo "$ours" | grep EXPLOIT | awk '{print $2}')
  ext=$(echo "$theirs" | grep EXPLOIT | awk '{print $2}')
  rso=$(echo "$ours" | grep ROOTSTRAT | cut -d' ' -f2-)
  rst=$(echo "$theirs" | grep ROOTSTRAT | cut -d' ' -f2-)
  # IP's first decision after OOP checks (decide node 1) vs the oracle's
  # same node (reached by playing the root check).
  nso=$(echo "$ours" | grep "^STRAT" | cut -d' ' -f3-)
  nst=$(echo "$theirs" | grep NODESTRAT | cut -d' ' -f2-)
  result=$(awk -v a="$ev0o" -v b="$ev0t" -v x="$exo" -v y="$ext" \
              -v ro="$rso" -v rt="$rst" -v gate="$EVGATE" \
              -v no="$nso" -v nt="$nst" \
              -v xlo="$EXLO" -v xhi="$EXHI" -v sgate="$STRGATE" 'BEGIN {
    d = a - b; if (d < 0) d = -d
    if (y <= 0) r = 1; else r = x / y
    m = 0; n = split(ro, A, " "); split(rt, B, " ")
    for (i = 1; i <= n; i++) { t = A[i] - B[i]; if (t < 0) t = -t; if (t > m) m = t }
    nm = -1; k = split(no, C, " "); tl = split(nt, D, " ")
    if (k == 0 && tl > 0) nm = 1; else if (k == 0) nm = 0; else {
      nm = 0
      for (i = 1; i <= k; i++) { t = C[i] - D[i]; if (t < 0) t = -t; if (t > nm) nm = t }
    }
    evok = (d < gate) ? 1 : 0
    exok = (r >= xlo && r <= xhi) ? 1 : 0
    rsok = (m < sgate) ? 1 : 0
    nsok = (nm >= 0 && nm < sgate) ? 1 : 0
    status = "OK"
    if (!evok) status = "FAIL(ev)"
    if (!exok) status = status "=FAIL(expl)"
    if (!rsok) status = status "=FAIL(strat)"
    if (!nsok) status = status "=FAIL(nstrat)"
    printf "%.6f %.4f %.4f %.4f %s", d, r, m, nm, status
  }')
  status=$(echo "$result" | awk '{print $5}')
  echo "$board pot=$pot stack=$stack iters=$iters: EV diff=$(echo "$result" | awk '{print $1}')  expl ratio=$(echo "$result" | awk '{print $2}')  strat maxdiff=$(echo "$result" | awk '{print $3}')  node1 maxdiff=$(echo "$result" | awk '{print $4}')  [$status]"
  if [ "$status" != "OK" ]; then fail=1; fi
}

# Tiny turn ground truth (independent of the oracle): uniform and two
# seeded strategy profiles, hand-computed in tools/tiny_check.cpp.
tiny_check() {
  label="$1"; got="$2"; want="$3"
  result=$(awk -v a="$got" -v b="$want" 'BEGIN {
    d = a - b; if (d < 0) d = -d
    printf "%.2e %s", d, (d < 1e-4) ? "OK" : "FAIL"
  }')
  status=$(echo "$result" | awk '{print $2}')
  echo "tiny $label: EV0=$got want=$want diff=$(echo "$result" | awk '{print $1}')  [$status]"
  if [ "$status" != "OK" ]; then fail=1; fi
}

run_tiny() {
  tout=$("$TINY")
  tiny_check uniform \
    "$("$GPU" postflop Qs9h2d7c "TT+" "99+" 200 3000 0.5 "" 0 2>/dev/null | grep '^EV 0' | awk '{print $3}')" \
    "$(echo "$tout" | grep '^TINY:' | awk '{print $3}')"
  tiny_check seed-ipTop \
    "$(GPU_CFR_SEED_STRAT=7 "$GPU" postflop Qs9h2d7c "TT+" "99+" 200 3000 0.5 "" 0 2>/dev/null | grep '^EV 0' | awk '{print $3}')" \
    "$(echo "$tout" | grep '^M7:' | awk '{print $3}')"
  tiny_check seed-ipFace \
    "$(GPU_CFR_SEED_STRAT=9 "$GPU" postflop Qs9h2d7c "TT+" "99+" 200 3000 0.5 "" 0 2>/dev/null | grep '^EV 0' | awk '{print $3}')" \
    "$(echo "$tout" | grep '^M9:' | awk '{print $3}')"
}

if [ "$MODE" = quick ]; then
  echo "== turn spots vs postflop-solver (quick)"
  run_spot Qs9h2d7c 200 500 "0.5,0.75" "2.5,3.0" "$TURN_ITERS"
  run_spot AhTd9s6h 300 1500 "0.25,0.5,0.75,1.25" "2.5,3.0" "$TURN_ITERS"
  # short stack: all-in on the first street, the rest of the board is a
  # pure chance runout (regression for the dead-street path in the build).
  run_spot Qs9h2d7c 500 150 "0.5,0.75" "2.5,3.0" "$TURN_ITERS"
  run_spot 7h8h9c2c 200 900 "0.33,0.6,1.0" "2.5,3.0" "$TURN_ITERS"
  run_tiny
else
  echo "== turn spots vs postflop-solver (full)"
  run_spot Qs9h2d7c 200 500 "0.5,0.75" "2.5,3.0" "$TURN_ITERS"
  run_spot 7h8h9c2c 200 900 "0.33,0.6,1.0" "2.5,3.0" "$TURN_ITERS"
  run_spot KdKc4cQh 1000 500 "0.5,0.75,1.0" "2.5,3.0" "$TURN_ITERS"
  run_spot AhTd9s6h 300 1500 "0.25,0.5,0.75,1.25" "2.5,3.0" "$TURN_ITERS"
  run_spot 2d7sJh3c 200 900 "0.33,0.6,1.0" "2.5,3.0" "$TURN_ITERS"
  # short stack: all-in on the first street, the rest of the board is a
  # pure chance runout (regression for the dead-street path in the build).
  run_spot Qs9h2d7c 500 150 "0.5,0.75" "2.5,3.0" "$TURN_ITERS"
  run_tiny
fi

if [ "$fail" = "0" ]; then
  echo "GPU POSTFLOP PARITY OK ($MODE)"
else
  echo "GPU POSTFLOP PARITY FAILED ($MODE)"
  exit 1
fi
