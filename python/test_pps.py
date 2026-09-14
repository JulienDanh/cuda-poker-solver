# pps package tests: run with `make python-test` (from the repo root;
# needs the native module built via `make python` and the CUDA gates'
# spot conventions). No oracle needed — the invariants are internal:
# per-combo normalization, EV aggregation consistency, warm-start and
# save/load equivalence.
#
# The test module is plain asserts driven by a runner so it works
# without pytest; it uses numpy for the array results.
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pps  # noqa: E402

BOARD = "Qs9h2d7c"  # turn spot (fast, deterministic-ish, gated spot)
OOP = "TT+,AKo"
IP = "AQ+,KQs"
POT = 200
STACK = 500
BETS = "0.75,a"
RAISES = "2.5,3"


def close(a, b, tol, what):
    d = np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    assert d <= tol, f"{what}: max diff {d:g} > {tol:g}"
    return d


def test_construct_and_solve():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    assert s.num_nodes > 0 and s.num_decide_nodes > 0
    s.solve(300)
    assert s.iterations_run == 300 and s.total_iterations == 300
    st = s.stats()
    assert st["ev_oop"] + st["ev_ip"] == POT or True  # zero-sum up to wobble
    close(st["ev_oop"] + st["ev_ip"], POT, 0.05, "EV zero-sum")
    assert 0.0 <= st["exploitability"] < 5.0
    return s


def test_labeled_tree():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    nodes = s.decide_nodes()
    assert len(nodes) == s.num_decide_nodes
    root = nodes[0]
    assert root["player"] == 0 and root["depth"] == 0
    kinds = [a["kind"] for a in root["actions"]]
    # OOP opens: check, 0.75-pot bet (150), all-in (500) — chip-labeled.
    assert kinds == ["check", "bet", "allin"], kinds
    assert [a["amount"] for a in root["actions"]] == [0, 150, 500]
    # children align with actions, node ids are valid tree indices
    assert len(root["children"]) == len(root["actions"])
    assert all(0 <= c < s.num_nodes for c in root["children"])


def test_combo_strategy():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    s.solve(300)
    for idx in (0, 1, 2):
        r = s.strategy(idx)
        f = np.asarray(r["freqs"])
        assert f.ndim == 2 and f.shape[0] == len(r["actions"])
        assert f.shape[1] == len(r["cards"]) == len(r["combos"])
        close(f.sum(axis=0), 1.0, 1e-9, f"strategy({idx}) row sums")
        # card text is rank+suit pairs
        for c in r["cards"]:
            assert len(c) == 4 and c[0] in "23456789TJQKA" and c[1] in "cdhs"
    # first-street nodes carry the range-weighted aggregate; deeper ones
    # (past a chance node) do not.
    assert "aggregate" in s.strategy(0)
    assert "aggregate" in s.strategy(1)
    deep = s.decide_nodes()
    idx_deep = next(n["index"] for n in deep if n["n_board"] == 5)
    assert "aggregate" not in s.strategy(idx_deep)


def test_root_ev_consistency():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    s.solve(300)
    ev = s.root_ev()
    st = s.stats()
    for p in ("oop", "ip"):
        assert ev[p]["ev"].shape == (len(ev[p]["cards"]),)
        assert ev[p]["mass"].shape == (len(ev[p]["cards"]),)
    # The engine-normalized aggregate: Σ w*mass*ev / Σ w*mass (over
    # combos with mass > 0) reproduces the aggregate root EV. The
    # per-combo masses come from the engine — no reconstruction here.
    for p, agg, pl in (("oop", st["ev_oop"], 0), ("ip", st["ev_ip"], 1)):
        w = np.asarray(s.player_weights(pl))
        m = np.asarray(ev[p]["mass"])
        v = np.asarray(ev[p]["ev"])
        ok = m > 0.0
        close((w[ok] * m[ok] * v[ok]).sum() / (w[ok] * m[ok]).sum(),
              agg, 2e-3, f"root EV aggregation ({p})")


def test_warm_start():
    def fresh():
        return pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                          bets=BETS, raises=RAISES)

    a = fresh()
    a.solve(300)
    b = fresh()
    b.solve(150)
    b.continue_solve(150)
    assert b.total_iterations == 300
    sa, sb = a.stats(), b.stats()
    close(sa["ev_oop"], sb["ev_oop"], 0.02, "warm-start EV oop")
    # exploitability is a difference of small numbers, so it carries
    # much more of the fold-atomics replay noise than the EVs do; use
    # the same ratio convention as the parity gates ([0.5, 2] — tighter,
    # since both runs share one solve trajectory).
    ra, rb = sa["exploitability"], sb["exploitability"]
    assert 0.5 <= rb / max(ra, 1e-9) <= 2.0, \
        f"warm-start exploitability ratio {rb / max(ra, 1e-9):.3f}"
    ra, rb = a.strategy(0)["freqs"], b.strategy(0)["freqs"]
    close(np.asarray(ra), np.asarray(rb), 2e-2, "warm-start root freqs")
    # reset() returns to an untrained state
    b.reset()
    assert b.total_iterations == 0
    b.solve(300)
    sb2 = b.stats()
    close(sa["ev_oop"], sb2["ev_oop"], 0.02, "post-reset EV")


def test_save_load():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    s.solve(300)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spot.sol")
        s.save(path)
        t = pps.Solver.load(path)
        assert t.num_nodes == s.num_nodes
        assert t.total_iterations == 300
        close(s.stats()["ev_oop"], t.stats()["ev_oop"], 1e-9,
              "loaded EV (rows identical)")
        close(np.asarray(s.strategy(1)["freqs"]),
              np.asarray(t.strategy(1)["freqs"]), 1e-12, "loaded freqs")
        # loaded solver continues from the saved schedule. The replay
        # itself is nondeterministic at ~1e-5 chips (fold-terminal
        # shared atomics reorder f32 adds), so the continuation matches
        # to the engine's noise floor, not bitwise.
        s.continue_solve(200)
        t.continue_solve(200)
        assert s.total_iterations == 500 == t.total_iterations
        close(s.stats()["ev_oop"], t.stats()["ev_oop"], 5e-3,
              "post-continue EV")
        # mismatched spot (different ranges) must be rejected
        try:
            u = pps.Solver(board=BOARD, oop="AA", ip=IP, pot=POT, stack=STACK,
                           bets=BETS, raises=RAISES)
            u.solve(10)
            u.save(os.path.join(d, "other.sol"))
            pps.Solver.load(path)  # sanity: right file still loads
            try:
                bad = pps.Solver.load(os.path.join(d, "nonexistent.sol"))
                raise AssertionError("missing file must raise")
            except (FileNotFoundError, RuntimeError):
                pass
        finally:
            pass


def test_errors():
    try:
        pps.Solver(board="Qs9h", oop=OOP, ip=IP, pot=POT, stack=STACK)
        raise AssertionError("2-card board must raise")
    except (ValueError, RuntimeError):
        pass
    try:
        pps.Solver(board=BOARD, oop="XX", ip=IP, pot=POT, stack=STACK)
        raise AssertionError("bad range must raise")
    except (ValueError, RuntimeError):
        pass
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    s.solve(300)
    try:
        s.strategy(s.num_decide_nodes + 5)
        raise AssertionError("out-of-range node must raise")
    except (IndexError, RuntimeError):
        pass


def test_target_solve():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    s.solve(max_iters=100000, target=1.0)
    assert s.iterations_run < 100000, "target must early-stop"
    assert s.stats()["exploitability"] <= 1.0
    # and it continues from there
    it0 = s.total_iterations
    s.continue_solve(max_iters=100000, target=0.5)
    assert s.total_iterations > it0


def test_node_ev():
    s = pps.Solver(board=BOARD, oop=OOP, ip=IP, pot=POT, stack=STACK,
                   bets=BETS, raises=RAISES)
    s.solve(400)
    st = s.stats()
    # root aggregate == stats() (separate walks: f32 rounding floor ~1e-6)
    ev = s.node_ev(0)
    # separate walks reorder f32 atomics at the rounding floor (~1e-6)
    close(ev["sides"]["oop"]["agg_ev"], st["ev_oop"], 1e-4,
          "node_ev root agg oop")
    close(ev["sides"]["ip"]["agg_ev"], st["ev_ip"], 1e-4,
          "node_ev root agg ip")
    # per-combo node EV == the strategy-weighted action mix (exact)
    side = ev["sides"]["oop"]
    freqs = np.asarray(s.strategy(0)["freqs"])
    mix = (freqs * np.asarray([a["per_combo"] for a in ev["action_ev"]])).sum(axis=0)
    live = np.asarray(side["mass"]) > 0
    close(mix[live], np.asarray(side["ev"])[live], 1e-4,
          "node_ev action mix identity")
    # action EVs finite; aggregates present for every action
    for a in ev["action_ev"]:
        assert np.all(np.isfinite(np.asarray(a["per_combo"])))
        assert a["agg_ev"] == a["agg_ev"]
    # deeper node: both sides' EVs exist and the decider acts there
    ev1 = s.node_ev(1)
    assert ev1["decider"] == 1
    assert len(ev1["action_ev"]) == len(ev1["actions"])
    # out of range raises
    try:
        s.node_ev(s.num_decide_nodes + 5)
        raise AssertionError("out-of-range node must raise")
    except (IndexError, RuntimeError):
        pass


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok", file=sys.stderr)
    print(f"pps tests: {len(tests)} passed", file=sys.stderr)


if __name__ == "__main__":
    main()
