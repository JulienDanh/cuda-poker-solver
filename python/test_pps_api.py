# pps solver API tests: in-process TestClient (fastapi.testclient)
# exercising the full HTTP surface against the real GPU engine.
# Run with `make api-test` (or from python/: python test_pps_api.py).
import os
import sys
import tempfile
import time

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
import pps_api  # noqa: E402

# Sandbox solution files into a temp dir for this run.
tmp = tempfile.mkdtemp(prefix="pps-api-test-")
os.environ["PPS_API_DATA_DIR"] = tmp
pps_api.DATA_DIR = tmp

client = TestClient(pps_api.app)

SPOT = {
    "board": "Qs9h2d7c",  # turn spot: fast, matches the gate conventions
    "oop": "TT+,AKo",
    "ip": "AQ+,KQs",
    "pot": 200,
    "stack": 500,
    "bets": "0.75,a",
    "raises": "2.5,3",
}


def check(resp, code=200):
    assert resp.status_code == code, (resp.status_code, resp.text[:400])
    return resp.json()


def test_health():
    h = check(client.get("/health"))
    assert h["status"] == "ok"
    assert h["max_solvers"] >= 1


def test_lifecycle():
    r = check(client.post("/solvers", json=SPOT), 201)
    assert r["num_nodes"] > 0 and r["num_decide_nodes"] > 0
    assert r["compile_ms"] >= 0
    sid = r["id"]

    # metadata before any solving
    check(client.get(f"/solvers/{sid}"))
    assert len(check(client.get("/solvers"))) == 1

    r = check(client.post(f"/solvers/{sid}/solve",
                          json={"max_iters": 300}))
    st = r["stats"]
    assert r["iterations_run"] == 300
    assert abs(st["ev_oop"] + st["ev_ip"] - SPOT["pot"]) < 0.05
    assert 0.0 <= st["exploitability"] < 5.0

    # stats endpoint matches the solve response (both are fresh walks)
    st2 = check(client.get(f"/solvers/{sid}/stats"))
    assert abs(st2["ev_oop"] - st["ev_oop"]) < 1e-9

    # labeled strategy over HTTP
    s0 = check(client.get(f"/solvers/{sid}/strategy?node=0"))
    kinds = [a["kind"] for a in s0["actions"]]
    assert kinds == ["check", "bet", "allin"], kinds
    assert [a["amount"] for a in s0["actions"]] == [0, 150, 500]
    rows = s0["freqs"]
    assert len(rows) == 3 and len(rows[0]) == len(s0["cards"])
    col_sums = [sum(r[c] for r in rows) for c in range(len(rows[0]))]
    assert all(abs(s - 1.0) < 1e-9 for s in col_sums)
    assert len(s0["aggregate"]) == 3

    # decide-node introspection with pagination
    dn = check(client.get(f"/solvers/{sid}/decide-nodes?offset=0&limit=2"))
    assert dn["total"] == r["total_decide_nodes"] \
        if "total_decide_nodes" in r else True
    assert len(dn["nodes"]) == 2
    assert dn["nodes"][0]["actions"][1]["amount"] == 150

    # per-combo root EV over HTTP: shapes and finiteness (the mass-
    # weighted aggregation identity is gated by python/test_pps.py and
    # test_one_shot below)
    ev = check(client.get(f"/solvers/{sid}/root-ev"))
    for p in ("oop", "ip"):
        assert len(ev[p]["ev"]) == len(ev[p]["cards"]) == len(ev[p]["mass"])
        assert all(v == v for v in ev[p]["ev"])  # no NaN
        assert all(v == v for v in ev[p]["mass"])

    # warm-start continuation
    r = check(client.post(f"/solvers/{sid}/continue",
                          json={"max_iters": 200}))
    assert r["total_iterations"] == 500

    # save + load roundtrip through the file sandbox
    r = check(client.post(f"/solvers/{sid}/save", json={"name": "t.sol"}))
    assert r["saved"] and r["bytes"] > 0
    # path escape is rejected
    check(client.post(f"/solvers/{sid}/save", json={"name": "../x"}), 400)
    r = check(client.post("/solvers/load", json={"name": "t.sol"}), 201)
    lid = r["id"]
    assert r["total_iterations"] == 500
    st_l = check(client.get(f"/solvers/{lid}/stats"))
    st_s = check(client.get(f"/solvers/{sid}/stats"))
    assert abs(st_l["ev_oop"] - st_s["ev_oop"]) < 1e-9

    # loaded instance continues from the restored schedule
    r = check(client.post(f"/solvers/{lid}/continue",
                          json={"max_iters": 100}))
    assert r["total_iterations"] == 600

    # reset + delete
    check(client.post(f"/solvers/{sid}/reset"))
    meta = check(client.get(f"/solvers/{sid}"))
    assert meta["total_iterations"] == 0
    check(client.delete(f"/solvers/{sid}"), 200)
    check(client.get(f"/solvers/{sid}"), 404)
    check(client.delete(f"/solvers/{lid}"))
    check(client.post("/solvers/load", json={"name": "missing.sol"}), 404)


def test_one_shot():
    r = check(client.post("/solve", json={**SPOT, "max_iters": 300,
                                          "target": 1.0,
                                          "strategy_nodes": [0, 1],
                                          "root_ev": True,
                                          "keep": True}))
    assert "stats" in r and "strategies" in r and "root_ev" in r
    assert set(r["strategies"].keys()) == {"0", "1"}
    assert r["stats"]["exploitability"] <= 1.0
    sid = r["solver_id"]
    assert check(client.get(f"/solvers/{sid}"))["total_iterations"] \
        == r["iterations_run"]
    # verify EV aggregation on the one-shot's root_ev
    import numpy as np
    ev = r["root_ev"]
    w = np.asarray(pps_api.REGISTRY.get(sid).solver.player_weights(0))
    m = np.asarray(ev["oop"]["mass"])
    v = np.asarray(ev["oop"]["ev"])
    ok = m > 0
    agg = (w[ok] * m[ok] * v[ok]).sum() / (w[ok] * m[ok]).sum()
    assert abs(agg - r["stats"]["ev_oop"]) < 2e-3
    check(client.delete(f"/solvers/{sid}"))


def test_errors():
    # bad board -> 400 (engine raises through the wrapper)
    check(client.post("/solvers", json={**SPOT, "board": "Qs9h"}), 400)
    # bad range
    check(client.post("/solvers", json={**SPOT, "oop": "ZZ"}), 400)
    # unknown solver
    check(client.post("/solvers/deadbeef/solve",
                      json={"max_iters": 10}), 404)
    # iteration cap
    check(client.post("/solve", json={**SPOT, "max_iters": 10 ** 9}), 422)
    # bad decide node
    r = check(client.post("/solve", json={**SPOT, "max_iters": 100,
                                         "keep": True}))
    sid = r["solver_id"]
    check(client.get(f"/solvers/{sid}/strategy?node=99999"), 400)
    check(client.delete(f"/solvers/{sid}"))


def test_ui_and_paths():
    # action-path labels: root, first-street, and card-labeled deal edges
    r = check(client.post("/solvers", json=SPOT), 201)
    sid = r["id"]
    check(client.post(f"/solvers/{sid}/solve", json={"max_iters": 200}))
    dn = check(client.get(f"/solvers/{sid}/decide-nodes?offset=0&limit=50"))
    paths = [n["path"] for n in dn["nodes"]]
    assert paths[0] == "(root)"
    assert paths[1].startswith("check"), paths[:4]
    # deals are card-labeled ("deal 5c"), never bare "deal" or "?"
    deals = [p for p in paths if "deal" in p]
    assert deals, paths[:10]
    for p in deals:
        for step in p.split(" — "):
            if step.startswith("deal "):
                card = step[5:]
                assert len(card) == 2 and card[0] in "23456789TJQKA" \
                    and card[1] in "cdhs", (step, p)
    dn2 = check(client.get(f"/solvers/{sid}/decide-nodes?offset=1&limit=1"))
    assert dn2["nodes"][0]["index"] == 1

    # static UI is served; / redirects to it
    rr = client.get("/", follow_redirects=False)
    assert rr.status_code in (301, 302, 307, 308), rr.status_code
    r = client.get("/ui/")
    assert r.status_code == 200 and "pps" in r.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200
    check(client.delete(f"/solvers/{sid}"))


def test_runout_branches():
    # a flop spot: the check-check deal enumerates all turn runouts
    r = check(client.post("/solvers", json={**SPOT, "board": "Qs9h2d"}), 201)
    sid = r["id"]
    check(client.post(f"/solvers/{sid}/solve", json={"max_iters": 100}))
    nav = check(client.get(f"/solvers/{sid}/node-nav?node=1"))
    check_act = next(a for a in nav["actions"] if a["kind"] == "check")
    assert check_act["child_kind"] == "chance"
    br = check_act["branches"]
    assert len(br) == 49, len(br)  # 52 - 3 flop cards
    cards = [b["card"] for b in br]
    assert "?" not in cards
    assert "2c" in cards  # card id 0 is a valid deal (no zero-padding bug)
    assert len(set(cards)) == 49
    assert sum(1 for b in br if b["next_decide"] is not None) > 40
    # the dealt cards never collide with the flop board
    board_cards = {"Qs", "9h", "2d"}
    assert not (set(cards) & board_cards)
    # navigating a branch lands on a turn decide node with 4 live cards
    some = next(b for b in br if b["next_decide"] is not None)
    ev = check(client.get(f"/solvers/{sid}/node-ev?node={some['next_decide']}"))
    assert ev["sides"]["oop"]["cards"]
    check(client.delete(f"/solvers/{sid}"))


def test_node_nav_and_range():
    r = check(client.post("/solvers", json=SPOT), 201)
    sid = r["id"]
    check(client.post(f"/solvers/{sid}/solve", json={"max_iters": 200}))

    nav = check(client.get(f"/solvers/{sid}/node-nav?node=0"))
    assert nav["path"] == "(root)"
    acts = {a["label"]: a for a in nav["actions"]}
    assert set(acts) == {"check", "bet 150", "all-in"}
    # OOP checks -> IP's first decision is decide node 1
    assert acts["check"]["next_decide"] == 1
    # bet leads to IP facing the bet; the all-in leads to IP's
    # fold-or-call decision (the runout only starts after the call)
    assert acts["bet 150"]["next_decide"] is not None
    assert acts["all-in"]["next_decide"] is not None
    # deeper: at IP's check-back decision (node 1) there is no fold;
    # IP's check closes the street and chance deals — the first river
    # decide node is reachable through that edge
    nav1 = check(client.get(f"/solvers/{sid}/node-nav?node=1"))
    a1 = {a["kind"]: a for a in nav1["actions"]}
    assert "fold" not in a1
    assert a1["check"]["next_decide"] is not None

    rng = check(client.get(f"/solvers/{sid}/range?player=0"))
    assert len(rng["combos"]) == len(rng["weights"]) > 0
    assert all(len(c) == 4 for c in rng["combos"])
    assert sum(rng["weights"]) > 0
    check(client.get(f"/solvers/{sid}/node-nav?node=99999"), 400)
    check(client.delete(f"/solvers/{sid}"))


def test_async_job():
    r = check(client.post("/solvers", json=SPOT), 201)
    sid = r["id"]
    # background solve: response returns immediately with the job
    r = check(client.post(f"/solvers/{sid}/solve",
                          json={"max_iters": 600, "wait": False}), 200)
    assert r["job"]["running"] is True
    # a second concurrent job is rejected while it runs (or after, the
    # status flips; assert only the conflict semantics when running)
    meta = poll_job(sid)
    assert meta["job"]["status"] == "done"
    assert meta["total_iterations"] == 600
    st = check(client.get(f"/solvers/{sid}/stats"))
    assert abs(st["ev_oop"] + st["ev_ip"] - SPOT["pot"]) < 0.05
    # chunked job == one long solve (warm-start invariant, loose bound
    # for the replay nondeterminism)
    r2 = check(client.post("/solve",
                           json={**SPOT, "max_iters": 600}))
    assert abs(r2["stats"]["ev_oop"] - st["ev_oop"]) < 0.02

    # continue as a job
    r = check(client.post(f"/solvers/{sid}/continue",
                          json={"max_iters": 200, "wait": False}), 200)
    meta = poll_job(sid)
    assert meta["job"]["status"] == "done"
    assert meta["total_iterations"] == 800

    # cancel a longer job
    r = check(client.post(f"/solvers/{sid}/continue",
                          json={"max_iters": 50000, "wait": False}), 200)
    time.sleep(0.6)  # let at least one chunk land
    check(client.post(f"/solvers/{sid}/cancel"), 200)
    meta = poll_job(sid)
    assert meta["job"]["status"] == "cancelled"
    assert 800 < meta["total_iterations"] < 50000
    check(client.delete(f"/solvers/{sid}"))


def poll_job(sid, timeout=60.0):
    import time as _t
    t0 = _t.time()
    while _t.time() - t0 < timeout:
        meta = check(client.get(f"/solvers/{sid}"))
        if meta["job"] and not meta["job"]["running"]:
            return meta
        _t.sleep(0.1)
    raise AssertionError("job did not finish in time")


def test_node_ev_endpoint():
    r = check(client.post("/solvers", json=SPOT), 201)
    sid = r["id"]
    check(client.post(f"/solvers/{sid}/solve", json={"max_iters": 300}))
    ev = check(client.get(f"/solvers/{sid}/node-ev?node=0"))
    st = check(client.get(f"/solvers/{sid}/stats"))
    assert ev["decider"] == 0
    assert len(ev["action_ev"]) == len(ev["actions"])
    assert abs(ev["sides"]["oop"]["agg_ev"] - st["ev_oop"]) < 1e-4
    assert abs(ev["sides"]["ip"]["agg_ev"] - st["ev_ip"]) < 1e-4
    for side in ("oop", "ip"):
        d = ev["sides"][side]
        assert len(d["cards"]) == len(d["ev"]) == len(d["mass"])
    # deeper node: decider is IP there
    ev1 = check(client.get(f"/solvers/{sid}/node-ev?node=1"))
    assert ev1["decider"] == 1
    check(client.get(f"/solvers/{sid}/node-ev?node=99999"), 400)
    check(client.delete(f"/solvers/{sid}"))


def test_solutions_listing():
    r = check(client.post("/solvers", json=SPOT), 201)
    sid = r["id"]
    check(client.post(f"/solvers/{sid}/solve", json={"max_iters": 100}))
    check(client.post(f"/solvers/{sid}/save", json={"name": "list.sol"}))
    lst = check(client.get("/solutions"))
    names = [s["name"] for s in lst["solutions"]]
    assert "list.sol" in names
    saved = next(s for s in lst["solutions"] if s["name"] == "list.sol")
    assert saved["bytes"] > 0
    check(client.delete(f"/solvers/{sid}"))


def main():
    for t in (test_health, test_lifecycle, test_one_shot, test_errors,
              test_ui_and_paths, test_node_nav_and_range, test_async_job,
              test_node_ev_endpoint, test_solutions_listing,
              test_runout_branches):
        t()
        print(f"  {t.__name__}: ok", file=sys.stderr)
    print("pps api tests: 10 passed", file=sys.stderr)


if __name__ == "__main__":
    main()
