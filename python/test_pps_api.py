# pps solver API tests: in-process TestClient (fastapi.testclient)
# exercising the full HTTP surface against the real GPU engine.
# Run with `make api-test` (or from python/: python test_pps_api.py).
import os
import sys
import tempfile

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
    # action-path labels: root, first-street, and a deal edge for
    # nodes past a chance branch
    r = check(client.post("/solvers", json=SPOT), 201)
    sid = r["id"]
    check(client.post(f"/solvers/{sid}/solve", json={"max_iters": 200}))
    dn = check(client.get(f"/solvers/{sid}/decide-nodes?offset=0&limit=50"))
    paths = [n["path"] for n in dn["nodes"]]
    assert paths[0] == "(root)"
    assert paths[1].startswith("check"), paths[:4]
    assert any("deal" in p for p in paths), paths[:10]
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


def main():
    for t in (test_health, test_lifecycle, test_one_shot, test_errors,
              test_ui_and_paths):
        t()
        print(f"  {t.__name__}: ok", file=sys.stderr)
    print("pps api tests: 5 passed", file=sys.stderr)


if __name__ == "__main__":
    main()
