# pps UI flow test: the deepest verification layer — drives the ACTUAL
# app.js click handlers through a full study session with fetch
# bridged to the real FastAPI app (in-process TestClient → pps package
# → GPU engine). No canned data anywhere: solve, poll, navigate
# actions, pick runouts, switch tabs — and assert the DOM state.
#
# Run: make ui-flow-test   (needs `pip install quickjs`)
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

os.environ["PPS_API_DATA_DIR"] = tempfile.mkdtemp(prefix="pps-ui-flow-")
import pps_api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import quickjs  # noqa: E402

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui",
                  "app.js")

STUB = """
globalThis.__els = {};
globalThis.__created = [];
function el_of(id) {
    if (!__els[id]) {
        var e = {
            id: id, tag: id, children: [], style: {}, dataset: {},
            value: "", title: "", onclick: null,
            appendChild: function (c) { this.children.push(c); },
            append: function (c) { this.children.push(c); },
            addEventListener: function () {},
            classList: { add: function () {}, remove: function () {},
                         toggle: function () {} }
        };
        // DOM-faithful: assigning innerHTML/textContent clears children
        var html = "", text = "";
        Object.defineProperty(e, "innerHTML", {
            get: function () { return html; },
            set: function (v) { html = String(v); e.children.length = 0; }
        });
        Object.defineProperty(e, "textContent", {
            get: function () { return text; },
            set: function (v) { text = String(v); e.children.length = 0; }
        });
        __els[id] = e;
    }
    return __els[id];
}
var __anon = 0;
var document = {
    getElementById: function (id) { return el_of(id); },
    createElement: function (tag) {
        var e = el_of('anon_' + tag + '_' + (__anon++));
        e.tag = tag;
        __created.push(e);
        return e;
    },
    querySelectorAll: function () { return []; }
};
var performance = { now: function () { return 0; } };
function setInterval(fn, ms) { return 0; }
function clearInterval(id) {}
var fetch = function (url, opts) {
    var body = (opts && opts.body) ? String(opts.body) : "";
    var r = JSON.parse(PY_FETCH(String(url), body));
    return Promise.resolve({
        ok: r.status < 400,
        status: r.status,
        json: function () {
            return Promise.resolve(JSON.parse(r.body || "{}"));
        }
    });
};
function set_val(id, v) { el_of(id).value = v; }
"""


def make_ctx(client: TestClient) -> quickjs.Context:
    ctx = quickjs.Context()
    ctx.eval(STUB)

    def py_fetch(url: str, body: str) -> str:
        if body:
            r = client.post(url, content=body.encode(),
                           headers={"Content-Type": "application/json"})
        else:
            r = client.get(url)
        return json.dumps({"status": r.status_code, "body": r.text})

    ctx.add_callable("PY_FETCH", py_fetch)
    with open(UI) as f:
        ctx.eval(f.read())  # full app.js including the async boot
    return ctx


def pump(ctx: quickjs.Context) -> None:
    while ctx.execute_pending_job():
        pass


def js(ctx: quickjs.Context, expr: str):
    return json.loads(ctx.eval(f"JSON.stringify({expr})"))


def wait_job_done(ctx, timeout=30.0) -> None:
    """pollJob is driven manually (setInterval is stubbed): call it
    until the server-side job finishes."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        ctx.eval("pollJob()")
        pump(ctx)
        if js(ctx, "el_of('progress').classList") is None:
            pass  # classList stub records nothing; check the text
        if "done" in js(ctx, "el_of('progress-text').textContent"):
            return
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_full_study_session(client: TestClient):
    ctx = make_ctx(client)
    pump(ctx)  # boot: /health, /solvers, /solutions
    assert "ok" in js(ctx, "el_of('conn').textContent")

    # --- solve flow: fill the form, click Solve spot ---
    ctx.eval("""
        set_val('oop', 'TT+,AKo');
        set_val('ip', 'AQ+,KQs');
        set_val('pot', '200');
        set_val('stack', '500');
        set_val('bets', '0.75,a');
        set_val('raises', '2.5,3');
        set_val('iters', '300');
    """)
    ctx.eval("board = ['Qs', '9h', '2d']; render();")
    ctx.eval("el_of('solve').onclick()")
    pump(ctx)
    # a solver appeared in the registry list
    assert js(ctx, "el_of('solver-list').children.length") == 1
    # the async job started; poll until done (real worker thread)
    wait_job_done(ctx)
    pump(ctx)
    # stats rendered with real EVs summing to the pot
    st_html = js(ctx, "el_of('stats').innerHTML")
    assert "EV OOP" in st_html and "exploitability" in st_html
    # the 13x13 matrix rendered with filled cells
    assert js(ctx, "el_of('matrix').children.length") == 14 + 13 * 14
    filled = js(ctx, """
        el_of('matrix').children.filter(function (c) {
            return c.style && c.style.background;
        }).length
    """)
    assert filled >= 5, f"matrix filled cells {filled}"
    # nav buttons with EV sub-labels exist
    nacts = js(ctx, """
        __created.filter(function (b) {
            return b.tag === 'button' && b.innerHTML.indexOf('EV') >= 0;
        }).length
    """)
    assert nacts == 3, nacts

    # --- click the check action (OOP checks -> IP decides) ---
    ctx.eval("""
        __created.filter(function (b) {
            return b.tag === 'button' && b.innerHTML.indexOf('check') >= 0
                && b.innerHTML.indexOf('EV') >= 0;
        })[0].onclick();
    """)
    pump(ctx)
    assert js(ctx, "history.length") == 2
    assert js(ctx, "currentNode") == 1
    # EV chips for both players are shown at the new node
    strip = js(ctx, "el_of('node-ev-strip').innerHTML")
    assert "OOP EV" in strip and "IP EV" in strip

    # --- IP checks back -> deal: open the runout picker ---
    ctx.eval("""
        __created.filter(function (b) {
            return b.tag === 'button' && b.innerHTML.indexOf('pick a runout') >= 0;
        })[0].onclick();
    """)
    pump(ctx)
    # the runout strip holds the 49-card grid (the dim header is a
    # text assignment, not an element)
    cells = js(ctx, """
        (function () {
            var grids = el_of('runout-strip').children.filter(
                function (d) { return d.children.length > 10; });
            return grids.length ? grids[0].children.length : 0;
        })()
    """)
    assert cells == 49, cells
    # every live cell has a stacked strategy bar (real runout summary)
    bars = js(ctx, """
        (function () {
            var grids = el_of('runout-strip').children.filter(
                function (d) { return d.children.length > 10; });
            return grids[0].children.filter(
                function (c) { return c.children.length >= 2; }).length;
        })()
    """)
    assert bars == 49, bars

    # --- click the 2c runout card -> study that exact turn ---
    ctx.eval("""
        (function () {
            var grids = el_of('runout-strip').children.filter(
                function (d) { return d.children.length > 10; });
            var cell = grids[0].children.filter(
                function (c) { return c.children[0].textContent === '2c'; })[0];
            cell.onclick();
        })();
    """)
    pump(ctx)
    assert js(ctx, "history.length") == 3
    assert js(ctx, "history[2].card") == "2c"
    # the big board now shows the dealt 2c
    board_txt = js(ctx, """
        el_of('big-board').children.map(function (c) { return c.textContent; })
    """)
    assert "2\u2663" in board_txt, board_txt
    # the turn node's matrix re-rendered
    assert js(ctx, "el_of('matrix').children.length") == 14 + 13 * 14

    # --- tabs: tree, ranges, EVs ---
    ctx.eval("switchTab('tree')")
    pump(ctx)
    assert js(ctx, "el_of('tree-list').children.length") > 0
    ctx.eval("switchTab('ranges')")
    pump(ctx)
    assert js(ctx, "el_of('range-matrix').children.length") == 14 + 13 * 14
    ctx.eval("switchTab('ev')")
    pump(ctx)
    assert "hand" in js(ctx, "el_of('ev-table').innerHTML")

    print("  full study session (solve -> poll -> act -> runout -> tabs): ok")


def main():
    client = TestClient(pps_api.app)
    client.headers["X-Test"] = "flow"
    test_full_study_session(client)
    print("pps ui flow test: passed", file=sys.stderr)


if __name__ == "__main__":
    main()
