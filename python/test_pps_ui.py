# pps UI logic tests: run python/ui/app.js inside a real JS engine
# (quickjs) with a minimal DOM stub, driving the data-path functions
# with REAL solver data (solved in-process through the pps package —
# no HTTP). Verifies what static parsing cannot: the 13x13 hand-class
# aggregation, the freq/EV matrix rendering, and class naming.
#
# Run: make ui-logic-test   (needs `pip install quickjs`)
import json
import os
import sys

import numpy as np
import quickjs

sys.path.insert(0, os.path.dirname(__file__))
import pps  # noqa: E402

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui",
                  "app.js")

STUB = """
globalThis.__els = {};
globalThis.__out = [];
function el_of(id) {
    if (!__els[id]) {
        __els[id] = {
            id: id, tag: id, children: [], style: {}, dataset: {},
            textContent: "", innerHTML: "", value: "",
            appendChild: function (c) { this.children.push(c); },
            append: function (c) { this.children.push(c); },
            addEventListener: function () {},
            classList: { add: function () {}, remove: function () {},
                         toggle: function () {} }
        };
    }
    return __els[id];
}
var __anon = 0;
var document = {
    getElementById: function (id) { return el_of(id); },
    createElement: function (tag) {
        var e = el_of('anon_' + tag + '_' + (__anon++));
        e.tag = tag;
        return e;
    },
    querySelectorAll: function () { return []; }
};
var performance = { now: function () { return 0; } };
function setInterval(fn, ms) { return 0; }
function clearInterval(id) {}
var fetch = function () {
    return Promise.reject(new Error("no fetch in tests"));
};
"""


def load_app():
    """A quickjs context with the DOM stub and app.js loaded (the async
    boot block, which uses fetch, is stripped)."""
    ctx = quickjs.Context()
    ctx.eval(STUB)
    with open(UI) as f:
        src = f.read()
    boot = src.index("// ---------------- boot")
    ctx.eval(src[:boot])
    return ctx


def j(ctx, expr):
    return json.loads(ctx.eval(f"JSON.stringify({expr})"))


def _jsonify(o):
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def test_combo_class_and_colors(ctx):
    assert j(ctx, 'comboClass("AhKd").cls') == "AKo"
    assert j(ctx, 'comboClass("AhKh").cls') == "AKs"
    assert j(ctx, 'comboClass("KdKc").cls') == "KK"
    assert j(ctx, 'comboClass("2c3d").cls') == "32o"
    assert j(ctx, 'comboClass("2c3c").cls') == "32s"
    # a hand maps to exactly one matrix cell (13x13 = 169 classes)
    classes = set()
    for r in "AKQJT98765432":
        for r2 in "AKQJT98765432":
            for su in ("c", "d", "h", "s"):
                c1 = r + su
                c2 = r2 + ("c" if su != "c" else "d")
                classes.add(j(ctx, f'comboClass("{c1}{c2}").cls'))
    assert classes <= {a + b + s for a in "AKQJT98765432"
                       for b in "AKQJT98765432"
                       for s in ("", "s", "o")}, "bad class names"
    assert j(ctx, 'RANK_VAL.A') == 12
    assert j(ctx, 'kindColor("bet", 0)').startswith("hsl(212")
    print("  comboClass / matrix class naming: ok")


def test_freq_matrix(ctx, strat, nev):
    ctx.eval(f"globalThis.__s = {strat};")
    ctx.eval(f"globalThis.__ev = {nev};")
    ctx.eval("""
        currentStrategy = __s; currentEv = __ev; matrixMode = "freq";
        renderMatrixFromStrategy(__s);
        __m = el_of('matrix');
        __cells = __m.children.filter(function (c) {
            return c.style && c.style.background &&
                   String(c.style.background).startsWith("linear-gradient");
        });
        // each filled cell's stacked gradient must end at ~100%
        __sums = __cells.map(function (c) {
            var stops = String(c.style.background).split(",");
            var last = stops[stops.length - 1].trim().split(" ");
            return parseFloat(last[2]);
        });
    """)
    n_kids = j(ctx, "__m.children.length")
    assert n_kids == 14 + 13 * 14, f"matrix grid children {n_kids}"
    sums = j(ctx, "__sums")
    assert len(sums) > 30, f"only {len(sums)} filled classes"
    assert all(abs(v - 100.0) < 2.0 for v in sums), sums[:8]
    print(f"  freq matrix: ok ({len(sums)} filled classes, "
          "stacked bars sum to 100)")


def test_ev_matrix(ctx, strat, nev):
    ctx.eval("""
        matrixMode = "ev";
        renderMatrixFromStrategy(__s);
        __cells = el_of('matrix').children.filter(function (c) {
            return c.style && c.style.background &&
                   String(c.style.background).startsWith("rgba");
        });
        __vals = __cells.map(function (c) { return c.textContent; });
    """)
    vals = j(ctx, "__vals")
    assert len(vals) > 30, f"EV cells {len(vals)}"
    assert all(isinstance(v, (int, float)) for v in vals), vals[:5]
    # the decider's action EVs: finite, and differ across actions
    agg = [a["agg_ev"] for a in json.loads(nev)["action_ev"]]
    assert all(v == v for v in agg), agg
    assert max(agg) != min(agg), agg
    print(f"  EV matrix: ok ({len(vals)} classes), action EVs "
          f"{[round(v, 1) for v in agg]}")


def main():
    ctx = load_app()
    test_combo_class_and_colors(ctx)

    s = pps.Solver(board="Qs9h2d7c", oop="22+,A2s+,K9s+,Q9s+,JT+",
                   ip="AQ+,KQs", pot=200, stack=500, bets="0.75,a",
                   raises="2.5,3")
    s.solve(300)
    strat = json.dumps(_jsonify(s.strategy(0)))
    nev = json.dumps(_jsonify(s.node_ev(0)))
    test_freq_matrix(ctx, strat, nev)
    test_ev_matrix(ctx, strat, nev)
    print("pps ui logic tests: passed", file=sys.stderr)


if __name__ == "__main__":
    main()
