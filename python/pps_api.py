"""FastAPI app exposing the pps GPU postflop solver.

Run from python/ (or `make api` from the repo root):

    uvicorn pps_api:app --host 127.0.0.1 --port 8070

Interactive docs at /docs. The solver instances hold GPU state, so the
app keeps a registry: create a solver, solve, query, continue, save.

    POST /solvers            {"board": "Qs9h2d7c", "oop": "TT+,AKo", ...}
    POST /solvers/{id}/solve {"max_iters": 2000}   (or {"target": 0.1})
    GET  /solvers/{id}/stats
    GET  /solvers/{id}/strategy?node=0
    GET  /solvers/{id}/decide-nodes?offset=0&limit=200
    GET  /solvers/{id}/root-ev
    POST /solvers/{id}/continue   POST /solvers/{id}/reset
    POST /solvers/{id}/save {"name": "spot.sol"}
    POST /solvers/load       {"name": "spot.sol"}
    POST /solve              one-shot: compile + solve + results
    DELETE /solvers/{id}

Concurrency: the native module releases the GIL during engine calls;
solves additionally take a global lock (one GPU), while queries lock
the individual solver. Solution files are sandboxed to
PPS_API_DATA_DIR (default: build-cuda/api-solutions, auto-created).

Env: PPS_API_MAX_SOLVERS (default 8, LRU-evicted), PPS_API_MAX_ITERS
(default 1000000), PPS_API_DATA_DIR.
"""
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pps
from pps import Solver

MAX_SOLVERS = int(os.environ.get("PPS_API_MAX_SOLVERS", "8"))
MAX_ITERS = int(os.environ.get("PPS_API_MAX_ITERS", "1000000"))
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "build-cuda",
    "api-solutions")
DATA_DIR = os.path.abspath(
    os.environ.get("PPS_API_DATA_DIR", DEFAULT_DATA_DIR))

app = FastAPI(
    title="pps solver API",
    description="GPU postflop solver (DCFR, compile-to-dataflow + CUDA "
                "graph replay): solve spots, query labeled per-combo "
                "strategies and EVs, save/load and warm-start solutions.",
    version=pps.__version__,
)

RangeArg = Union[str, Dict[str, float], List[float]]
BetSizesArg = Union[str, List[Union[float, str]]]


class SpotSpec(BaseModel):
    """A spot to compile: board, ranges, pot/stack and bet abstraction."""
    board: str = Field(..., description='e.g. "Qs9h2d7c" (3-5 cards)')
    oop: RangeArg = Field(..., description='player 0 range, e.g. "TT+,AKo"')
    ip: RangeArg = Field(..., description='player 1 range, e.g. "AQ+,KQs"')
    pot: int = Field(..., ge=0)
    stack: int = Field(..., gt=0)
    bets: BetSizesArg = "0.75,a"
    raises: Union[str, List[float]] = "2.5,3"
    max_raises: int = Field(0, ge=0)


class SolveParams(BaseModel):
    max_iters: int = Field(2000, gt=0, le=MAX_ITERS)
    algo: str = "dcfr"
    target: Optional[float] = Field(None, gt=0.0,
                                    description="exploitability target "
                                    "(chips); stops early once reached")
    wait: bool = Field(True, description="false: run as a background job "
                      "in iteration chunks (progress via GET "
                      "/solvers/{id}); the response returns immediately")


class ContinueParams(BaseModel):
    max_iters: int = Field(2000, gt=0, le=MAX_ITERS)
    target: Optional[float] = Field(None, gt=0.0)
    wait: bool = Field(True, description="see solve(wait)")


class SaveParams(BaseModel):
    name: str = Field(..., description="file name under the data dir")


class LoadParams(BaseModel):
    name: str


class _Entry:
    __slots__ = ("solver", "lock", "created", "last_used", "board", "paths",
                 "tree", "decide_idx", "job")

    def __init__(self, solver: Solver, board: str):
        self.solver = solver
        self.lock = threading.Lock()
        self.created = time.time()
        self.last_used = self.created
        self.board = board
        self.paths = None   # cached action-path labels (never change)
        self.tree = None    # cached tree structure (kinds + CSR children)
        self.decide_idx = None  # node_id -> decide index
        self.job = None     # async solve job dict while running


class Registry:
    """Solver instances by id, LRU-evicted past MAX_SOLVERS."""

    def __init__(self):
        self._d: "OrderedDict[str, _Entry]" = OrderedDict()
        self._mu = threading.Lock()

    def add(self, solver: Solver, board: str) -> str:
        with self._mu:
            sid = uuid.uuid4().hex[:12]
            self._d[sid] = _Entry(solver, board)
            self._d.move_to_end(sid)
            evicted = []
            while len(self._d) > MAX_SOLVERS:
                old, e = self._d.popitem(last=False)
                evicted.append(old)
                del e
        return sid

    def get(self, sid: str) -> _Entry:
        with self._mu:
            e = self._d.get(sid)
            if e is None:
                raise HTTPException(404, f"no solver {sid}")
            e.last_used = time.time()
            self._d.move_to_end(sid)
            return e

    def remove(self, sid: str) -> bool:
        with self._mu:
            return self._d.pop(sid, None) is not None

    def describe_all(self) -> List[Dict[str, Any]]:
        with self._mu:
            return [dict(id=sid, board=e.board, created=e.created,
                         last_used=e.last_used,
                         total_iterations=e.solver.total_iterations,
                         iterations_run=e.solver.iterations_run,
                         num_nodes=e.solver.num_nodes)
                    for sid, e in self._d.items()]


REGISTRY = Registry()
# One GPU: solves serialize (they saturate it anyway). Queries run
# concurrently, each under its solver's lock.
SOLVE_LOCK = threading.Lock()


def _engine(fn, *args, **kwargs):
    """Run an engine call, mapping its errors to HTTP 400."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e


def _new_solver(spec: SpotSpec) -> Solver:
    return _engine(
        Solver, board=spec.board, oop=spec.oop, ip=spec.ip, pot=spec.pot,
        stack=spec.stack, bets=spec.bets, raises=spec.raises,
        max_raises=spec.max_raises)


def _solve(solver: Solver, p: SolveParams):
    target = p.target if p.target is not None else -1.0
    with SOLVE_LOCK:
        _engine(solver.solve, p.max_iters, p.algo, target)


def _jsonable(obj: Any) -> Any:
    """numpy arrays / pybind types -> plain JSON values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _action_text(a: Dict[str, Any]) -> str:
    k = a["kind"]
    amt = a.get("amount", 0)
    if k in ("bet", "raise"):
        return f"{k} {amt}"
    if k == "allin":
        return "all-in"
    return k


def _node_paths(e: "_Entry") -> List[str]:
    """Action-path label per tree node, computed once per solver.

    Child node ids are always greater than their parent's (the tree
    builder allocates parents first), so one forward pass works.
    Decide edges carry their action label; chance edges are the deal.
    Also caches the tree structure and the node_id -> decide index map
    for node-nav.
    """
    if e.paths is not None:
        return e.paths
    with e.lock:
        tree = _engine(e.solver.tree_structure)
        dns = _engine(e.solver.decide_nodes)
    kinds, cb, ch = tree["kinds"], tree["child_base"], tree["children"]
    n = len(kinds)
    acts = {d["node_id"]: d["actions"] for d in dns}
    paths = [""] * n
    for u in range(n):
        p = paths[u]
        nxt = cb[u + 1] if u + 1 < n else len(ch)
        cnt = nxt - cb[u]
        if cnt == 0:
            continue
        if kinds[u] == 0:  # decide: one labeled edge per action
            for i, a in enumerate(acts[u]):
                lab = _action_text(a)
                c = ch[cb[u] + i]
                paths[c] = f"{p} — {lab}" if p else lab
        elif kinds[u] == 3:  # chance: the deal
            for i in range(cnt):
                c = ch[cb[u] + i]
                paths[c] = f"{p} — deal"
        # fold/showdown are leaves
    e.paths = paths
    e.tree = tree
    e.decide_idx = {d["node_id"]: d["index"] for d in dns}
    return paths


def _first_decide(e: "_Entry", start: int) -> Optional[int]:
    """Node id of the first decide node in `start`'s subtree (BFS from
    its children), or None for a runout that ends in showdown.
    Chance branches are followed; fold/showdown are terminal."""
    kinds, cb, ch = e.tree["kinds"], e.tree["child_base"], e.tree["children"]
    n = len(kinds)
    q = [start]
    while q:
        u = q.pop(0)
        nxt = cb[u + 1] if u + 1 < n else len(ch)
        for i in range(cb[u], nxt):
            c = ch[i]
            if kinds[c] == 0:
                return c
            if kinds[c] == 3:
                q.append(c)
    return None


def _solver_meta(sid: str, e: _Entry) -> Dict[str, Any]:
    s = e.solver
    return {
        "id": sid,
        "board": e.board,
        "num_nodes": s.num_nodes,
        "num_decide_nodes": s.num_decide_nodes,
        "max_depth": s.max_depth,
        "total_iterations": s.total_iterations,
        "iterations_run": s.iterations_run,
        "job": dict(e.job) if e.job else None,
    }


def _data_path(name: str) -> str:
    """Resolve `name` under DATA_DIR; reject escapes and separators."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(400, f"bad solution name {name!r}")
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "solvers": len(REGISTRY.describe_all()),
            "max_solvers": MAX_SOLVERS, "max_iters": MAX_ITERS,
            "data_dir": DATA_DIR}


@app.post("/solvers", status_code=201)
def create_solver(spec: SpotSpec) -> Dict[str, Any]:
    t0 = time.perf_counter()
    solver = _new_solver(spec)
    ms = (time.perf_counter() - t0) * 1000
    sid = REGISTRY.add(solver, spec.board)
    out = _solver_meta(sid, REGISTRY.get(sid))
    out["compile_ms"] = round(ms, 1)
    return _jsonable(out)


@app.get("/solvers")
def list_solvers() -> List[Dict[str, Any]]:
    return REGISTRY.describe_all()


@app.get("/solvers/{sid}")
def get_solver(sid: str) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    return _solver_meta(sid, e)


@app.delete("/solvers/{sid}")
def delete_solver(sid: str) -> Dict[str, bool]:
    if not REGISTRY.remove(sid):
        raise HTTPException(404, f"no solver {sid}")
    return {"removed": True}


def _job_worker(e: "_Entry", kind: str, p: SolveParams) -> None:
    """Background solve: iteration chunks with progress reporting.

    Chunked continuation follows the exact same DCFR schedule as one
    long solve (the staircase depends only on the cumulative t —
    gated by python/test_pps.py's warm-start test), so this is purely
    a progress/UX split. Each chunk takes the global GPU lock and the
    solver lock, so queries interleave between chunks.
    """
    s = e.solver
    target = p.target if p.target is not None else -1.0
    chunk = max(128, p.max_iters // 50)
    remaining = p.max_iters
    e.job["start"] = s.total_iterations
    try:
        if kind == "solve":
            take = min(chunk, remaining)
            with SOLVE_LOCK, e.lock:
                s.solve(take, p.algo, target)
            e.job["done"] = s.total_iterations - e.job["start"]
            remaining -= s.iterations_run
            if s.iterations_run < take:  # target hit
                remaining = 0
        while remaining > 0 and not e.job["cancel"]:
            take = min(chunk, remaining)
            with SOLVE_LOCK, e.lock:
                s.continue_solve(take, target)
            e.job["done"] = s.total_iterations - e.job["start"]
            ran = s.iterations_run
            remaining -= ran
            if ran < take:  # early stop on the target
                remaining = 0
        e.job["status"] = "cancelled" if e.job["cancel"] else "done"
    except Exception as ex:  # engine errors end the job, not the server
        e.job["status"] = "error"
        e.job["error"] = str(ex)
    finally:
        e.job["running"] = False


def _start_job(sid: str, e: "_Entry", kind: str,
               p: SolveParams) -> Dict[str, Any]:
    if e.job and e.job["running"]:
        raise HTTPException(409, "a solve job is already running on this "
                                 "solver (POST /solvers/{id}/cancel first)")
    e.job = {"running": True, "status": "running", "kind": kind,
             "done": 0, "start": 0,
             "total": p.max_iters, "target": p.target, "error": None,
             "cancel": False}
    t = threading.Thread(target=_job_worker, args=(e, kind, p),
                         daemon=True)
    t.start()
    return {"id": sid, "job": dict(e.job)}


@app.post("/solvers/{sid}/solve")
def solve(sid: str, p: SolveParams):
    e = REGISTRY.get(sid)
    if not p.wait:
        return _start_job(sid, e, "solve", p)
    with e.lock:
        t0 = time.perf_counter()
        _solve(e.solver, p)
        ms = (time.perf_counter() - t0) * 1000
        st = _engine(e.solver.stats)
    return _jsonable({"solve_ms": round(ms, 1),
                      "iterations_run": e.solver.iterations_run,
                      "total_iterations": e.solver.total_iterations,
                      "stats": st})


@app.post("/solvers/{sid}/continue")
def continue_solve(sid: str, p: ContinueParams):
    e = REGISTRY.get(sid)
    if not p.wait:
        return _start_job(sid, e, "continue", p)
    with e.lock:
        t0 = time.perf_counter()
        target = p.target if p.target is not None else -1.0
        with SOLVE_LOCK:
            _engine(e.solver.continue_solve, p.max_iters, target)
        ms = (time.perf_counter() - t0) * 1000
        st = _engine(e.solver.stats)
    return _jsonable({"continue_ms": round(ms, 1),
                      "iterations_run": e.solver.iterations_run,
                      "total_iterations": e.solver.total_iterations,
                      "stats": st})


@app.post("/solvers/{sid}/cancel")
def cancel_job(sid: str) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    if not (e.job and e.job["running"]):
        return {"cancelled": False}
    e.job["cancel"] = True
    return {"cancelled": True}


@app.post("/solvers/{sid}/reset")
def reset_solver(sid: str) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    with e.lock:
        _engine(e.solver.reset)
    return _solver_meta(sid, e)


@app.get("/solvers/{sid}/stats")
def solver_stats(sid: str) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    with e.lock:
        st = _engine(e.solver.stats)
    return _jsonable(st)


@app.get("/solvers/{sid}/strategy")
def solver_strategy(sid: str, node: int = Query(0, ge=0)) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    with e.lock:
        st = _engine(e.solver.strategy, node)
    return _jsonable(st)


@app.get("/solvers/{sid}/decide-nodes")
def solver_decide_nodes(sid: str, offset: int = Query(0, ge=0),
                        limit: int = Query(200, gt=0, le=2000)) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    with e.lock:
        total = e.solver.num_decide_nodes
        nodes = _engine(e.solver.decide_nodes)
    paths = _node_paths(e)
    for d in nodes:
        d["path"] = paths[d["node_id"]] or "(root)"
    return {"total": total,
            "nodes": nodes[offset:offset + limit]}


@app.get("/solvers/{sid}/node-nav")
def solver_node_nav(sid: str, node: int = Query(0, ge=0)) -> Dict[str, Any]:
    """Navigation out of decide node `node`: one entry per action with
    the child's kind and the first DECIDE node reachable through that
    edge (chance branches are followed to the next street; fold and
    showdown are terminal; an all-in runout may reach no decide node).
    Drives the UI's action-history bar."""
    e = REGISTRY.get(sid)
    _node_paths(e)  # builds/caches tree + decide index too
    with e.lock:
        dns = _engine(e.solver.decide_nodes)
    if node >= len(dns):
        raise HTTPException(400, f"decide node {node} out of range")
    d = dns[node]
    kinds, cb, ch = e.tree["kinds"], e.tree["child_base"], e.tree["children"]
    kind_name = {0: "decide", 1: "showdown", 2: "fold", 3: "chance"}
    actions = []
    for a, child in zip(d["actions"], d["children"]):
        item = {"label": _action_text(a), "kind": a["kind"],
                "amount": a.get("amount", 0),
                "child_node": child, "child_kind": kind_name[kinds[child]]}
        nxt = child if kinds[child] == 0 else _first_decide(e, child)
        item["next_decide"] = \
            e.decide_idx[nxt] if nxt is not None else None
        actions.append(item)
    return {"node": node, "path": e.paths[d["node_id"]] or "(root)",
            "player": d["player"], "n_board": d["n_board"],
            "actions": actions}


_RSRC = "23456789TJQKA"
_SUITS = "cdhs"


def _card_text(cid: int) -> str:
    return _RSRC[cid % 13] + _SUITS[cid // 13]


@app.get("/solvers/{sid}/range")
def solver_range(sid: str, player: int = Query(0, ge=0, le=1)) -> Dict[str, Any]:
    """The compiled range of `player`: combo card pairs (text) with
    weights, aligned (board-filtered, positive-weight only)."""
    e = REGISTRY.get(sid)
    with e.lock:
        cards = _engine(e.solver.player_cards, player)
        w = _engine(e.solver.player_weights, player)
    return {"player": player,
            "combos": [_card_text(cards[2 * i]) + _card_text(cards[2 * i + 1])
                       for i in range(len(w))],
            "weights": _jsonable(np.asarray(w))}


@app.get("/solvers/{sid}/root-ev")
def solver_root_ev(sid: str) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    with e.lock:
        ev = _engine(e.solver.root_ev)
    return _jsonable(ev)


@app.post("/solvers/{sid}/save")
def solver_save(sid: str, p: SaveParams) -> Dict[str, Any]:
    e = REGISTRY.get(sid)
    path = _data_path(p.name)
    with e.lock:
        _engine(e.solver.save, path)
    return {"saved": True, "name": p.name, "bytes": os.path.getsize(path)}


@app.post("/solvers/load", status_code=201)
def solver_load(p: LoadParams) -> Dict[str, Any]:
    path = _data_path(p.name)
    if not os.path.isfile(path):
        raise HTTPException(404, f"no solution file {p.name!r}")
    solver = _engine(Solver.load, path)
    sid = REGISTRY.add(solver, p.name)
    out = _solver_meta(sid, REGISTRY.get(sid))
    out["name"] = p.name
    return _jsonable(out)


class OneShot(SpotSpec, SolveParams):
    """One-shot solve: compile, solve, return results."""
    strategy_nodes: List[int] = Field(default_factory=list,
                                      description="decide indices for "
                                      "labeled strategies")
    root_ev: bool = False
    keep: bool = Field(False, description="keep the solver registered "
                       "(default: discard)")


@app.post("/solve")
def solve_oneshot(body: OneShot) -> Dict[str, Any]:
    t0 = time.perf_counter()
    solver = _new_solver(body)
    compile_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    _solve(solver, body)
    solve_ms = (time.perf_counter() - t1) * 1000
    out: Dict[str, Any] = {
        "compile_ms": round(compile_ms, 1),
        "solve_ms": round(solve_ms, 1),
        "iterations_run": solver.iterations_run,
        "stats": _jsonable(_engine(solver.stats)),
    }
    for n in body.strategy_nodes:
        out.setdefault("strategies", {})[n] = \
            _jsonable(_engine(solver.strategy, n))
    if body.root_ev:
        out["root_ev"] = _jsonable(_engine(solver.root_ev))
    if body.keep:
        sid = REGISTRY.add(solver, body.board)
        out["solver_id"] = sid
    return out


# ---- web UI (python/ui): static, no build step ----
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
if os.path.isdir(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/ui/")
