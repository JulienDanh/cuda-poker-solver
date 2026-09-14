// pps solver UI: vanilla JS over the pps_api HTTP surface.
// GTO-Wizard-style study flow: build a spot, solve in the background
// with progress, walk the tree by clicking actions, read 13x13 hand
// matrices (stacked action frequencies), per-combo tables and root EVs.
"use strict";

const $ = (id) => document.getElementById(id);
const SUITS = "cdhs";
const SUIT_GLYPH = { c: "\u2663", d: "\u2666", h: "\u2665", s: "\u2660" };
const SUIT_RED = { d: true, h: true };
const RANKS_HIGH = "AKQJT98765432";  // matrix order (A first)
const RANK_VAL = { "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
  "9": 7, T: 8, J: 9, Q: 10, K: 11, A: 12 };

// GTO-ish action colors; repeated kinds in one node shift hue.
const KIND_HUE = { check: 172, call: 152, bet: 212, raise: 28, allin: 356,
  fold: 215 };
const kindColor = (kind, i) =>
  `hsl(${(KIND_HUE[kind] || 0) + i * 26}, 62%, ${45 + (i % 2) * 6}%)`;

// ---------------- board picker ----------------
let board = ["Qs", "9h", "2d"];

const clsOf = (c) => (SUIT_RED[c[1]] ? "red" : "blk");

function renderBoard() {
  const el = $("board-slots");
  el.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const s = document.createElement("div");
    s.className = "slot " + (board[i] ? clsOf(board[i]) : "");
    s.textContent = board[i] || "";
    s.onclick = () => { if (board[i]) { board.splice(i, 1); render(); } };
    el.appendChild(s);
  }
  $("clear-board").style.display = board.length ? "inline-block" : "none";
}

function renderPicker() {
  const el = $("card-picker");
  el.innerHTML = "";
  for (const su of SUITS) {
    for (const r of RANKS_HIGH) {
      const c = r + su;
      const b = document.createElement("div");
      b.className = "pk " + clsOf(c) + (board.includes(c) ? " used" : "");
      b.textContent = c;
      b.onclick = () => {
        if (board.includes(c) || board.length >= 5) return;
        board.push(c);
        render();
      };
      el.appendChild(b);
    }
  }
}

function render() { renderBoard(); renderPicker(); }

// ---------------- fetch helpers ----------------
let overlayTimer = null;
async function api(method, url, body, showOverlay) {
  if (showOverlay) {
    $("overlay").classList.remove("hidden");
    const t0 = performance.now();
    clearInterval(overlayTimer);
    overlayTimer = setInterval(() => {
      $("elapsed").textContent =
        ((performance.now() - t0) / 1000).toFixed(1) + "s";
    }, 100);
  }
  try {
    const opts = method === "GET" ? {} :
      { method, headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) };
    const r = await fetch(url, opts);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return data;
  } finally {
    clearInterval(overlayTimer);
    if (showOverlay) $("overlay").classList.add("hidden");
  }
}

const err = (where) => (e) => {
  $(where).innerHTML = `<span class="error">${e.message}</span>`;
};

// ---------------- state ----------------
let active = null;         // current solver id
let history = [{ node: 0, label: "(root)", who: 0 }];
let jobTimer = null;
let currentStrategy = null;
let currentEv = null;      // node-ev at the current node
let matrixMode = "freq";   // "freq" | "ev"

function spotBody() {
  return {
    board: board.join(""),
    oop: $("oop").value.trim(),
    ip: $("ip").value.trim(),
    pot: parseInt($("pot").value, 10),
    stack: parseInt($("stack").value, 10),
    bets: $("bets").value.trim(),
    raises: $("raises").value.trim(),
  };
}

// ---------------- stats ----------------
function showStats(st, extra) {
  const el = $("stats");
  const fmt = (x, p) => (typeof x === "number" ? x.toFixed(p) : "—");
  el.innerHTML = `
    <div class="stat"><div class="v oop">${fmt(st.ev_oop, 2)}</div><div class="k">EV OOP</div></div>
    <div class="stat"><div class="v ip">${fmt(st.ev_ip, 2)}</div><div class="k">EV IP</div></div>
    <div class="stat"><div class="v">${fmt(st.exploitability, 4)}</div><div class="k">exploitability</div></div>
    ${extra || ""}`;
}

// ---------------- solve (background job) ----------------
async function startJob(path, body) {
  try {
    const r = await api("POST", `/solvers/${active}/${path}`,
      { ...body, wait: false }, path === "solve" && body.fresh);
    $("progress").classList.remove("hidden");
    clearInterval(jobTimer);
    jobTimer = setInterval(pollJob, 500);
    return r;
  } catch (e) { err("stats")(e); }
}

async function pollJob() {
  if (!active) return;
  let meta;
  try {
    meta = await api("GET", `/solvers/${active}`);
  } catch (e) { return; }
  const job = meta.job;
  if (!job) { clearInterval(jobTimer); $("progress").classList.add("hidden"); return; }
  const pct = job.total ? Math.min(100, 100 * job.done / job.total) : 0;
  $("bar-fill").style.width = pct.toFixed(1) + "%";
  $("progress-text").textContent =
    `${job.status} — ${job.done}/${job.total} iters` +
    `${job.error ? " — " + job.error : ""}`;
  if (!job.running) {
    clearInterval(jobTimer);
    $("progress").classList.add("hidden");
    if (job.status === "error") err("stats")(new Error(job.error));
    else {
      const st = await api("GET", `/solvers/${active}/stats`);
      showStats(st, `<div class="stat"><div class="v">${meta.total_iterations}</div><div class="k">iters total</div></div>`);
      await selectNode(history[history.length - 1].node);
    }
    refreshSolvers();
  }
}

async function solve() {
  if (board.length < 3 || board.length > 5) {
    err("stats")(new Error("board must be 3-5 cards")); return;
  }
  try {
    // compile + register, then solve as a background job
    const created = await api("POST", "/solvers", spotBody(), true);
    active = created.id;
    history = [{ node: 0, label: "(root)", who: 0 }];
    await refreshSolvers();
    await startJob("solve", {
      max_iters: parseInt($("iters").value, 10),
      target: $("target").value ? parseFloat($("target").value) : null,
    });
  } catch (e) { err("stats")(e); }
}

// ---------------- solver registry ----------------
async function refreshSolvers() {
  const list = await api("GET", "/solvers");
  const el = $("solver-list");
  if (!list.length) { el.innerHTML = '<span class="dim">none</span>'; return; }
  el.innerHTML = "";
  for (const s of list) {
    const d = document.createElement("div");
    d.className = "sent" + (s.id === active ? " sel" : "");
    const busy = s.job && s.job.running ? " ●" : "";
    d.innerHTML = `<span class="id">${s.id}${busy}</span> ${s.board} ` +
      `<span class="id">${s.total_iterations} iters</span>`;
    d.onclick = async () => {
      active = s.id;
      // show THIS solver's board (the spot it was compiled with)
      board = (s.board.match(/.{2}/g) || []);
      render();
      history = [{ node: 0, label: "(root)", who: 0 }];
      await refreshSolvers();
      const st = await api("GET", `/solvers/${active}/stats`);
      showStats(st, `<div class="stat"><div class="v">${s.total_iterations}</div><div class="k">iters total</div></div>`);
      await selectNode(0);
    };
    el.appendChild(d);
  }
}

async function solverAction(path, body) {
  if (!active) return err("stats")(new Error("no active solver"));
  if (path === "reset") {
    try {
      await api("POST", `/solvers/${active}/reset`, {});
      history = [{ node: 0, label: "(root)", who: 0 }];
      await refreshSolvers();
      await selectNode(0);
    } catch (e) { err("stats")(e); }
    return;
  }
  await startJob(path, body);
}

// ---------------- hand classes / matrix ----------------
function comboClass(c) {  // "AhKd" -> {r: hi, c: lo, cls: "AKo"}
  const r1 = RANK_VAL[c[0]], r2 = RANK_VAL[c[2]];
  const hi = Math.max(r1, r2), lo = Math.min(r1, r2);
  const suited = c[1] === c[3];
  const cls = hi === lo ? RANKS_HIGH[12 - hi] + RANKS_HIGH[12 - hi]
    : RANKS_HIGH[12 - hi] + RANKS_HIGH[12 - lo] + (suited ? "s" : "o");
  return { hi, lo, cls };
}

function renderMatrix(classes) {
  // classes: {"AKs": {freqs, evs, n}, ...}; evs present in EV mode
  const m = $("matrix");
  const pot = Math.max(1, parseInt($("pot").value, 10) || 1);
  m.innerHTML = "";
  m.appendChild(el("div", "mxh", ""));
  for (const r of RANKS_HIGH) m.appendChild(el("div", "mxh", r));
  for (const ri of RANKS_HIGH) {
    const h = el("div", "mxh rowh", ri);
    h.style.display = "flex"; h.style.alignItems = "center";
    m.appendChild(h);
    for (const ci of RANKS_HIGH) {
      const suited = ri !== ci && (RANK_VAL[ri] > RANK_VAL[ci]);
      const cls = ri === ci ? ri + ci
        : (suited ? ri + ci + "s" : ci + ri + "o");
      const cell = el("div", "mxcell");
      const d = classes ? classes[cls] : null;
      if (d && d.n > 0 && d.evs && d.evs.length) {
        const mean = d.evs.reduce((a, b) => a + b, 0) / d.evs.length;
        const ratio = Math.max(-1, Math.min(1, mean / pot));
        const alpha = (0.12 + 0.72 * Math.abs(ratio)).toFixed(3);
        cell.style.background = ratio >= 0
          ? `rgba(62,201,122,${alpha})` : `rgba(255,107,107,${alpha})`;
        cell.textContent = Math.round(mean);
        cell.title = `${cls}: EV ${mean.toFixed(1)} (${d.evs.length} live combos)`;
        cell.onclick = () => {
          document.querySelectorAll(".mxcell").forEach((x) =>
            x.classList.remove("sel"));
          cell.classList.add("sel");
          $("cell-detail").innerHTML =
            `<b>${cls}</b> — EV <b>${mean.toFixed(1)}</b> ` +
            `(mean of ${d.evs.length} live combos)`;
        };
      } else if (d && d.n > 0) {
        // frequency mode: stacked action frequencies as the background
        const stops = [];
        let acc = 0;
        d.freqs.forEach((f, i) => {
          if (f <= 0.001) return;
          stops.push(`${d.colors[i]} ${acc * 100}% ${(acc + f) * 100}%`);
          acc += f;
        });
        cell.style.background = stops.length
          ? `linear-gradient(90deg, ${stops.join(",")})` : "#0d1016";
        cell.textContent = (Math.max(...d.freqs) * 100).toFixed(0);
        cell.title = d.freqs.map((f, i) =>
          `${d.labels[i]} ${(f * 100).toFixed(1)}%`).join(" | ");
        cell.onclick = () => {
          document.querySelectorAll(".mxcell").forEach((x) =>
            x.classList.remove("sel"));
          cell.classList.add("sel");
          $("cell-detail").innerHTML =
            `<b>${cls}</b> (${d.n} combos): ` + d.freqs.map((f, i) =>
              `${d.labels[i]} <b>${(f * 100).toFixed(1)}%</b>`).join(" · ");
        };
      } else {
        cell.innerHTML = '<span class="dead">–</span>';
        cell.title = cls + ": not in range";
      }
      m.appendChild(cell);
    }
  }
}

const el = (tag, cls, text) => {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text != null) d.textContent = text;
  return d;
};

// ---------------- strategy / navigation ----------------
async function selectNode(idx, step) {
  if (!active) return;
  if (step) history.push(step);
  currentNode = idx;
  const s = await api("GET", `/solvers/${active}/strategy?node=${idx}`);
  const nav = await api("GET", `/solvers/${active}/node-nav?node=${idx}`);
  let ev = null;
  try {
    ev = await api("GET", `/solvers/${active}/node-ev?node=${idx}`);
  } catch (e) { /* unsolved or empty: EVs unavailable */ }
  currentStrategy = s;
  currentEv = ev;
  renderSpot(nav);
  renderMatrixFromStrategy(s);
  renderCombosTable(s);
  renderEvTab();
  $("strat-title").textContent = `Strategy — ${nav.path || "(root)"}`;
}

let currentNode = 0;

function renderSpot(nav) {
  // big board cards; undealt cards are dimmed (n_board = cards live
  // at this node — deeper nodes have seen the deal)
  const bb = $("big-board");
  bb.innerHTML = "";
  board.forEach((c, i) => {
    const dealt = i < (nav ? nav.n_board : board.length);
    const d = el("div", "bcard" + (SUIT_RED[c[1]] ? " red" : ""),
      c[0] + SUIT_GLYPH[c[1]]);
    if (!dealt) d.style.opacity = ".25";
    bb.appendChild(d);
  });
  for (let i = board.length; i < 5; i++) bb.appendChild(el("div", "bcard"));

  // breadcrumb
  const h = $("history");
  h.innerHTML = "";
  history.forEach((s, i) => {
    if (i > 0) h.appendChild(el("span", "dim", "→"));
    const c = el("div", "crumb" + (i === history.length - 1 ? " here" : ""));
    const whoSpan = s.who == null ? "" :
      `<span class="who p${s.who}">${s.who === 0 ? "OOP" : "IP"}</span> `;
    c.innerHTML = whoSpan + s.label;
    c.onclick = () => {
      history = history.slice(0, i + 1);
      selectNode(s.node);
    };
    h.appendChild(c);
  });

  // action buttons (with per-action EVs when available)
  const na = $("nav-actions");
  na.innerHTML = "";
  nav.actions.forEach((a, i) => {
    const b = document.createElement("button");
    const kindIdx = nav.actions.filter((x, j) => x.kind === a.kind && j <= i).length - 1;
    b.className = "nact";
    b.style.background = kindColor(a.kind, Math.max(0, kindIdx));
    const parts = [];
    const ae = currentEv && currentEv.action_ev[i];
    if (ae && ae.agg_ev != null) parts.push(`EV ${ae.agg_ev.toFixed(1)}`);
    if (a.next_decide == null) {
      parts.push({ fold: "terminal", showdown: "terminal",
        chance: "runout — terminal" }[a.child_kind] || "terminal");
    } else if (a.child_kind === "chance") {
      parts.push("deals next street");
    }
    b.innerHTML = `${a.label}<small>${parts.join(" · ")}</small>`;
    b.disabled = a.next_decide == null;
    b.onclick = () => {
      if (a.next_decide == null) return;
      // the chip records who acted: the current node's deciding player
      const who = currentStrategy ? currentStrategy.player : 0;
      selectNode(a.next_decide,
        { node: a.next_decide, label: a.label, who });
    };
    na.appendChild(b);
  });

  // per-player node EV chips
  if (currentEv && currentEv.sides) {
    const strip = $("node-ev-strip");
    strip.innerHTML =
      `<span class="evchip p0">OOP EV <b>${currentEv.sides.oop.agg_ev.toFixed(1)}</b></span>` +
      `<span class="evchip p1">IP EV <b>${currentEv.sides.ip.agg_ev.toFixed(1)}</b></span>`;
  }
}

function renderMatrixFromStrategy(s) {
  const colors = s.actions.map((a, i) => kindColor(a.kind,
    s.actions.filter((x, j) => x.kind === a.kind && j <= i).length - 1));
  const labels = s.actions.map((a) =>
    a.kind + (a.amount ? " " + a.amount : ""));
  // per-class unweighted mean of its combos' frequencies or EVs
  const evMode = matrixMode === "ev" && currentEv;
  const dec = currentEv ? currentEv.decider : s.player;
  const evSide = evMode ? currentEv.sides[dec === 0 ? "oop" : "ip"] : null;
  const classes = {};
  for (let c = 0; c < s.cards.length; c++) {
    const k = comboClass(s.cards[c]).cls;
    if (!classes[k]) {
      classes[k] = {
        n: 0, freqs: new Array(s.actions.length).fill(0),
        evs: [],
      };
    }
    const d = classes[k];
    d.n++;
    for (let a = 0; a < s.actions.length; a++)
      d.freqs[a] += s.freqs[a][c];
    if (evSide) {
      // map strategy combo slot -> ev combo slot by hand text
      d.evKey = d.evKey || {};
      d.evKey[s.cards[c]] = c;
    }
  }
  if (evSide) {
    // node-ev combos align by card text; collect per-class EV values
    const evByCard = {};
    evSide.cards.forEach((c, i) => { evByCard[c] = i; });
    for (const cls in classes) {
      const d = classes[cls];
      d.evs = [];
    }
    for (let c = 0; c < s.cards.length; c++) {
      const d = classes[comboClass(s.cards[c]).cls];
      const i = evByCard[s.cards[c]];
      if (i != null && evSide.mass[i] > 0)
        d.evs.push(evSide.ev[i]);
    }
  }
  for (const cls in classes) {
    const d = classes[cls];
    for (let a = 0; a < d.freqs.length; a++) d.freqs[a] /= d.n;
    d.colors = colors;
    d.labels = labels;
  }
  renderMatrix(classes);


  // legend with range-weighted aggregate where available
  const lg = $("legend");
  lg.innerHTML = "";
  s.actions.forEach((a, i) => {
    const l = el("div", "lg");
    const sw = el("div", "sw");
    sw.style.background = colors[i];
    const t = a.kind + (a.amount ? " " + a.amount : "");
    const agg = s.aggregate ? ` — ${(s.aggregate[i] * 100).toFixed(1)}%` : "";
    l.appendChild(sw);
    l.appendChild(el("span", "", t + agg));
    lg.appendChild(l);
  });
  if (!s.aggregate) {
    const note = el("div", "dim",
      "deeper node: per-combo frequencies only (aggregate is exact on " +
      "first-street nodes)");
    note.style.marginTop = "6px";
    $("aggregate").innerHTML = "";
    $("aggregate").appendChild(note);
  } else {
    $("aggregate").innerHTML = "<b>aggregate:</b> " + s.actions.map((a, i) =>
      `${a.kind}${a.amount ? " " + a.amount : ""} ` +
      `<b>${(s.aggregate[i] * 100).toFixed(1)}%</b>`).join(" · ");
  }
}

function renderCombosTable(s) {
  const t = $("strat-table");
  const heads = s.actions.map((a) =>
    `${a.kind}${a.amount ? " " + a.amount : ""}`).join("</th><th>");
  let html = `<thead><tr><th>hand</th><th>${heads}</th></tr></thead><tbody>`;
  for (let c = 0; c < s.cards.length; c++) {
    html += `<tr><td class="combo">${s.cards[c]}</td>`;
    for (let a = 0; a < s.actions.length; a++) {
      const f = s.freqs[a][c];
      html += `<td class="freq" style="background:${kindColor(
        s.actions[a].kind, a)}aa"><b>${(f * 100).toFixed(1)}</b></td>`;
    }
    html += "</tr>";
  }
  t.innerHTML = html + "</tbody>";
}

// ---------------- tree view ----------------
const PAGE = 200;
let treePage = 0;

async function renderTree() {
  if (!active) return;
  try {
    const r = await api("GET",
      `/solvers/${active}/decide-nodes?offset=${treePage * PAGE}&limit=${PAGE}`);
    const flt = $("node-filter").value.toLowerCase();
    const el2 = $("tree-list");
    el2.innerHTML = "";
    for (const n of r.nodes) {
      if (flt && !n.path.toLowerCase().includes(flt)) continue;
      const d = document.createElement("div");
      d.className = "tnode" + (n.index === currentNode ? " sel" : "");
      d.innerHTML = `<span class="idx">#${n.index}</span>` +
        `<span class="who p${n.player}">${n.player === 0 ? "OOP" : "IP"}</span>` +
        `<span>${n.path}</span>`;
      d.onclick = () => {
        // rebuild the history from the path: decide actions alternate
        // the actor (heads-up), deal steps carry no actor
        const steps = n.path.split(" — ");
        history = [{ node: 0, label: "(root)", who: 0 }];
        let who = 0;
        for (const st of steps.slice(1)) {
          if (st === "deal") {
            history.push({ node: null, label: "deal", who: null });
            continue;
          }
          history.push({ node: null, label: st, who });
          who = 1 - who;
        }
        history[history.length - 1].node = n.index;
        selectNode(n.index);
        switchTab("matrix");
      };
      el2.appendChild(d);
    }
    $("tree-page").textContent =
      `page ${treePage + 1} / ${Math.ceil(r.total / PAGE)} (${r.total})`;
  } catch (e) { err("tree-list")(e); }
}

// ---------------- ranges ----------------
async function renderRange(p) {
  if (!active) return;
  const r = await api("GET", `/solvers/${active}/range?player=${p === "oop" ? 0 : 1}`);
  const by = {};
  r.combos.forEach((c, i) => { by[comboClass(c).cls] = (by[comboClass(c).cls] || 0) + r.weights[i]; });
  // per-class mean weight over its combos present in the range
  const cnt = {};
  r.combos.forEach((c) => { const k = comboClass(c).cls; cnt[k] = (cnt[k] || 0) + 1; });
  const m = $("range-matrix");
  m.innerHTML = "";
  m.appendChild(el("div", "mxh", ""));
  for (const r2 of RANKS_HIGH) m.appendChild(el("div", "mxh", r2));
  let maxw = 0;
  for (const cls in by) maxw = Math.max(maxw, by[cls] / cnt[cls]);
  for (const ri of RANKS_HIGH) {
    const h = el("div", "mxh rowh", ri);
    h.style.display = "flex"; h.style.alignItems = "center";
    m.appendChild(h);
    for (const ci of RANKS_HIGH) {
      const suited = ri !== ci && (RANK_VAL[ri] > RANK_VAL[ci]);
      const cls = ri === ci ? ri + ci : (suited ? ri + ci + "s" : ci + ri + "o");
      const cell = el("div", "mxcell");
      if (by[cls]) {
        const w = by[cls] / cnt[cls] / maxw;
        cell.style.background = `rgba(62,201,122,${(0.12 + 0.75 * w).toFixed(3)})`;
        cell.textContent = (by[cls] / cnt[cls]).toFixed(2);
        cell.title = `${cls}: total weight ${(by[cls]).toFixed(2)} over ${cnt[cls]} combos`;
      } else {
        cell.innerHTML = '<span class="dead">–</span>';
        cell.title = cls + ": not in range";
      }
      m.appendChild(cell);
    }
  }
  $("range-detail").textContent =
    `cells show the mean weight of the class's combos (max ${maxw.toFixed(3)})`;
}

// ---------------- node EVs ----------------
let evSideSel = "oop";

function renderEvTab(p) {
  const t = $("ev-table");
  if (!active || !currentEv) { t.innerHTML = ""; return; }
  const ev = currentEv;
  const sideName = p || evSideSel || "oop";
  evSideSel = sideName;
  const side = ev.sides[sideName];
  const isDec = (sideName === "oop" ? 0 : 1) === ev.decider;
  let heads = isDec
    ? ["hand", "EV"].concat(ev.actions.map((a) =>
        `${a.kind}${a.amount ? " " + a.amount : ""}`))
    : ["hand", "EV", "range mass"];
  let html = "<thead><tr>" + heads.map((h) =>
    `<th>${h}</th>`).join("") + "</tr></thead><tbody>";
  const n = side.cards.length;
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => side.ev[b] - side.ev[a]);
  for (const i of order) {
    if (side.mass[i] <= 0) continue;
    const col = side.ev[i] > 0 ? "var(--green)" : "var(--red)";
    html += `<tr><td class="combo">${side.cards[i]}</td>` +
      `<td style="color:${col}"><b>${side.ev[i].toFixed(2)}</b></td>`;
    if (isDec) {
      for (let a = 0; a < ev.actions.length; ++a) {
        const v = ev.action_ev[a].per_combo[i] || 0;
        html += `<td>${v.toFixed(2)}</td>`;
      }
    } else {
      html += `<td>${side.mass[i].toFixed(0)}</td>`;
    }
    html += "</tr>";
  }
  t.innerHTML = html + "</tbody>";
}

// ---------------- tabs ----------------
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((t) =>
    t.classList.toggle("hidden", t.id !== "tab-" + name));
  if (name === "tree") renderTree();
  if (name === "ev") renderEvTab();
  if (name === "ranges") renderRange("oop");
  if (name === "matrix" && active && !currentStrategy) selectNode(0);
}
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => switchTab(t.dataset.tab);
});
document.querySelectorAll("[data-evp]").forEach((b) => {
  b.onclick = () => renderEvTab(b.dataset.evp);
});
document.querySelectorAll("[data-mx]").forEach((b) => {
  b.onclick = () => {
    matrixMode = b.dataset.mx;
    document.querySelectorAll("[data-mx]").forEach((x) =>
      x.classList.toggle("active", x.dataset.mx === matrixMode));
    if (currentStrategy) renderMatrixFromStrategy(currentStrategy);
  };
});
document.querySelectorAll("[data-rp]").forEach((b) => {
  b.onclick = () => renderRange(b.dataset.rp);
});

// ---------------- controls ----------------
$("solve").onclick = solve;
$("clear-board").onclick = () => { board = []; render(); };
$("cancel-job").onclick = async () => {
  if (active) await api("POST", `/solvers/${active}/cancel`, {});
};
$("continue-btn").onclick = () =>
  solverAction("continue", { max_iters: parseInt($("iters").value, 10) });
$("to-target-btn").onclick = () => {
  const t = $("target").value ? parseFloat($("target").value) : null;
  if (!t) return err("stats")(new Error("set a target first"));
  solverAction("continue", { max_iters: 100000, target: t });
};
$("reset-btn").onclick = () => solverAction("reset", {});
$("delete-btn").onclick = async () => {
  if (!active) return;
  await api("DELETE", `/solvers/${active}`);
  active = null;
  $("stats").textContent = "no solution loaded";
  $("matrix").innerHTML = "";
  await refreshSolvers();
};
$("save-btn").onclick = async () => {
  if (!active) return;
  const name = prompt("solution file name:", "spot.sol");
  if (!name) return;
  try {
    const r = await api("POST", `/solvers/${active}/save`, { name });
    $("stats").innerHTML = `saved ${name} (${r.bytes} bytes)`;
  } catch (e) { err("stats")(e); }
};
$("load-btn").onclick = async () => {
  const name = prompt("solution file name:", "spot.sol");
  if (!name) return;
  try {
    const r = await api("POST", "/solvers/load", { name }, true);
    active = r.id;
    history = [{ node: 0, label: "(root)", who: 0 }];
    await refreshSolvers();
    const st = await api("GET", `/solvers/${active}/stats`);
    showStats(st, `<div class="stat"><div class="v">${r.total_iterations}</div><div class="k">iters total</div></div>`);
    await selectNode(0);
  } catch (e) { err("stats")(e); }
};
$("node-filter").addEventListener("input", () => renderTree());
$("tree-prev").onclick = () => { if (treePage > 0) { treePage--; renderTree(); } };
$("tree-next").onclick = () => { treePage++; renderTree(); };

// ---------------- boot ----------------
(async () => {
  render();
  document.querySelectorAll("[data-mx]").forEach((x) =>
    x.classList.toggle("active", x.dataset.mx === matrixMode));
  try {
    const h = await api("GET", "/health");
    $("conn").textContent = `ok — up to ${h.max_solvers} solvers`;
  } catch (e) { $("conn").textContent = "offline"; }
  await refreshSolvers();
})();
