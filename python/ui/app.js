// pps solver UI: vanilla JS over the pps_api HTTP surface.
// Workflow mirrors desktop-postflop's: build a spot (board picker +
// ranges), solve, browse the labeled decision tree, read per-combo
// strategy grids and EVs, save/load and warm-start solutions.
"use strict";

const $ = (id) => document.getElementById(id);
const RANKS = "23456789TJQKA";
const SUITS = "cdhs";
const SUIT_RED = { d: true, h: true };

// ---------------- board picker ----------------
let board = ["Qs", "9h", "2d"];  // turn spot default

function cardTextCls(c) { return SUIT_RED[c[1]] ? "red" : "blk"; }

function renderBoard() {
  const el = $("board-slots");
  el.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const s = document.createElement("div");
    s.className = "slot " + (board[i] ? cardTextCls(board[i]) : "");
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
    for (const r of RANKS) {
      const c = r + su;
      const b = document.createElement("div");
      b.className = "pk " + cardTextCls(c) + (board.includes(c) ? " used" : "");
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
let busyTimer = null;
async function api(method, url, body) {
  const overlay = method === "POST" || url.includes("solve");
  if (overlay) {
    $("overlay").classList.remove("hidden");
    const t0 = performance.now();
    clearInterval(busyTimer);
    busyTimer = setInterval(() => {
      $("elapsed").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s";
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
    clearInterval(busyTimer);
    if (overlay) $("overlay").classList.add("hidden");
  }
}

function err(where) {
  return (e) => {
    const el = $(where);
    el.innerHTML = `<span class="error">${e.message}</span>`;
  };
}

// ---------------- spot & solve ----------------
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

function solveParams() {
  return {
    max_iters: parseInt($("iters").value, 10),
    target: $("target").value ? parseFloat($("target").value) : null,
  };
}

let active = null;  // current solver id

function showStats(r) {
  const st = r.stats || r;
  const el = $("stats");
  el.innerHTML = `
    <div class="stat"><div class="v oop">${st.ev_oop?.toFixed(2)}</div><div class="k">EV OOP</div></div>
    <div class="stat"><div class="v ip">${st.ev_ip?.toFixed(2)}</div><div class="k">EV IP</div></div>
    <div class="stat"><div class="v">${st.exploitability?.toFixed(4)}</div><div class="k">exploitability</div></div>
    ${r.iterations_run != null ? `
      <div class="stat"><div class="v">${r.iterations_run}</div><div class="k">iters (last)</div></div>
      <div class="stat"><div class="v">${r.total_iterations ?? r.iterations_run}</div><div class="k">iters (total)</div></div>
      <div class="stat"><div class="v">${r.compile_ms != null ? r.compile_ms + "ms" : ""}</div><div class="k">compile</div></div>
      <div class="stat"><div class="v">${(r.solve_ms ?? r.continue_ms ?? "")}</div><div class="k">solve ms</div></div>` : ""}`;
}

async function solve() {
  if (board.length < 3 || board.length > 5) {
    err("stats")(new Error("board must be 3-5 cards")); return;
  }
  try {
    const r = await api("POST", "/solve",
      { ...spotBody(), ...solveParams(), keep: true });
    active = r.solver_id;
    showStats(r);
    await refreshSolvers();
    await selectNode(0);
    treePage = 0;
    await renderTree();
  } catch (e) { err("stats")(e); }
}

// ---------------- solver registry ----------------
async function refreshSolvers() {
  const list = await api("GET", "/solvers");
  const el = $("solver-list");
  if (!list.length) { el.innerHTML = '<span class="id">none</span>'; return; }
  el.innerHTML = "";
  for (const s of list) {
    const d = document.createElement("div");
    d.className = "sent" + (s.id === active ? " sel" : "");
    d.innerHTML =
      `<span class="id">${s.id}</span> ${s.board} <span class="id">` +
      `${s.total_iterations} iters</span>`;
    d.onclick = async () => {
      active = s.id;
      await refreshSolvers();
      const st = await api("GET", `/solvers/${active}/stats`);
      showStats(st);
      await selectNode(0);
    };
    el.appendChild(d);
  }
}

async function solverAction(path, body, label) {
  if (!active) return err("stats")(new Error("no active solver"));
  try {
    const r = await api("POST", `/solvers/${active}/${path}`, body);
    if (r.stats) showStats(r);
    await refreshSolvers();
    await selectNode(currentNode);
    return r;
  } catch (e) { err("stats")(e); }
}

// ---------------- strategy view ----------------
let currentNode = 0;

function freqColor(f) {
  // white -> accent blue by frequency
  const a = Math.round(f * 100);
  return `rgba(79,140,255,${(0.08 + 0.75 * f).toFixed(3)})`;
}

async function selectNode(idx) {
  if (!active) return;
  currentNode = idx;
  const s = await api("GET", `/solvers/${active}/strategy?node=${idx}`);
  $("strat-title").textContent = `Strategy — node ${idx}`;
  const agg = s.aggregate
    ? "<b>aggregate:</b> " + s.actions.map((a, i) =>
        `${a.kind}${a.amount ? " " + a.amount : ""} ` +
        `${(s.aggregate[i] * 100).toFixed(1)}%`).join(" &nbsp; ")
    : "(deeper node: per-combo frequencies only — the range-weighted " +
      "aggregate is exact on first-street nodes)";
  $("aggregate").innerHTML = agg;

  const t = $("strat-table");
  const heads = s.actions.map((a) =>
    `${a.kind}${a.amount ? " " + a.amount : ""}`).join("</th><th>");
  let html = `<thead><tr><th>hand</th><th>${heads}</th></tr></thead><tbody>`;
  for (let c = 0; c < s.cards.length; c++) {
    html += `<tr><td class="combo">${s.cards[c]}</td>`;
    for (let a = 0; a < s.actions.length; a++) {
      const f = s.freqs[a][c];
      html += `<td class="freq" style="background:${freqColor(f)}">` +
        `<b>${(f * 100).toFixed(1)}</b></td>`;
    }
    html += "</tr>";
  }
  t.innerHTML = html + "</tbody>";
}

// ---------------- tree view ----------------
const PAGE = 200;
let treePage = 0;
let treeNodes = [];

async function renderTree() {
  if (!active) return;
  try {
    const r = await api("GET",
      `/solvers/${active}/decide-nodes?offset=${treePage * PAGE}&limit=${PAGE}`);
    treeNodes = r.nodes;
    const el = $("tree-list");
    el.innerHTML = "";
    const flt = $("node-filter").value.toLowerCase();
    for (const n of r.nodes) {
      if (flt && !n.path.toLowerCase().includes(flt)) continue;
      const d = document.createElement("div");
      d.className = "tnode" + (n.index === currentNode ? " sel" : "");
      d.innerHTML = `<span class="idx">#${n.index}</span>` +
        `<span class="who p${n.player}">${n.player === 0 ? "OOP" : "IP"}</span>` +
        `<span>${n.path}</span>`;
      d.onclick = async () => {
        document.querySelectorAll(".tnode").forEach((x) => x.classList.remove("sel"));
        d.classList.add("sel");
        switchTab("strategy");
        await selectNode(n.index);
      };
      el.appendChild(d);
    }
    $("tree-page").textContent = `page ${treePage + 1} / ${Math.ceil(r.total / PAGE)} (${r.total})`;
  } catch (e) { err("tree-list")(e); }
}

$("node-filter").addEventListener("input", () => renderTree());
$("tree-prev").onclick = () => { if (treePage > 0) { treePage--; renderTree(); } };
$("tree-next").onclick = () => { treePage++; renderTree(); };

// ---------------- root EV view ----------------
async function renderEv(p) {
  if (!active) return;
  const ev = await api("GET", `/solvers/${active}/root-ev`);
  const d = ev[p];
  const t = $("ev-table");
  let html = "<thead><tr><th>hand</th><th>EV (chips)</th><th>vs range mass</th></tr></thead><tbody>";
  // sort by EV descending
  const order = d.ev.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  for (const [v, i] of order) {
    const w = v === 0 && d.mass[i] === 0 ? "dead" : (v / 1).toFixed(2);
    const col = v > 0 ? "var(--green)" : "var(--red)";
    html += `<tr><td class="combo">${d.cards[i]}</td>` +
      `<td style="color:${col}"><b>${v.toFixed(2)}</b></td>` +
      `<td>${d.mass[i].toFixed(0)}</td></tr>`;
  }
  t.innerHTML = html + "</tbody>";
}
document.querySelectorAll("[data-evp]").forEach((b) => {
  b.onclick = () => renderEv(b.dataset.evp);
});

// ---------------- tabs ----------------
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((t) =>
    t.classList.toggle("hidden", t.id !== "tab-" + name));
  if (name === "ev") renderEv("oop");
}
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => switchTab(t.dataset.tab);
});

// ---------------- session actions ----------------
$("solve").onclick = solve;
$("clear-board").onclick = () => { board = []; render(); };
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
    const r = await api("POST", "/solvers/load", { name });
    active = r.id;
    await refreshSolvers();
    const st = await api("GET", `/solvers/${active}/stats`);
    showStats(st);
    await selectNode(0);
  } catch (e) { err("stats")(e); }
};

// ---------------- boot ----------------
(async () => {
  render();
  try {
    const h = await api("GET", "/health");
    $("conn").textContent = `ok — up to ${h.max_solvers} solvers, ` +
      `data dir ${h.data_dir}`;
  } catch (e) { $("conn").textContent = "offline"; }
  await refreshSolvers();
})();
