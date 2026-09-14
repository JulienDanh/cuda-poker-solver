// pps_native: Python bindings for the GPU postflop solver
// (pps::gpu::GpuPostflopSolver, see cuda/gpu_cfr.h). Exposed through
// the thin `pps` package (python/pps/__init__.py).
//
//   import pps
//   s = pps.Solver(board="Qs9h2d", oop="TT+,AKo", ip="AQ+,KQs",
//                 pot=200, stack=500, bets="0.75,a", raises="2.5,3")
//   s.solve(2000)                  # or s.solve(max_iters=100000, target=0.1)
//   s.stats()                      # EV / exploitability
//   s.strategy()                   # labeled per-combo root strategy
//   s.decide_nodes()               # tree introspection
//   s.save("spot.sol"); t = pps.Solver.load("spot.sol"); t.continue_solve(500)
//
// Card / range syntax matches the gpu_pfflop CLI: cards are rank+suit
// ("Ah"), boards are concatenated cards, ranges are postflop-solver
// syntax ("TT+,AKs", "AA:0.5"), bet sizes are pot fractions with "a"
// (all-in) and "e" (geometric) tokens.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_cfr.h"
#include "postflop_cfr.h"
#include "range.h"

namespace py = pybind11;
using pps::gpu::GpuPostflopSolver;
using pps::Card;
using pps::Range;

namespace {

char rankChar(int r) {
  static const char* kRanks = "23456789TJQKA";
  return kRanks[r];
}
char suitChar(int s) {
  static const char* kSuits = "cdhs";
  return kSuits[s];
}

Card cardFromText(const std::string& s, size_t i) {
  if (i + 1 >= s.size() || s[i] == ',')
    throw std::runtime_error("bad card text '" + s.substr(0, 2) + "'");
  int rank;
  switch (s[i]) {
    case '2': rank = 0; break;
    case '3': rank = 1; break;
    case '4': rank = 2; break;
    case '5': rank = 3; break;
    case '6': rank = 4; break;
    case '7': rank = 5; break;
    case '8': rank = 6; break;
    case '9': rank = 7; break;
    case 'T': case 't': rank = 8; break;
    case 'J': case 'j': rank = 9; break;
    case 'Q': case 'q': rank = 10; break;
    case 'K': case 'k': rank = 11; break;
    case 'A': case 'a': rank = 12; break;
    default: throw std::runtime_error("bad card rank '" + std::string(1, s[i]) + "'");
  }
  int suit;
  switch (s[i + 1]) {
    case 'c': case 'C': suit = 0; break;
    case 'd': case 'D': suit = 1; break;
    case 'h': case 'H': suit = 2; break;
    case 's': case 'S': suit = 3; break;
    default: throw std::runtime_error("bad card suit '" + std::string(1, s[i + 1]) + "'");
  }
  return static_cast<Card>(suit * 13 + rank);
}

std::string cardText(uint8_t c) {
  return std::string{rankChar(c % 13), suitChar(c / 13)};
}

std::string comboText(uint8_t a, uint8_t b) {
  return cardText(a) + cardText(b);
}

// Board: "Qs9h2d" or ["Qs", "9h", "2d"] (3-5 cards).
bool isListLike(const py::object& o) {
  return py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o);
}

std::vector<Card> parseBoard(const py::object& board) {
  std::vector<Card> out;
  std::string joined;
  if (py::isinstance<py::str>(board)) {
    joined = board.cast<std::string>();
  } else if (isListLike(board)) {
    for (auto c : board)
      joined += py::str(c).cast<std::string>();
  } else {
    throw std::runtime_error("board must be a string or a list of card strings");
  }
  if (joined.size() % 2 != 0 || joined.size() < 6 || joined.size() > 10)
    throw std::runtime_error("board must be 3, 4 or 5 cards (e.g. \"Qs9h2d\")");
  for (size_t i = 0; i < joined.size(); i += 2) out.push_back(cardFromText(joined, i));
  return out;
}

// Range: "TT+,AKs" / "AA:0.5" syntax, a dict {hand: weight}, or a list
// of 1326 floats (combo weights, see pps::Range).
Range parseRangeArg(const py::object& r) {
  Range out;
  if (py::isinstance<py::str>(r)) {
    out = pps::parseRange(r.cast<std::string>());
  } else if (py::isinstance<py::dict>(r)) {
    std::string joined;
    for (auto item : r.cast<py::dict>()) {
      if (!joined.empty()) joined += ",";
      joined += py::str(item.first).cast<std::string>() + ":" +
                std::to_string(item.second.cast<double>());
    }
    out = pps::parseRange(joined);
  } else if (isListLike(r)) {
    auto v = r.cast<std::vector<double>>();
    if (v.size() != pps::kCombos)
      throw std::runtime_error("range list must have exactly 1326 weights");
    for (int i = 0; i < pps::kCombos; ++i) out.w[i] = v[i];
  } else {
    throw std::runtime_error("range must be a string, dict or 1326-float list");
  }
  return out;
}

// Bet sizes: "0.75,a" or [0.75, "a", "e"] (fractions of the
// call-inclusive pot; "a" = explicit all-in, "e" = geometric).
void parseBetSizes(const py::object& arg, pps::pf::BetConfig& cfg,
                   bool& sawAllIn) {
  if (py::isinstance<py::str>(arg)) {
    const std::string s = arg.cast<std::string>();
    size_t start = 0;
    while (start <= s.size()) {
      size_t end = s.find(',', start);
      std::string tok = end == std::string::npos ? s.substr(start)
                                                 : s.substr(start, end - start);
      if (!tok.empty()) {
        if (tok == "a" || tok == "A" || tok == "allin") sawAllIn = true;
        else if (tok == "e" || tok == "E" || tok == "geo")
          cfg.betGeometric = true;
        else {
          try { cfg.betFracs.push_back(std::stod(tok)); }
          catch (...) {
            throw std::runtime_error("bad bet size token '" + tok + "'");
          }
        }
      }
      if (end == std::string::npos) break;
      start = end + 1;
    }
  } else if (isListLike(arg)) {
    for (auto item : arg) {
      if (py::isinstance<py::str>(item)) {
        const std::string tok = item.cast<std::string>();
        if (tok == "a" || tok == "A" || tok == "allin") sawAllIn = true;
        else if (tok == "e" || tok == "E" || tok == "geo")
          cfg.betGeometric = true;
        else throw std::runtime_error("bad bet size token '" + tok + "'");
      } else {
        cfg.betFracs.push_back(item.cast<double>());
      }
    }
  } else {
    throw std::runtime_error("bets must be a string or a list");
  }
}

std::vector<double> parseDoubleList(const py::object& arg) {
  if (py::isinstance<py::str>(arg)) {
    std::vector<double> out;
    const std::string s = arg.cast<std::string>();
    size_t start = 0;
    while (start <= s.size()) {
      size_t end = s.find(',', start);
      std::string tok = end == std::string::npos ? s.substr(start)
                                                 : s.substr(start, end - start);
      if (!tok.empty()) out.push_back(std::stod(tok));
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return out;
  }
  return arg.cast<std::vector<double>>();
}

py::dict actionDict(const pps::gpu::ActionLabel& a) {
  const char* kind;
  switch (a.kind) {
    case pps::pf::ActionKind::Fold: kind = "fold"; break;
    case pps::pf::ActionKind::Check: kind = "check"; break;
    case pps::pf::ActionKind::Call: kind = "call"; break;
    case pps::pf::ActionKind::Bet: kind = "bet"; break;
    case pps::pf::ActionKind::Raise: kind = "raise"; break;
    case pps::pf::ActionKind::AllIn: kind = "allin"; break;
    default: kind = "?";
  }
  py::dict d;
  d["kind"] = kind;
  d["amount"] = a.amount;  // street total contribution in chips
  return d;
}

py::array_t<double> toArray(const std::vector<double>& v) {
  py::array_t<double> a((py::ssize_t)v.size());
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

class Solver {
 public:
  Solver(const py::object& board, const py::object& oop,
         const py::object& ip, int64_t pot, int64_t stack,
         const py::object& bets, const py::object& raises, int maxRaises) {
    std::vector<Card> bd = parseBoard(board);
    pps::gpu::PostflopSpot spot;
    spot.nBoard = (int)bd.size();
    for (int i = 0; i < spot.nBoard; ++i) spot.board[i] = bd[i];
    spot.oop = parseRangeArg(oop);
    spot.ip = parseRangeArg(ip);
    spot.pot = pot;
    spot.stack = stack;
    spot.oop.filterBoard(spot.board, spot.nBoard);
    spot.ip.filterBoard(spot.board, spot.nBoard);
    pps::pf::BetConfig cfg;
    cfg.betFracs.clear();
    bool sawAllIn = false;
    parseBetSizes(bets, cfg, sawAllIn);
    cfg.betAllIn = sawAllIn;
    cfg.raiseMults = parseDoubleList(raises);
    cfg.maxRaises = maxRaises;
    // The compile (tree build + upload) is the heavy part and touches
    // no Python objects — release the GIL so other threads (an API
    // server) keep serving during it.
    py::gil_scoped_release rel;
    solver_ = std::make_unique<GpuPostflopSolver>(spot, cfg);
  }

  static std::unique_ptr<Solver> load(const std::string& path) {
    auto s = std::unique_ptr<Solver>(new Solver());
    py::gil_scoped_release rel;
    s->solver_ = std::make_unique<GpuPostflopSolver>(path);
    return s;
  }

  // The engine calls below run without the GIL (they never touch
  // Python objects), so a long solve does not freeze the interpreter
  // thread pool. Concurrency between threads must still be handled by
  // the caller: two threads driving one solver interleave CUDA work.
  void solve(int maxIters, const std::string& algo, double target) {
    py::gil_scoped_release rel;
    solver_->solve(maxIters, algo, target);
  }
  void continueSolve(int maxIters, double target) {
    py::gil_scoped_release rel;
    solver_->continueSolve(maxIters, target);
  }
  void reset() {
    py::gil_scoped_release rel;
    solver_->reset();
  }

  py::dict stats() {
    pps::pf::NodeStats st;
    {
      py::gil_scoped_release rel;
      st = solver_->stats();
    }
    py::dict d;
    d["ev_oop"] = st.ev0;
    d["ev_ip"] = st.ev1;
    d["exploitability"] = st.expl;
    d["pair_mass"] = st.pairMass;
    return d;
  }

  int numNodes() const { return solver_->numNodes(); }
  int maxDepth() const { return solver_->maxDepth(); }
  int numDecideNodes() const { return solver_->numDecideNodes(); }
  int iterationsRun() const { return solver_->iterationsRun(); }
  int64_t totalIterations() const { return solver_->totalIterations(); }

  py::list decideNodes() const {
    std::vector<pps::gpu::DecideNodeInfo> ns;
    {
      py::gil_scoped_release rel;
      ns.reserve(solver_->numDecideNodes());
      for (int i = 0; i < solver_->numDecideNodes(); ++i)
        ns.push_back(solver_->decideNode(i));
    }
    py::list out;
    for (size_t i = 0; i < ns.size(); ++i) {
      const auto& n = ns[i];
      py::dict d;
      d["index"] = i;
      d["node_id"] = n.nodeId;
      d["player"] = n.player;
      d["depth"] = n.depth;
      d["n_board"] = n.nBoard;
      d["n_combos"] = n.nCombos;
      py::list acts, kids;
      for (const auto& a : n.actions) acts.append(actionDict(a));
      for (int c : n.children) kids.append(c);
      d["actions"] = acts;
      d["children"] = kids;
      out.append(d);
    }
    return out;
  }

  // Labeled per-combo strategy of decide node `decideIdx`: freqs is an
  // (n_actions, n_combos) array of average-strategy frequencies, cards
  // aligns with its columns, combos are base-list slots. `aggregate`
  // (present only for first-street nodes, where base range weights are
  // the exact reach weights) is the range-weighted frequency vector.
  py::dict strategy(int decideIdx) {
    pps::gpu::ComboStrategy cs;
    pps::gpu::DecideNodeInfo n;
    std::vector<uint8_t> cards;
    std::vector<double> agg;
    {
      py::gil_scoped_release rel;
      cs = solver_->comboStrategy(decideIdx);
      n = solver_->decideNode(decideIdx);
      cards = solver_->playerCards(n.player);
      agg = solver_->nodeStrategy(decideIdx);
    }
    const int na = (int)cs.actions.size();
    const int nC = (int)cs.combos.size();
    py::dict d;
    d["index"] = decideIdx;
    d["node_id"] = n.nodeId;
    d["player"] = n.player;
    d["depth"] = n.depth;
    py::list acts;
    for (const auto& a : cs.actions) acts.append(actionDict(a));
    d["actions"] = acts;
    // Column labels: the deciding player's combo at this node.
    py::list cardList;
    for (int c : cs.combos)
      cardList.append(comboText(cards[2 * c], cards[2 * c + 1]));
    d["cards"] = cardList;
    py::array_t<double> freqs({(py::ssize_t)na, (py::ssize_t)nC});
    std::copy(cs.freqs.begin(), cs.freqs.end(), freqs.mutable_data());
    d["freqs"] = freqs;
    d["combos"] = cs.combos;  // base-list slots, for joining with weights
    if (!agg.empty()) d["aggregate"] = toArray(agg);
    return d;
  }

  // Per-combo root EVs vs the average strategy (chips), aligned with
  // each player's base list.
  py::dict rootEv() {
    pps::gpu::ComboEv ev;
    std::vector<uint8_t> cards[2];
    {
      py::gil_scoped_release rel;
      ev = solver_->rootEvPerCombo();
      for (int p = 0; p < 2; ++p) cards[p] = solver_->playerCards(p);
    }
    py::dict out;
    const char* names[2] = {"oop", "ip"};
    for (int p = 0; p < 2; ++p) {
      py::list cardList;
      for (size_t i = 0; i < ev.combos[p].size(); ++i) {
        const int c = ev.combos[p][i];
        cardList.append(comboText(cards[p][2 * c], cards[p][2 * c + 1]));
      }
      py::dict d;
      d["combos"] = toVectorList(ev.combos[p]);
      d["cards"] = cardList;
      d["ev"] = toArray(ev.ev[p]);
      d["mass"] = toArray(ev.mass[p]);
      out[names[p]] = d;
    }
    return out;
  }

  // Per-combo EVs of any decide node vs the average strategy, plus
  // the decider's per-action EVs (chips).
  py::dict nodeEv(int decideIdx) {
    pps::gpu::NodeEv ev;
    std::vector<uint8_t> cards[2];
    {
      py::gil_scoped_release rel;
      ev = solver_->nodeEv(decideIdx);
      for (int p = 0; p < 2; ++p) cards[p] = solver_->playerCards(p);
    }
    py::dict out;
    out["decider"] = ev.decider;
    py::list acts;
    for (const auto& a : ev.actions) acts.append(actionDict(a));
    out["actions"] = acts;
    py::list aev;
    for (const auto& a : ev.actionEv) {
      py::dict d;
      d["per_combo"] = toArray(a.perCombo);
      d["agg_ev"] = a.aggEv;
      aev.append(d);
    }
    out["action_ev"] = aev;
    const char* names[2] = {"oop", "ip"};
    py::dict sides;
    for (int p = 0; p < 2; ++p) {
      py::dict d;
      py::list cardList;
      for (int c : ev.side[p].combos)
        cardList.append(comboText(cards[p][2 * c], cards[p][2 * c + 1]));
      d["combos"] = toVectorList(ev.side[p].combos);
      d["cards"] = cardList;
      d["ev"] = toArray(ev.side[p].ev);
      d["mass"] = toArray(ev.side[p].mass);
      d["agg_ev"] = ev.side[p].aggEv;
      sides[names[p]] = d;
    }
    out["sides"] = sides;
    return out;
  }

  void save(const std::string& path) const {
    py::gil_scoped_release rel;
    solver_->save(path);
  }

  // Raw card ids per player: flat list cards[2i], cards[2i+1] of combo
  // slot i (the engine's 0..51 card index: suit*13 + rank).
  py::list playerCardsList(int player) const {
    auto cards = solver_->playerCards(player);
    py::list out;
    for (uint8_t c : cards) out.append((int)c);
    return out;
  }
  py::array_t<double> playerWeightsArr(int player) const {
    return toArray(solver_->playerWeights(player));
  }
  int numBaseCombos(int player) const {
    return solver_->numBaseCombos(player);
  }

  // Per-node kind (0 decide, 1 showdown, 2 fold, 3 chance) + CSR
  // children table — enough to walk the whole tree from the root.
  py::dict treeStructure() const {
    pps::gpu::TreeStructure t;
    {
      py::gil_scoped_release rel;
      t = solver_->treeStructure();
    }
    py::list kinds, childBase, children;
    for (uint8_t k : t.kinds) kinds.append((int)k);
    for (int c : t.childBase) childBase.append(c);
    for (int c : t.children) children.append(c);
    py::dict d;
    d["kinds"] = kinds;
    d["child_base"] = childBase;
    d["children"] = children;
    return d;
  }

 private:
  Solver() = default;
  static py::list toVectorList(const std::vector<int>& v) {
    py::list out;
    for (int x : v) out.append(x);
    return out;
  }
  std::unique_ptr<GpuPostflopSolver> solver_;
};

}  // namespace

PYBIND11_MODULE(pps_native, m) {
  m.doc() = "GPU postflop solver (compile-to-dataflow DCFR, CUDA graph replay)";

  py::class_<Solver>(m, "Solver")
      .def(py::init([](const py::object& board, const py::object& oop,
                       const py::object& ip, int64_t pot, int64_t stack,
                       const py::object& bets, const py::object& raises,
                       int max_raises) {
             return std::make_unique<Solver>(board, oop, ip, pot, stack, bets,
                                             raises, max_raises);
           }),
           py::arg("board"), py::arg("oop"), py::arg("ip"), py::arg("pot"),
           py::arg("stack"), py::arg("bets") = "0.75,a",
           py::arg("raises") = "2.5,3", py::arg("max_raises") = 0)
      .def_static("load", &Solver::load, py::arg("path"))
      .def("solve", &Solver::solve, py::arg("max_iters"),
           py::arg("algo") = "dcfr", py::arg("target") = -1.0)
      .def("continue_solve", &Solver::continueSolve, py::arg("max_iters"),
           py::arg("target") = -1.0)
      .def("reset", &Solver::reset)
      .def("stats", &Solver::stats)
      .def_property_readonly("num_nodes", &Solver::numNodes)
      .def_property_readonly("max_depth", &Solver::maxDepth)
      .def_property_readonly("num_decide_nodes", &Solver::numDecideNodes)
      .def_property_readonly("iterations_run", &Solver::iterationsRun)
      .def_property_readonly("total_iterations", &Solver::totalIterations)
      .def("decide_nodes", &Solver::decideNodes)
      .def("strategy", &Solver::strategy, py::arg("decide_idx") = 0)
      .def("root_ev", &Solver::rootEv)
      .def("node_ev", &Solver::nodeEv, py::arg("decide_idx") = 0)
      .def("save", &Solver::save, py::arg("path"))
      .def("player_cards", &Solver::playerCardsList, py::arg("player"))
      .def("player_weights", &Solver::playerWeightsArr, py::arg("player"))
      .def("num_base_combos", &Solver::numBaseCombos, py::arg("player"))
      .def("tree_structure", &Solver::treeStructure);
}
