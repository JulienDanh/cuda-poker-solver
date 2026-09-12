#include "postflop_cfr.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>

namespace pps {
namespace pf {

namespace {

uint64_t handValueKey(const HandValue& v) {
  uint64_t key = v.category;
  for (int i = 0; i < 5; ++i) key = key * 13 + static_cast<uint64_t>(v.tiebreak[i]);
  return key;
}

int64_t roundTo(double x) { return static_cast<int64_t>(std::llround(x)); }

}  // namespace

RiverSolver::RiverSolver(const Spot& spot, const BetConfig& cfg)
    : spot_(spot), cfg_(cfg) {
  initScratch();
  for (int p = 0; p < 2; ++p) {
    const Range& r = p == 0 ? spot_.oop : spot_.ip;
    Side& s = sides_[p];
    for (int i = 0; i < kCombos; ++i) {
      uint8_t a, b;
      comboCards(i, a, b);
      bool blocked = false;
      for (int c = 0; c < 5; ++c) {
        if (a == spot_.board[c] || b == spot_.board[c]) blocked = true;
      }
      if (blocked || r.w[i] <= 0.0) continue;
      Card seven[7] = {a, b, spot_.board[0], spot_.board[1], spot_.board[2],
                       spot_.board[3], spot_.board[4]};
      s.cards.push_back(a);
      s.cards.push_back(b);
      s.w.push_back(r.w[i]);
      s.strength.push_back(handValueKey(evaluateN(seven, 7)));
      s.sorted.push_back(s.n);
      ++s.n;
    }
    std::sort(s.sorted.begin(), s.sorted.end(), [&](int x, int y) {
      if (s.strength[x] != s.strength[y]) return s.strength[x] < s.strength[y];
      return x < y;
    });
  }
  buildTree();
}

// ---------------------------------------------------------------------------
// Tree construction (mirrors postflop-solver's action_tree.rs).
// ---------------------------------------------------------------------------

std::vector<TreeAction> RiverSolver::possibleActions(int64_t sc0, int64_t sc1,
                                                     int actor,
                                                     bool afterAllin) const {
  const int64_t S = spot_.stack;
  int opp = actor ^ 1;
  int64_t mySc = actor == 0 ? sc0 : sc1;
  int64_t oppSc = actor == 0 ? sc1 : sc0;
  int64_t myRemaining = S - mySc;
  int64_t oppRemaining = S - oppSc;
  int64_t prevAmount = std::max(sc0, sc1);
  int64_t toCall = mySc < oppSc ? oppSc - mySc : 0;
  // Sizing pot: pot after the current player calls
  // (their formula: starting_pot + 2 * (my_sc + to_call)).
  int64_t pot = spot_.pot + 2 * (mySc + toCall);
  int64_t maxAmount = oppRemaining + prevAmount;
  int64_t minAmount = prevAmount + toCall;
  if (minAmount < 1) minAmount = 1;
  if (minAmount > maxAmount) minAmount = maxAmount;

  std::vector<TreeAction> actions;
  auto isAboveThreshold = [&](int64_t amount) {
    int64_t diff = amount - prevAmount;
    int64_t newPot = pot + 2 * diff;
    int64_t threshold =
        roundTo(static_cast<double>(newPot) * cfg_.forceAllinThreshold);
    return maxAmount <= amount + threshold;
  };

  if (toCall == 0) {
    actions.push_back({ActionKind::Check, 0});
    for (double f : cfg_.betFracs) {
      actions.push_back({ActionKind::Bet, roundTo(f * static_cast<double>(pot))});
    }
    if (maxAmount <= roundTo(cfg_.addAllinThreshold * static_cast<double>(pot))) {
      actions.push_back({ActionKind::AllIn, maxAmount});
    }
  } else {
    actions.push_back({ActionKind::Fold, 0});
    actions.push_back({ActionKind::Call, 0});
    if (!afterAllin) {
      for (double m : cfg_.raiseMults) {
        actions.push_back(
            {ActionKind::Raise, roundTo(m * static_cast<double>(prevAmount))});
      }
      int64_t allinThreshold =
          prevAmount + roundTo(cfg_.addAllinThreshold * static_cast<double>(pot));
      if (maxAmount <= allinThreshold) {
        actions.push_back({ActionKind::AllIn, maxAmount});
      }
    }
  }

  for (auto& a : actions) {
    if (a.kind == ActionKind::Bet || a.kind == ActionKind::Raise) {
      int64_t clamped = std::min(std::max(a.amount, minAmount), maxAmount);
      if (isAboveThreshold(clamped)) {
        a = {ActionKind::AllIn, maxAmount};
      } else {
        a.amount = clamped;
      }
    }
  }

  std::sort(actions.begin(), actions.end(),
            [](const TreeAction& x, const TreeAction& y) {
              if (x.kind != y.kind) return x.kind < y.kind;
              return x.amount < y.amount;
            });
  actions.erase(std::unique(actions.begin(), actions.end(),
                            [](const TreeAction& x, const TreeAction& y) {
                              return x.kind == y.kind && x.amount == y.amount;
                            }),
                actions.end());

  // Merge close bet amounts (mirrors merge_bet_actions).
  if (cfg_.mergingThreshold > 0) {
    std::vector<TreeAction> bets, others;
    for (auto& a : actions) {
      if (a.kind == ActionKind::Bet || a.kind == ActionKind::Raise ||
          a.kind == ActionKind::AllIn) {
        bets.push_back(a);
      } else {
        others.push_back(a);
      }
    }
    std::sort(bets.begin(), bets.end(),
              [](const TreeAction& x, const TreeAction& y) {
                if (x.amount != y.amount) return x.amount > y.amount;
                return x.kind < y.kind;
              });
    std::vector<TreeAction> kept;
    int64_t curAmount = INT64_MAX;
    for (auto& a : bets) {
      double ratio =
          static_cast<double>(a.amount - prevAmount) / static_cast<double>(pot);
      double curRatio = static_cast<double>(curAmount - prevAmount) /
                        static_cast<double>(pot);
      double thresholdRatio =
          (curRatio - cfg_.mergingThreshold) / (1.0 + cfg_.mergingThreshold);
      if (ratio < thresholdRatio * (1.0 - 1e-12)) {
        kept.push_back(a);
        curAmount = a.amount;
      }
    }
    std::reverse(kept.begin(), kept.end());
    // Reassemble in canonical order: non-bet actions (sorted) then kept bets.
    std::sort(others.begin(), others.end(),
              [](const TreeAction& x, const TreeAction& y) {
                if (x.kind != y.kind) return x.kind < y.kind;
                return x.amount < y.amount;
              });
    others.insert(others.end(), kept.begin(), kept.end());
    actions = others;
  }
  return actions;
}

int RiverSolver::buildNode(int64_t sc0, int64_t sc1, int actor, bool afterAllin,
                           int depth) {
  int idx = static_cast<int>(nodes_.size());
  nodes_.emplace_back();
  Node& nd = nodes_[idx];
  nd.sc[0] = sc0;
  nd.sc[1] = sc1;

  // Terminal when betting is complete (equal non-zero contributions).
  if (sc0 == sc1 && sc0 > 0) {
    nd.kind = Node::SHOWDOWN;
    return idx;
  }
  if (depth > 24) {
    nd.kind = Node::SHOWDOWN;  // safety; should not trigger
    return idx;
  }

  auto acts = possibleActions(sc0, sc1, actor, afterAllin);
  if (acts.empty()) {
    nd.kind = Node::SHOWDOWN;
    return idx;
  }
  nd.kind = Node::DECIDE;
  nd.player = actor;
  int n = sides_[actor].n;
  nd.regret.assign(acts.size() * n, 0.0);
  nd.stratSum.assign(acts.size() * n, 0.0);
  nd.children.reserve(acts.size());

  for (size_t a = 0; a < acts.size(); ++a) {
    const TreeAction& act = acts[a];
    int child = -1;
    int64_t nsc0 = sc0, nsc1 = sc1;
    int nActor = actor ^ 1;
    bool nAfterAllin = afterAllin;
    switch (act.kind) {
      case ActionKind::Check:
        if (actor == 0) {
          // OOP checks pre-bet: IP now acts with no bet to face. IP's only
          // "check" here closes the betting (showdown); represent by
          // recursing with actor = 1 — its Check leads to a showdown child.
          child = buildNode(sc0, sc1, 1, afterAllin, depth + 1);
        } else {
          // IP checks behind: betting over, showdown with no river money.
          int term = static_cast<int>(nodes_.size());
          nodes_.emplace_back();
          nodes_[term].kind = Node::SHOWDOWN;
          nodes_[term].sc[0] = sc0;
          nodes_[term].sc[1] = sc1;
          child = term;
        }
        break;
      case ActionKind::Fold: {
        int term = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
        nodes_[term].kind = Node::FOLD;
        nodes_[term].foldBy = actor;
        nodes_[term].sc[0] = sc0;
        nodes_[term].sc[1] = sc1;
        child = term;
        break;
      }
      case ActionKind::Call: {
        int64_t target = std::max(sc0, sc1);
        if (actor == 0) nsc0 = target;
        else nsc1 = target;
        int term = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
        nodes_[term].kind = Node::SHOWDOWN;
        nodes_[term].sc[0] = nsc0;
        nodes_[term].sc[1] = nsc1;
        child = term;
        break;
      }
      case ActionKind::Bet:
      case ActionKind::Raise:
      case ActionKind::AllIn:
        if (act.kind == ActionKind::AllIn) nAfterAllin = true;
        if (actor == 0) nsc0 = act.amount;
        else nsc1 = act.amount;
        child = buildNode(nsc0, nsc1, nActor, nAfterAllin, depth + 1);
        break;
    }
    nd.children.push_back(child);
  }
  nd.actions = std::move(acts);
  return idx;
}

void RiverSolver::buildTree() {
  nodes_.clear();
  // River start: OOP (player 0) acts first.
  buildNode(0, 0, 0, false, 0);
  numNodes_ = static_cast<int>(nodes_.size());
}


void RiverSolver::dumpTree(std::FILE* f) const {
  std::function<void(int, const std::string&)> rec = [&](int idx, const std::string& h) {
    const Node& nd = nodes_[idx];
    if (nd.kind == Node::SHOWDOWN) {
      std::fprintf(f, "%s SHOWDOWN sc=%lld,%lld\n", h.c_str(),
                   (long long)nd.sc[0], (long long)nd.sc[1]);
      return;
    }
    if (nd.kind == Node::FOLD) {
      std::fprintf(f, "%s FOLD by p%d\n", h.c_str(), nd.foldBy);
      return;
    }
    for (size_t a = 0; a < nd.actions.size(); ++a) {
      const char* k = "?";
      switch (nd.actions[a].kind) {
        case ActionKind::Fold: k = "Fold"; break;
        case ActionKind::Check: k = "Check"; break;
        case ActionKind::Call: k = "Call"; break;
        case ActionKind::Bet: k = "Bet"; break;
        case ActionKind::Raise: k = "Raise"; break;
        case ActionKind::AllIn: k = "AllIn"; break;
      }
      std::string line = h + (h.empty() ? "" : " | ") + "p" +
                         std::to_string(nd.player) + ":" + k;
      if (nd.actions[a].kind == ActionKind::Bet ||
          nd.actions[a].kind == ActionKind::Raise ||
          nd.actions[a].kind == ActionKind::AllIn) {
        line += "(" + std::to_string(nd.actions[a].amount) + ")";
      }
      std::fprintf(f, "%s\n", line.c_str());
      rec(nd.children[a], line);
    }
  };
  rec(0, "");
}

// ---------------------------------------------------------------------------
// DCFR parameters.
// ---------------------------------------------------------------------------

RiverSolver::Discount RiverSolver::makeDiscount(int t, int iters) const {
  double posCoef, negCoef, avgCoef;
  if (algo_ == "hs30") {
    double n = static_cast<double>(iters);
    double tt = static_cast<double>(t + 1);
    double alpha = 1.0 + 3.0 * tt / n;
    double beta = -1.0 - 2.0 * tt / n;
    double gamma = 30.0 - 5.0 * tt / n;
    double pa = std::pow(tt, alpha);
    double pb = std::pow(tt, beta);
    posCoef = pa / (pa + 1.0);
    negCoef = pb / (pb + 1.0);
    avgCoef = std::pow(tt / (tt + 1.0), gamma);
  } else {
    // postflop-solver parity.
    double ta = std::max(t - 1, 0);
    double pa = ta * std::sqrt(ta);
    uint32_t k = 0;
    if (t > 0) {
      uint32_t x = static_cast<uint32_t>(t);
      k = 1u << ((31 - __builtin_clz(x)) & ~1u);
    }
    double tg = static_cast<double>(t - static_cast<int>(k));
    posCoef = pa / (pa + 1.0);
    negCoef = 0.5;
    avgCoef = std::pow(tg / (tg + 1.0), 3.0);
  }
  return {posCoef, negCoef, avgCoef};
}

// ---------------------------------------------------------------------------
// Regret matching+ / average strategy.
// ---------------------------------------------------------------------------

void RiverSolver::regretMatching(const Node& nd, int n, double* sigma) const {
  int na = static_cast<int>(nd.actions.size());
  for (int c = 0; c < n; ++c) {
    double sum = 0.0;
    for (int a = 0; a < na; ++a) {
      double r = nd.regret[static_cast<size_t>(a) * n + c];
      if (r > 0.0) sum += r;
    }
    if (sum > 0.0) {
      for (int a = 0; a < na; ++a) {
        double r = nd.regret[static_cast<size_t>(a) * n + c];
        sigma[static_cast<size_t>(a) * n + c] = r > 0.0 ? r / sum : 0.0;
      }
    } else {
      for (int a = 0; a < na; ++a) sigma[static_cast<size_t>(a) * n + c] = 1.0 / na;
    }
  }
}

void RiverSolver::avgStrategy(const Node& nd, int n, double* sigma) const {
  int na = static_cast<int>(nd.actions.size());
  for (int c = 0; c < n; ++c) {
    double sum = 0.0;
    for (int a = 0; a < na; ++a) sum += nd.stratSum[static_cast<size_t>(a) * n + c];
    if (sum > 0.0) {
      for (int a = 0; a < na; ++a) {
        sigma[static_cast<size_t>(a) * n + c] =
            nd.stratSum[static_cast<size_t>(a) * n + c] / sum;
      }
    } else {
      for (int a = 0; a < na; ++a) sigma[static_cast<size_t>(a) * n + c] = 1.0 / na;
    }
  }
}

// ---------------------------------------------------------------------------
// Showdown sweep.
// ---------------------------------------------------------------------------

void RiverSolver::initScratch() {
  for (int k = 0; k < 4; ++k) {
    for (int d = 0; d < kMaxDepth; ++d) {
      scratch_[k][d].assign(kScratchStride, 0.0);
    }
  }
}

void RiverSolver::showdownValues(int nodeIdx, int tr, const double* reachOpp,
                                 double winV, double tieV, double loseV,
                                 double* out) const {
  (void)nodeIdx;
  const Side& ts = sides_[tr];
  const Side& os = sides_[tr ^ 1];
  int nt = ts.n, no = os.n;
  if (nt == 0 || no == 0) return;

  thread_local std::vector<double> wless, weq, wgreater;
  wless.assign(nt, 0.0);
  weq.assign(nt, 0.0);
  wgreater.assign(nt, 0.0);

  double runSum = 0.0;
  double minus[52] = {0.0};
  int j = 0;
  for (int i = 0; i < nt; ++i) {
    int ti = ts.sorted[i];
    uint64_t s = ts.strength[ti];
    while (j < no && os.strength[os.sorted[j]] < s) {
      int oi = os.sorted[j];
      double r = reachOpp[oi];
      runSum += r;
      minus[os.cards[2 * oi]] += r;
      minus[os.cards[2 * oi + 1]] += r;
      ++j;
    }
    double v = runSum - minus[ts.cards[2 * ti]] - minus[ts.cards[2 * ti + 1]];
    wless[ti] = v > 0.0 ? v : 0.0;
  }
  runSum = 0.0;
  std::memset(minus, 0, sizeof(minus));
  j = no - 1;
  for (int i = nt - 1; i >= 0; --i) {
    int ti = ts.sorted[i];
    uint64_t s = ts.strength[ti];
    while (j >= 0 && os.strength[os.sorted[j]] > s) {
      int oi = os.sorted[j];
      double r = reachOpp[oi];
      runSum += r;
      minus[os.cards[2 * oi]] += r;
      minus[os.cards[2 * oi + 1]] += r;
      --j;
    }
    double v = runSum - minus[ts.cards[2 * ti]] - minus[ts.cards[2 * ti + 1]];
    wgreater[ti] = v > 0.0 ? v : 0.0;
  }
  int g = 0;
  for (int i = 0; i < nt; ++i) {
    int ti = ts.sorted[i];
    uint64_t s = ts.strength[ti];
    while (g < no && os.strength[os.sorted[g]] < s) ++g;
    int gEnd = g;
    while (gEnd < no && os.strength[os.sorted[gEnd]] == s) ++gEnd;
    double eq = 0.0;
    for (int q = g; q < gEnd; ++q) {
      int oi = os.sorted[q];
      if (os.cards[2 * oi] == ts.cards[2 * ti] ||
          os.cards[2 * oi] == ts.cards[2 * ti + 1] ||
          os.cards[2 * oi + 1] == ts.cards[2 * ti] ||
          os.cards[2 * oi + 1] == ts.cards[2 * ti + 1]) {
        continue;
      }
      eq += reachOpp[oi];
    }
    weq[ti] = eq;
  }
  for (int c = 0; c < nt; ++c) {
    out[c] = wless[c] * winV + weq[c] * tieV + wgreater[c] * loseV;
  }
}

// ---------------------------------------------------------------------------
// CFR pass (alternating half-step for the traverser).
// ---------------------------------------------------------------------------

void RiverSolver::passRec(int nodeIdx, int tr, const double* reachOpp,
                          const Discount& d, double* outVal, int depth) {
  Node& nd = nodes_[nodeIdx];
  int ntr = sides_[tr].n;
  if (nd.kind == Node::SHOWDOWN) {
    int64_t u = std::max(nd.sc[0], nd.sc[1]) - std::min(nd.sc[0], nd.sc[1]);
    double totalPot = static_cast<double>(spot_.pot + nd.sc[0] + nd.sc[1] - u);
    double mySc = static_cast<double>(nd.sc[tr]);
    showdownValues(nodeIdx, tr, reachOpp, totalPot - mySc, totalPot / 2.0 - mySc,
                   -mySc, outVal);
    return;
  }
  if (nd.kind == Node::FOLD) {
    // The fold payoff is hand-independent, but the counterfactual value
    // carries the opponent's DISJOINT reach mass at this terminal (card-
    // conflicting combos cannot be dealt; showdowns already restrict to
    // disjoint mass via the sweep).
    double v = tr == nd.foldBy ? -static_cast<double>(nd.sc[nd.foldBy])
                               : static_cast<double>(spot_.pot + nd.sc[nd.foldBy]);
    showdownValues(nodeIdx, tr, reachOpp, 1.0, 1.0, 1.0, outVal);
    for (int c = 0; c < ntr; ++c) outVal[c] *= v;
    return;
  }

  int n = sides_[nd.player].n;
  int na = static_cast<int>(nd.actions.size());
  double* sigma = scratch_[0][depth].data();
  regretMatching(nd, n, sigma);
  if (nd.player == tr) {
    double* cfv = scratch_[1][depth].data();
    for (int a = 0; a < na; ++a) {
      passRec(nd.children[a], tr, reachOpp, d, cfv + static_cast<size_t>(a) * ntr,
              depth + 1);
    }
    for (int c = 0; c < ntr; ++c) {
      double v = 0.0;
      for (int a = 0; a < na; ++a) {
        v += sigma[static_cast<size_t>(a) * n + c] *
             cfv[static_cast<size_t>(a) * ntr + c];
      }
      outVal[c] = v;
    }
    for (int a = 0; a < na; ++a) {
      for (int c = 0; c < n; ++c) {
        double& s = nd.stratSum[static_cast<size_t>(a) * n + c];
        s = s * d.avgCoef + sigma[static_cast<size_t>(a) * n + c];
      }
    }
    for (int a = 0; a < na; ++a) {
      for (int c = 0; c < n; ++c) {
        double& r = nd.regret[static_cast<size_t>(a) * n + c];
        double coef = r >= 0.0 ? d.posCoef : d.negCoef;
        r = r * coef + (cfv[static_cast<size_t>(a) * ntr + c] -
                        outVal[c]);
      }
    }
  } else {
    // Opponent's node: the counterfactual value is the plain sum over
    // actions of the child cfvs; the opponent's strategy enters through
    // the updated cfreach, NOT through an extra marginal probability.
    double* childReach = scratch_[2][depth].data();
    double* cv = scratch_[3][depth].data();
    for (int c = 0; c < ntr; ++c) outVal[c] = 0.0;
    for (int a = 0; a < na; ++a) {
      for (int c = 0; c < n; ++c) {
        childReach[c] = reachOpp[c] * sigma[static_cast<size_t>(a) * n + c];
      }
      passRec(nd.children[a], tr, childReach, d, cv, depth + 1);
      for (int c = 0; c < ntr; ++c) outVal[c] += cv[c];
    }
  }
}

void RiverSolver::solve(int iterations, const std::string& algo) {
  algo_ = algo;
  std::vector<double> rootVal0(sides_[0].n), rootVal1(sides_[1].n);
  for (int t = 0; t < iterations; ++t) {
    Discount d = makeDiscount(t, iterations);
    passRec(0, 0, sides_[1].w.data(), d, rootVal0.data(), 0);
    passRec(0, 1, sides_[0].w.data(), d, rootVal1.data(), 0);
  }
  computeStats();
}

// ---------------------------------------------------------------------------
// Final walks.
// ---------------------------------------------------------------------------

void RiverSolver::evOne(int nodeIdx, int tr, const double* reachOpp,
                        double* outVal) const {
  const Node& nd = nodes_[nodeIdx];
  int ntr = sides_[tr].n;
  if (nd.kind == Node::SHOWDOWN) {
    int64_t u = std::max(nd.sc[0], nd.sc[1]) - std::min(nd.sc[0], nd.sc[1]);
    double totalPot = static_cast<double>(spot_.pot + nd.sc[0] + nd.sc[1] - u);
    double mySc = static_cast<double>(nd.sc[tr]);
    showdownValues(nodeIdx, tr, reachOpp, totalPot - mySc, totalPot / 2.0 - mySc,
                   -mySc, outVal);
    return;
  }
  if (nd.kind == Node::FOLD) {
    double v = tr == nd.foldBy ? -static_cast<double>(nd.sc[nd.foldBy])
                               : static_cast<double>(spot_.pot + nd.sc[nd.foldBy]);
    showdownValues(nodeIdx, tr, reachOpp, 1.0, 1.0, 1.0, outVal);
    for (int c = 0; c < ntr; ++c) outVal[c] *= v;
    return;
  }
  int n = sides_[nd.player].n;
  int na = static_cast<int>(nd.actions.size());
  // evOne/brRec run once per solve: allocation is irrelevant here.
  std::vector<double> sigma(static_cast<size_t>(na) * n, 0.0);
  avgStrategy(nd, n, sigma.data());
  std::vector<double> cfv(static_cast<size_t>(na) * ntr, 0.0);
  if (nd.player == tr) {
    for (int a = 0; a < na; ++a) {
      evOne(nd.children[a], tr, reachOpp, &cfv[static_cast<size_t>(a) * ntr]);
    }
    for (int c = 0; c < ntr; ++c) {
      double v = 0.0;
      for (int a = 0; a < na; ++a) {
        v += sigma[static_cast<size_t>(a) * n + c] *
             cfv[static_cast<size_t>(a) * ntr + c];
      }
      outVal[c] = v;
    }
  } else {
    std::vector<double> childReach(n, 0.0);
    for (int c = 0; c < ntr; ++c) outVal[c] = 0.0;
    for (int a = 0; a < na; ++a) {
      for (int c = 0; c < n; ++c) {
        childReach[c] = reachOpp[c] * sigma[static_cast<size_t>(a) * n + c];
      }
      std::vector<double> cv(ntr, 0.0);
      evOne(nd.children[a], tr, childReach.data(), cv.data());
      for (int c = 0; c < ntr; ++c) outVal[c] += cv[c];
    }
  }
}

void RiverSolver::brRec(int nodeIdx, int br, const double* reachOpp,
                        double* v) const {
  const Node& nd = nodes_[nodeIdx];
  int nbr = sides_[br].n;
  if (nd.kind == Node::SHOWDOWN) {
    int64_t u = std::max(nd.sc[0], nd.sc[1]) - std::min(nd.sc[0], nd.sc[1]);
    double totalPot = static_cast<double>(spot_.pot + nd.sc[0] + nd.sc[1] - u);
    double mySc = static_cast<double>(nd.sc[br]);
    showdownValues(nodeIdx, br, reachOpp, totalPot - mySc, totalPot / 2.0 - mySc,
                   -mySc, v);
    return;
  }
  if (nd.kind == Node::FOLD) {
    double val = br == nd.foldBy ? -static_cast<double>(nd.sc[nd.foldBy])
                                 : static_cast<double>(spot_.pot + nd.sc[nd.foldBy]);
    showdownValues(nodeIdx, br, reachOpp, 1.0, 1.0, 1.0, v);
    for (int c = 0; c < nbr; ++c) v[c] *= val;
    return;
  }
  int n = sides_[nd.player].n;
  int na = static_cast<int>(nd.actions.size());
  if (nd.player == br) {
    std::vector<double> cfv(static_cast<size_t>(na) * nbr, 0.0);
    for (int a = 0; a < na; ++a) {
      brRec(nd.children[a], br, reachOpp, &cfv[static_cast<size_t>(a) * nbr]);
    }
    for (int c = 0; c < nbr; ++c) {
      double best = -1e300;
      for (int a = 0; a < na; ++a) {
        best = std::max(best, cfv[static_cast<size_t>(a) * nbr + c]);
      }
      v[c] = best;
    }
  } else {
    std::vector<double> sigma(static_cast<size_t>(na) * n, 0.0);
    avgStrategy(nd, n, sigma.data());
    for (int c = 0; c < nbr; ++c) v[c] = 0.0;
    for (int a = 0; a < na; ++a) {
      std::vector<double> r(reachOpp, reachOpp + n);
      for (int c = 0; c < n; ++c) {
        r[c] = reachOpp[c] * sigma[static_cast<size_t>(a) * n + c];
      }
      std::vector<double> cv(nbr, 0.0);
      brRec(nd.children[a], br, r.data(), cv.data());
      for (int c = 0; c < nbr; ++c) v[c] += cv[c];
    }
  }
}

void RiverSolver::computeStats() {
  int n0 = sides_[0].n, n1 = sides_[1].n;
  // Disjoint pair mass Z = sum over disjoint pairs of w0*w1.
  double Z = 0.0;
  {
    std::vector<double> v(n0, 0.0);
    // Counting every disjoint opponent combo: win=tie=lose coefficient 1.
    showdownValues(0, 0, sides_[1].w.data(), 1.0, 1.0, 1.0, v.data());
    for (int c = 0; c < n0; ++c) Z += sides_[0].w[c] * v[c];
  }
  if (Z <= 0.0) {
    stats_ = NodeStats{};
    return;
  }
  std::vector<double> v0(n0), v1(n1), br0(n0), br1(n1);
  evOne(0, 0, sides_[1].w.data(), v0.data());
  evOne(0, 1, sides_[0].w.data(), v1.data());
  double ev0 = 0.0, ev1 = 0.0;
  for (int c = 0; c < n0; ++c) ev0 += sides_[0].w[c] * v0[c];
  for (int c = 0; c < n1; ++c) ev1 += sides_[1].w[c] * v1[c];
  ev0 /= Z;
  ev1 /= Z;

  brRec(0, 0, sides_[1].w.data(), br0.data());
  double V0 = 0.0;
  for (int c = 0; c < n0; ++c) V0 += sides_[0].w[c] * br0[c];
  V0 /= Z;
  brRec(0, 1, sides_[0].w.data(), br1.data());
  double V1 = 0.0;
  for (int c = 0; c < n1; ++c) V1 += sides_[1].w[c] * br1[c];
  V1 /= Z;

  stats_.ev0 = ev0;
  stats_.ev1 = ev1;
  stats_.expl = (V0 + V1 - static_cast<double>(spot_.pot)) / 2.0;
  stats_.pairMass = static_cast<int64_t>(Z * (1LL << 20));
}

}  // namespace pf
}  // namespace pps
