#include <cmath>
#include <vector>

#include "common.h"
#include "eval.h"
#include "showdown.h"

namespace pps {

namespace {

// All-in equity of seat 0 vs seat 1 for one deal: P(win), P(lose), P(tie)
// of seat 0, estimated by Monte Carlo over boards from the remaining deck.
struct DealEquity {
  double win = 0.0;
  double lose = 0.0;
  double tie = 0.0;
};

DealEquity computeEquity(const PokerGame& g, const PokerGame::State& deal, int boards,
                         uint64_t seed) {
  DealEquity eq;
  RNG rng(seed);
  int n = deal.numPlayers * 2;
  bool dealt[52] = {false};
  for (int i = 0; i < n; ++i) dealt[deal.hole[i]] = true;
  std::vector<Card> avail;
  avail.reserve(52 - n);
  for (int c = 0; c < 52; ++c) {
    if (!dealt[c]) avail.push_back(static_cast<Card>(c));
  }
  Card h0[2] = {deal.hole[0], deal.hole[1]};
  Card h1[2] = {deal.hole[2], deal.hole[3]};
  int w = 0, l = 0, t = 0;
  for (int b = 0; b < boards; ++b) {
    // Sample 5 distinct cards.
    for (int i = 0; i < 5; ++i) {
      int j = i + static_cast<int>(rng.nextBelow(static_cast<uint32_t>(avail.size() - i)));
      std::swap(avail[i], avail[j]);
    }
    Card seven0[7] = {h0[0], h0[1], avail[0], avail[1], avail[2], avail[3], avail[4]};
    Card seven1[7] = {h1[0], h1[1], avail[0], avail[1], avail[2], avail[3], avail[4]};
    HandValue v0 = evaluate7(seven0);
    HandValue v1 = evaluate7(seven1);
    if (v1 < v0) ++w;
    else if (v0 < v1) ++l;
    else ++t;
  }
  double B = static_cast<double>(boards);
  eq.win = w / B;
  eq.lose = l / B;
  eq.tie = t / B;
  return eq;
}

// Expected payoff of a stage-2 (board chance) node, using the deal equity
// instead of sampling: exact given (eq, contributions).
double showdownExpectedPayoff(const PokerGame& g, const PokerGame::State& s,
                              const DealEquity& eq, int player) {
  int64_t c0 = s.contributed[0];
  int64_t c1 = s.contributed[1];
  int64_t b0 = s.stack[0];
  int64_t b1 = s.stack[1];
  int64_t hi = std::max(c0, c1);
  int64_t lo = std::min(c0, c1);
  int64_t uncalled = hi - lo;  // returned to the larger contributor
  int64_t pot = c0 + c1 - uncalled;
  // Seat 0 wins: takes pot (minus its own returned chips are separate).
  auto finalStack = [&](int winner) {
    std::array<int64_t, 2> fs = {b0, b1};
    if (c0 > c1) fs[0] += uncalled;
    else if (c1 > c0) fs[1] += uncalled;
    fs[winner] += pot;
    return fs;
  };
  std::array<int64_t, 2> fsWin = finalStack(0);
  std::array<int64_t, 2> fsLose = finalStack(1);
  // Tie: uncalled still returned, pot split evenly.
  std::array<int64_t, 2> fsTie = {b0, b1};
  if (c0 > c1) fsTie[0] += uncalled;
  else if (c1 > c0) fsTie[1] += uncalled;
  fsTie[0] += pot / 2;
  fsTie[1] += pot - pot / 2;
  int64_t start = g.startStackOf(player);
  double v = eq.win * static_cast<double>(fsWin[player] - start);
  v += eq.lose * static_cast<double>(fsLose[player] - start);
  v += eq.tie * static_cast<double>(fsTie[player] - start);
  return v;
}

// Expected payoff of a continuation terminal (FGS stub = showdown to the
// river). Called pots have equal contributions; the pot goes to the
// better hand with the deal's equity.
double continuationExpectedPayoff(const PokerGame& g, const PokerGame::State& s,
                                   const DealEquity& eq, int player) {
  // Contributions are equal in a called pot with both players covered;
  // fall back to uncalled-return handling if they are not (defensive).
  int64_t c0 = s.contributed[0];
  int64_t c1 = s.contributed[1];
  int64_t c = std::min(c0, c1);
  int64_t hi = std::max(c0, c1);
  int64_t uncalled = hi - c;
  int64_t pot = 2 * c;
  double v = 0.0;
  int64_t start = g.startStackOf(player);
  // Seat 0 wins.
  {
    std::array<int64_t, 2> fs = {s.stack[0], s.stack[1]};
    if (c0 > c1) fs[0] += uncalled;
    if (c1 > c0) fs[1] += uncalled;
    fs[0] += pot;
    v += eq.win * static_cast<double>(fs[player] - start);
  }
  // Seat 1 wins.
  {
    std::array<int64_t, 2> fs = {s.stack[0], s.stack[1]};
    if (c0 > c1) fs[0] += uncalled;
    if (c1 > c0) fs[1] += uncalled;
    fs[1] += pot;
    v += eq.lose * static_cast<double>(fs[player] - start);
  }
  // Tie.
  {
    std::array<int64_t, 2> fs = {s.stack[0], s.stack[1]};
    if (c0 > c1) fs[0] += uncalled;
    if (c1 > c0) fs[1] += uncalled;
    fs[0] += pot / 2;
    fs[1] += pot - pot / 2;
    v += eq.tie * static_cast<double>(fs[player] - start);
  }
  return v;
}

enum class Mode { Avg, Br };

// Full tree DP from state s against the opponent's fixed average
// strategy. Chance and continuation terminals are evaluated with the
// per-deal equity, so no decision in the tree can condition on a
// particular board and no max is taken over noisy samples.
double evalNode(const PokerGame& g, const PokerGame::State& s, int brPlayer, Mode mode,
                const std::unordered_map<uint64_t, InfosetStrategy>& avg,
                const DealEquity& eq) {
  if (g.isTerminal(s)) {
    if (s.terminalKind == 3) return continuationExpectedPayoff(g, s, eq, brPlayer);
    return g.payoff(s, brPlayer);
  }
  if (g.isChance(s)) return showdownExpectedPayoff(g, s, eq, brPlayer);
  int p = g.currentPlayer(s);
  auto acts = g.legalActions(s);
  std::vector<double> sigma(acts.size(), 1.0 / acts.size());
  auto it = avg.find(g.infosetKey(s, p));
  if (it != avg.end() &&
      static_cast<int>(it->second.avg.size()) == static_cast<int>(acts.size())) {
    sigma = it->second.avg;
  }
  if (mode == Mode::Br && p == brPlayer) {
    double best = -1e300;
    for (const auto& a : acts) {
      best = std::max(best, evalNode(g, g.apply(s, a), brPlayer, mode, avg, eq));
    }
    return best;
  }
  double v = 0.0;
  for (size_t a = 0; a < acts.size(); ++a) {
    if (sigma[a] == 0.0) continue;
    v += sigma[a] * evalNode(g, g.apply(s, acts[a]), brPlayer, mode, avg, eq);
  }
  return v;
}

}  // namespace

ExploitabilityResult huExploitability(
    const PokerGame& game,
    const std::unordered_map<uint64_t, InfosetStrategy>& avg,
    int deals, int boardsPerEval, uint64_t seed) {
  ExploitabilityResult res;
  res.deals = deals;
  RNG rng(seed);
  double sum = 0.0;
  double sumSq = 0.0;
  double vAvgSum = 0.0;
  for (int d = 0; d < deals; ++d) {
    PokerGame::State deal = game.sampleChance(game.rootState(), rng);
    DealEquity eq = computeEquity(game, deal, boardsPerEval, rng.nextU64());
    double vAvg0 = evalNode(game, deal, 0, Mode::Avg, avg, eq);
    double vAvg1 = evalNode(game, deal, 1, Mode::Avg, avg, eq);
    double vBr0 = evalNode(game, deal, 0, Mode::Br, avg, eq);
    double vBr1 = evalNode(game, deal, 1, Mode::Br, avg, eq);
    double e = (vBr0 - vAvg0) + (vBr1 - vAvg1);
    sum += e;
    sumSq += e * e;
    vAvgSum += 0.5 * (vAvg0 - vAvg1);
  }
  double n = static_cast<double>(deals);
  res.exploitability = sum / n;
  res.vAvg0 = vAvgSum / n;
  if (deals > 1) {
    double var = (sumSq - n * res.exploitability * res.exploitability) / (n - 1);
    res.sem = std::sqrt(std::max(0.0, var) / n);
  }
  return res;
}

}  // namespace pps
