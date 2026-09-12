// Unit tests: cards, ICM, hand169, showdown, MCCFR-on-Kuhn, poker state machine.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>

#include "cards.h"
#include "framework.h"
#include "hand169.h"
#include "icm.h"
#include "kuhn.h"
#include "poker.h"
#include "showdown.h"
#include "solver.h"

using namespace pps;

// ---------------------------------------------------------------- cards ---

TEST(HandEvalCategories) {
  auto C = [](int rank, int suit) { return static_cast<Card>(suit * 13 + rank); };
  Card royalFlush[7] = {C(12, 0), C(11, 0), C(10, 0), C(9, 0), C(8, 0), C(5, 1), C(3, 2)};
  Card quads[7] = {C(7, 0), C(7, 1), C(7, 2), C(7, 3), C(12, 0), C(11, 1), C(5, 2)};
  Card fullHouse[7] = {C(7, 0), C(7, 1), C(7, 2), C(3, 3), C(3, 1), C(12, 0), C(11, 1)};
  Card flush[7] = {C(12, 2), C(10, 2), C(8, 2), C(5, 2), C(3, 2), C(7, 1), C(6, 0)};
  Card straight[7] = {C(4, 0), C(5, 1), C(6, 2), C(7, 3), C(8, 0), C(12, 1), C(2, 2)};
  Card wheel[7] = {C(12, 0), C(0, 1), C(1, 2), C(2, 3), C(3, 0), C(7, 1), C(9, 2)};
  Card trips[7] = {C(9, 0), C(9, 1), C(9, 2), C(4, 3), C(6, 1), C(11, 0), C(12, 1)};
  Card twoPair[7] = {C(5, 0), C(5, 1), C(2, 2), C(2, 3), C(12, 0), C(11, 1), C(10, 2)};
  Card pair[7] = {C(11, 0), C(11, 1), C(2, 2), C(4, 3), C(6, 1), C(9, 0), C(7, 1)};
  Card highCard[7] = {C(12, 0), C(10, 1), C(8, 2), C(6, 3), C(4, 0), C(2, 1), C(5, 2)};
  CHECK(evaluate7(royalFlush).category == 8);
  CHECK(evaluate7(quads).category == 7);
  CHECK(evaluate7(fullHouse).category == 6);
  CHECK(evaluate7(flush).category == 5);
  CHECK(evaluate7(straight).category == 4);
  CHECK(evaluate7(wheel).category == 4);
  CHECK(evaluate7(trips).category == 3);
  CHECK(evaluate7(twoPair).category == 2);
  CHECK(evaluate7(pair).category == 1);
  CHECK(evaluate7(highCard).category == 0);
  CHECK(evaluate7(wheel) < evaluate7(straight));
  Card quadsLow[7] = {C(3, 0), C(3, 1), C(3, 2), C(3, 3), C(2, 0), C(11, 1), C(12, 2)};
  CHECK(evaluate7(quadsLow) < evaluate7(quads));
}

// ------------------------------------------------------------------ ICM ---

TEST(IcmMatchesBruteForce) {
  RNG rng(7);
  for (int trial = 0; trial < 25; ++trial) {
    int n = 3 + static_cast<int>(rng.nextBelow(3));
    std::vector<int64_t> stacks(n);
    for (auto& s : stacks) s = 100 + rng.nextBelow(10000);
    std::vector<double> payouts{0.5, 0.3, 0.2};
    auto fast = icmEquities(stacks, payouts);
    auto slow = icmBruteForce(stacks, payouts);
    for (int i = 0; i < n; ++i) CHECK_NEAR(fast[i], slow[i], 1e-12);
    double sum = 0.0;
    for (double e : fast) sum += e;
    CHECK_NEAR(sum, 1.0, 1e-12);
  }
}

TEST(IcmEqualStacks) {
  std::vector<int64_t> stacks(8, 10000);
  std::vector<double> payouts{0.5, 0.3, 0.2};
  auto eq = icmEquities(stacks, payouts);
  for (double e : eq) CHECK_NEAR(e, 1.0 / 8.0, 1e-12);
}

TEST(IcmBiggerStackMoreEquity) {
  std::vector<int64_t> stacks{50000, 10000, 10000, 10000, 10000, 10000, 10000, 10000};
  std::vector<double> payouts{0.5, 0.3, 0.2};
  auto eq = icmEquities(stacks, payouts);
  CHECK(eq[0] > eq[1]);
  for (int i = 1; i < 8; ++i) CHECK_NEAR(eq[i], eq[1], 1e-12);
}

// -------------------------------------------------------------- hand169 ---

TEST(Hand169Indexing) {
  auto C = [](int rank, int suit) { return static_cast<Card>(suit * 13 + rank); };
  CHECK(handIndex169(C(0, 0), C(0, 1)) == 0);
  CHECK(handIndex169(C(12, 0), C(12, 1)) == 12);
  CHECK(handIndex169(C(12, 0), C(11, 0)) == 90);
  CHECK(handIndex169(C(12, 0), C(11, 1)) == 168);
  CHECK(handIndex169(C(11, 1), C(12, 0)) == 168);
  std::map<int, int> counts;
  for (Card a = 0; a < 52; ++a) {
    for (Card b = a + 1; b < 52; ++b) ++counts[handIndex169(a, b)];
  }
  for (int i = 0; i < 169; ++i) {
    int expect = i < 13 ? 6 : (i < 91 ? 4 : 12);
    CHECK(counts[i] == expect);
  }
  CHECK(handName169(0) == "22");
  CHECK(handName169(12) == "AA");
  CHECK(handName169(90) == "AKs");
  CHECK(handName169(168) == "AKo");
}

TEST(BucketClustering) {
  CHECK(bucketOf(12, 15) == 0);
  CHECK(bucketOf(0, 15) == 2);
  CHECK(bucketOf(90, 15) >= 3 && bucketOf(90, 15) <= 8);
  CHECK(bucketOf(168, 15) >= 9);
  // Every bucket is non-empty and every hand maps into range.
  int seen[15] = {0};
  for (int i = 0; i < 169; ++i) {
    int b = bucketOf(i, 15);
    CHECK(b >= 0 && b < 15);
    seen[b] = 1;
  }
  for (int b = 0; b < 15; ++b) CHECK(seen[b] == 1);
}

TEST(IcmZeroStacks) {
  // Zero-chip players deterministically occupy the bottom places.
  std::vector<int64_t> stacks{5000, 0, 5000};
  std::vector<double> payouts{0.5, 0.3, 0.2};
  auto eq = icmEquities(stacks, payouts);
  CHECK_NEAR(eq[0], 0.4, 1e-12);
  CHECK_NEAR(eq[2], 0.4, 1e-12);
  CHECK_NEAR(eq[1], 0.2, 1e-12);

  std::vector<int64_t> stacks2{5000, 0, 0};
  auto eq2 = icmEquities(stacks2, payouts);
  CHECK_NEAR(eq2[0], 0.5, 1e-12);
  CHECK_NEAR(eq2[1], 0.25, 1e-12);
  CHECK_NEAR(eq2[2], 0.25, 1e-12);

  // Uneven positive stacks with a short stack: no NaN anywhere.
  std::vector<int64_t> stacks3{9000, 0, 1000, 3000};
  std::vector<double> payouts3{0.5, 0.3, 0.15, 0.05};
  auto eq3 = icmEquities(stacks3, payouts3);
  double sum = 0.0;
  for (double e : eq3) {
    CHECK(e == e);  // NaN check
    sum += e;
  }
  CHECK_NEAR(sum, 1.0, 1e-12);
  CHECK(eq3[0] > eq3[3] && eq3[3] > eq3[2]);
  CHECK_NEAR(eq3[1], 0.05, 1e-12);  // zero stack gets the last payout
}

// ------------------------------------------------------------- showdown ---

TEST(ShowdownWinnerTakesPot) {
  std::array<Card, kMaxSeats * 2> hole{};
  hole[0] = 0 * 13 + 12;
  hole[1] = 1 * 13 + 12;
  hole[2] = 0 * 13 + 11;
  hole[3] = 1 * 13 + 11;
  Card board[5] = {2 * 13 + 5, 3 * 13 + 6, 0 * 13 + 2, 1 * 13 + 3, 2 * 13 + 9};
  std::vector<int64_t> contrib{1000, 1000, 0, 0, 0, 0, 0, 0};
  auto w = distributeShowdown(2, hole, board, contrib.data(), {0, 1});
  CHECK(w[0] == 2000);
  CHECK(w[1] == 0);
}

TEST(ShowdownUncalledBetReturn) {
  std::array<Card, kMaxSeats * 2> hole{};
  hole[0] = 0 * 13 + 12;
  hole[1] = 1 * 13 + 12;
  hole[2] = 0 * 13 + 11;
  hole[3] = 1 * 13 + 11;
  Card board[5] = {2 * 13 + 5, 3 * 13 + 6, 0 * 13 + 2, 1 * 13 + 3, 2 * 13 + 9};
  std::vector<int64_t> contrib{500, 900, 0, 0, 0, 0, 0, 0};
  auto w = distributeShowdown(2, hole, board, contrib.data(), {0, 1});
  CHECK(w[0] == 1000);
  CHECK(w[1] == 400);
}

TEST(ShowdownSidePotThreeWay) {
  std::array<Card, kMaxSeats * 2> hole{};
  hole[0] = 0 * 13 + 12;
  hole[1] = 1 * 13 + 12;
  hole[2] = 0 * 13 + 11;
  hole[3] = 1 * 13 + 11;
  hole[4] = 0 * 13 + 10;
  hole[5] = 1 * 13 + 10;
  Card board[5] = {2 * 13 + 5, 3 * 13 + 6, 0 * 13 + 2, 1 * 13 + 3, 2 * 13 + 9};
  std::vector<int64_t> contrib{100, 300, 300, 0, 0, 0, 0, 0};
  auto w = distributeShowdown(3, hole, board, contrib.data(), {0, 1, 2});
  CHECK(w[0] == 300);
  CHECK(w[1] == 400);
  CHECK(w[2] == 0);
}

TEST(ShowdownSplitPot) {
  std::array<Card, kMaxSeats * 2> hole{};
  hole[0] = 0 * 13 + 12;  // both players hold A on a board where they tie
  hole[1] = 1 * 13 + 12;
  hole[2] = 0 * 13 + 12;
  hole[3] = 1 * 13 + 12;
  Card board[5] = {2 * 13 + 5, 3 * 13 + 6, 0 * 13 + 2, 1 * 13 + 3, 2 * 13 + 9};
  std::vector<int64_t> contrib{1000, 1000, 0, 0, 0, 0, 0, 0};
  auto w = distributeShowdown(2, hole, board, contrib.data(), {0, 1});
  CHECK(w[0] == 1000);
  CHECK(w[1] == 1000);
}

// ----------------------------------------------------------------- Kuhn ---

namespace {

using KuhnTable = std::map<uint64_t, std::vector<double>>;

// All 6 ordered deals, deterministically (never sample them: a fixed-seed
// RNG can miss whole deals and skew every value computed from them).
std::vector<KuhnGame::State> kuhnAllDeals() {
  const int pairs[6][2] = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}};
  std::vector<KuhnGame::State> out;
  for (auto& pr : pairs) {
    KuhnGame::State s;
    s.card0 = pr[0];
    s.card1 = pr[1];
    out.push_back(s);
  }
  return out;
}

double kuhnValue(const KuhnTable& s0, const KuhnTable& s1) {
  KuhnGame game;
  double total = 0.0;
  for (const auto& root : kuhnAllDeals()) {
    std::function<double(const KuhnGame::State&)> ev =
        [&](const KuhnGame::State& st) -> double {
      if (game.isTerminal(st)) return game.payoff(st, 0);
      int p = game.currentPlayer(st);
      uint64_t key = game.infosetKey(st, p);
      const auto& table = p == 0 ? s0 : s1;
      auto it = table.find(key);
      auto acts = game.legalActions(st);
      std::vector<double> sigma;
      if (it == table.end() || it->second.size() != acts.size()) {
        sigma.assign(acts.size(), 1.0 / acts.size());
      } else {
        sigma = it->second;
      }
      double v = 0.0;
      for (size_t a = 0; a < acts.size(); ++a) v += sigma[a] * ev(game.apply(st, acts[a]));
      return v;
    };
    total += ev(root);
  }
  return total / 6.0;
}

// Exact exploitability: enumerate all pure strategies of one player
// (2 actions at up to 6 infosets -> 2^6) against the other's average.
double kuhnBestResponse(const KuhnTable& fixed, int responder) {
  KuhnGame game;
  std::vector<uint64_t> keys;
  for (const auto& root : kuhnAllDeals()) {
    std::function<void(const KuhnGame::State&)> walk = [&](const KuhnGame::State& st) {
      if (game.isTerminal(st)) return;
      int p = game.currentPlayer(st);
      if (p == responder) {
        uint64_t k = game.infosetKey(st, p);
        if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
        for (auto a : game.legalActions(st)) walk(game.apply(st, a));
      } else {
        uint64_t key = game.infosetKey(st, p);
        auto it = fixed.find(key);
        auto acts = game.legalActions(st);
        std::vector<double> sigma;
        if (it == fixed.end() || it->second.size() != acts.size()) {
          sigma.assign(acts.size(), 1.0 / acts.size());
        } else {
          sigma = it->second;
        }
        for (size_t a = 0; a < acts.size(); ++a) {
          if (sigma[a] > 0) walk(game.apply(st, acts[a]));
        }
      }
    };
    walk(root);
  }
  CHECK(keys.size() <= 8);
  int n = static_cast<int>(keys.size());
  std::vector<int> assign(n, 0);
  double best = -1e18;
  while (true) {
    KuhnTable pure;
    for (int i = 0; i < n; ++i) {
      // All Kuhn infosets have exactly 2 legal actions.
      pure[keys[i]] = {1.0 - assign[i], 0.0 + assign[i]};
    }
    // kuhnValue always returns the value to player 0; for the second
    // responder maximize their own (negated) value.
    double v = responder == 0 ? kuhnValue(pure, fixed) : -kuhnValue(fixed, pure);
    if (v > best) best = v;
    int i = 0;
    while (i < n) {
      if (++assign[i] < 2) break;
      assign[i] = 0;
      ++i;
    }
    if (i == n) break;
  }
  return best;
}

double kuhnExploitability(const KuhnTable& s0, const KuhnTable& s1) {
  double vAvg = kuhnValue(s0, s1);       // value to player 0
  double br0 = kuhnBestResponse(s1, 0);  // player 0 deviating
  double br1 = kuhnBestResponse(s0, 1);   // player 1 deviating (their payoff)
  return (br0 - vAvg) + (br1 - (-vAvg));
}

}  // namespace

TEST(KuhnHarnessSanity) {
  // Validate the value/exploitability harness on hand-computable strategy
  // pairs. Action codes: 0 = first listed action (check or fold),
  // 1 = second listed action (bet or call).
  KuhnGame game;
  auto uniform = [](size_t n) { return std::vector<double>(n, 1.0 / n); };

  // Build "always action 0" and "always action 1" tables for a player.
  auto fixedTable = [&](int which) {
    KuhnTable t;
    for (const auto& root : kuhnAllDeals()) {
      std::function<void(const KuhnGame::State&)> walk = [&](const KuhnGame::State& st) {
        if (game.isTerminal(st)) return;
        int p = game.currentPlayer(st);
        uint64_t k = game.infosetKey(st, p);
        auto acts = game.legalActions(st);
        std::vector<double> sigma(acts.size(), 0.0);
        sigma[which] = 1.0;
        t[k] = sigma;
        for (auto a : game.legalActions(st)) walk(game.apply(st, a));
      };
      walk(root);
    }
    return t;
  };

  KuhnTable always0 = fixedTable(0);  // check / fold
  KuhnTable always1 = fixedTable(1);  // bet / call

  // (check, check) and (bet, call) symmetric showdowns: value 0.
  CHECK_NEAR(kuhnValue(always0, always0), 0.0, 1e-12);
  CHECK_NEAR(kuhnValue(always1, always1), 0.0, 1e-12);
  // P0 always bets, P1 always folds: P0 nets +1 on every deal.
  CHECK_NEAR(kuhnValue(always1, always0), 1.0, 1e-12);
  // P1's best response to "always bet": call with K and Q (J folds).
  // Per deal: (J,Q)+2 (J,K)+2 (Q,J)-1 (Q,K)+2 (K,J)-1 (K,Q)-2 -> 1/3.
  CHECK_NEAR(kuhnBestResponse(always1, 1), 1.0 / 3.0, 1e-12);
  // P0's best response to "always fold" is always bet: value 1.
  CHECK_NEAR(kuhnBestResponse(always0, 0), 1.0, 1e-12);
  // Exploitability of (always bet, always fold): P0 cannot improve (0),
  // P1 gains 1/3 + 1 = 4/3 total.
  CHECK_NEAR((kuhnBestResponse(always0, 0) - kuhnValue(always1, always0)) +
                 (kuhnBestResponse(always1, 1) + kuhnValue(always1, always0)),
             4.0 / 3.0, 1e-12);
}

TEST(KuhnMCCFRConverges) {
  KuhnGame game;
  MCCFRConfig cfg;
  cfg.iterations = 100000;
  cfg.threads = 1;
  MCCFR<KuhnGame> solver(game, cfg);
  solver.train();
  auto avg = solver.averageStrategies();

  // Build per-player strategy tables by walking the reachable tree.
  KuhnTable s0, s1;
  for (const auto& root : kuhnAllDeals()) {
    std::function<void(const KuhnGame::State&)> walk = [&](const KuhnGame::State& st) {
      if (game.isTerminal(st)) return;
      int p = game.currentPlayer(st);
      uint64_t k = game.infosetKey(st, p);
      auto it = avg.find(k);
      if (it == avg.end()) return;
      if (p == 0 && s0.find(k) == s0.end()) s0[k] = it->second.avg;
      if (p == 1 && s1.find(k) == s1.end()) s1[k] = it->second.avg;
      for (auto a : game.legalActions(st)) walk(game.apply(st, a));
    };
    walk(root);
  }
  CHECK(s0.size() == 6);
  CHECK(s1.size() == 6);
  double expl = kuhnExploitability(s0, s1);
  std::printf("  kuhn exploitability after %llu iters: %.4f\n", cfg.iterations, expl);
  CHECK(expl < 0.05);
}

// ---------------------------------------------------------------- poker ---

namespace {

PokerGame::Config testConfig(int numPlayers, int64_t stack, int64_t ante,
                             ContinuationModel* cont) {
  PokerGame::Config cfg;
  cfg.numPlayers = numPlayers;
  cfg.startStack = stack;
  cfg.sb = 500;
  cfg.bb = 1000;
  cfg.ante = ante;
  cfg.payouts = {0.5, 0.3, 0.2};
  cfg.numBuckets = 169;
  cfg.continuation = cont;
  return cfg;
}

// Deal cards deterministically and enter the betting stage.
PokerGame::State dealtState(const PokerGame& game, RNG& rng) {
  return game.sampleChance(game.rootState(), rng);
}

}  // namespace

TEST(PokerInitialState) {
  ShowdownContinuation cont;
  PokerGame game(testConfig(8, 100000, 125, &cont));
  RNG rng(5);
  PokerGame::State s = dealtState(game, rng);
  CHECK(!game.isTerminal(s));
  CHECK(!game.isChance(s));
  CHECK(game.currentPlayer(s) == 3);  // UTG first to act
  CHECK(s.pendingActors == 8);
  CHECK(s.currentBet == 1000);
  // Antes: 8 * 125 + blinds 500 + 1000 = 2500 total in pot.
  int64_t pot = 0;
  for (int i = 0; i < 8; ++i) pot += s.contributed[i];
  CHECK(pot == 8 * 125 + 500 + 1000);
  // UTG legal actions: fold, call, raises.
  auto acts = game.legalActions(s);
  CHECK(acts.size() >= 3);
  CHECK(acts[0].type == PokerGame::Action::FOLD);
}

TEST(PokerFoldAroundBigBlindWins) {
  ShowdownContinuation cont;
  PokerGame game(testConfig(8, 100000, 125, &cont));
  RNG rng(5);
  PokerGame::State s = dealtState(game, rng);
  // Fold seats 3..7 (UTG..CO), 0 (BTN), 1 (SB): BB wins.
  for (int f = 0; f < 7; ++f) {
    PokerGame::State n = game.apply(s, {PokerGame::Action::FOLD, 0});
    s = n;
  }
  CHECK(game.isTerminal(s));
  CHECK(s.terminalKind == 1);
  // BB final stack = behind (stack - ante - bb) + pot (all contributions).
  int64_t pot = 0;
  for (int i = 0; i < 8; ++i) pot += s.contributed[i];
  CHECK(pot == 8 * 125 + 500 + 1000);
  CHECK(s.finalStack[2] == 100000 - 125 - 1000 + pot);
  CHECK(s.finalStack[0] == 100000 - 125);        // BTN: ante only
  CHECK(s.finalStack[1] == 100000 - 125 - 500);  // SB: ante + sb
  CHECK(s.finalStack[3] == 100000 - 125);        // UTG: ante only
}

TEST(PokerHeadsUpAllIn) {
  ShowdownContinuation cont;
  PokerGame game(testConfig(2, 5000, 0, &cont));
  RNG rng(9);
  PokerGame::State s = dealtState(game, rng);
  // SB (seat 0) shoves; BB (seat 1) calls. Both all-in -> board chance.
  auto acts = game.legalActions(s);
  const PokerGame::Action* shove = nullptr;
  for (auto& a : acts) {
    if (a.type == PokerGame::Action::RAISE && a.raiseTo == 5000) shove = &a;
  }
  CHECK(shove != nullptr);
  s = game.apply(s, *shove);
  CHECK(game.currentPlayer(s) == 1);
  acts = game.legalActions(s);
  const PokerGame::Action* call = nullptr;
  for (auto& a : acts) {
    if (a.type == PokerGame::Action::MATCH) call = &a;
  }
  s = game.apply(s, *call);
  // BB has 4000 behind; calling 4500 uses all of it -> all-in.
  CHECK(game.isChance(s));
  CHECK(!game.isTerminal(s));
  PokerGame::State t = game.sampleChance(s, rng);
  CHECK(game.isTerminal(t));
  CHECK(t.terminalKind == 2);
  int64_t pot = t.contributed[0] + t.contributed[1];
  CHECK(pot == 5000 + 5000);
  // Winner takes the pot; loser has 0.
  int64_t total = t.finalStack[0] + t.finalStack[1];
  CHECK(total == 10000);
  bool oneWinner = (t.finalStack[0] == 10000) || (t.finalStack[1] == 10000);
  CHECK(oneWinner || (t.finalStack[0] == 5000 && t.finalStack[1] == 5000));
}

TEST(PokerHeadsUpCalledPotIsContinuation) {
  ShowdownContinuation cont;
  PokerGame game(testConfig(2, 50000, 0, &cont));
  RNG rng(21);
  PokerGame::State s = dealtState(game, rng);
  // SB raises 2.5bb, BB calls: both keep chips -> FGS continuation terminal.
  auto acts = game.legalActions(s);
  const PokerGame::Action* open = nullptr;
  for (auto& a : acts) {
    if (a.type == PokerGame::Action::RAISE && a.raiseTo == 2500) open = &a;
  }
  CHECK(open != nullptr);
  s = game.apply(s, *open);
  acts = game.legalActions(s);
  const PokerGame::Action* call = nullptr;
  for (auto& a : acts) {
    if (a.type == PokerGame::Action::MATCH) call = &a;
  }
  s = game.apply(s, *call);
  CHECK(game.isTerminal(s));
  CHECK(s.terminalKind == 3);
  // Outcome distribution sanity: probabilities sum to 1, and expected
  // chips conserved (sum of winnings == pot for each outcome).
  double psum = 0.0;
  int64_t pot = s.contributed[0] + s.contributed[1];
  for (auto& o : s.contOutcomes) {
    psum += o.probability;
    int64_t sum = 0;
    for (auto w : o.winnings) sum += w;
    CHECK(sum == pot);
  }
  CHECK_NEAR(psum, 1.0, 1e-9);
  // Payoffs sum to zero across players (ICM is constant-sum).
  double p0 = game.payoff(s, 0);
  double p1 = game.payoff(s, 1);
  CHECK_NEAR(p0 + p1, 0.0, 1e-9);
}

TEST(PokerUncalledRaiseReturned) {
  ShowdownContinuation cont;
  PokerGame game(testConfig(3, 50000, 0, &cont));
  RNG rng(33);
  PokerGame::State s = dealtState(game, rng);
  // 3-max: BTN (seat 0) opens 3bb, SB folds, BB folds. Uncalled 2bb
  // returns to BTN.
  auto acts = game.legalActions(s);
  const PokerGame::Action* open = nullptr;
  for (auto& a : acts) {
    if (a.type == PokerGame::Action::RAISE && a.raiseTo == 3000) open = &a;
  }
  CHECK(open != nullptr);
  s = game.apply(s, *open);
  CHECK(game.currentPlayer(s) == 1);  // SB next
  s = game.apply(s, {PokerGame::Action::FOLD, 0});
  CHECK(game.currentPlayer(s) == 2);  // BB next
  s = game.apply(s, {PokerGame::Action::FOLD, 0});
  CHECK(game.isTerminal(s));
  CHECK(s.terminalKind == 1);
  // BTN final = 50000 - 3000 + pot(500 + 1000 + 3000) = 50000 + 1500.
  CHECK(s.finalStack[0] == 50000 + 500 + 1000);
  CHECK(s.finalStack[1] == 50000 - 500);
  CHECK(s.finalStack[2] == 50000 - 1000);
}

TEST(PokerMinRaiseEnforced) {
  ShowdownContinuation cont;
  PokerGame game(testConfig(2, 50000, 0, &cont));
  RNG rng(2);
  PokerGame::State s = dealtState(game, rng);
  auto acts = game.legalActions(s);
  // All raise targets are at least currentBet + lastRaiseIncrement (2bb).
  for (auto& a : acts) {
    if (a.type == PokerGame::Action::RAISE) {
      bool ok = a.raiseTo >= 2000 || a.raiseTo == 50000;
      CHECK(ok);
    }
  }
}

TEST(PokerBustedPlayerExcluded) {
  ShowdownContinuation cont;
  PokerGame::Config cfg = testConfig(3, 50000, 0, &cont);
  cfg.stacks = {0, 50000, 50000};  // BTN is busted
  PokerGame game(cfg);
  RNG rng(4);
  PokerGame::State s = dealtState(game, rng);
  CHECK(s.folded[0] == 1);
  // First to act is SB (seat 1) since BTN busted.
  CHECK(game.currentPlayer(s) == 1);
}

int main() { return runAllTests(); }
