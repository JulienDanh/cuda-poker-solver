// CLI for the 8-max tournament preflop ICM solver.
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>

#include "eval.h"
#include "fgs.h"
#include "hand169.h"
#include "poker.h"
#include "solver.h"

using namespace pps;

namespace {

struct Args {
  int seats = 8;
  int64_t startStack = 1000000;
  std::string stacksCsv;  // optional per-player stacks
  int64_t sb = 5000;
  int64_t bb = 10000;
  int64_t ante = 1250;
  std::string payouts = "0.5,0.3,0.2";
  uint64_t iterations = 50000;
  int threads = 4;
  int buckets = 169;
  std::string outCsv = "strategy.csv";
  long dumpNodes = 50000;
  std::string openSizes = "2.2,2.5,3.0";
  std::string raiseMult = "2.5,3.0";
  int maxBets = 4;
  int contSamples = 16;
  bool postflop = true;  // preflop+postflop by default; --stub reverts
  bool postflopExact = false;
  std::string streetBets = "0.5,0.75,1.0";
  std::string streetMult = "2.5,3.0";
  int streetMaxBets = 3;
  bool chipEv = false;
  int explDeals = 0;  // >0: measure HU exploitability after training
  int explBoards = 4;
  int mcValue = 0;   // >0: Monte-Carlo rollout value of the avg strategy
  bool printUtgOpen = true;
};

Args parseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", s.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "--seats") a.seats = std::stoi(next());
    else if (s == "--stack") a.startStack = std::stoll(next());
    else if (s == "--stacks") a.stacksCsv = next();
    else if (s == "--sb") a.sb = std::stoll(next());
    else if (s == "--bb") a.bb = std::stoll(next());
    else if (s == "--ante") a.ante = std::stoll(next());
    else if (s == "--payouts") a.payouts = next();
    else if (s == "--iters") a.iterations = std::stoull(next());
    else if (s == "--threads") a.threads = std::stoi(next());
    else if (s == "--buckets") a.buckets = std::stoi(next());
    else if (s == "--out") a.outCsv = next();
    else if (s == "--dump-nodes") a.dumpNodes = std::stol(next());
    else if (s == "--open-sizes") a.openSizes = next();
    else if (s == "--raise-mult") a.raiseMult = next();
    else if (s == "--max-bets") a.maxBets = std::stoi(next());
    else if (s == "--cont-samples") a.contSamples = std::stoi(next());
    else if (s == "--postflop") a.postflop = true;
    else if (s == "--stub") a.postflop = false;
    else if (s == "--postflop-exact") a.postflopExact = true;
    else if (s == "--street-bets") a.streetBets = next();
    else if (s == "--street-mult") a.streetMult = next();
    else if (s == "--street-max-bets") a.streetMaxBets = std::stoi(next());
    else if (s == "--chip-ev") a.chipEv = true;
    else if (s == "--exploitability") a.explDeals = std::stoi(next());
    else if (s == "--mc-value") a.mcValue = std::stoi(next());
    else if (s == "--expl-boards") a.explBoards = std::stoi(next());
    else if (s == "--help") {
      std::printf(
          "usage: ppsolve [--seats N] [--stack chips] [--stacks a,b,..] [--sb c] [--bb c]\n"
          "               [--ante c] [--payouts p,p,..] [--iters N] [--threads N]\n"
          "               [--buckets 169|15] [--chip-ev] [--exploitability N]\n"
          "               [--open-sizes a,b,c] [--raise-mult a,b] [--max-bets N]\n"
          "               [--postflop [--postflop-exact] --street-bets a,b]\n"
          "               [--out strategy.csv]\n");
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown arg %s\n", s.c_str());
      std::exit(2);
    }
  }
  return a;
}

std::string actionLabel(const PokerGame& game, const PokerGame::Action& a, int64_t bb) {
  char buf[64];
  if (a.type == PokerGame::Action::FOLD) return "fold";
  if (a.type == PokerGame::Action::MATCH) return "check/call";
  std::snprintf(buf, sizeof(buf), "raise %.1fbb",
                static_cast<double>(a.raiseTo) / bb);
  return buf;
}

// Representative canonical hand index for each bucket id.
std::vector<int> bucketRepHands(int numBuckets) {
  std::vector<int> reps(numBuckets, -1);
  for (int h = 0; h < kNumHands169; ++h) {
    int b = bucketOf(h, numBuckets);
    if (reps[b] < 0) reps[b] = h;
  }
  return reps;
}

// Walk the betting tree with representative hands, dumping the average
// strategy for every (seat, bucket, history) infoset, up to a node budget:
// the 8-max public tree is combinatorially large, so the dump enumerates
// at most maxNodes decision nodes.
void dumpStrategy(const PokerGame& game, int numBuckets,
                  const std::unordered_map<uint64_t, InfosetStrategy>& avg,
                  const std::string& path, long maxNodes) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return;
  }
  std::fprintf(f, "position,history,bucket");
  for (int k = 0; k < 6; ++k) std::fprintf(f, ",a%d,a%d_freq", k, k);
  std::fprintf(f, "\n");

  RNG rng(1);
  PokerGame::State s0 = game.sampleChance(game.rootState(), rng);
  int64_t bb = game.config().bb;
  std::vector<int> reps = bucketRepHands(numBuckets);
  long nodes = 0;
  long rows = 0;

  std::function<void(const PokerGame::State&, const std::string&)> rec =
      [&](const PokerGame::State& s, const std::string& hist) {
        if (game.isTerminal(s) || game.isChance(s)) return;
        if (nodes >= maxNodes) return;
        ++nodes;
        int seat = game.currentPlayer(s);
        auto actions = game.legalActions(s);
        for (int b = 0; b < numBuckets; ++b) {
          if (reps[b] < 0) continue;
          auto rep = PokerGame::representativeCards(reps[b]);
          PokerGame::State s2 = s;
          s2.hole[2 * seat] = rep[0];
          s2.hole[2 * seat + 1] = rep[1];
          uint64_t key = game.infosetKey(s2, seat);
          auto it = avg.find(key);
          if (it == avg.end()) continue;
          ++rows;
          std::fprintf(f, "%s,%s,%s", game.positionName(seat).c_str(), hist.c_str(),
                       bucketName(b, numBuckets).c_str());
          int k = 0;
          for (; k < static_cast<int>(actions.size()) && k < 6; ++k) {
            std::fprintf(f, ",%s,%.4f", actionLabel(game, actions[k], bb).c_str(),
                         it->second.avg[k]);
          }
          for (; k < 6; ++k) std::fprintf(f, ",,");
          std::fprintf(f, "\n");
        }
        for (const auto& a : actions) {
          PokerGame::State n = game.apply(s, a);
          if (n.nhist == s.nhist) continue;
          std::string lbl = hist;
          if (!lbl.empty()) lbl += "|";
          lbl += game.positionName(seat) + ":" + actionLabel(game, a, bb);
          rec(n, lbl);
          if (nodes >= maxNodes) break;
        }
      };
  rec(s0, "");
  std::fclose(f);
  std::printf("strategy written to %s (%ld decision nodes, %ld rows, "
              "budget %ld; the full 8-max tree is larger)\n",
              path.c_str(), nodes, rows, maxNodes);
}

// UTG opening range summary at the first decision node.
void printUtgRange(const PokerGame& game, int numBuckets,
                   const std::unordered_map<uint64_t, InfosetStrategy>& avg) {
  RNG rng(1);
  PokerGame::State s = game.sampleChance(game.rootState(), rng);
  if (game.isTerminal(s) || game.isChance(s)) return;
  int seat = game.currentPlayer(s);
  if (game.positionName(seat) != "UTG") return;
  auto actions = game.legalActions(s);
  std::vector<int> reps = bucketRepHands(numBuckets);
  std::printf("\n%s first-decision range (seat %d):\n", game.positionName(seat).c_str(),
              seat);
  std::printf("%-6s", "hand");
  for (const auto& a : actions) std::printf("%-14s", actionLabel(game, a, game.config().bb).c_str());
  std::printf("\n");
  for (int b = 0; b < numBuckets; ++b) {
    if (reps[b] < 0) continue;
    auto rep = PokerGame::representativeCards(reps[b]);
    PokerGame::State s2 = s;
    s2.hole[2 * seat] = rep[0];
    s2.hole[2 * seat + 1] = rep[1];
    auto it = avg.find(game.infosetKey(s2, seat));
    if (it == avg.end()) continue;
    std::printf("%-6s", bucketName(b, numBuckets).c_str());
    for (double p : it->second.avg) std::printf("%-14.3f", p);
    std::printf("\n");
  }
}

// Monte-Carlo rollout value of the trained average strategy: sample
// deals and chance, sample actions from the average strategy, and average
// the terminal payoffs. Unbiased for any mode (handles all chance stages).
std::vector<double> mcRolloutValue(
    const PokerGame& game,
    const std::unordered_map<uint64_t, InfosetStrategy>& avg, int samples,
    uint64_t seed) {
  std::vector<double> sum(game.numPlayers(), 0.0);
  RNG rng(seed);
  for (int i = 0; i < samples; ++i) {
    PokerGame::State s = game.rootState();
    while (true) {
      if (game.isTerminal(s)) {
        for (int p = 0; p < game.numPlayers(); ++p) sum[p] += game.payoff(s, p);
        break;
      }
      if (game.isChance(s)) {
        s = game.sampleChance(s, rng);
        continue;
      }
      int p = game.currentPlayer(s);
      auto acts = game.legalActions(s);
      std::vector<double> sigma(acts.size(), 1.0 / acts.size());
      auto it = avg.find(game.infosetKey(s, p));
      if (it != avg.end() &&
          static_cast<int>(it->second.avg.size()) == static_cast<int>(acts.size())) {
        sigma = it->second.avg;
      }
      double r = rng.nextDouble();
      double acc = 0.0;
      size_t pick = acts.size() - 1;
      for (size_t a = 0; a < acts.size(); ++a) {
        acc += sigma[a];
        if (r <= acc) {
          pick = a;
          break;
        }
      }
      s = game.apply(s, acts[pick]);
    }
  }
  for (auto& v : sum) v /= samples;
  return sum;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parseArgs(argc, argv);

  PokerGame::Config cfg;
  cfg.numPlayers = a.seats;
  cfg.startStack = a.startStack;
  if (!a.stacksCsv.empty()) {
    for (auto& tok : splitString(a.stacksCsv, ',')) cfg.stacks.push_back(std::stoll(tok));
  }
  cfg.sb = a.sb;
  cfg.bb = a.bb;
  cfg.ante = a.ante;
  cfg.payouts.clear();
  for (auto& tok : splitString(a.payouts, ',')) cfg.payouts.push_back(std::stod(tok));
  cfg.numBuckets = a.buckets;
  cfg.icm = !a.chipEv;
  cfg.bets.openSizes.clear();
  for (auto& tok : splitString(a.openSizes, ',')) cfg.bets.openSizes.push_back(std::stod(tok));
  cfg.bets.raiseMultipliers.clear();
  for (auto& tok : splitString(a.raiseMult, ',')) cfg.bets.raiseMultipliers.push_back(std::stod(tok));
  cfg.bets.maxBets = a.maxBets;
  cfg.continuationSamples = a.contSamples;
  cfg.postflop = a.postflop;
  cfg.postflopAbstraction = a.postflopExact ? 0 : 1;
  cfg.bets.streetBetSizes.clear();
  for (auto& tok : splitString(a.streetBets, ','))
    cfg.bets.streetBetSizes.push_back(std::stod(tok));
  cfg.bets.streetRaiseMultipliers.clear();
  for (auto& tok : splitString(a.streetMult, ','))
    cfg.bets.streetRaiseMultipliers.push_back(std::stod(tok));
  cfg.bets.streetMaxBets = a.streetMaxBets;

  ShowdownContinuation cont;
  cfg.continuation = &cont;

  PokerGame game(cfg);
  std::printf("solving %d-max tournament preflop: stacks=%lld sb=%lld bb=%lld ante=%lld\n",
              cfg.numPlayers, static_cast<long long>(cfg.startStack),
              static_cast<long long>(cfg.sb), static_cast<long long>(cfg.bb),
              static_cast<long long>(cfg.ante));
  std::printf("payouts:");
  for (double p : cfg.payouts) std::printf(" %.2f", p);
  std::printf("  buckets=%d iters=%llu threads=%d\n", cfg.numBuckets,
              static_cast<unsigned long long>(a.iterations), a.threads);

  MCCFRConfig mcfg;
  mcfg.iterations = a.iterations;
  mcfg.threads = a.threads;
  MCCFR<PokerGame> solver(game, mcfg);
  solver.train();

  auto avg = solver.averageStrategies();
  std::printf("infosets visited: %zu\n", avg.size());
  if (a.mcValue > 0) {
    auto v = mcRolloutValue(game, avg, a.mcValue, 0x5EEDULL);
    std::printf("mc rollout value (%d samples, bb = %lld):", a.mcValue,
                static_cast<long long>(cfg.bb));
    for (int p = 0; p < cfg.numPlayers; ++p)
      std::printf("  %s %.4f bb", game.positionName(p).c_str(), v[p] / cfg.bb);
    std::printf("\n");
  }
  if (a.explDeals > 0) {
    if (a.postflop) {
      std::fprintf(stderr,
                    "exploitability measurement does not support postflop "
                    "mode (street chances cannot be collapsed analytically)\n");
      return 1;
    }
    if (cfg.numPlayers != 2) {
      std::fprintf(stderr,
                    "exploitability is only implemented for heads-up\n");
      return 1;
    }
    auto ex = huExploitability(game, avg, a.explDeals, a.explBoards, 0xABCD);
    std::printf("exploitability (upper bound, both players): %.6f bb/hand "
                "(sem %.6f)  [v_avg0 %.6f bb]\n",
                ex.exploitability / cfg.bb, ex.sem / cfg.bb, ex.vAvg0 / cfg.bb);
  }
  if (a.printUtgOpen) printUtgRange(game, cfg.numBuckets, avg);
  dumpStrategy(game, cfg.numBuckets, avg, a.outCsv, a.dumpNodes);
  return 0;
}
