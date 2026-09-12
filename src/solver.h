#pragma once
// Game-agnostic external-sampling MCCFR solver.
//
// The Game type must provide:
//   using State = ...;          (value type, cheap to copy)
//   using Action = ...;
//   State rootState() const;              // initial (usually chance) state
//   int numPlayers() const;
//   bool isTerminal(const State&) const;
//   bool isChance(const State&) const;
//   State sampleChance(const State&, RNG&) const;
//   int  currentPlayer(const State&) const;   // valid if not terminal/chance
//   std::vector<Action> legalActions(const State&) const;
//   State apply(const State&, const Action&) const;
//   double payoff(const State&, int player) const;  // valid if terminal
//   uint64_t infosetKey(const State&, int player) const;
//
// Payoffs may be any reals (the poker game uses ICM-equity deltas, which
// are constant-sum across players). External sampling: for each traverser,
// opponent actions and chance outcomes are sampled; the traverser's own
// actions are all explored, producing regret updates. Average strategies
// accumulate from the sampled visits (visit-weighted).
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace pps {

struct MCCFRConfig {
  uint64_t iterations = 100000;
  int threads = 1;
  uint64_t seed = 0x1234567;
  // Weight later iterations more when accumulating the average strategy.
  bool linearAveraging = true;
};

struct InfosetStrategy {
  std::vector<double> avg;
  int actionCount = 0;
};

template <class Game>
class MCCFR {
 public:
  using State = typename Game::State;
  using Action = typename Game::Action;

  explicit MCCFR(const Game& game, MCCFRConfig cfg) : game_(game), cfg_(cfg) {}

  void train() {
    std::vector<std::thread> threads;
    for (int t = 0; t < cfg_.threads; ++t) threads.emplace_back([this, t]() { run(t); });
    for (auto& th : threads) th.join();
    mergeThreads();
  }

  std::unordered_map<uint64_t, InfosetStrategy> averageStrategies() const {
    std::unordered_map<uint64_t, InfosetStrategy> out;
    for (const auto& [key, nd] : merged_) {
      InfosetStrategy s;
      s.avg = nd.strategySum;
      s.actionCount = static_cast<int>(nd.regret.size());
      double sum = 0.0;
      for (double v : s.avg) sum += v;
      if (sum > 0.0) {
        for (double& v : s.avg) v /= sum;
      } else {
        for (double& v : s.avg) v = 1.0 / s.avg.size();
      }
      out[key] = s;
    }
    return out;
  }

  static std::vector<double> regretMatching(const std::vector<double>& regret) {
    std::vector<double> s(regret.size());
    double sum = 0.0;
    for (double r : regret)
      if (r > 0.0) sum += r;
    if (sum <= 0.0) {
      for (double& v : s) v = 1.0 / s.size();
    } else {
      for (size_t i = 0; i < regret.size(); ++i)
        s[i] = regret[i] > 0.0 ? regret[i] / sum : 0.0;
    }
    return s;
  }

 private:
  struct NodeData {
    std::vector<double> regret;
    std::vector<double> strategySum;
  };
  using Table = std::unordered_map<uint64_t, NodeData>;

  void run(int threadId) {
    Table table;
    RNG rng(cfg_.seed + 0x9E3779B9ULL * (threadId + 1));
    for (uint64_t it = 1; it <= cfg_.iterations; ++it) {
      State root = game_.rootState();
      for (int p = 0; p < game_.numPlayers(); ++p) {
        walk(root, p, rng, table, it);
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    threadTables_.push_back(std::move(table));
  }

  double walk(const State& s, int t, RNG& rng, Table& table, uint64_t it) {
    if (game_.isTerminal(s)) return game_.payoff(s, t);
    if (game_.isChance(s)) return walk(game_.sampleChance(s, rng), t, rng, table, it);

    int p = game_.currentPlayer(s);
    std::vector<Action> actions = game_.legalActions(s);
    uint64_t key = game_.infosetKey(s, p);
    NodeData& nd = table[key];
    if (nd.regret.empty()) {
      nd.regret.assign(actions.size(), 0.0);
      nd.strategySum.assign(actions.size(), 0.0);
    }
    std::vector<double> sigma = regretMatching(nd.regret);
    double w = cfg_.linearAveraging ? static_cast<double>(it) : 1.0;

    if (p == t) {
      std::vector<double> vals(actions.size());
      double nodeVal = 0.0;
      for (size_t a = 0; a < actions.size(); ++a) {
        vals[a] = walk(game_.apply(s, actions[a]), t, rng, table, it);
        nodeVal += sigma[a] * vals[a];
      }
      for (size_t a = 0; a < actions.size(); ++a) nd.regret[a] += vals[a] - nodeVal;
      return nodeVal;
    }
    for (size_t a = 0; a < actions.size(); ++a) nd.strategySum[a] += w * sigma[a];
    uint32_t idx = sampleAction(sigma, rng);
    return walk(game_.apply(s, actions[idx]), t, rng, table, it);
  }

  static uint32_t sampleAction(const std::vector<double>& sigma, RNG& rng) {
    double r = rng.nextDouble();
    double acc = 0.0;
    for (size_t i = 0; i < sigma.size(); ++i) {
      acc += sigma[i];
      if (r <= acc) return static_cast<uint32_t>(i);
    }
    return static_cast<uint32_t>(sigma.size() - 1);
  }

  void mergeThreads() {
    merged_.clear();
    for (auto& table : threadTables_) {
      for (auto& [key, nd] : table) {
        NodeData& m = merged_[key];
        if (m.regret.empty()) {
          m.regret = nd.regret;
          m.strategySum = nd.strategySum;
        } else {
          for (size_t i = 0; i < m.regret.size(); ++i) {
            m.regret[i] += nd.regret[i];
            m.strategySum[i] += nd.strategySum[i];
          }
        }
      }
    }
    threadTables_.clear();
  }

  const Game& game_;
  MCCFRConfig cfg_;
  std::mutex mutex_;
  std::vector<Table> threadTables_;
  std::unordered_map<uint64_t, NodeData> merged_;
};

}  // namespace pps
