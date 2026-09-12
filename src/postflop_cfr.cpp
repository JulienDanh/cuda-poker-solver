#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#endif

#include "postflop_cfr.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

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
  // sameOther maps: identical combos across the two sides.
  {
    std::map<uint16_t, int> idx1;
    for (int i = 0; i < sides_[1].n; ++i) {
      idx1[(static_cast<uint16_t>(sides_[1].cards[2 * i]) << 8) |
           sides_[1].cards[2 * i + 1]] = i;
    }
    sides_[0].sameOther.assign(sides_[0].n, -1);
    for (int i = 0; i < sides_[0].n; ++i) {
      auto it = idx1.find((static_cast<uint16_t>(sides_[0].cards[2 * i]) << 8) |
                           sides_[0].cards[2 * i + 1]);
      if (it != idx1.end()) sides_[0].sameOther[i] = it->second;
    }
    std::map<uint16_t, int> idx0;
    for (int i = 0; i < sides_[0].n; ++i) {
      idx0[(static_cast<uint16_t>(sides_[0].cards[2 * i]) << 8) |
           sides_[0].cards[2 * i + 1]] = i;
    }
    sides_[1].sameOther.assign(sides_[1].n, -1);
    for (int i = 0; i < sides_[1].n; ++i) {
      auto it = idx0.find((static_cast<uint16_t>(sides_[1].cards[2 * i]) << 8) |
                          sides_[1].cards[2 * i + 1]);
      if (it != idx0.end()) sides_[1].sameOther[i] = it->second;
    }
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
  maxDepth_ = std::max(maxDepth_, depth);

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
  maxNa_ = std::max(maxNa_, static_cast<int>(acts.size()));
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
  computeSubNodes();
}

void RiverSolver::computeSubNodes() {
  subNodes_.assign(numNodes_, 0);
  // Iterative post-order: children are always built after their parent, so
  // accumulating in reverse node order visits every node after its subtree.
  for (int i = numNodes_ - 1; i >= 0; --i) {
    int cnt = 1;
    for (int ch : nodes_[i].children) cnt += subNodes_[ch];
    subNodes_[i] = cnt;
  }
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

// ---------------------------------------------------------------------------
// Fork-join task pool for passRec (parallel CFR).
//
// Only the passRec fan-out uses this: a DECIDE node with enough subtree work
// submits its per-action child evaluations as jobs and waits for the group.
// Waiting threads help run queued jobs (work-sharing), so the task tree makes
// progress as long as any job is runnable. Jobs carry their own PassCtx
// checked out of a freelist at submit time; a spawner that cannot check a
// context out runs that child serially with its own context, so exhaustion
// degrades to less parallelism, never to a deadlock.
//
// Determinism: children write disjoint cfv rows and disjoint nodes; the
// spawner aggregates in fixed action order after the join, so the result is
// bit-identical to the serial path regardless of scheduling.
// ---------------------------------------------------------------------------

class RiverSolver::ForkPool {
 public:
  ForkPool(int workers, int maxNa, int maxN, int depths, int freeCtxs) {
    for (int i = 0; i < freeCtxs; ++i) {
      freeCtxs_.push_back(std::make_unique<PassCtx>());
      freeCtxs_.back()->init(maxNa, maxN, depths);
    }
    for (int i = 0; i < workers; ++i) {
      threads_.emplace_back([this] { workerLoop(); });
    }
  }

  ~ForkPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }

  ForkPool(const ForkPool&) = delete;
  ForkPool& operator=(const ForkPool&) = delete;

  // Returns null when all free contexts are checked out.
  PassCtx* tryCheckout() {
    std::lock_guard<std::mutex> lk(m_);
    if (freeCtxs_.empty()) return nullptr;
    PassCtx* c = freeCtxs_.back().release();
    freeCtxs_.pop_back();
    return c;
  }

  void submit(std::atomic<int>& group, PassCtx* ctx,
              std::function<void()> fn) {
    group.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(m_);
      queue_.push_back(Job{std::move(fn), &group, ctx});
      qsize_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_one();
  }

  // Helps run queued jobs until `group` reaches zero.
  void wait(std::atomic<int>& group) {
    debugSites_.fetch_add(1, std::memory_order_relaxed);
    while (group.load(std::memory_order_acquire) > 0) {
      Job j = popJob();
      if (j.fn) {
        runJob(j);
      } else {
        // Children still in flight; spin on the queue/group counters.
        spin();
      }
    }
  }

  long debugJobs() const { return debugJobs_.load(); }
  long debugSites() const { return debugSites_.load(); }

 private:
  struct Job {
    std::function<void()> fn;
    std::atomic<int>* group;
    PassCtx* ctx;  // null when the job shares no free-list context
  };

  static void spin() {
    for (volatile int i = 0; i < 100; ++i) {
    }
  }

  Job popJob() {
    // Fast path: take the lock only when the queue looks non-empty, so
    // idle spinners never contend the mutex.
    if (qsize_.load(std::memory_order_acquire) == 0) return Job{};
    std::lock_guard<std::mutex> lk(m_);
    if (queue_.empty()) return Job{};
    Job j = std::move(queue_.front());
    queue_.pop_front();
    qsize_.fetch_sub(1, std::memory_order_release);
    return j;
  }

  void runJob(Job& j) {
    debugJobs_.fetch_add(1, std::memory_order_relaxed);
    j.fn();
    if (j.ctx) checkin(j.ctx);
    j.group->fetch_sub(1, std::memory_order_acq_rel);
  }

  void checkin(PassCtx* c) {
    std::unique_ptr<PassCtx> p(c);
    std::lock_guard<std::mutex> lk(m_);
    freeCtxs_.push_back(std::move(p));
  }

  void workerLoop() {
#ifdef __APPLE__
    // Keep workers on performance cores: on Apple Silicon, threads without
    // an interactive QoS get parked on efficiency cores, which run these
    // microsecond-scale jobs 3-5x slower and straggle at every join.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    // Jobs are microsecond-scale, so spin-poll the atomic queue counter for
    // a while before sleeping on the condition variable: a cv wake costs
    // more than most jobs.
    for (;;) {
      for (;;) {
        Job j = popJob();
        if (!j.fn) break;
        runJob(j);
      }
      if (stop_.load(std::memory_order_relaxed)) return;
      // Spin while idle: the solver submits job waves every few dozen
      // microseconds; a condition-variable wake costs more than that.
      // (Workers only spin while solve() is running; the pool dies with it.)
      spin();
    }
  }

  std::vector<std::unique_ptr<PassCtx>> freeCtxs_;
  std::vector<std::thread> threads_;
  std::deque<Job> queue_;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<size_t> qsize_{0};
  std::atomic<bool> stop_{false};
  std::atomic<long> debugJobs_{0};
  std::atomic<long> debugSites_{0};
};

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

void RiverSolver::disjointMass(int tr, const double* reachOpp,
                               double* out) const {
  const Side& ts = sides_[tr];
  const Side& os = sides_[tr ^ 1];
  double minus[52] = {0.0};
  double total = 0.0;
  for (int i = 0; i < os.n; ++i) {
    double r = reachOpp[i];
    total += r;
    minus[os.cards[2 * i]] += r;
    minus[os.cards[2 * i + 1]] += r;
  }
  for (int t = 0; t < ts.n; ++t) {
    double w = total - minus[ts.cards[2 * t]] - minus[ts.cards[2 * t + 1]];
    int s = ts.sameOther[t];
    if (s >= 0) w += reachOpp[s];  // identical combo subtracted twice
    out[t] = w > 0.0 ? w : 0.0;
  }
}

// ---------------------------------------------------------------------------
// CFR pass (alternating half-step for the traverser).
// ---------------------------------------------------------------------------

namespace {
// A DECIDE node fans out its per-action child evaluations as parallel jobs
// when the subtree work (nodes x combos) is large enough to amortize the
// fork-join overhead (jobs must be tens of microseconds). Tuned against
// tools/quality/bench_river.sh.
constexpr long kSpawnWork = 8192;
// Fan-out cap: with very large action lists the per-node bookkeeping would
// dominate; keep such nodes serial.
constexpr int kMaxSpawnFan = 16;
// Auto mode skips the pool below this estimated per-iteration work.
constexpr long kAutoParallelWork = 8192;

// On Apple Silicon only the performance cores pay off for these
// microsecond-scale fork-join jobs; efficiency cores run them 3-5x slower
// and straggle at every join. Default to the P-core count when detectable.
int detectComputeCores() {
#ifdef __APPLE__
  int n = 0;
  size_t sz = sizeof(n);
  if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &sz, nullptr, 0) == 0 &&
      n > 0) {
    return n;
  }
#endif
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  return hw > 0 ? hw : 1;
}
}  // namespace

void RiverSolver::passRec(int nodeIdx, int tr, const double* reachOpp,
                          const Discount& d, double* outVal, ForkPool* pool,
                          PassCtx& ctx, int depth) {
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
    // carries the opponent's DISJOINT reach mass at this terminal.
    double v = tr == nd.foldBy ? -static_cast<double>(nd.sc[nd.foldBy])
                               : static_cast<double>(spot_.pot + nd.sc[nd.foldBy]);
    disjointMass(tr, reachOpp, outVal);
    for (int c = 0; c < ntr; ++c) outVal[c] *= v;
    return;
  }

  assert(depth + 1 < ctx.depths);
  int n = sides_[nd.player].n;
  int na = static_cast<int>(nd.actions.size());
  double* sigma = ctx.sigma(depth);
  regretMatching(nd, n, sigma);

  bool oppNode = nd.player != tr;
  double* aval = ctx.aval(depth);  // per-action child values (na rows of ntr)
  double* reachRows = ctx.reach(depth);

  // Per-action reach into each child. At the traverser's own nodes all
  // children see the same reach; at opponent nodes each action reweights.
  if (oppNode) {
    for (int a = 0; a < na; ++a) {
      for (int c = 0; c < n; ++c) {
        reachRows[static_cast<size_t>(a) * n + c] =
            reachOpp[c] * sigma[static_cast<size_t>(a) * n + c];
      }
    }
  }

  // Fan-out: submit per-action child evaluations as jobs. Each job runs on
  // its own pool context; actions without a free context run serially here
  // (same values, fewer threads).
  std::atomic<int> group(0);
  int nSpawned = 0;
  bool jobDone[kMaxSpawnFan] = {false};
  if (pool != nullptr && na >= 2 && na <= kMaxSpawnFan &&
      static_cast<long>(subNodes_[nodeIdx]) * (sides_[0].n + sides_[1].n) >=
          kSpawnWork) {
    for (int a = 0; a < na; ++a) {
      PassCtx* jc = pool->tryCheckout();
      if (jc == nullptr) break;
      const double* ra = oppNode ? reachRows + static_cast<size_t>(a) * n
                                : reachOpp;
      double* dst = aval + static_cast<size_t>(a) * ntr;
      pool->submit(group, jc, [this, nodeIdx, tr, ra, &d, dst, pool, jc, depth,
                              a] {
        passRec(nodes_[nodeIdx].children[a], tr, ra, d, dst, pool, *jc,
                depth + 1);
      });
      jobDone[a] = true;
      ++nSpawned;
    }
  }

  for (int a = 0; a < na; ++a) {
    if (jobDone[a]) continue;
    const double* ra = oppNode ? reachRows + static_cast<size_t>(a) * n
                               : reachOpp;
    double* dst = aval + static_cast<size_t>(a) * ntr;
    passRec(nd.children[a], tr, ra, d, dst, pool, ctx, depth + 1);
  }
  if (nSpawned > 0) pool->wait(group);

  // Aggregate in fixed action order: bit-identical to the serial path.
  if (!oppNode) {
    for (int c = 0; c < ntr; ++c) {
      double v = 0.0;
      for (int a = 0; a < na; ++a) {
        v += sigma[static_cast<size_t>(a) * n + c] *
             aval[static_cast<size_t>(a) * ntr + c];
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
        r = r * coef + (aval[static_cast<size_t>(a) * ntr + c] - outVal[c]);
      }
    }
  } else {
    // Opponent's node: the counterfactual value is the plain sum over
    // actions of the child cfvs; the opponent's strategy enters through
    // the updated cfreach, NOT through an extra marginal probability.
    for (int c = 0; c < ntr; ++c) {
      double v = 0.0;
      for (int a = 0; a < na; ++a) {
        v += aval[static_cast<size_t>(a) * ntr + c];
      }
      outVal[c] = v;
    }
  }
}

void RiverSolver::solve(int iterations, const std::string& algo, int threads) {
  algo_ = algo;
  int maxN = std::max(sides_[0].n, sides_[1].n);
  int depths = std::min(PassCtx::kMaxDepth, maxDepth_ + 2);
  PassCtx mainCtx;
  mainCtx.init(maxNa_, maxN, depths);

  std::unique_ptr<ForkPool> pool;
  // threads: 0 = auto (one worker per performance core minus one; serial
  // on tiny trees), 1 = serial, N >= 2 = N-1 workers. Workers raise their
  // QoS to stay on P-cores and spin rather than sleep (waves arrive every
  // few dozen microseconds), which is what makes fan-out profitable here.
  if (threads != 1 && numNodes_ > 1) {
    int workers = 0;
    if (threads == 0) {
      long work = static_cast<long>(numNodes_) * (sides_[0].n + sides_[1].n);
      if (work >= kAutoParallelWork) {
        workers = std::max(1, detectComputeCores() - 1);
      }
    } else {
      workers = threads - 1;
    }
    if (workers > 0) {
      pool = std::make_unique<ForkPool>(workers, maxNa_, maxN, depths,
                                        4 * workers);
    }
    if (std::getenv("PFSOLVER_DEBUG")) {
      std::fprintf(stderr,
                   "pool workers=%d nodes=%d n0=%d n1=%d maxNa=%d maxDepth=%d\n",
                   workers, numNodes_, sides_[0].n, sides_[1].n, maxNa_,
                   maxDepth_);
    }
  }

  std::vector<double> rootVal0(sides_[0].n), rootVal1(sides_[1].n);
  double dbgHalf = 0.0;
  int dbgN = 0;
  for (int t = 0; t < iterations; ++t) {
    Discount d = makeDiscount(t, iterations);
    if (std::getenv("PFSOLVER_DEBUG")) {
      auto t0 = std::chrono::steady_clock::now();
      passRec(0, 0, sides_[1].w.data(), d, rootVal0.data(), pool.get(),
              mainCtx, 0);
      passRec(0, 1, sides_[0].w.data(), d, rootVal1.data(), pool.get(),
              mainCtx, 0);
      dbgHalf += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
      ++dbgN;
      continue;
    }
    passRec(0, 0, sides_[1].w.data(), d, rootVal0.data(), pool.get(), mainCtx,
            0);
    passRec(0, 1, sides_[0].w.data(), d, rootVal1.data(), pool.get(), mainCtx,
            0);
  }
  if (dbgN > 0) {
    std::fprintf(stderr, "half-step avg %.1fus jobs/half=%.1f sites/half=%.1f\n",
                 dbgHalf / dbgN * 1e6,
                 pool ? pool->debugJobs() / static_cast<double>(dbgN) : 0.0,
                 pool ? pool->debugSites() / static_cast<double>(dbgN) : 0.0);
  }
  pool.reset();  // join workers before the serial final walks
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
    disjointMass(tr, reachOpp, outVal);
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
    disjointMass(br, reachOpp, v);
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

std::vector<double> RiverSolver::rootStrategy() const {
  const Node& nd = nodes_[0];
  if (nd.kind != Node::DECIDE) return {};
  int n = sides_[0].n;
  int na = static_cast<int>(nd.actions.size());
  std::vector<double> sigma(static_cast<size_t>(na) * n, 0.0);
  avgStrategy(nd, n, sigma.data());
  std::vector<double> freq(na, 0.0);
  double z = 0.0;
  for (int c = 0; c < n; ++c) z += sides_[0].w[c];
  for (int a = 0; a < na; ++a) {
    for (int c = 0; c < n; ++c) freq[a] += sides_[0].w[c] * sigma[static_cast<size_t>(a) * n + c];
    freq[a] /= z;
  }
  return freq;
}

void RiverSolver::computeStats() {
  int n0 = sides_[0].n, n1 = sides_[1].n;
  // Disjoint pair mass Z = sum over disjoint pairs of w0*w1.
  double Z = 0.0;
  {
    std::vector<double> v(n0, 0.0);
    disjointMass(0, sides_[1].w.data(), v.data());
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
