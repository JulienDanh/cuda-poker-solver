// Bet-tree construction mirroring b-inary/postflop-solver's action_tree.rs
// (see postflop_cfr.h). Shared by the GPU postflop engine.
#include "postflop_cfr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pps {
namespace pf {

namespace {

int64_t roundTo(double x) { return static_cast<int64_t>(std::llround(x)); }

}  // namespace

std::vector<TreeAction> betActions(int64_t potBase, int64_t stack,
                                   const BetConfig& cfg, int64_t sc0,
                                   int64_t sc1, int actor, bool afterAllin) {
  const int64_t S = stack;
  int64_t mySc = actor == 0 ? sc0 : sc1;
  int64_t oppSc = actor == 0 ? sc1 : sc0;
  int64_t oppRemaining = S - oppSc;
  int64_t prevAmount = std::max(sc0, sc1);
  int64_t toCall = mySc < oppSc ? oppSc - mySc : 0;
  // Sizing pot: pot after the current player calls
  // (their formula: starting_pot + 2 * (my_sc + to_call)).
  int64_t pot = potBase + 2 * (mySc + toCall);
  int64_t maxAmount = oppRemaining + prevAmount;
  int64_t minAmount = prevAmount + toCall;
  if (minAmount < 1) minAmount = 1;
  if (minAmount > maxAmount) minAmount = maxAmount;

  std::vector<TreeAction> actions;
  auto isAboveThreshold = [&](int64_t amount) {
    int64_t diff = amount - prevAmount;
    int64_t newPot = pot + 2 * diff;
    int64_t threshold =
        roundTo(static_cast<double>(newPot) * cfg.forceAllinThreshold);
    return maxAmount <= amount + threshold;
  };

  if (toCall == 0) {
    actions.push_back({ActionKind::Check, 0});
    for (double f : cfg.betFracs) {
      actions.push_back({ActionKind::Bet, roundTo(f * static_cast<double>(pot))});
    }
    if (cfg.betAllIn ||
        maxAmount <= roundTo(cfg.addAllinThreshold * static_cast<double>(pot))) {
      actions.push_back({ActionKind::AllIn, maxAmount});
    }
  } else {
    actions.push_back({ActionKind::Fold, 0});
    actions.push_back({ActionKind::Call, 0});
    if (!afterAllin) {
      for (double m : cfg.raiseMults) {
        actions.push_back(
            {ActionKind::Raise, roundTo(m * static_cast<double>(prevAmount))});
      }
      int64_t allinThreshold =
          prevAmount + roundTo(cfg.addAllinThreshold * static_cast<double>(pot));
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
  if (cfg.mergingThreshold > 0) {
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
          (curRatio - cfg.mergingThreshold) / (1.0 + cfg.mergingThreshold);
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

}  // namespace pf
}  // namespace pps
