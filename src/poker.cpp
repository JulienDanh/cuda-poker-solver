#include "poker.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <map>

#include "showdown.h"

namespace pps {

namespace {
std::atomic<uint64_t> g_nextGameId{1};

int nonFolded(const PokerGame::State& s) {
  int n = 0;
  for (int i = 0; i < s.numPlayers; ++i) n += s.folded[i] ? 0 : 1;
  return n;
}
}  // namespace

PokerGame::PokerGame(Config cfg) : cfg_(std::move(cfg)) {
  gameId_ = g_nextGameId.fetch_add(1);
  if (cfg_.numPlayers < 2 || cfg_.numPlayers > kMaxPlayers) {
    throw std::runtime_error("numPlayers must be in [2, 8]");
  }
  if (cfg_.numBuckets != kNumHands169 && cfg_.numBuckets != kNumCoarseBuckets) {
    throw std::runtime_error("numBuckets must be 169 or 15");
  }
  if (cfg_.continuation == nullptr) {
    throw std::runtime_error("continuation model required (FGS)");
  }
  if (!cfg_.stacks.empty() &&
      static_cast<int>(cfg_.stacks.size()) != cfg_.numPlayers) {
    throw std::runtime_error("stacks vector must have numPlayers entries");
  }
  std::vector<int64_t> stacks(cfg_.numPlayers, cfg_.startStack);
  if (!cfg_.stacks.empty()) stacks = cfg_.stacks;
  double paySum = 0.0;
  for (double p : cfg_.payouts) paySum += p;
  if (paySum <= 0.999 || paySum >= 1.001) {
    throw std::runtime_error("payouts must sum to 1");
  }
  icm0_ = icmEquities(stacks, cfg_.payouts);
}

std::string PokerGame::positionName(int seat) const {
  // Fixed seat labels (action-order naming): seat 0 BTN, 1 SB, 2 BB, then
  // preflop action order UTG, UTG+1, MP, HJ, CO. For tables smaller than
  // 8-handed the same labels apply; seat 3 is always first to act preflop.
  static const char* kNames[8] = {"BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"};
  if (cfg_.numPlayers == 2) return seat == 0 ? "BTN/SB" : "BB";
  return kNames[seat];
}

std::array<Card, 2> PokerGame::representativeCards(int handIdx169) {
  auto decode = [](int idx, int& hi, int& lo, bool& suited) {
    if (idx < 13) {
      hi = idx;
      lo = idx;
      suited = true;  // pairs are rank-equal; suits picked below
      return;
    }
    bool s = idx < 91;
    int t = s ? idx - 13 : idx - 91;
    hi = 1;
    while ((hi + 1) * hi / 2 <= t) ++hi;
    lo = t - hi * (hi - 1) / 2;
    suited = s;
  };
  int hi, lo;
  bool suited;
  decode(handIdx169, hi, lo, suited);
  if (hi == lo) return {static_cast<Card>(0 * 13 + hi), static_cast<Card>(1 * 13 + hi)};
  if (suited) return {static_cast<Card>(0 * 13 + hi), static_cast<Card>(0 * 13 + lo)};
  return {static_cast<Card>(0 * 13 + hi), static_cast<Card>(1 * 13 + lo)};
}

PokerGame::State PokerGame::rootState() const {
  State s;
  s.numPlayers = cfg_.numPlayers;
  std::vector<int64_t> stacks(cfg_.numPlayers, cfg_.startStack);
  if (!cfg_.stacks.empty()) stacks = cfg_.stacks;
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    s.stack[i] = stacks[i];
    if (s.stack[i] <= 0) {
      s.folded[i] = 1;  // busted player: cannot participate
    }
  }
  // Post antes.
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    if (s.folded[i]) continue;
    int64_t post = std::min(cfg_.ante, s.stack[i]);
    s.stack[i] -= post;
    s.contributed[i] += post;
  }
  // Post blinds.
  auto post = [&s](int seat, int64_t amt) {
    if (s.folded[seat]) return;
    int64_t p = std::min(amt, s.stack[seat]);
    s.stack[seat] -= p;
    s.contributed[seat] += p;
  };
  post(smallBlindSeat(), cfg_.sb);
  post(bigBlindSeat(), cfg_.bb);
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    s.allin[i] = (!s.folded[i] && s.stack[i] == 0) ? 1 : 0;
  }
  s.currentBet = cfg_.bb;
  s.lastRaiseIncrement = cfg_.bb;
  s.current = (bigBlindSeat() + 1) % cfg_.numPlayers;
  s.pendingActors = 0;
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    if (!s.folded[i] && !s.allin[i]) ++s.pendingActors;
  }
  s.stage = 0;  // hole-card chance first
  return s;
}

bool PokerGame::isChance(const State& s) const {
  return s.stage == 0 || s.stage == 2;
}

PokerGame::State PokerGame::sampleChance(const State& s, RNG& rng) const {
  State n = s;
  if (n.stage == 0) {
    // Deal 2n distinct cards via partial Fisher-Yates over the full deck.
    int need = n.numPlayers * 2;
    std::array<Card, 52> deck;
    for (int c = 0; c < 52; ++c) deck[c] = static_cast<Card>(c);
    for (int i = 0; i < need; ++i) {
      int j = i + static_cast<int>(rng.nextBelow(52 - i));
      std::swap(deck[i], deck[j]);
    }
    for (int i = 0; i < need; ++i) n.hole[i] = deck[i];
    n.cardsDealt = need;
    n.stage = 1;
    // Advance the pointer to the first seat that can actually act.
    for (int step = 0; step < n.numPlayers; ++step) {
      int seat = (n.current + step) % n.numPlayers;
      if (!n.folded[seat] && !n.allin[seat]) {
        n.current = seat;
        break;
      }
    }
    if (n.pendingActors == 0 || nonFolded(n) == 1) streetEnd(n);
    return n;
  }
  // stage == 2: deal a 5-card board from the remaining deck.
  int need = n.numPlayers * 2;
  std::vector<Card> avail;
  avail.reserve(52 - need);
  bool dealt[52] = {false};
  for (int i = 0; i < need; ++i) dealt[n.hole[i]] = true;
  for (int c = 0; c < 52; ++c) {
    if (!dealt[c]) avail.push_back(static_cast<Card>(c));
  }
  for (int i = 0; i < 5; ++i) {
    int j = i + static_cast<int>(rng.nextBelow(static_cast<uint32_t>(avail.size() - i)));
    std::swap(avail[i], avail[j]);
  }
  for (int i = 0; i < 5; ++i) n.board[i] = avail[i];
  n.boardDealt = 5;
  resolveShowdown(n);
  return n;
}

bool PokerGame::anyActiveChips(const State& s) const {
  int withChips = 0;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (!s.folded[i] && !s.allin[i]) ++withChips;
  }
  return withChips >= 2;
}

void PokerGame::streetEnd(State& s) const {
  int inHand = 0;
  for (int i = 0; i < s.numPlayers; ++i) inHand += s.folded[i] ? 0 : 1;
  if (inHand <= 1) {
    resolveFoldWin(s);
    return;
  }
  if (!anyActiveChips(s)) {
    s.stage = 2;  // board chance, then showdown
    return;
  }
  resolveContinuation(s);
}

std::vector<PokerGame::Action> PokerGame::legalActions(const State& s) const {
  assert(s.stage == 1 && s.terminalKind == 0);
  std::vector<Action> acts;
  int me = s.current;
  int64_t behind = s.stack[me];
  int64_t myContrib = s.contributed[me];
  int64_t facing = s.currentBet - myContrib;

  if (facing > 0) acts.push_back({Action::FOLD, 0});
  acts.push_back({Action::MATCH, 0});

  if (s.raisesSoFar >= cfg_.bets.maxBets) return acts;

  int64_t myTotal = behind + myContrib;
  if (myTotal <= s.currentBet) return acts;  // cannot raise
  // Cap meaningful raises at the largest opponent total.
  int64_t maxOpp = 0;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (i == me || s.folded[i]) continue;
    maxOpp = std::max(maxOpp, s.stack[i] + s.contributed[i]);
  }
  int64_t capTo = std::min(myTotal, maxOpp);
  if (capTo <= s.currentBet) return acts;

  int64_t minRaiseTo = s.currentBet + s.lastRaiseIncrement;
  int64_t effTotal = myTotal;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (i == me || s.folded[i]) continue;
    effTotal = std::min(effTotal, s.stack[i] + s.contributed[i]);
  }
  int64_t shoveLine = static_cast<int64_t>(
      cfg_.bets.shoveThreshold * static_cast<double>(effTotal));

  std::vector<int64_t> candidates;
  if (s.raisesSoFar == 0) {
    for (double o : cfg_.bets.openSizes) {
      candidates.push_back(static_cast<int64_t>(o * cfg_.bb));
    }
  } else {
    for (double m : cfg_.bets.raiseMultipliers) {
      candidates.push_back(static_cast<int64_t>(m * s.currentBet));
    }
  }
  std::vector<int64_t> raises;
  for (int64_t c : candidates) {
    int64_t to = std::min(c, capTo);
    if (to <= s.currentBet) continue;
    if (to != myTotal && to > shoveLine) continue;  // collapse into all-in
    if (to < minRaiseTo && to != myTotal) continue;  // below min-raise
    bool dup = false;
    for (int64_t r : raises) dup |= (r == to);
    if (!dup) raises.push_back(to);
  }
  // Always offer the "all-in effective" raise when the effective cap is a
  // real shove and hasn't been added yet.
  int64_t shoveTo = std::min(myTotal, capTo);
  if (shoveTo > shoveLine) {
    bool dup = false;
    for (int64_t r : raises) dup |= (r == shoveTo);
    if (!dup && (shoveTo >= minRaiseTo || shoveTo == myTotal)) raises.push_back(shoveTo);
  }
  std::sort(raises.begin(), raises.end());
  for (int64_t r : raises) acts.push_back({Action::RAISE, r});
  return acts;
}

PokerGame::State PokerGame::apply(const State& s, const Action& a) const {
  State n = s;
  int me = n.current;
  if (a.type == Action::FOLD) {
    n.folded[me] = 1;
    n.hist[n.nhist++] = 0;
    --n.pendingActors;
  } else if (a.type == Action::MATCH) {
    int64_t facing = n.currentBet - n.contributed[me];
    int64_t chips = std::min(facing, n.stack[me]);
    n.stack[me] -= chips;
    n.contributed[me] += chips;
    if (n.stack[me] == 0) n.allin[me] = 1;
    n.hist[n.nhist++] = 1;
    --n.pendingActors;
  } else {  // RAISE
    int64_t to = std::min(a.raiseTo, n.stack[me] + n.contributed[me]);
    int64_t chips = to - n.contributed[me];
    assert(chips >= 0);
    n.stack[me] -= chips;
    n.contributed[me] += chips;
    if (n.stack[me] == 0) n.allin[me] = 1;
    int64_t prevBet = n.currentBet;
    n.currentBet = std::max(n.currentBet, n.contributed[me]);
    n.lastRaiseIncrement = std::max(cfg_.bb, n.currentBet - prevBet);
    ++n.raisesSoFar;
    n.hist[n.nhist++] = 0x80000000u | static_cast<uint32_t>(to);
    // Everyone active except the raiser must respond again.
    n.pendingActors = 0;
    for (int i = 0; i < n.numPlayers; ++i) {
      if (!n.folded[i] && !n.allin[i] && i != me) ++n.pendingActors;
    }
  }
  int nonFolded = 0;
  for (int i = 0; i < n.numPlayers; ++i) nonFolded += n.folded[i] ? 0 : 1;
  if (n.pendingActors <= 0 || nonFolded == 1) {
    streetEnd(n);
    return n;
  }
  // Advance to the next seat that can act.
  for (int step = 1; step <= n.numPlayers; ++step) {
    int next = (me + step) % n.numPlayers;
    if (!n.folded[next] && !n.allin[next]) {
      n.current = next;
      break;
    }
  }
  return n;
}

void PokerGame::resolveFoldWin(State& s) const {
  int winner = -1;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (!s.folded[i]) {
      assert(winner == -1);
      winner = i;
    }
  }
  assert(winner >= 0);
  // Winner's final stack = chips behind + the whole pot (their own
  // contribution included, uncalled excess included).
  int64_t pot = 0;
  for (int i = 0; i < s.numPlayers; ++i) {
    pot += s.contributed[i];
    s.finalStack[i] = s.stack[i];
  }
  s.finalStack[winner] += pot;
  s.terminalKind = 1;
  s.stage = 3;
}

void PokerGame::resolveShowdown(State& s) const {
  std::vector<int> inHand;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (!s.folded[i]) inHand.push_back(i);
  }
  auto w = distributeShowdown(s.numPlayers, s.hole, s.board.data(),
                              s.contributed.data(), inHand);
  for (int i = 0; i < s.numPlayers; ++i) s.finalStack[i] = s.stack[i] + w[i];
  s.terminalKind = 2;
  s.stage = 3;
}

void PokerGame::resolveContinuation(State& s) const {
  ContinuationContext ctx;
  ctx.numPlayers = s.numPlayers;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (!s.folded[i]) ctx.inHand.push_back(i);
  }
  ctx.holeCards.resize(s.numPlayers);
  for (int i = 0; i < s.numPlayers; ++i) {
    ctx.holeCards[i] = {s.hole[2 * i], s.hole[2 * i + 1]};
  }
  ctx.contributed.assign(s.contributed.begin(), s.contributed.begin() + s.numPlayers);
  ctx.samples = 16;
  uint64_t seed = fnv1a(0xCBF29CE484222325ULL, s.hist.data(), s.nhist * 4);
  s.contOutcomes = cfg_.continuation->continuationEV(ctx, seed);
  s.terminalKind = 3;
  s.stage = 3;
}

double PokerGame::icmDelta(
    const std::vector<std::array<int64_t, kMaxPlayers>>& finalStacks,
    const std::vector<double>& probs, int player) const {
  // Thread-local ICM cache keyed by (gameId, stack vector).
  struct CacheKey {
    uint64_t game;
    std::array<int64_t, kMaxPlayers> stacks;
    bool operator<(const CacheKey& o) const {
      if (game != o.game) return game < o.game;
      return stacks < o.stacks;
    }
  };
  thread_local std::map<CacheKey, std::vector<double>> cache;
  double v = 0.0;
  for (size_t k = 0; k < finalStacks.size(); ++k) {
    CacheKey key{gameId_, finalStacks[k]};
    auto it = cache.find(key);
    if (it == cache.end()) {
      std::vector<int64_t> stacks(finalStacks[k].begin(),
                                  finalStacks[k].begin() + cfg_.numPlayers);
      it = cache.emplace(key, icmEquities(stacks, cfg_.payouts)).first;
    }
    v += probs[k] * (it->second[player] - icm0_[player]);
  }
  return v;
}

double PokerGame::payoff(const State& s, int player) const {
  if (s.terminalKind == 1 || s.terminalKind == 2) {
    std::vector<std::array<int64_t, kMaxPlayers>> fs(1, s.finalStack);
    return icmDelta(fs, {1.0}, player);
  }
  assert(s.terminalKind == 3);
  std::vector<std::array<int64_t, kMaxPlayers>> fs;
  std::vector<double> probs;
  fs.reserve(s.contOutcomes.size());
  probs.reserve(s.contOutcomes.size());
  for (const auto& o : s.contOutcomes) {
    std::array<int64_t, kMaxPlayers> st = s.stack;
    for (int i = 0; i < s.numPlayers; ++i) st[i] += o.winnings[i];
    fs.push_back(st);
    probs.push_back(o.probability);
  }
  return icmDelta(fs, probs, player);
}

uint64_t PokerGame::infosetKey(const State& s, int player) const {
  int h = handIndex169(s.hole[2 * player], s.hole[2 * player + 1]);
  int bucket = bucketOf(h, cfg_.numBuckets);
  uint64_t key = 0xCBF29CE484222325ULL;
  key = hashCombine(key, static_cast<uint64_t>(player));
  key = hashCombine(key, static_cast<uint64_t>(bucket));
  key = fnv1a(key, s.hist.data(), s.nhist * sizeof(uint32_t));
  return key;
}

}  // namespace pps
