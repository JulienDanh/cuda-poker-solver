#include "poker.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <map>
#include <stdexcept>

#include "showdown.h"

namespace pps {

namespace {
std::atomic<uint64_t> g_nextGameId{1};

int nonFolded(const PokerGame::State& s) {
  int n = 0;
  for (int i = 0; i < s.numPlayers; ++i) n += s.folded[i] ? 0 : 1;
  return n;
}

int chipHolders(const PokerGame::State& s) {
  int n = 0;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (!s.folded[i] && !s.allin[i]) ++n;
  }
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
  if (cfg_.continuation == nullptr && !cfg_.postflop) {
    throw std::runtime_error("continuation model required (FGS) unless postflop=true");
  }
  if (cfg_.postflopAbstraction != 0 && cfg_.postflopAbstraction != 1) {
    throw std::runtime_error("postflopAbstraction must be 0 (exact) or 1 (buckets)");
  }
  if (cfg_.postflop) {
    if (cfg_.bets.streetBetSizes.empty() || cfg_.bets.streetRaiseMultipliers.empty()) {
      throw std::runtime_error("postflop mode requires street bet sizes");
    }
  }
  if (!cfg_.stacks.empty() &&
      static_cast<int>(cfg_.stacks.size()) != cfg_.numPlayers) {
    throw std::runtime_error("stacks vector must have numPlayers entries");
  }
  std::vector<int64_t> stacks(cfg_.numPlayers, cfg_.startStack);
  if (!cfg_.stacks.empty()) stacks = cfg_.stacks;
  startStacks_ = stacks;
  if (!cfg_.icm) {
    return;  // chip-EV mode: no payout table or baseline equity needed
  }
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
  // Post antes: antes are pot money, not street bets.
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    if (s.folded[i]) continue;
    int64_t post = std::min(cfg_.ante, s.stack[i]);
    s.stack[i] -= post;
    s.contributed[i] += post;
  }
  // Post blinds: blinds are preflop street bets.
  auto post = [&s](int seat, int64_t amt) {
    if (s.folded[seat]) return;
    int64_t p = std::min(amt, s.stack[seat]);
    s.stack[seat] -= p;
    s.contributed[seat] += p;
    s.streetContrib[seat] += p;
  };
  post(smallBlindSeat(), cfg_.sb);
  post(bigBlindSeat(), cfg_.bb);
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    s.allin[i] = (!s.folded[i] && s.stack[i] == 0) ? 1 : 0;
  }
  s.streetBet = cfg_.bb;
  s.minRaiseIncrement = cfg_.bb;
  s.current = (bigBlindSeat() + 1) % cfg_.numPlayers;
  s.pendingActors = 0;
  for (int i = 0; i < cfg_.numPlayers; ++i) {
    if (!s.folded[i] && !s.allin[i]) ++s.pendingActors;
  }
  s.street = 0;
  s.stage = 0;  // hole-card chance first
  return s;
}

bool PokerGame::isChance(const State& s) const {
  return s.stage == 0 || s.stage == 2 || s.stage == 3;
}

namespace {
// Deal k distinct cards from the cards not in `used` (52 bitmap).
void dealUnused(const bool* used, int k, Card* out, RNG& rng) {
  Card avail[52];
  int nAvail = 0;
  for (int c = 0; c < 52; ++c) {
    if (!used[c]) avail[nAvail++] = static_cast<Card>(c);
  }
  for (int i = 0; i < k; ++i) {
    int j = i + static_cast<int>(rng.nextBelow(static_cast<uint32_t>(nAvail - i)));
    std::swap(avail[i], avail[j]);
  }
  for (int i = 0; i < k; ++i) out[i] = avail[i];
}
}  // namespace

PokerGame::State PokerGame::sampleChance(const State& s, RNG& rng) const {
  State n = s;
  if (n.stage == 0) {
    // Deal 2n distinct hole cards via partial Fisher-Yates.
    int need = n.numPlayers * 2;
    bool used[52] = {false};
    Card hole[16];
    dealUnused(used, need, hole, rng);
    for (int i = 0; i < need; ++i) n.hole[i] = hole[i];
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
  if (n.stage == 2) {
    // Deal the next street's board cards.
    int k = (n.street == 0) ? 3 : 1;
    bool used[52] = {false};
    for (int i = 0; i < n.numPlayers * 2; ++i) used[n.hole[i]] = true;
    for (int i = 0; i < n.boardDealt; ++i) used[n.board[i]] = true;
    Card dealt[3];
    dealUnused(used, k, dealt, rng);
    for (int i = 0; i < k; ++i) n.board[n.boardDealt + i] = dealt[i];
    n.boardDealt += k;
    ++n.street;
    n.stage = 1;
    startStreet(n);
    return n;
  }
  // stage == 3: runout. Deal the rest of the board, then showdown.
  assert(n.stage == 3);
  int k = 5 - n.boardDealt;
  bool used[52] = {false};
  for (int i = 0; i < n.numPlayers * 2; ++i) used[n.hole[i]] = true;
  for (int i = 0; i < n.boardDealt; ++i) used[n.board[i]] = true;
  Card dealt[5];
  dealUnused(used, k, dealt, rng);
  for (int i = 0; i < k; ++i) n.board[n.boardDealt + i] = dealt[i];
  n.boardDealt = 5;
  resolveShowdown(n);
  return n;
}

void PokerGame::startStreet(State& s) const {
  s.streetBet = 0;
  s.minRaiseIncrement = cfg_.bb;
  s.raisesSoFar = 0;
  for (int i = 0; i < s.numPlayers; ++i) s.streetContrib[i] = 0;
  s.pendingActors = chipHolders(s);
  // Postflop action starts to the left of the button (the SB seat), i.e.
  // seat 1 in this layout; heads-up that is the BB.
  s.current = -1;
  for (int step = 0; step < s.numPlayers; ++step) {
    int seat = (1 + step) % s.numPlayers;
    if (!s.folded[seat] && !s.allin[seat]) {
      s.current = seat;
      break;
    }
  }
  if (s.pendingActors == 0) streetEnd(s);
}

void PokerGame::streetEnd(State& s) const {
  if (nonFolded(s) <= 1) {
    resolveFoldWin(s);
    return;
  }
  if (!cfg_.postflop) {
    // Preflop-only mode: flop-bound pots resolve through the FGS model.
    if (chipHolders(s) >= 2) {
      resolveContinuation(s);
    } else {
      s.stage = 3;  // runout, then showdown
    }
    return;
  }
  if (s.street >= 3) {
    resolveShowdown(s);  // river done: board complete
    return;
  }
  if (chipHolders(s) >= 2) {
    s.stage = 2;  // next street's chance, then betting
  } else {
    s.stage = 3;  // no more betting possible: runout
  }
}

std::vector<PokerGame::Action> PokerGame::legalActions(const State& s) const {
  assert(s.stage == 1 && s.terminalKind == 0);
  std::vector<Action> acts;
  int me = s.current;
  int64_t behind = s.stack[me];
  int64_t myStreet = s.streetContrib[me];
  int64_t myStreetCap = myStreet + behind;
  int64_t facing = s.streetBet - myStreet;

  if (facing > 0) acts.push_back({Action::FOLD, 0});
  acts.push_back({Action::MATCH, 0});

  int maxBets = (s.street == 0) ? cfg_.bets.maxBets : cfg_.bets.streetMaxBets;
  if (s.raisesSoFar >= maxBets) return acts;
  if (myStreetCap <= s.streetBet) return acts;  // cannot bet/raise

  // Largest meaningful street target: capped by the largest opponent.
  int64_t maxOpp = 0;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (i == me || s.folded[i]) continue;
    maxOpp = std::max(maxOpp, s.streetContrib[i] + s.stack[i]);
  }
  int64_t capTo = std::min(myStreetCap, maxOpp);
  if (capTo <= s.streetBet) return acts;

  int64_t minRaiseTo = s.streetBet + s.minRaiseIncrement;
  int64_t effStack = myStreetCap;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (i == me || s.folded[i]) continue;
    effStack = std::min(effStack, s.streetContrib[i] + s.stack[i]);
  }
  int64_t shoveLine = static_cast<int64_t>(
      cfg_.bets.shoveThreshold * static_cast<double>(effStack));

  std::vector<int64_t> candidates;
  if (s.street == 0) {
    if (s.raisesSoFar == 0) {
      for (double o : cfg_.bets.openSizes) {
        candidates.push_back(static_cast<int64_t>(o * cfg_.bb));
      }
    } else {
      for (double m : cfg_.bets.raiseMultipliers) {
        candidates.push_back(static_cast<int64_t>(m * s.streetBet));
      }
    }
  } else {
    int64_t pot = 0;
    for (int i = 0; i < s.numPlayers; ++i) pot += s.contributed[i];
    if (s.streetBet == 0) {
      for (double f : cfg_.bets.streetBetSizes) {
        candidates.push_back(s.streetBet + static_cast<int64_t>(f * pot));
      }
    } else {
      for (double m : cfg_.bets.streetRaiseMultipliers) {
        candidates.push_back(static_cast<int64_t>(m * s.streetBet));
      }
    }
  }

  std::vector<int64_t> raises;
  for (int64_t c : candidates) {
    int64_t to = std::min(c, capTo);
    if (to <= s.streetBet) continue;
    if (to != myStreetCap && to > shoveLine) continue;  // collapse into all-in
    if (to < minRaiseTo && to != myStreetCap) continue;  // below min-raise
    bool dup = false;
    for (int64_t r : raises) dup |= (r == to);
    if (!dup) raises.push_back(to);
  }
  int64_t shoveTo = capTo;
  if (shoveTo > shoveLine) {
    bool dup = false;
    for (int64_t r : raises) dup |= (r == shoveTo);
    if (!dup && (shoveTo >= minRaiseTo || shoveTo == myStreetCap)) raises.push_back(shoveTo);
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
    int64_t facing = n.streetBet - n.streetContrib[me];
    int64_t chips = std::min(facing, n.stack[me]);
    n.stack[me] -= chips;
    n.contributed[me] += chips;
    n.streetContrib[me] += chips;
    if (n.stack[me] == 0) n.allin[me] = 1;
    n.hist[n.nhist++] = 1;
    --n.pendingActors;
  } else {  // RAISE (street-level target)
    int64_t to = std::min(a.raiseTo, n.streetContrib[me] + n.stack[me]);
    int64_t chips = to - n.streetContrib[me];
    assert(chips >= 0);
    n.stack[me] -= chips;
    n.contributed[me] += chips;
    n.streetContrib[me] += chips;
    if (n.stack[me] == 0) n.allin[me] = 1;
    int64_t prevBet = n.streetBet;
    n.streetBet = std::max(n.streetBet, n.streetContrib[me]);
    n.minRaiseIncrement = std::max(cfg_.bb, n.streetBet - prevBet);
    ++n.raisesSoFar;
    n.hist[n.nhist++] = 0x80000000u | static_cast<uint32_t>(to);
    // Everyone active except the raiser must respond again.
    n.pendingActors = 0;
    for (int i = 0; i < n.numPlayers; ++i) {
      if (!n.folded[i] && !n.allin[i] && i != me) ++n.pendingActors;
    }
  }
  if (n.pendingActors <= 0 || nonFolded(n) == 1) {
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
  s.stage = 4;
}

void PokerGame::resolveShowdown(State& s) const {
  assert(s.boardDealt == 5);
  std::vector<int> inHand;
  for (int i = 0; i < s.numPlayers; ++i) {
    if (!s.folded[i]) inHand.push_back(i);
  }
  auto w = distributeShowdown(s.numPlayers, s.hole, s.board.data(),
                              s.contributed.data(), inHand);
  for (int i = 0; i < s.numPlayers; ++i) s.finalStack[i] = s.stack[i] + w[i];
  s.terminalKind = 2;
  s.stage = 4;
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
  ctx.samples = cfg_.continuationSamples;
  uint64_t seed = fnv1a(0xCBF29CE484222325ULL, s.hist.data(), s.nhist * 4);
  s.contOutcomes = cfg_.continuation->continuationEV(ctx, seed);
  s.terminalKind = 3;
  s.stage = 4;
}

int PokerGame::bucketOfPostflop(const State& s, int player) const {
  Card cards[7];
  int n = 2 + s.boardDealt;
  cards[0] = s.hole[2 * player];
  cards[1] = s.hole[2 * player + 1];
  for (int i = 0; i < s.boardDealt; ++i) cards[2 + i] = s.board[i];
  HandValue v = evaluateN(cards, n);
  // Rank tier of the primary rank: broadway / mid / low.
  int tier = v.tiebreak[0] >= 10 ? 2 : (v.tiebreak[0] >= 5 ? 1 : 0);
  return v.category * 3 + tier;
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
    if (!cfg_.icm) return static_cast<double>(s.finalStack[player] - startStacks_[player]);
    std::vector<std::array<int64_t, kMaxPlayers>> fs(1, s.finalStack);
    return icmDelta(fs, {1.0}, player);
  }
  assert(s.terminalKind == 3);
  if (!cfg_.icm) {
    // Chip delta = chips won from the pot minus everything contributed
    // (antes, blinds, bets). `stack` is the behind-stack, so the final
    // stack is stack + winnings and the delta vs. start is
    // winnings - contributed.
    double v = -static_cast<double>(s.contributed[player]);
    for (const auto& o : s.contOutcomes) {
      v += o.probability * static_cast<double>(o.winnings[player]);
    }
    return v;
  }
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
  uint64_t key = 0xCBF29CE484222325ULL;
  key = hashCombine(key, static_cast<uint64_t>(player));
  if (s.street == 0) {
    int h = handIndex169(s.hole[2 * player], s.hole[2 * player + 1]);
    int bucket = bucketOf(h, cfg_.numBuckets);
    key = hashCombine(key, static_cast<uint64_t>(bucket));
  } else {
    key = hashCombine(key, static_cast<uint64_t>(s.street));
    if (cfg_.postflopAbstraction == 0) {
      // Exact mode: the board and hole cards are part of the key.
      Card cards[7];
      cards[0] = s.hole[2 * player];
      cards[1] = s.hole[2 * player + 1];
      for (int i = 0; i < s.boardDealt; ++i) cards[2 + i] = s.board[i];
      key = fnv1a(key, cards, 2 + s.boardDealt);
    } else {
      int bucket = bucketOfPostflop(s, player);
      key = hashCombine(key, static_cast<uint64_t>(bucket));
    }
  }
  key = fnv1a(key, s.hist.data(), s.nhist * sizeof(uint32_t));
  return key;
}

}  // namespace pps
