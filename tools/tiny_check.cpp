// Deterministic ground truth for the tiny betting turn spot:
//   board Qs9h2d7c + river chance; turn pot P=200, bet B1=100; river after a
//   called turn bet: pot P2=P+2*B1=400, river bet B2=0.5*P2; no raises/all-ins.
//   Uniform strategies (sigma = 1/2). Chance weight 1/44, per-pair masking.
// Values measured from the STREET-START baseline:
//   fold:  v(folder) = -sc(folder), v(winner) = pb + sc(folder)
//   bet-call showdown: v(win) = (pb + 2*bet) - bet = pb + bet
//                      v(lose) = -bet, tie = pb/2
//   chance after called street: each player deducts their own street bet.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "cards.h"
#include "range.h"

using namespace pps;

struct C {
  uint8_t a, b;
  double w;
  uint64_t s5[52];
};

static Card cardFromText(const std::string& s, int i) {
  char r = s[2 * i];
  char su = s[2 * i + 1];
  int rank = -1;
  const char* ranks = "23456789TJQKA";
  for (int k = 0; k < 13; ++k)
    if (r == ranks[k]) rank = k;
  int suit = -1;
  const char* suits = "cdhs";
  for (int k = 0; k < 4; ++k)
    if (su == suits[k] || su == suits[k] - 32) suit = k;
  return (Card)(suit * 13 + rank);
}

static uint64_t keyOf(const HandValue& v) {
  uint64_t key = v.category;
  for (int i = 0; i < 5; ++i) key = key * 13 + v.tiebreak[i];
  return key;
}

// River subtree value (uniform sigma), traverser t, river pot pb, bet b.
// Baselines: check-check showdown: win=pb, lose=0, tie=pb/2.
// Bet-call showdown: win=pb+b, lose=-b, tie=pb/2. Fold: folder loses 0.
static double riverVal(const C& i, const C& j, int br, double pb, double b,
                       int t) {
  auto cc = [&](int who) {
    if (who == 0)
      return i.s5[br] > j.s5[br] ? pb : (i.s5[br] == j.s5[br] ? pb / 2 : 0.0);
    return i.s5[br] > j.s5[br] ? 0.0 : (i.s5[br] == j.s5[br] ? pb / 2 : pb);
  };
  auto bc = [&](int who) {
    if (who == 0)
      return i.s5[br] > j.s5[br] ? pb + b : (i.s5[br] == j.s5[br] ? pb / 2 : -b);
    return i.s5[br] > j.s5[br] ? -b : (i.s5[br] == j.s5[br] ? pb / 2 : pb + b);
  };
  if (t == 0) {
    double oopCheck = 0.5 * cc(0) +
                      0.5 * (0.5 * 0.0 /*OOP folds*/ + 0.5 * bc(0));
    double oopBet = 0.5 * (pb + 0.0) /*IP folds*/ + 0.5 * bc(0);
    return 0.5 * oopCheck + 0.5 * oopBet;
  }
  double oopCheck = 0.5 * cc(1) +
                    0.5 * (0.5 * (pb + 0.0) /*OOP folds*/ + 0.5 * bc(1));
  double oopBet = 0.5 * 0.0 /*IP folds*/ + 0.5 * bc(1);
  return 0.5 * oopCheck + 0.5 * oopBet;
}

static int c0r(int x) { return x == 0 ? 1 : 0; }
int main() {
  Card board[4] = {cardFromText("Qs9h2d7c", 0), cardFromText("Qs9h2d7c", 1),
                   cardFromText("Qs9h2d7c", 2), cardFromText("Qs9h2d7c", 3)};
  Range r0 = parseRange("TT+");
  Range r1 = parseRange("99+");
  double P = 200, B1 = 100, P2 = P + 2 * B1, B2 = P2 / 2;
  std::vector<C> s0, s1;
  for (int i = 0; i < kCombos; ++i) {
    uint8_t a, b;
    comboCards(i, a, b);
    bool blocked = false;
    for (int c = 0; c < 4; ++c)
      if (a == board[c] || b == board[c]) blocked = true;
    if (blocked || r0.w[i] <= 0) continue;
    s0.push_back({a, b, r0.w[i], {}});
  }
  for (int i = 0; i < kCombos; ++i) {
    uint8_t a, b;
    comboCards(i, a, b);
    bool blocked = false;
    for (int c = 0; c < 4; ++c)
      if (a == board[c] || b == board[c]) blocked = true;
    if (blocked || r1.w[i] <= 0) continue;
    s1.push_back({a, b, r1.w[i], {}});
  }
  for (int br = 0; br < 52; ++br) {
    bool onBoard = false;
    for (int c = 0; c < 4; ++c)
      if (board[c] == br) onBoard = true;
    if (onBoard) continue;
    Card five[5] = {board[0], board[1], board[2], board[3], (Card)br};
    for (auto& s : s0) {
      Card seven[7] = {s.a, s.b, five[0], five[1], five[2], five[3], five[4]};
      s.s5[br] = keyOf(evaluate7(seven));
    }
    for (auto& s : s1) {
      Card seven[7] = {s.a, s.b, five[0], five[1], five[2], five[3], five[4]};
      s.s5[br] = keyOf(evaluate7(seven));
    }
  }

  double Z = 0, A0 = 0, A1 = 0;
  for (auto& i : s0) {
    for (auto& j : s1) {
      if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
      double pw = i.w * j.w;
      Z += pw;

      // Chance-averaged river values for the two reachable river states.
      double r0a = 0, r1a = 0, r0b = 0, r1b = 0;  // a: pb=P; b: pb=P2-B1 net
      for (int br = 0; br < 52; ++br) {
        bool onBoard = false;
        for (int c = 0; c < 4; ++c)
          if (board[c] == br) onBoard = true;
        if (onBoard) continue;
        if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
        double w = 1.0 / 44;
        r0a += w * riverVal(i, j, br, P, P / 2, 0);
        r1a += w * riverVal(i, j, br, P, P / 2, 1);
        r0b += w * (riverVal(i, j, br, P2, B2, 0) - B1);
        r1b += w * (riverVal(i, j, br, P2, B2, 1) - B1);
      }
      // Turn play, uniform.
      double v0 = 0, v1 = 0;
      // OOP checks (1/2):
      //   IP checks (1/2): chance, river pb=P, no deduction.
      //   IP bets (1/2): OOP folds (1/2): v0=0, v1=P.
      //                   OOP calls (1/2): chance, river pb=P2, deduct B1.
      // OOP bets (1/2):
      //   IP folds (1/2): v0=P, v1=0.
      //   IP calls (1/2): chance, river pb=P2, deduct B1.
      v0 = 0.5 * (0.5 * r0a + 0.5 * (0.5 * 0.0 + 0.5 * r0b)) +
           0.5 * (0.5 * (P + 0.0) + 0.5 * r0b);
      v1 = 0.5 * (0.5 * r1a + 0.5 * (0.5 * (P + 0.0) + 0.5 * r1b)) +
           0.5 * (0.5 * 0.0 + 0.5 * r1b);
      A0 += pw * v0;
      A1 += pw * v1;

      if (v0 + v1 > 200.001 || v0 + v1 < 199.999) {
        static int shown = 0;
        if (shown < 5) {
          std::printf("PAIRDBG v0=%.3f v1=%.3f sum=%.3f r0a=%.2f r1a=%.2f r0b=%.2f r1b=%.2f\n",
                      v0, v1, v0 + v1, r0a, r1a, r0b, r1b);
          ++shown;
        }
      }
    }
  }
  {
    // per-combo r0a/r0b for the first side-0 combo (engine-comparable)
    const C& i0 = s0[0];
    double ra = 0, rb = 0;
    for (auto& j : s1) {
      if (i0.a == j.a || i0.a == j.b || i0.b == j.a || i0.b == j.b) continue;
      for (int br = 0; br < 52; ++br) {
        bool onBoard = false;
        for (int c = 0; c < 4; ++c)
          if (board[c] == br) onBoard = true;
        if (onBoard) continue;
        if (i0.a == br || i0.b == br || j.a == br || j.b == br) continue;
        double w = 1.0 / 44;
        ra += w * riverVal(i0, j, br, P, P / 2, 0);
        rb += w * (riverVal(i0, j, br, P2, B2, 0) - B1);
      }
    }
    std::printf("COMBO0: r0a=%.4f r0b=%.4f\n", ra, rb);
  }
  std::printf("TINY: EV0 %.6f  EV1 %.6f  sum %.4f (Z=%.1f)\n", A0 / Z,
              A1 / Z, (A0 + A1) / Z, Z);
  // Mode-7 profile: root 50/50; ipTop: check with p0(c)=0.15+0.7*(c%8)/7,
  // bet with 1-p0(c); deeper uniform. EV walk per pair.
  {
    auto ipTopSeed = [](int c) {
      return 0.15 + 0.7 * ((c % 8) / 7.0);
    };
    double M0 = 0, M1 = 0;
    // index side-1 combos
    int jc = 0;
    (void)jc;
    for (auto& i : s0) {
      for (auto& j : s1) {
        if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
        double pw = i.w * j.w;
        // find j's index
        int jidx = (int)(&j - &s1[0]);
        double pCk = ipTopSeed(jidx);
        double pBt = 1.0 - pCk;
        // chance-averaged river values (uniform deeper)
        double r0a = 0, r1a = 0, r0b = 0, r1b = 0;
        for (int br = 0; br < 52; ++br) {
          bool onBoard = false;
          for (int c = 0; c < 4; ++c)
            if (board[c] == br) onBoard = true;
          if (onBoard) continue;
          if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
          double w = 1.0 / 44;
          r0a += w * riverVal(i, j, br, P, P / 2, 0);
          r1a += w * riverVal(i, j, br, P, P / 2, 1);
          r0b += w * (riverVal(i, j, br, P2, B2, 0) - B1);
          r1b += w * (riverVal(i, j, br, P2, B2, 1) - B1);
        }
        // Turn paths with the profile:
        // OOP checks (1/2) -> ipTop: check pCk -> chance river pb=P (r?a);
        //   bet pBt -> oopFace: fold/call 50/50 ->
        //     OOP folds: v0=0, v1=P; OOP calls -> chance pb=P2, deduct B1
        // OOP bets (1/2) -> ipFace: fold/call 50/50 (uniform) ->
        //   IP folds: v0=P, v1=0; IP calls -> chance pb=P2 (r?b)
        double v0 = 0.5 * (pCk * r0a +
                           pBt * (0.5 * 0.0 + 0.5 * r0b)) +
                   0.5 * (0.5 * (P + 0.0) + 0.5 * r0b);
        double v1 = 0.5 * (pCk * r1a +
                           pBt * (0.5 * (P + 0.0) + 0.5 * r1b)) +
                   0.5 * (0.5 * 0.0 + 0.5 * r1b);
        M0 += pw * v0;
        M1 += pw * v1;
      }
    }
    std::printf("M7: EV0 %.6f  EV1 %.6f  sum %.4f\n", M0 / Z, M1 / Z,
                (M0 + M1) / Z);
    // per-i values under the M7 profile
    // Mode-9: ipFace seeded: fold=p0(j), call=1-p0(j); all else uniform.
    {
      double N0 = 0, N1 = 0;
      for (auto& i : s0) {
        for (auto& j : s1) {
          if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
          double pw = i.w * j.w;
          int jidx = (int)(&j - &s1[0]);
          double pF = ipTopSeed(jidx);
          double pC = 1.0 - pF;
          double r0a = 0, r1a = 0, r0b = 0, r1b = 0;
          for (int br = 0; br < 52; ++br) {
            bool onBoard = false;
            for (int c = 0; c < 4; ++c)
              if (board[c] == br) onBoard = true;
            if (onBoard) continue;
            if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
            double w = 1.0 / 44;
            r0a += w * riverVal(i, j, br, P, P / 2, 0);
            r1a += w * riverVal(i, j, br, P, P / 2, 1);
            r0b += w * (riverVal(i, j, br, P2, B2, 0) - B1);
            r1b += w * (riverVal(i, j, br, P2, B2, 1) - B1);
          }
          double v0 = 0.5 * (0.5 * r0a + 0.5 * (0.5 * 0.0 + 0.5 * r0b)) +
                     0.5 * (pF * (P + 0.0) + pC * r0b);
          double v1 = 0.5 * (0.5 * r1a + 0.5 * (0.5 * (P + 0.0) + 0.5 * r1b)) +
                     0.5 * (pF * 0.0 + pC * r1b);
          N0 += pw * v0;
          N1 += pw * v1;
        }
      }
      std::printf("M9: EV0 %.6f  EV1 %.6f  sum %.4f\n", N0 / Z, N1 / Z,
                  (N0 + N1) / Z);
      // per-i for mode 9
      std::printf("M9PERCOMBO:");
      {
        int idx = 0;
        for (auto& i : s0) {
          double v0 = 0;
          for (auto& j : s1) {
            if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
            int jidx = (int)(&j - &s1[0]);
            double pF = ipTopSeed(jidx);
            double pC = 1.0 - pF;
            double r0a = 0, r0b = 0;
            for (int br = 0; br < 52; ++br) {
              bool onBoard = false;
              for (int c = 0; c < 4; ++c)
                if (board[c] == br) onBoard = true;
              if (onBoard) continue;
              if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
              double w = 1.0 / 44;
              r0a += w * riverVal(i, j, br, P, P / 2, 0);
              r0b += w * (riverVal(i, j, br, P2, B2, 0) - B1);
            }
            v0 += 0.5 * (0.5 * r0a + 0.5 * (0.5 * 0.0 + 0.5 * r0b)) +
                 0.5 * (pF * (P + 0.0) + pC * r0b);
          }
          std::printf(" %.2f", v0);
          ++idx;
          if (idx >= 27) break;
        }
      }
      std::printf("\n");
    }
    // fold 871 (tr=0, OOP winner) and chance-872 per-i under mode 9.
    std::printf("M9KIDS:");
    {
      int idx = 0;
      for (auto& i : s0) {
        double fw = 0;
        double cc = 0;
        for (auto& j : s1) {
          if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
          int jidx = (int)(&j - &s1[0]);
          double pF = ipTopSeed(jidx);
          double pC = 1.0 - pF;
          fw += pF * P;
          for (int br = 0; br < 52; ++br) {
            bool onBoard = false;
            for (int c = 0; c < 4; ++c)
              if (board[c] == br) onBoard = true;
            if (onBoard) continue;
            if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
            cc += (1.0 / 44) * pC * (riverVal(i, j, br, P2, B2, 0) - B1);
          }
        }
        std::printf(" %.2f/%.2f", fw, cc);
        ++idx;
        if (idx >= 6) break;
      }
    }
    std::printf("\n");
    // N873 = river decide (branch card 0) under chance 872, tr=0.
    // SD875 (check-check showdown under 873, branch 0) masses for i=0.
    {
      int br = 0;
      const C& i0 = s0[0];
      double mTie = 0, mGreat = 0, mLess = 0;
      for (auto& j : s1) {
        if (i0.a == j.a || i0.a == j.b || i0.b == j.a || i0.b == j.b) continue;
        if (j.a == br || j.b == br) continue;
        int jidx = (int)(&j - &s1[0]);
        double pC = 1.0 - ipTopSeed(jidx);
        double w = pC * 0.5 / 44.0;  // IP checked (0.5) at the river decide
        if (j.s5[br] > i0.s5[br]) mGreat += w;
        else if (j.s5[br] == i0.s5[br]) mTie += w;
        else mLess += w;
      }
      std::printf("SD875MASSES: tm=%.6f gm=%.6f lm=%.6f out=%.4f\n", mTie,
                  mGreat, mLess,
                  mTie * 100.0 + mGreat * (-100.0));
    }
    // Expected le/te for side-0 combo 0 on the branch-0 board.
    {
      int br = 0;
      const C& i0 = s0[0];
      int le = 0, te = 0;
      for (auto& j : s1) {
        if (j.a == br || j.b == br) continue;  // branch filter
        // count over the BRANCH-FILTERED opponent list? No — the engine's
        // tables count over the FULL base list (minus board blockers).
        if (j.s5[br] < i0.s5[br]) ++le;
        if (j.s5[br] <= i0.s5[br]) ++te;
      }
      std::printf("LE0: le=%d te=%d\n", le, te);
    }
    std::printf("N873VAL:");
    {
      int br = 0;  // card 0 = 2c
      int idx = 0;
      for (auto& i : s0) {
        double v = 0;
        double reach0 = 0;
        for (auto& j : s1) {
          if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
          if (j.a == br || j.b == br) continue;  // j must survive the branch
          int jidx = (int)(&j - &s1[0]);
          double pC = 1.0 - ipTopSeed(jidx);
          double w = pC / 44.0;
          // river decide at pb=400, pc=100: value measured from root
          // baseline = riverVal(pb=400, bet=200) - 100
          v += w * (riverVal(i, j, br, P2, B2, 0) - B1);
          if (c0r(jidx) == 0) reach0 = w;
        }
        std::printf(" %.2f", v);
        ++idx;
        if (idx >= 3) break;
      }
    }
    std::printf("\n");
    std::printf("CH2VAL:");
    {
      int idx2 = 0;
      for (auto& i : s0) {
        double v = 0;
        for (auto& j : s1) {
          if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
          int jidx = (int)(&j - &s1[0]);
          double pCk = ipTopSeed(jidx);
          for (int br = 0; br < 52; ++br) {
            bool onBoard = false;
            for (int c = 0; c < 4; ++c)
              if (board[c] == br) onBoard = true;
            if (onBoard) continue;
            if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
            v += (1.0 / 44) * pCk * riverVal(i, j, br, P, P / 2, 0);
          }
        }
        std::printf(" %.2f", v);
        ++idx2;
        if (idx2 >= 27) break;
      }
    }
    std::printf("\n");
    std::printf("M7PERCOMBO:");
    {
      int idx = 0;
      for (auto& i : s0) {
        double v0 = 0;
        for (auto& j : s1) {
          if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
          int jidx = (int)(&j - &s1[0]);
          double pCk = ipTopSeed(jidx);
          double pBt = 1.0 - pCk;
          double r0a = 0, r0b = 0;
          for (int br = 0; br < 52; ++br) {
            bool onBoard = false;
            for (int c = 0; c < 4; ++c)
              if (board[c] == br) onBoard = true;
            if (onBoard) continue;
            if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
            double w = 1.0 / 44;
            r0a += w * riverVal(i, j, br, P, P / 2, 0);
            r0b += w * (riverVal(i, j, br, P2, B2, 0) - B1);
          }
          v0 += 0.5 * (pCk * r0a + pBt * (0.5 * 0.0 + 0.5 * r0b)) +
               0.5 * (0.5 * (P + 0.0) + 0.5 * r0b);
        }
        std::printf(" %.2f", v0);
        ++idx;
        if (idx >= 27) break;
      }
    }
    std::printf("\n");
  }
  // per-i (side-0) value: recompute summing over j
  {
    int idx = 0;
    std::printf("PERCOMBO0:");
    for (auto& i : s0) {
      double v0 = 0;
      double wsum = 0;
      for (auto& j : s1) {
        if (i.a == j.a || i.a == j.b || i.b == j.a || i.b == j.b) continue;
        double r0a = 0, r0b = 0;
        for (int br = 0; br < 52; ++br) {
          bool onBoard = false;
          for (int c = 0; c < 4; ++c)
            if (board[c] == br) onBoard = true;
          if (onBoard) continue;
          if (i.a == br || i.b == br || j.a == br || j.b == br) continue;
          double w = 1.0 / 44;
          r0a += w * riverVal(i, j, br, P, P / 2, 0);
          r0b += w * (riverVal(i, j, br, P2, B2, 0) - B1);
        }
        v0 += 0.5 * (0.5 * r0a + 0.5 * (0.5 * 0.0 + 0.5 * r0b)) +
              0.5 * (0.5 * (P + 0.0) + 0.5 * r0b);
        wsum += 1;
      }
      std::printf(" %.2f", v0);
      ++idx;
      if (idx >= 27) break;
    }
    std::printf("\n");
  }
  return 0;
}
