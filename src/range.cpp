#include "range.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace pps {

const uint8_t kComboCards[kCombos][2] = [] {
  // Cards use our encoding: suit * 13 + rank.
  static uint8_t table[kCombos][2];
  int idx = 0;
  for (int a = 0; a < 52; ++a) {
    for (int b = a + 1; b < 52; ++b) {
      table[idx][0] = static_cast<uint8_t>(a);
      table[idx][1] = static_cast<uint8_t>(b);
      ++idx;
    }
  }
  return table;
}();

int comboIndex(uint8_t a, uint8_t b) {
  if (a > b) std::swap(a, b);
  // a < b: index = a * (51 - a) + (b - a - 1) - (offset of a-th block).
  // Simple: precompute inverse table on demand.
  static std::vector<int> inv(52 * 52, -1);
  if (inv[a * 52 + b] < 0) {
    int idx = 0;
    for (int i = 0; i < 52; ++i) {
      for (int j = i + 1; j < 52; ++j) {
        inv[i * 52 + j] = idx++;
      }
    }
  }
  return inv[a * 52 + b];
}

Range::Range() {
  for (int i = 0; i < kCombos; ++i) w[i] = 0.0;
}

Range Range::ones() {
  Range r;
  for (int i = 0; i < kCombos; ++i) r.w[i] = 1.0;
  return r;
}

double Range::total() const {
  double s = 0.0;
  for (int i = 0; i < kCombos; ++i) s += w[i];
  return s;
}

void Range::normalize() {
  double s = total();
  if (s > 0) {
    for (int i = 0; i < kCombos; ++i) w[i] /= s;
  }
}

void Range::filterBoard(const Card* board, int n) {
  for (int i = 0; i < kCombos; ++i) {
    for (int b = 0; b < n; ++b) {
      if (kComboCards[i][0] == board[b] || kComboCards[i][1] == board[b]) {
        w[i] = 0.0;
        break;
      }
    }
  }
}

int rankFromChar(char c) {
  switch (c) {
    case '2': return 0;
    case '3': return 1;
    case '4': return 2;
    case '5': return 3;
    case '6': return 4;
    case '7': return 5;
    case '8': return 6;
    case '9': return 7;
    case 'T': return 8;
    case 'J': return 9;
    case 'Q': return 10;
    case 'K': return 11;
    case 'A': return 12;
  }
  return -1;
}

namespace {

struct ClassRef {
  int hi = -1;
  int lo = -1;
  char suitKind = 'x';  // 'p' pair, 's' suited, 'o' offsuit
  double weight = 1.0;
};

void addCombo(Range& r, int a, int b, double weight) {
  if (a == b) return;
  r.w[comboIndex(static_cast<uint8_t>(a), static_cast<uint8_t>(b))] += weight;
}

// Adds all combos of the class (hi, lo, kind).
void addClassCombos(Range& r, int hi, int lo, char kind, double weight) {
  if (kind == 'p') {
    for (int s1 = 0; s1 < 4; ++s1) {
      for (int s2 = s1 + 1; s2 < 4; ++s2) {
        addCombo(r, s1 * 13 + hi, s2 * 13 + hi, weight);
      }
    }
    return;
  }
  if (kind == 's') {
    for (int s = 0; s < 4; ++s) addCombo(r, s * 13 + hi, s * 13 + lo, weight);
    return;
  }
  // offsuit
  for (int s1 = 0; s1 < 4; ++s1) {
    for (int s2 = 0; s2 < 4; ++s2) {
      if (s1 != s2) addCombo(r, s1 * 13 + hi, s2 * 13 + lo, weight);
    }
  }
}

ClassRef parseClass(const std::string& tok) {
  // Accepted: "TT", "AKs", "AKo" (optionally with ":w" already stripped).
  ClassRef c;
  size_t pos = 0;
  std::string t = tok;
  double weight = 1.0;
  size_t colon = t.find(':');
  if (colon != std::string::npos) {
    weight = std::stod(t.substr(colon + 1));
    t = t.substr(0, colon);
  }
  c.weight = weight;
  if (t.size() < 2) throw std::runtime_error("bad range token: " + tok);
  int r1 = rankFromChar(t[0]);
  int r2 = rankFromChar(t[1]);
  if (t.size() == 2) {
    if (r1 < 0 || r2 < 0) throw std::runtime_error("bad range token: " + tok);
    c.suitKind = (r1 == r2) ? 'p' : 'o';
    c.hi = std::max(r1, r2);
    c.lo = std::min(r1, r2);
    return c;
  }
  if (t.size() == 3 && (t[2] == 's' || t[2] == 'o') && r1 >= 0 && r2 >= 0 && r1 != r2) {
    c.suitKind = t[2];
    c.hi = std::max(r1, r2);
    c.lo = std::min(r1, r2);
    return c;
  }
  throw std::runtime_error("bad range token: " + tok);
}

}  // namespace

Range parseRange(const std::string& s) {
  Range r;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(',', start);
    std::string tok;
    if (end == std::string::npos) {
      tok = s.substr(start);
      start = s.size() + 1;
    } else {
      tok = s.substr(start, end - start);
      start = end + 1;
    }
    while (!tok.empty() && tok.back() == ' ') tok.pop_back();
    while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
    if (tok.empty()) continue;

    bool plus = false;
    if (tok.back() == '+') {
      plus = true;
      tok.pop_back();
    }
    size_t dash = tok.find('-');
    if (dash != std::string::npos && !plus) {
      // Dash ranges: same gap -> slide (e.g. "KQo-JTo"); same first rank ->
      // vary the second card (e.g. "A5s-A2s"); descending order required.
      ClassRef a = parseClass(tok.substr(0, dash));
      ClassRef b = parseClass(tok.substr(dash + 1));
      if (a.suitKind != b.suitKind) throw std::runtime_error("range dash mismatch: " + tok);
      int gapA = a.hi - a.lo;
      int gapB = b.hi - b.lo;
      if (gapA == gapB) {
        if (a.hi < b.hi) throw std::runtime_error("range must be descending: " + tok);
        for (int hi = b.hi; hi <= a.hi; ++hi) {
          if (a.suitKind == 'p') {
            addClassCombos(r, hi, hi, 'p', a.weight);
          } else {
            addClassCombos(r, hi, hi - gapA, a.suitKind, a.weight);
          }
        }
      } else if (a.hi == b.hi) {
        if (a.lo < b.lo) throw std::runtime_error("range must be descending: " + tok);
        if (a.suitKind == 'p') throw std::runtime_error("pair dash needs same gap: " + tok);
        for (int lo = b.lo; lo <= a.lo; ++lo) {
          addClassCombos(r, a.hi, lo, a.suitKind, a.weight);
        }
      } else {
        throw std::runtime_error("invalid range: " + tok);
      }
      continue;
    }
    ClassRef c = parseClass(tok);
    if (plus) {
      int gap = c.hi - c.lo;
      if (gap <= 1) {
        // Pairs and connectors slide up preserving the gap
        // (e.g. "88+" = 88..AA, "T9s+" = T9s, JTs, QJs, AKs).
        for (int hi = c.hi; hi <= 12; ++hi) {
          if (c.suitKind == 'p') {
            addClassCombos(r, hi, hi, 'p', c.weight);
          } else {
            addClassCombos(r, hi, hi - gap, c.suitKind, c.weight);
          }
        }
      } else {
        // Wider ranges keep the first card fixed
        // (e.g. "ATo+" = ATo..AKo, "A9s+" = A9s..AKs).
        for (int lo = c.lo; lo < c.hi; ++lo) {
          addClassCombos(r, c.hi, lo, c.suitKind, c.weight);
        }
      }
      continue;
    }
    addClassCombos(r, c.hi, c.lo, c.suitKind, c.weight);
  }
  return r;
}

}  // namespace pps
