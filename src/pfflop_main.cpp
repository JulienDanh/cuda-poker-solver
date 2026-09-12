// pfflop: range-based CFR solver for heads-up river spots
// (parity target: b-inary/postflop-solver).
//
// usage:
//   pfflop river <board> <oop_range> <ip_range> <pot> <stack> <bets> <raises>
//               <iters> [--algo dcfr|hs30] [--dump-tree]
//   board: 5 cards e.g. "Qs9h2d7c8d"; ranges: postflop-solver syntax;
//   bets/raises: e.g. "0.5,0.75,1.0" and "2.5,3.0" (see docs).
//
// Prints:
//   EV 0 <chips> / EV 1 <chips> / EXPLOITABILITY <chips>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cards.h"
#include "postflop_cfr.h"
#include "range.h"

using namespace pps;

namespace {

Card cardFromText(const std::string& s, int i) {
  char r = s[2 * i];
  char su = s[2 * i + 1];
  int rank;
  switch (r) {
    case '2': rank = 0; break;
    case '3': rank = 1; break;
    case '4': rank = 2; break;
    case '5': rank = 3; break;
    case '6': rank = 4; break;
    case '7': rank = 5; break;
    case '8': rank = 6; break;
    case '9': rank = 7; break;
    case 'T': case 't': rank = 8; break;
    case 'J': case 'j': rank = 9; break;
    case 'Q': case 'q': rank = 10; break;
    case 'K': case 'k': rank = 11; break;
    case 'A': case 'a': rank = 12; break;
    default: std::fprintf(stderr, "bad card rank '%c'\n", r); std::exit(2);
  }
  int suit;
  switch (su) {
    case 'c': case 'C': suit = 0; break;
    case 'd': case 'D': suit = 1; break;
    case 'h': case 'H': suit = 2; break;
    case 's': case 'S': suit = 3; break;
    default: std::fprintf(stderr, "bad card suit '%c'\n", su); std::exit(2);
  }
  return static_cast<Card>(suit * 13 + rank);
}

std::vector<double> parseDoubles(const std::string& s) {
  std::vector<double> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(',', start);
    std::string tok = end == std::string::npos ? s.substr(start)
                                               : s.substr(start, end - start);
    if (!tok.empty()) out.push_back(std::stod(tok));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 9 || std::strcmp(argv[1], "river") != 0) {
    std::fprintf(stderr,
                  "usage: pfflop river <board> <oop_range> <ip_range> <pot> "
                  "<stack> <bets> <raises> <iters> [--algo dcfr|hs30] "
                  "[--dump-tree]\n");
    return 2;
  }
  pf::Spot spot;
  std::string board = argv[2];
  for (int i = 0; i < 5; ++i) spot.board[i] = cardFromText(board, i);
  spot.oop = parseRange(argv[3]);
  spot.ip = parseRange(argv[4]);
  spot.pot = std::stoll(argv[5]);
  spot.stack = std::stoll(argv[6]);
  pf::BetConfig cfg;
  cfg.betFracs = parseDoubles(argv[7]);
  cfg.raiseMults = parseDoubles(argv[8]);
  int iters = std::stoi(argv[9]);
  std::string algo = "dcfr";
  bool dumpTree = false;
  for (int i = 10; i < argc; ++i) {
    if (std::strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
      algo = argv[++i];
    } else if (std::strcmp(argv[i], "--dump-tree") == 0) {
      dumpTree = true;
    }
  }

  pf::RiverSolver solver(spot, cfg);
  if (dumpTree) {
    solver.dumpTree(stdout);
  }
  solver.solve(iters, algo);
  auto st = solver.stats();
  std::printf("EV 0 %.6f\n", st.ev0);
  std::printf("EV 1 %.6f\n", st.ev1);
  std::printf("EXPLOITABILITY %.6f\n", st.expl);
  return 0;
}
