// gpu_pfflop: GPU-CFR postflop solver (compile-to-static-dataflow +
// CUDA graph replay; see cuda/gpu_cfr.h). Multi-street: the board
// argument is 3, 4 or 5 cards (flop / turn / river spot).
//
//   gpu_pfflop postflop <board> <oop_range> <ip_range> <pot> <stack>
//             <bets> <raises> <iters> [--algo dcfr|hs30] [--root]
//
// Prints (plus GPU timings on stderr):
//   EV 0 <chips> / EV 1 <chips> / EXPLOITABILITY <chips>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_cfr.h"
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
  if (argc < 9 ||
      (std::strcmp(argv[1], "postflop") != 0 &&
       std::strcmp(argv[1], "river") != 0)) {
    std::fprintf(stderr,
                  "usage: gpu_pfflop postflop <board> <oop_range> <ip_range> "
                  "<pot> <stack> <bets> <raises> <iters> "
                  "[--algo dcfr|hs30] [--root]\n"
                  "  board: 3, 4 or 5 cards (flop / turn / river)\n");
    return 2;
  }
  const std::string boardStr = argv[2];
  if (boardStr.size() != 6 && boardStr.size() != 8 && boardStr.size() != 10) {
    std::fprintf(stderr, "board must be 3, 4 or 5 cards\n");
    return 2;
  }
  gpu::PostflopSpot spot;
  spot.nBoard = (int)(boardStr.size() / 2);
  for (int i = 0; i < spot.nBoard; ++i)
    spot.board[i] = cardFromText(boardStr, i);
  spot.oop = parseRange(argv[3]);
  spot.ip = parseRange(argv[4]);
  spot.pot = std::stoll(argv[5]);
  spot.stack = std::stoll(argv[6]);
  pf::BetConfig cfg;
  cfg.betFracs = parseDoubles(argv[7]);
  cfg.raiseMults = parseDoubles(argv[8]);
  int iters = std::stoi(argv[9]);
  std::string algo = "dcfr";
  bool rootStrat = false;
  for (int i = 10; i < argc; ++i) {
    if (std::strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
      algo = argv[++i];
    } else if (std::strcmp(argv[i], "--root") == 0) {
      rootStrat = true;
    }
  }

  auto wall0 = std::chrono::steady_clock::now();
  gpu::GpuPostflopSolver solver(spot, cfg);
  auto wall1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "gpu_cfr: nodes=%d depth=%d board=%d cards\n",
               solver.numNodes(), solver.maxDepth(), spot.nBoard);
  std::fprintf(stderr, "gpu_cfr: compile %.1f ms\n",
               std::chrono::duration<double, std::milli>(wall1 - wall0).count());

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  solver.solve(iters, algo);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  std::fprintf(stderr, "gpu_cfr: %d iters in %.1f ms (%.0f iters/s)\n", iters,
               ms, iters / (ms / 1000.0f));

  if (std::getenv("GPU_CFR_DEBUG")) solver.debugDump();

  auto wall2 = std::chrono::steady_clock::now();
  if (rootStrat) {
    auto rs = solver.rootStrategy();
    std::printf("ROOTSTRAT");
    for (double f : rs) std::printf(" %.6f", f);
    std::printf("\n");
  }
  auto st = solver.stats();
  auto wall3 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "gpu_cfr: stats walk %.1f ms\n",
               std::chrono::duration<double, std::milli>(wall3 - wall2)
                   .count());
  std::printf("EV 0 %.6f\n", st.ev0);
  std::printf("EV 1 %.6f\n", st.ev1);
  std::printf("EXPLOITABILITY %.6f\n", st.expl);
  return 0;
}
