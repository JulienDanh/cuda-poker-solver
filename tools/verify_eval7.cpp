// verify_eval7: cross-validates our 7-card hand evaluator (src/cards.cpp)
// against b-inary/postflop-solver's Hand::evaluate() via the pfs-verify
// eval7 chunked streaming protocol.
//
// For N random 7-card hands (fixed seed), it computes our HandValue
// (encoded as a total-order key: category and five tiebreak ranks in
// base 13) and their strength. The two orderings must agree exactly:
// - sorting by our key, their strengths must be non-decreasing
// - equal keys must have equal strengths
// Any disagreement means one of the evaluators ranks poker hands
// differently — a hard failure.
//
// usage: verify_eval7 [num_hands] [path-to-pfs-verify]
#include <algorithm>
#include <cstdint>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cards.h"
#include "common.h"

using namespace pps;

namespace {

uint64_t handKey(const HandValue& v) {
  uint64_t key = v.category;
  for (int i = 0; i < 5; ++i) {
    key = key * 13 + static_cast<uint64_t>(v.tiebreak[i]);
  }
  return key;
}

}  // namespace

// macOS popen has no "w+" (bidirectional) mode, so spawn the child with
// two pipes manually.
struct Child {
  int inFd = -1;    // parent writes hand chunks here (child stdin)
  int outFd = -1;   // parent reads strengths here (child stdout)
  pid_t pid = -1;
};

bool spawnChild(Child& c, const std::string& cmd) {
  int inPipe[2], outPipe[2];
  if (pipe(inPipe) != 0) return false;
  if (pipe(outPipe) != 0) return false;
  pid_t pid = fork();
  if (pid < 0) return false;
  if (pid == 0) {
    // child
    ::close(inPipe[1]);
    ::close(outPipe[0]);
    dup2(inPipe[0], STDIN_FILENO);
    dup2(outPipe[1], STDOUT_FILENO);
    execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(inPipe[0]);
  ::close(outPipe[1]);
  c.inFd = inPipe[1];
  c.outFd = outPipe[0];
  c.pid = pid;
  return true;
}

void closeChild(Child& c) {
  if (c.inFd >= 0) ::close(c.inFd);
  if (c.outFd >= 0) ::close(c.outFd);
  int status = 0;
  if (c.pid > 0) waitpid(c.pid, &status, 0);
}

int main(int argc, char** argv) {
  size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 4000000;
  const char* pfsPath =
      argc > 2 ? argv[2] : "tools/pfs-verify/target/release/pfs-verify";

  std::string cmd = std::string(pfsPath) + " eval7";
  Child child;
  if (!spawnChild(child, cmd)) {
    std::fprintf(stderr, "cannot spawn %s\n", cmd.c_str());
    return 2;
  }

  constexpr size_t kChunk = 4096;
  RNG rng(0xC0FFEE12);
  Card deck[52];
  std::vector<std::pair<uint64_t, uint64_t>> hands;
  hands.reserve(n);
  std::vector<uint8_t> out(kChunk * 7);
  std::vector<uint8_t> in(kChunk * 8);

  auto writeAll = [&](const void* p, size_t len) -> bool {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    size_t off = 0;
    while (off < len) {
      ssize_t w = ::write(child.inFd, b + off, len - off);
      if (w <= 0) return false;
      off += static_cast<size_t>(w);
    }
    return true;
  };
  auto readAll = [&](void* p, size_t len) -> bool {
    uint8_t* b = static_cast<uint8_t*>(p);
    size_t off = 0;
    while (off < len) {
      ssize_t r = ::read(child.outFd, b + off, len - off);
      if (r <= 0) return false;
      off += static_cast<size_t>(r);
    }
    return true;
  };
  auto sendChunks = [&]() -> bool {
    size_t done = 0;
    while (done < n) {
      size_t chunk = std::min(kChunk, n - done);
      uint32_t cnt = static_cast<uint32_t>(chunk);
      if (!writeAll(&cnt, 4)) return false;
      for (size_t i = 0; i < chunk; ++i) {
        for (int c = 0; c < 52; ++c) deck[c] = static_cast<Card>(c);
        for (int i2 = 0; i2 < 7; ++i2) {
          int j = i2 + static_cast<int>(rng.nextBelow(52 - i2));
          std::swap(deck[i2], deck[j]);
        }
        HandValue hv = evaluate7(deck);
        hands.emplace_back(handKey(hv), 0);
        for (int j = 0; j < 7; ++j) out[i * 7 + j] = deck[j];
      }
      if (!writeAll(out.data(), chunk * 7)) return false;
      if (!readAll(in.data(), chunk * 8)) return false;
      for (size_t i = 0; i < chunk; ++i) {
        uint64_t s = 0;
        for (int b = 0; b < 8; ++b) s |= static_cast<uint64_t>(in[i * 8 + b]) << (8 * b);
        hands[done + i].second = s;
      }
      done += chunk;
    }
    uint32_t zero = 0;
    return writeAll(&zero, 4);
  };
  if (!sendChunks()) {
    std::fprintf(stderr, "io error talking to child\n");
    closeChild(child);
    return 2;
  }
  closeChild(child);

  std::sort(hands.begin(), hands.end());
  size_t violations = 0;
  size_t ties = 0;
  for (size_t i = 1; i < n; ++i) {
    const auto& a = hands[i - 1];
    const auto& b = hands[i];
    if (a.first == b.first) {
      ++ties;
      if (a.second != b.second) {
        if (violations < 5) {
          std::fprintf(stderr, "TIE MISMATCH: same key, strengths %llu vs %llu\n",
                       (unsigned long long)a.second, (unsigned long long)b.second);
        }
        ++violations;
      }
    } else if (a.second > b.second) {
      if (violations < 5) {
        std::fprintf(stderr, "ORDER MISMATCH: keys %llu < %llu but strengths "
                             "%llu > %llu\n",
                     (unsigned long long)a.first, (unsigned long long)b.first,
                     (unsigned long long)a.second, (unsigned long long)b.second);
      }
      ++violations;
    }
  }
  std::printf("eval7 cross-check: %zu hands, %zu equal-key pairs, "
              "%zu violations -> %s\n",
              n, ties, violations, violations == 0 ? "OK" : "FAIL");
  return violations == 0 ? 0 : 1;
}
