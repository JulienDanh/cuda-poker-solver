#pragma once
// Minimal test harness: register tests with TEST(name), run via main().
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct TestRegistry {
  std::vector<std::pair<std::string, std::function<void()>>> tests;
  static TestRegistry& get() {
    static TestRegistry r;
    return r;
  }
};

#define TEST(name)                                                                 \
  static void test_##name();                                                      \
  struct Register_##name {                                                        \
    Register_##name() {                                                           \
      TestRegistry::get().tests.push_back({#name, test_##name});                   \
    }                                                                              \
  } register_##name;                                                              \
  static void test_##name()

#define CHECK(cond)                                                               \
  do {                                                                             \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      throw std::runtime_error("check failed");                                      \
    }                                                                              \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                      \
  do {                                                                             \
    double da = (a), db = (b);                                                     \
    if (da > db + (tol) || da < db - (tol)) {                                       \
      std::fprintf(stderr, "CHECK_NEAR failed at %s:%d: %f vs %f (tol %f)\n",       \
                   __FILE__, __LINE__, da, db, (double)(tol));                      \
      throw std::runtime_error("check_near failed");                                 \
    }                                                                              \
  } while (0)

inline int runAllTests() {
  int failed = 0;
  for (auto& [name, fn] : TestRegistry::get().tests) {
    std::printf("[ RUN ] %s\n", name.c_str());
    try {
      fn();
      std::printf("[ OK  ] %s\n", name.c_str());
    } catch (const std::exception& e) {
      std::printf("[FAIL ] %s: %s\n", name.c_str(), e.what());
      ++failed;
    } catch (...) {
      std::printf("[FAIL ] %s\n", name.c_str());
      ++failed;
    }
  }
  std::printf("\n%d test(s) failed\n", failed);
  return failed ? 1 : 0;
}
