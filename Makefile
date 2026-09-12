# cuda-poker-solver
# CPU build (default). For the optional CUDA build use CMake with
# -DENABLE_CUDA=ON on a machine with nvcc and an NVIDIA GPU.
CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?=
THREADFLAGS := -pthread

SRCS := src/cards.cpp src/common.cpp src/eval.cpp src/fgs.cpp src/hand169.cpp \
        src/icm.cpp src/poker.cpp src/showdown.cpp src/range.cpp \
        src/postflop_cfr.cpp
HEADERS := $(wildcard src/*.h)
BUILD := build

.PHONY: all test verify-pfs stub-bias river-parity river-quality river-baseline river-bench clean

all: $(BUILD)/ppsolve $(BUILD)/pfflop

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/ppsolve: $(SRCS) src/main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -o $@ $(SRCS) src/main.cpp $(LDFLAGS)

$(BUILD)/pfflop: $(SRCS) src/pfflop_main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -o $@ $(SRCS) src/pfflop_main.cpp $(LDFLAGS)

$(BUILD)/tests: $(SRCS) tests/test_main.cpp tests/framework.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -Itests -o $@ $(SRCS) tests/test_main.cpp $(LDFLAGS)

test: $(BUILD)/tests
	./$(BUILD)/tests

$(BUILD)/verify_eval7: tools/verify_eval7.cpp $(SRCS) $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -o $@ tools/verify_eval7.cpp src/cards.cpp src/common.cpp

# Cross-validation of the hand evaluator against b-inary/postflop-solver
# over N random 7-card hands (requires cargo). See tools/pfs-verify/.
verify-pfs: $(BUILD)/verify_eval7
	tools/pfs-verify/check_eval7.sh

# Measures the FGS continuation stub's EV bias against a full postflop
# solve (see tools/pfs-verify/stub_bias.sh).
stub-bias:
	tools/pfs-verify/stub_bias.sh

# Head-to-head parity of the range-based river solver against
# b-inary/postflop-solver (requires cargo).
river-parity:
	tools/pfs-verify/check_river_parity.sh

# Golden-baseline quality gate (no oracle needed) + monotone convergence.
river-quality:
	tools/quality/river_quality.sh

# Regenerate the golden baseline after an intentional quality change.
river-baseline:
	tools/quality/river_quality.sh --generate

# River solver performance benchmark.
river-bench:
	tools/quality/bench_river.sh

clean:
	rm -rf $(BUILD)
