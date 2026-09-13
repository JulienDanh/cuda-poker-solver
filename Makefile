# cuda-poker-solver
# CPU build (default): the engine library + unit tests + the MCCFR CLI.
# The postflop solver is GPU-only: build it with CMake and
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

.PHONY: all test verify-pfs stub-bias gpu-quick gpu-parity turn-bench clean

all: $(BUILD)/ppsolve

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/ppsolve: $(SRCS) src/main.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -o $@ $(SRCS) src/main.cpp $(LDFLAGS)

$(BUILD)/tests: $(SRCS) tests/test_main.cpp tests/framework.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -Itests -o $@ $(SRCS) tests/test_main.cpp $(LDFLAGS)

test: $(BUILD)/tests
	./$(BUILD)/tests

# Hand-computed ground truth for a tiny turn spot (uniform + seeded
# profiles); compared against the GPU value walk by
# tools/quality/gpu_postflop_parity.sh.
$(BUILD)/tiny_check: tools/tiny_check.cpp $(SRCS) $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(THREADFLAGS) -Isrc -o $@ tools/tiny_check.cpp $(SRCS) $(LDFLAGS)

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

# GPU postflop solver quality gates (need the cmake CUDA build:
# cmake -B build-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89,
# plus cargo for the postflop-solver oracle).
# gpu-quick is the fast debug loop (~20 s): a river spot, a flop spot,
# turn spots and the tiny-turn ground truth at low iteration counts,
# with loose gates. gpu-parity is the full commit gate: all streets,
# tight gates, a few minutes.
gpu-quick:
	tools/quality/gpu_postflop_parity.sh quick

gpu-parity:
	tools/quality/gpu_postflop_parity.sh full

# Turn-solver performance tracker (phase breakdown + iters/s).
turn-bench:
	tools/quality/bench_turn.sh

clean:
	rm -rf $(BUILD)
