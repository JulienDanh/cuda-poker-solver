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

.PHONY: all test verify-pfs stub-bias gpu-quick gpu-parity turn-bench flop-bench perf-loop python python-test api api-test ui-test ui-logic-test clean

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

# Performance trackers (phase breakdown + iters/s).
turn-bench:
	tools/quality/bench_turn.sh

# One-command perf feedback loop: rebuild, gpu-quick gate, turn-bench,
# and a per-spot iters/s comparison against the previous run (history in
# build-cuda/.bench_last). Exits nonzero on any failure or regression.
perf-loop:
	tools/quality/perf_loop.sh

# Flop benchmark: bets 0.4 pot + all-in, raises 2.5x (see bench_flop.sh).
flop-bench:
	tools/quality/bench_flop.sh

# Python bindings (the pps package, python/): builds pps_native into
# python/pps/. Requires pybind11 + numpy in the active Python env
# (python -m pip install pybind11 numpy) and the CUDA toolkit.
PYTHON ?= $(shell command -v python3)
python:
	cmake -B build-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
	  -DPython3_EXECUTABLE=$(PYTHON) .
	cmake --build build-cuda --target pps_native -j8

# pps package tests (see python/test_pps.py): API invariants — labeled
# actions, per-combo normalization, EV aggregation, save/load and
# warm-start equivalence. Complements the oracle gates.
python-test: python
	$(PYTHON) python/test_pps.py

# Solver HTTP API (python/pps_api.py) + web UI: run the server on
# localhost:8070 (docs at /docs, UI at /ui). Requires the pps module
# (make python) plus fastapi + uvicorn + httpx in the Python env.
api: python
	cd python && $(PYTHON) -m uvicorn pps_api:app --host 127.0.0.1 --port 8070

# HTTP API tests (python/test_pps_api.py, in-process TestClient) —
# includes the served-UI and action-path checks.
api-test: python
	$(PYTHON) python/test_pps_api.py

# Alias: the UI's checks live in the api test suite (static serving +
# path labels). `make api` serves the UI at http://127.0.0.1:8070/ui/.
ui-test: api-test

# UI logic tests (python/test_pps_ui.py): runs ui/app.js inside a real
# JS engine (quickjs) with a DOM stub against real solver data — the
# 13x13 aggregation, freq/EV matrix rendering, class naming.
# Requires `pip install quickjs` (build needs a C compiler on PATH).
ui-logic-test:
	$(PYTHON) python/test_pps_ui.py

clean:
	rm -rf $(BUILD)
