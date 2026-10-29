#!/usr/bin/env bash
#
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Builds the WebAssembly subgraph MobileNet benchmark (FP32/FP16/QS8 variants)
# with relaxed-SIMD FP16 enabled.
#
# Usage: scripts/build-wasm-mobilenet.sh [extra cmake args]
# Env:   EMSDK (optional) path to emsdk root; sourced if emcmake is not in PATH.
#        BUILD_DIR (default build/wasm)

set -e

if ! command -v emcmake >/dev/null 2>&1; then
  if [[ -n "${EMSDK:-}" && -f "${EMSDK}/emsdk_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${EMSDK}/emsdk_env.sh" >/dev/null 2>&1
  else
    echo "emcmake not found in PATH; set EMSDK=/path/to/emsdk or add emsdk to PATH" >&2
    exit 1
  fi
fi

BUILD_DIR="${BUILD_DIR:-build/wasm}"
mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  "-DXNNPACK_LIBRARY_TYPE=static"
  "-DXNNPACK_BUILD_BENCHMARKS=ON"
  # Benchmarks depend on gtest from the test tree.
  "-DXNNPACK_BUILD_TESTS=ON"
  "-DXNNPACK_ENABLE_WASMRELAXEDSIMDFP16=ON"
  "-DCMAKE_EXE_LINKER_FLAGS=-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sMAXIMUM_MEMORY=2147483648 -sENVIRONMENT=shell"
)
CMAKE_ARGS+=("$@")

(cd "${BUILD_DIR}" && emcmake cmake "$(git rev-parse --show-toplevel)" "${CMAKE_ARGS[@]}")

if [[ "$(uname)" == "Darwin" ]]; then
  JOBS="$(sysctl -n hw.ncpu)"
else
  JOBS="$(nproc)"
fi
cmake --build "${BUILD_DIR}" --target subgraph-mobilenet-bench -- "-j${JOBS}"

echo "Built: ${BUILD_DIR}/bench/subgraph/subgraph-mobilenet-bench.{js,wasm}"
