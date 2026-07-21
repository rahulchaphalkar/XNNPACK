#!/usr/bin/env bash
# WASM-focused f16/f32 comparison with reduced shapes to avoid OOM in Node runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_BUILD_DIR="${REPO_ROOT}/build/wasm"
WASM_RUNTIME="${1:-$HOME/tools/emsdk/node/22.16.0_64bit/bin/node}"
REPEATS="${REPEATS:-3}"
MIN_ITERS="${MIN_ITERS:-50}"

OUT_DIR="${REPO_ROOT}/build/bench-results/f16-f32/$(date +%Y%m%d-%H%M%S)-wasm-small"
mkdir -p "${OUT_DIR}"

echo "backend,op,precision,shape,repeat,json_path,stdout_path" > "${OUT_DIR}/manifest.csv"

# Smaller GEMM shapes for WASM memory limits.
GEMM_SHAPES=(
  "256 64 64"
  "512 128 128"
  "1024 128 128"
  "2048 64 128"
)

# Smaller IGEMM shapes for WASM memory limits.
IGEMM_SHAPES=(
  "28 28 3 3 1 1 1 1 64 64"
  "14 14 3 3 1 1 1 1 128 128"
  "14 14 3 3 1 1 1 1 192 192"
  "7 7 3 3 1 1 1 1 256 256"
)

run_one() {
  local op="$1"
  local precision="$2"
  local exe="$3"
  local shape="$4"
  local repeat_idx="$5"

  local safe_shape
  safe_shape="$(echo "$shape" | tr ' ' 'x')"
  local json_out="${OUT_DIR}/wasm_${op}_${precision}_${safe_shape}_r${repeat_idx}.json"
  local txt_out="${OUT_DIR}/wasm_${op}_${precision}_${safe_shape}_r${repeat_idx}.txt"
  local wasm_local_json="$(basename "$json_out")"

  local shape_args=()
  # shellcheck disable=SC2206
  shape_args=($shape)

  local args=(
    "${shape_args[@]}"
    "--num_threads=1"
    "--benchmark_min_iters=${MIN_ITERS}"
    "--benchmark_out=${wasm_local_json}"
    "--benchmark_out_format=json"
  )

  local js_dir="${WASM_BUILD_DIR}/bench"
  local js_file="./${exe}.js"

  (cd "${js_dir}" && "${WASM_RUNTIME}" "${js_file}" "${args[@]}") >"${txt_out}" 2>&1

  if [[ -f "${js_dir}/${wasm_local_json}" ]]; then
    mv -f "${js_dir}/${wasm_local_json}" "${json_out}"
  else
    echo "Missing JSON output: ${json_out}" >&2
    exit 1
  fi

  echo "wasm,${op},${precision},${shape},${repeat_idx},${json_out},${txt_out}" >> "${OUT_DIR}/manifest.csv"
}

for op in gemm gemm_minmax igemm; do
  if [[ "${op}" == "gemm" ]]; then
    f32="f32-gemm-bench"
    f16="f16-gemm-bench"
    shapes=("${GEMM_SHAPES[@]}")
  elif [[ "${op}" == "gemm_minmax" ]]; then
    f32="f32-gemm-minmax-bench"
    f16="f16-gemm-minmax-bench"
    shapes=("${GEMM_SHAPES[@]}")
  else
    f32="f32-igemm-bench"
    f16="f16-igemm-bench"
    shapes=("${IGEMM_SHAPES[@]}")
  fi

  for shape in "${shapes[@]}"; do
    for ((r = 1; r <= REPEATS; r++)); do
      echo "[wasm-small] ${op} f32 shape=[${shape}] repeat=${r}"
      run_one "${op}" "f32" "${f32}" "${shape}" "${r}"
      echo "[wasm-small] ${op} f16 shape=[${shape}] repeat=${r}"
      run_one "${op}" "f16" "${f16}" "${shape}" "${r}"
    done
  done
done

echo "[wasm-small] completed: ${OUT_DIR}/manifest.csv"
