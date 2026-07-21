#!/usr/bin/env bash
#
# Runs paired f32/f16 XNNPACK microkernel benchmarks and stores raw JSON output.
# Supports native binaries and wasm JS runners (via d8, node, or another JS runtime).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BACKEND="both"
NATIVE_BUILD_DIR="${REPO_ROOT}/build/local"
WASM_BUILD_DIR="${REPO_ROOT}/build/wasm"
WASM_RUNTIME="d8"
WASM_RUNTIME_FLAGS=""
NUM_THREADS=1
MIN_ITERS=50
REPEATS=3
DRY_RUN=0

# Each shape entry is "M N K".
SHAPES=(
  "12544 64 64"
  "3136 128 256"
  "784 256 512"
  "49 1024 1024"
)

# IGEMM uses convolution-style shapes:
# input_h input_w kernel_h kernel_w pad_h pad_w stride dilation cin cout
IGEMM_SHAPES=(
  "112 112 3 3 1 1 1 1 64 64"
  "56 56 3 3 1 1 1 1 128 128"
  "28 28 3 3 1 1 1 1 256 256"
  "14 14 3 3 1 1 1 1 512 512"
)

# Depthwise convolution shapes (dwconv, dwconv2d-chw):
# input_h input_w kernel_h kernel_w pad_h pad_w subsampling dilation channels
DWCONV_SHAPES=(
  "112 112 3 3 1 1 1 1 64"
  "56 56 3 3 1 1 1 1 128"
  "28 28 3 3 1 1 1 1 256"
  "14 14 3 3 1 1 1 1 512"
)

# HWC->CHW convolution shapes (conv-hwc2chw):
# input_h input_w output_channels
CONV_HWC2CHW_SHAPES=(
  "224 224 32"
  "192 192 32"
  "128 128 64"
  "96 96 128"
)

# Supported operation groups.
# Name maps to binary stem without backend-specific extension.
declare -A OP_TO_PAIR
OP_TO_PAIR[gemm]="f32-gemm-bench:f16-gemm-bench"
OP_TO_PAIR[gemm_minmax]="f32-gemm-minmax-bench:f16-gemm-minmax-bench"
OP_TO_PAIR[igemm]="f32-igemm-bench:f16-igemm-bench"
OP_TO_PAIR[dwconv]="f32-dwconv-bench:f16-dwconv-bench"
OP_TO_PAIR[dwconv2d_chw]="f32-dwconv2d-chw-bench:f16-dwconv2d-chw-bench"
OP_TO_PAIR[conv_hwc2chw]="f32-conv-hwc2chw-bench:f16-conv-hwc2chw-bench"
OP_TO_PAIR[raddstoreexpminusmax]="f32-raddstoreexpminusmax-bench:f16-raddstoreexpminusmax-bench"
OP_TO_PAIR[vcmul]="f32-vcmul-bench:f16-vcmul-bench"
OP_TO_PAIR[qd8_qb4w_gemm]="qd8-f32-qb4w-gemm-bench:qd8-f16-qb4w-gemm-bench"
OP_TO_PAIR[qd8_qc2w_gemm]="qd8-f32-qc2w-gemm-bench:qd8-f16-qc2w-gemm-bench"
OP_TO_PAIR[qd8_qc4w_gemm]="qd8-f32-qc4w-gemm-bench:qd8-f16-qc4w-gemm-bench"
OP_TO_PAIR[qd8_qc8w_gemm]="qd8-f32-qc8w-gemm-bench:qd8-f16-qc8w-gemm-bench"

# Maps each op to the shape schema it consumes (used by shapes_for_op).
#   mnk          -> SHAPES (M N K)
#   igemm        -> IGEMM_SHAPES
#   dwconv       -> DWCONV_SHAPES
#   conv_hwc2chw -> CONV_HWC2CHW_SHAPES
#   builtin      -> benchmark uses its own fixed shape sweep (no shape args)
declare -A OP_TO_SHAPE_KIND
OP_TO_SHAPE_KIND[gemm]="mnk"
OP_TO_SHAPE_KIND[gemm_minmax]="mnk"
OP_TO_SHAPE_KIND[igemm]="igemm"
OP_TO_SHAPE_KIND[dwconv]="dwconv"
OP_TO_SHAPE_KIND[dwconv2d_chw]="dwconv"
OP_TO_SHAPE_KIND[conv_hwc2chw]="conv_hwc2chw"
OP_TO_SHAPE_KIND[raddstoreexpminusmax]="builtin"
OP_TO_SHAPE_KIND[vcmul]="builtin"
OP_TO_SHAPE_KIND[qd8_qb4w_gemm]="mnk"
OP_TO_SHAPE_KIND[qd8_qc2w_gemm]="mnk"
OP_TO_SHAPE_KIND[qd8_qc4w_gemm]="mnk"
OP_TO_SHAPE_KIND[qd8_qc8w_gemm]="mnk"

OPS=(
  gemm gemm_minmax igemm
  dwconv dwconv2d_chw conv_hwc2chw raddstoreexpminusmax vcmul
  qd8_qb4w_gemm qd8_qc2w_gemm qd8_qc4w_gemm qd8_qc8w_gemm
)

usage() {
  cat <<'EOF'
Usage:
  scripts/bench-compare-f16-f32.sh [options]

Options:
  --backend <native|wasm|both>         Backend(s) to run (default: both)
  --native-build-dir <path>            Native build dir (default: build/local)
  --wasm-build-dir <path>              Wasm build dir (default: build/wasm)
  --wasm-runtime <command>             JS runtime for wasm (default: d8)
  --wasm-runtime-flags <flags>         Extra flags passed before the JS file (e.g. --experimental-wasm-relaxed-simd for d8)
  --num-threads <int>                  Benchmark --num_threads value (default: 1)
  --min-iters <int>                    Benchmark --benchmark_min_iters (default: 50)
  --repeats <int>                      Number of repeated runs per shape (default: 3)
  --ops <csv>                          Ops from: gemm,gemm_minmax,igemm,
                                       dwconv,dwconv2d_chw,conv_hwc2chw,
                                       raddstoreexpminusmax,vcmul,
                                       qd8_qb4w_gemm,qd8_qc2w_gemm,
                                       qd8_qc4w_gemm,qd8_qc8w_gemm
  --shape "M N K"                      Add one shape triple, may be repeated
  --dry-run                            Print commands without running
  -h, --help                           Show this message

Examples:
  scripts/bench-compare-f16-f32.sh --backend native --num-threads 4
  scripts/bench-compare-f16-f32.sh --backend wasm --wasm-runtime d8 --ops gemm,igemm
  scripts/bench-compare-f16-f32.sh --backend native --ops dwconv,qd8_qc8w_gemm
EOF
}

log() {
  echo "[bench-compare] $*"
}

die() {
  echo "[bench-compare] ERROR: $*" >&2
  exit 1
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

is_supported_op() {
  local op="$1"
  [[ -n "${OP_TO_PAIR[$op]:-}" ]]
}

run_or_echo() {
  local cmd=("$@")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

run_one() {
  local backend="$1"
  local op="$2"
  local precision="$3"
  local exe_path="$4"
  local shape="$5"
  local repeat_idx="$6"
  local out_dir="$7"

  local safe_shape
  safe_shape="$(echo "$shape" | tr ' ' 'x')"
  local json_out="${out_dir}/${backend}_${op}_${precision}_${safe_shape}_r${repeat_idx}.json"
  local txt_out="${out_dir}/${backend}_${op}_${precision}_${safe_shape}_r${repeat_idx}.txt"
  local wasm_local_json="${backend}_${op}_${precision}_${safe_shape}_r${repeat_idx}.json"

  local shape_args=()
  if [[ "$shape" != "builtin" ]]; then
    # shellcheck disable=SC2206
    shape_args=($shape)
  fi

  local common_args=(
    "${shape_args[@]}"
    "--num_threads=${NUM_THREADS}"
    "--benchmark_min_iters=${MIN_ITERS}"
    "--benchmark_out=${json_out}"
    "--benchmark_out_format=json"
  )

  if [[ "$backend" == "native" ]]; then
    run_or_echo "$exe_path" "${common_args[@]}" >"$txt_out" 2>&1
  else
    # Some Emscripten benchmark wrappers reject absolute paths for --benchmark_out.
    common_args[${#common_args[@]}-2]="--benchmark_out=${wasm_local_json}"
    # d8 resolves .wasm files relative to cwd, not the script path.
    # Run from the directory containing the JS file so the .wasm sidecar is found.
    local js_dir
    js_dir="$(dirname "$exe_path")"
    local js_file
    js_file="./$(basename "$exe_path")"
    # d8 requires runtime flags before the script path, and -- before script args.
    local runtime_flags_arr=()
    if [[ -n "$WASM_RUNTIME_FLAGS" ]]; then
      # shellcheck disable=SC2206
      runtime_flags_arr=($WASM_RUNTIME_FLAGS)
    fi
    local sep=()
    if [[ "$(basename "$WASM_RUNTIME")" == "d8" ]]; then
      sep=("--")
    fi
    (cd "$js_dir" && run_or_echo "$WASM_RUNTIME" "${runtime_flags_arr[@]+${runtime_flags_arr[@]}}" "$js_file" "${sep[@]+${sep[@]}}" "${common_args[@]}") >"$txt_out" 2>&1
    if [[ "$DRY_RUN" -eq 0 && -f "${js_dir}/${wasm_local_json}" ]]; then
      mv -f "${js_dir}/${wasm_local_json}" "$json_out"
    fi
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -s "$json_out" ]]; then
      if [[ "$backend" == "wasm" && -s "$txt_out" ]]; then
        log "WASM run did not emit JSON output; keeping text output only for ${backend}/${op}/${precision}/${safe_shape}/r${repeat_idx}"
      else
        die "Missing benchmark JSON output: ${json_out}"
      fi
    fi
  fi

  echo "${backend},${op},${precision},${shape},${repeat_idx},${json_out},${txt_out}" >> "${out_dir}/manifest.csv"
}

shapes_for_op() {
  local op="$1"
  case "${OP_TO_SHAPE_KIND[$op]:-mnk}" in
    igemm)
      printf '%s\n' "${IGEMM_SHAPES[@]}"
      ;;
    dwconv)
      printf '%s\n' "${DWCONV_SHAPES[@]}"
      ;;
    conv_hwc2chw)
      printf '%s\n' "${CONV_HWC2CHW_SHAPES[@]}"
      ;;
    builtin)
      # Benchmark defines its own fixed shape sweep; run once with no shape args.
      printf '%s\n' "builtin"
      ;;
    *)
      printf '%s\n' "${SHAPES[@]}"
      ;;
  esac
}

validate_backend() {
  [[ "$BACKEND" == "native" || "$BACKEND" == "wasm" || "$BACKEND" == "both" ]] || die "Invalid --backend: ${BACKEND}"
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || die "${name} must be a positive integer"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backend)
        BACKEND="${2:-}"
        shift 2
        ;;
      --native-build-dir)
        NATIVE_BUILD_DIR="${2:-}"
        shift 2
        ;;
      --wasm-build-dir)
        WASM_BUILD_DIR="${2:-}"
        shift 2
        ;;
      --wasm-runtime)
        WASM_RUNTIME="${2:-}"
        shift 2
        ;;
      --wasm-runtime-flags)
        WASM_RUNTIME_FLAGS="${2:-}"
        shift 2
        ;;
      --num-threads)
        NUM_THREADS="${2:-}"
        shift 2
        ;;
      --min-iters)
        MIN_ITERS="${2:-}"
        shift 2
        ;;
      --repeats)
        REPEATS="${2:-}"
        shift 2
        ;;
      --ops)
        IFS=',' read -r -a OPS <<< "${2:-}"
        shift 2
        ;;
      --shape)
        SHAPES+=("${2:-}")
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

resolve_exe() {
  local backend="$1"
  local stem="$2"

  if [[ "$backend" == "native" ]]; then
    local path="${NATIVE_BUILD_DIR}/${stem}"
    [[ -x "$path" ]] || die "Native benchmark binary not found or not executable: ${path}"
    echo "$path"
  else
    local js_path="${WASM_BUILD_DIR}/bench/${stem}.js"
    [[ -f "$js_path" ]] || die "Wasm benchmark runner not found: ${js_path}"
    echo "$js_path"
  fi
}

main() {
  parse_args "$@"

  validate_backend
  validate_positive_int "--num-threads" "$NUM_THREADS"
  validate_positive_int "--min-iters" "$MIN_ITERS"
  validate_positive_int "--repeats" "$REPEATS"

  for op in "${OPS[@]}"; do
    is_supported_op "$op" || die "Unsupported op in --ops: ${op}"
  done

  if [[ "$BACKEND" == "wasm" || "$BACKEND" == "both" ]]; then
    command -v "$WASM_RUNTIME" >/dev/null 2>&1 || die "Wasm runtime not found in PATH: ${WASM_RUNTIME}"
  fi

  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local out_dir="${REPO_ROOT}/build/bench-results/f16-f32/${stamp}"
  mkdir -p "$out_dir"

  echo "backend,op,precision,shape,repeat,json_path,stdout_path" > "${out_dir}/manifest.csv"

  log "Backend=${BACKEND} Ops=$(join_by_comma "${OPS[@]}") Repeats=${REPEATS}"
  log "Results directory: ${out_dir}"

  local backends=()
  if [[ "$BACKEND" == "both" ]]; then
    backends=(native wasm)
  else
    backends=("$BACKEND")
  fi

  for backend in "${backends[@]}"; do
    for op in "${OPS[@]}"; do
      IFS=':' read -r f32_stem f16_stem <<< "${OP_TO_PAIR[$op]}"

      local f32_exe
      local f16_exe
      f32_exe="$(resolve_exe "$backend" "$f32_stem")"
      f16_exe="$(resolve_exe "$backend" "$f16_stem")"

      mapfile -t op_shapes < <(shapes_for_op "$op")
      for shape in "${op_shapes[@]}"; do
        for ((r = 1; r <= REPEATS; r++)); do
          log "Running ${backend} ${op} f32 shape=[${shape}] repeat=${r}"
          run_one "$backend" "$op" "f32" "$f32_exe" "$shape" "$r" "$out_dir"

          log "Running ${backend} ${op} f16 shape=[${shape}] repeat=${r}"
          run_one "$backend" "$op" "f16" "$f16_exe" "$shape" "$r" "$out_dir"
        done
      done
    done
  done

  log "Completed. Manifest: ${out_dir}/manifest.csv"
  log "Next step: post-process JSON in ${out_dir} to compute median speedups (f32/f16) per op+shape."
}

main "$@"
