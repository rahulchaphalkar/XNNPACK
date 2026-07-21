#!/usr/bin/env bash
#
# Runs PRODUCTION-SELECTED f32/f16 XNNPACK microkernel benchmarks only.
#
# "Production-selected" means kernels listed in the PROD_<ISA>_MICROKERNEL_SRCS
# lists of gen/<isa>_microkernels.bzl, restricted to x86 and wasm ISAs. The set
# of kernels to run is derived at runtime by scripts/prod-kernel-list.py, which
# emits a Google Benchmark --benchmark_filter regex per op+precision. Only PROD
# kernels are executed; every other registered kernel is filtered out.
#
# This is a PROD-restricted variant of scripts/bench-compare-f16-f32.sh. Post-
# process the resulting manifest.csv with:
#   scripts/summarize-f16-f32-results.py --pairing same-name-gen ...
# which pairs f32 vs f16 by (tile, hardware-generation).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROD_LIST="${SCRIPT_DIR}/prod-kernel-list.py"

BACKEND="native"
NATIVE_BUILD_DIR="${REPO_ROOT}/build/native/bench"
WASM_BUILD_DIR="${REPO_ROOT}/build/wasm"
WASM_RUNTIME="d8"
WASM_RUNTIME_FLAGS=""
NUM_THREADS=1
MIN_ITERS=50
REPEATS=3
DRY_RUN=0
PYTHON_BIN="python3"

# Each shape entry is "M N K".
SHAPES=(
  "12544 64 64"
  "3136 128 256"
  "784 256 512"
  "49 1024 1024"
)

# IGEMM convolution-style shapes:
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

# HWC->CHW convolution shapes (conv-hwc2chw): input_h input_w output_channels
CONV_HWC2CHW_SHAPES=(
  "224 224 32"
  "192 192 32"
  "128 128 64"
  "96 96 128"
)

declare -A OP_TO_PAIR
OP_TO_PAIR[gemm]="f32-gemm-bench:f16-gemm-bench"
OP_TO_PAIR[gemm_minmax]="f32-gemm-minmax-bench:f16-gemm-minmax-bench"
OP_TO_PAIR[igemm]="f32-igemm-bench:f16-igemm-bench"
OP_TO_PAIR[dwconv]="f32-dwconv-bench:f16-dwconv-bench"
OP_TO_PAIR[dwconv2d_chw]="f32-dwconv2d-chw-bench:f16-dwconv2d-chw-bench"
OP_TO_PAIR[conv_hwc2chw]="f32-conv-hwc2chw-bench:f16-conv-hwc2chw-bench"
OP_TO_PAIR[raddstoreexpminusmax]="f32-raddstoreexpminusmax-bench:f16-raddstoreexpminusmax-bench"

declare -A OP_TO_SHAPE_KIND
OP_TO_SHAPE_KIND[gemm]="mnk"
OP_TO_SHAPE_KIND[gemm_minmax]="mnk"
OP_TO_SHAPE_KIND[igemm]="igemm"
OP_TO_SHAPE_KIND[dwconv]="dwconv"
OP_TO_SHAPE_KIND[dwconv2d_chw]="dwconv"
OP_TO_SHAPE_KIND[conv_hwc2chw]="conv_hwc2chw"
OP_TO_SHAPE_KIND[raddstoreexpminusmax]="builtin"

# Default op set: paired benches that can carry PROD kernels in both precisions.
# Ops with no PROD f16<->f32 overlap (or whose f16 bench does not register the
# PROD ISA, e.g. plain `gemm` on x86) self-skip at runtime via the precheck.
OPS=(
  gemm_minmax igemm
  dwconv dwconv2d_chw conv_hwc2chw raddstoreexpminusmax
)

usage() {
  cat <<'EOF'
Usage:
  scripts/bench-compare-f16-f32-prod.sh [options]

Runs only production-selected (PROD_*) x86/wasm microkernels, filtering each
benchmark binary to its PROD kernels for f32 and f16 separately.

Options:
  --backend <native|wasm>              Backend to run (default: native).
                                       native -> x86 PROD ISAs; wasm -> wasm PROD ISAs.
  --native-build-dir <path>            Native bench dir (default: build/native/bench)
  --wasm-build-dir <path>              Wasm build dir (default: build/wasm)
  --wasm-runtime <command>             JS runtime for wasm (default: d8)
  --wasm-runtime-flags <flags>         Extra flags before the JS file
  --num-threads <int>                  --num_threads (default: 1)
  --min-iters <int>                    --benchmark_min_iters (default: 50)
  --repeats <int>                      Repeats per shape (default: 3)
  --ops <csv>                          Subset of: gemm,gemm_minmax,igemm,dwconv,
                                       dwconv2d_chw,conv_hwc2chw,raddstoreexpminusmax
  --shape "M N K"                      Add one MNK shape, repeatable
  --python <bin>                       Python interpreter (default: python3)
  --dry-run                            Print commands and PROD filters, run nothing
  -h, --help                           Show this message

Examples:
  scripts/bench-compare-f16-f32-prod.sh --backend native --ops gemm_minmax,igemm
  scripts/bench-compare-f16-f32-prod.sh --backend native --dry-run
  scripts/bench-compare-f16-f32-prod.sh --backend wasm --wasm-runtime d8 \
      --wasm-runtime-flags --experimental-wasm-relaxed-simd
EOF
}

log()  { echo "[prod-compare] $*"; }
die()  { echo "[prod-compare] ERROR: $*" >&2; exit 1; }
join_by_comma() { local IFS=,; echo "$*"; }
is_supported_op() { [[ -n "${OP_TO_PAIR[$1]:-}" ]]; }

run_or_echo() {
  local cmd=("$@")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
}

# Backend -> prod-kernel-list backend name.
prod_backend() { [[ "$1" == "wasm" ]] && echo "wasm" || echo "x86"; }

# Emit the PROD --benchmark_filter for one op+precision, or empty if none.
prod_filter() {
  local pbackend="$1" op="$2" precision="$3"
  "$PYTHON_BIN" "$PROD_LIST" --backend "$pbackend" filter --op "$op" --precision "$precision" 2>/dev/null
}

# Count how many benchmarks a filter actually matches in a (native) binary for a
# representative shape. Used to skip ops whose f16 bench does not register the
# PROD ISA, or ISAs the running hardware does not support. Returns 0 on any
# error so the op is skipped rather than aborting the whole run. Native only.
count_matches() {
  local exe_path="$1" shape="$2" bench_filter="$3"
  local shape_args=()
  if [[ "$shape" != "builtin" ]]; then
    # shellcheck disable=SC2206
    shape_args=($shape)
  fi
  "$exe_path" "${shape_args[@]}" "--benchmark_filter=${bench_filter}" \
    --benchmark_list_tests 2>/dev/null | grep -c '/' || true
}

run_one() {
  local backend="$1" op="$2" precision="$3" exe_path="$4" shape="$5"
  local repeat_idx="$6" out_dir="$7" bench_filter="$8"

  local safe_shape; safe_shape="$(echo "$shape" | tr ' ' 'x')"
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
    "--benchmark_filter=${bench_filter}"
    "--num_threads=${NUM_THREADS}"
    "--benchmark_min_iters=${MIN_ITERS}"
    "--benchmark_out=${json_out}"
    "--benchmark_out_format=json"
  )

  if [[ "$backend" == "native" ]]; then
    run_or_echo "$exe_path" "${common_args[@]}" >"$txt_out" 2>&1
  else
    common_args[${#common_args[@]}-2]="--benchmark_out=${wasm_local_json}"
    local js_dir js_file
    js_dir="$(dirname "$exe_path")"
    js_file="./$(basename "$exe_path")"
    local runtime_flags_arr=()
    if [[ -n "$WASM_RUNTIME_FLAGS" ]]; then
      # shellcheck disable=SC2206
      runtime_flags_arr=($WASM_RUNTIME_FLAGS)
    fi
    local sep=()
    [[ "$(basename "$WASM_RUNTIME")" == "d8" ]] && sep=("--")
    (cd "$js_dir" && run_or_echo "$WASM_RUNTIME" "${runtime_flags_arr[@]+${runtime_flags_arr[@]}}" "$js_file" "${sep[@]+${sep[@]}}" "${common_args[@]}") >"$txt_out" 2>&1
    if [[ "$DRY_RUN" -eq 0 && -f "${js_dir}/${wasm_local_json}" ]]; then
      mv -f "${js_dir}/${wasm_local_json}" "$json_out"
    fi
  fi

  if [[ "$DRY_RUN" -eq 0 && ! -s "$json_out" ]]; then
    if [[ "$backend" == "wasm" && -s "$txt_out" ]]; then
      log "WASM run produced no JSON; keeping text only for ${op}/${precision}/${safe_shape}/r${repeat_idx}"
    else
      die "Missing benchmark JSON output: ${json_out}"
    fi
  fi

  echo "${backend},${op},${precision},${shape},${repeat_idx},${json_out},${txt_out}" >> "${out_dir}/manifest.csv"
}

shapes_for_op() {
  case "${OP_TO_SHAPE_KIND[$1]:-mnk}" in
    igemm)        printf '%s\n' "${IGEMM_SHAPES[@]}" ;;
    dwconv)       printf '%s\n' "${DWCONV_SHAPES[@]}" ;;
    conv_hwc2chw) printf '%s\n' "${CONV_HWC2CHW_SHAPES[@]}" ;;
    builtin)      printf '%s\n' "builtin" ;;
    *)            printf '%s\n' "${SHAPES[@]}" ;;
  esac
}

validate_backend() { [[ "$BACKEND" == "native" || "$BACKEND" == "wasm" ]] || die "Invalid --backend: ${BACKEND} (use native or wasm)"; }
validate_positive_int() { [[ "$2" =~ ^[0-9]+$ && "$2" -gt 0 ]] || die "$1 must be a positive integer"; }

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backend)            BACKEND="${2:-}"; shift 2 ;;
      --native-build-dir)   NATIVE_BUILD_DIR="${2:-}"; shift 2 ;;
      --wasm-build-dir)     WASM_BUILD_DIR="${2:-}"; shift 2 ;;
      --wasm-runtime)       WASM_RUNTIME="${2:-}"; shift 2 ;;
      --wasm-runtime-flags) WASM_RUNTIME_FLAGS="${2:-}"; shift 2 ;;
      --num-threads)        NUM_THREADS="${2:-}"; shift 2 ;;
      --min-iters)          MIN_ITERS="${2:-}"; shift 2 ;;
      --repeats)            REPEATS="${2:-}"; shift 2 ;;
      --ops)                IFS=',' read -r -a OPS <<< "${2:-}"; shift 2 ;;
      --shape)              SHAPES+=("${2:-}"); shift 2 ;;
      --python)             PYTHON_BIN="${2:-}"; shift 2 ;;
      --dry-run)            DRY_RUN=1; shift ;;
      -h|--help)            usage; exit 0 ;;
      *)                    die "Unknown argument: $1" ;;
    esac
  done
}

resolve_exe() {
  local backend="$1" stem="$2"
  if [[ "$backend" == "native" ]]; then
    local path="${NATIVE_BUILD_DIR}/${stem}"
    [[ -x "$path" ]] || return 1
    echo "$path"
  else
    local js_path="${WASM_BUILD_DIR}/bench/${stem}.js"
    [[ -f "$js_path" ]] || return 1
    echo "$js_path"
  fi
}

main() {
  parse_args "$@"
  validate_backend
  validate_positive_int "--num-threads" "$NUM_THREADS"
  validate_positive_int "--min-iters" "$MIN_ITERS"
  validate_positive_int "--repeats" "$REPEATS"
  [[ -f "$PROD_LIST" ]] || die "PROD kernel lister not found: ${PROD_LIST}"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: ${PYTHON_BIN}"

  for op in "${OPS[@]}"; do is_supported_op "$op" || die "Unsupported op in --ops: ${op}"; done

  if [[ "$BACKEND" == "wasm" ]]; then
    command -v "$WASM_RUNTIME" >/dev/null 2>&1 || die "Wasm runtime not found in PATH: ${WASM_RUNTIME}"
  fi

  local pbackend; pbackend="$(prod_backend "$BACKEND")"
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  local out_dir="${REPO_ROOT}/build/bench-results/f16-f32-prod/${stamp}"
  mkdir -p "$out_dir"
  echo "backend,op,precision,shape,repeat,json_path,stdout_path" > "${out_dir}/manifest.csv"

  # Snapshot the PROD kernel list used for this run for provenance.
  "$PYTHON_BIN" "$PROD_LIST" --backend "$pbackend" list > "${out_dir}/prod-kernels.csv" 2>/dev/null || true

  log "Backend=${BACKEND} (PROD ISAs=${pbackend}) Ops=$(join_by_comma "${OPS[@]}") Repeats=${REPEATS}"
  log "Results directory: ${out_dir}"

  for op in "${OPS[@]}"; do
    IFS=':' read -r f32_stem f16_stem <<< "${OP_TO_PAIR[$op]}"

    local f32_filter f16_filter
    f32_filter="$(prod_filter "$pbackend" "$op" f32)"
    f16_filter="$(prod_filter "$pbackend" "$op" f16)"

    if [[ -z "$f32_filter" || -z "$f16_filter" ]]; then
      log "SKIP ${op}: no PROD ${pbackend} kernels for $( [[ -z "$f32_filter" ]] && printf 'f32 ' )$( [[ -z "$f16_filter" ]] && printf 'f16' ) (need both precisions)."
      continue
    fi

    local f32_exe f16_exe
    if ! f32_exe="$(resolve_exe "$BACKEND" "$f32_stem")" || ! f16_exe="$(resolve_exe "$BACKEND" "$f16_stem")"; then
      log "SKIP ${op}: benchmark binary missing (looked for ${f32_stem} / ${f16_stem} under $( [[ "$BACKEND" == native ]] && echo "$NATIVE_BUILD_DIR" || echo "$WASM_BUILD_DIR/bench" )). Build it to include this op."
      continue
    fi

    mapfile -t op_shapes < <(shapes_for_op "$op")

    # Precheck (native only): confirm both precisions actually register PROD
    # kernels for this binary + hardware. Skips e.g. plain gemm on x86 (f16
    # bench lacks avx512fp16) or ISAs the CPU does not support.
    if [[ "$BACKEND" == "native" && "$DRY_RUN" -eq 0 ]]; then
      local probe_shape="${op_shapes[0]}"
      local n_f32 n_f16
      n_f32="$(count_matches "$f32_exe" "$probe_shape" "$f32_filter")"
      n_f16="$(count_matches "$f16_exe" "$probe_shape" "$f16_filter")"
      if [[ "$n_f32" -eq 0 || "$n_f16" -eq 0 ]]; then
        log "SKIP ${op}: PROD kernels not registered (f32 matches=${n_f32}, f16 matches=${n_f16}); check bench registration / CPU ISA support."
        continue
      fi
      log "${op}: PROD kernels registered (f32=${n_f32}, f16=${n_f16})."
    fi

    log "${op} PROD f32 filter: ${f32_filter}"
    log "${op} PROD f16 filter: ${f16_filter}"
    for shape in "${op_shapes[@]}"; do
      for ((r = 1; r <= REPEATS; r++)); do
        log "Running ${BACKEND} ${op} f32 shape=[${shape}] repeat=${r}"
        run_one "$BACKEND" "$op" "f32" "$f32_exe" "$shape" "$r" "$out_dir" "$f32_filter"
        log "Running ${BACKEND} ${op} f16 shape=[${shape}] repeat=${r}"
        run_one "$BACKEND" "$op" "f16" "$f16_exe" "$shape" "$r" "$out_dir" "$f16_filter"
      done
    done
  done

  log "Completed. Manifest: ${out_dir}/manifest.csv"
  log "PROD kernel snapshot: ${out_dir}/prod-kernels.csv"
  log "Next: scripts/summarize-f16-f32-results.py --pairing same-name-gen --manifest ${out_dir}/manifest.csv --output ${out_dir}/same-name-gen-summary.csv"
}

main "$@"
