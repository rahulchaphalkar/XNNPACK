#!/usr/bin/env bash
# Run the REMAINING production XNNPACK microkernels (native x86 + wasm) that are
# not part of the already-logged combined families, and grow the raw JSON log.
#
# Strategy (uniform with the combined runs): run each bench FULL (no PROD filter
# at runtime), capture JSON, and PROD-filter later in scripts/prod-collect-raw.py.
#
# Native benches write JSON via --benchmark_out. Wasm benches are JS-shell
# modules run under d8: they cannot persist files, so we capture stdout and
# strip the d8 preamble with scripts/extract-bench-json.py.
#
# Designed to be launched DETACHED (setsid/nohup) so it survives SSH close.
set -u

REPO="/home/gta/repos/XNNPACK"
cd "$REPO"

D8="/home/gta/repos/v8-build/release_vtune/d8"
NATIVE_DIR="build/native/bench"
WASM_DIR="build/wasm/bench"
MIN_TIME="0.05s"
EXTRACT="$REPO/scripts/extract-bench-json.py"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="build/bench-results/prod-remaining/$STAMP"
NATIVE_OUT="$OUT/native"
WASM_OUT="$OUT/wasm"
mkdir -p "$NATIVE_OUT" "$WASM_OUT"
LOG="$OUT/detached.log"
echo "$STAMP" > /tmp/prod_remaining_stamp

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- bench inventories --------------------------------------------------------
# Built-in benches (run with NO args) not already logged in the combined runs.
BUILTIN=(
  xN-transposec xx-transposev indirection x8-lut
  f32-softmax f32-raddexpminusmax f32-raddextexp
  f32-raddstoreexpminusmax f16-raddstoreexpminusmax
  f32-vscaleextexp f32-vscaleexpminusmax f32-vcmul
)

# Shape families -> shapes. Kinds: mnk / igemm / dwconv / conv.
SHAPES_MNK=( "12544 64 64" "3136 128 256" "784 256 512" "49 1024 1024" )
SHAPES_IGEMM=( "112 112 3 3 1 1 1 1 64 64" "56 56 3 3 1 1 1 1 128 128" \
               "28 28 3 3 1 1 1 1 256 256" "14 14 3 3 1 1 1 1 512 512" )
SHAPES_DWCONV=( "112 112 3 3 1 1 1 1 64" "56 56 3 3 1 1 1 1 128" \
                "28 28 3 3 1 1 1 1 256" "14 14 3 3 1 1 1 1 512" )
SHAPES_CONV=( "224 224 32" "192 192 32" "128 128 64" "96 96 128" )

MNK_BENCHES=(
  f32-gemm f32-gemm-minmax f16-gemm f16-gemm-minmax f16-f32acc-gemm
  f32-bgemm bf16-gemm f32-qc8w-gemm f32-qc4w-gemm
  qs8-gemm qs8-qc8w-gemm-fp32 qs8-qc4w-gemm-fp32
  qu8-gemm qu8-gemm-fp32 qu8-gemm-rndnu
  qd8-f32-qc8w-gemm qd8-f32-qc4w-gemm qd8-f32-qc2w-gemm qd8-f32-qb4w-gemm
  qd8-f16-qc8w-gemm qd8-f16-qc4w-gemm qd8-f16-qc2w-gemm qd8-f16-qb4w-gemm
  qp8-f32-qc8w-gemm qp8-f32-qc4w-gemm qp8-f32-qb4w-gemm
  pf32-gemm-minmax pf16-gemm-minmax
)
IGEMM_BENCHES=( f32-igemm f16-igemm f16-f32acc-igemm )
DWCONV_BENCHES=( f32-dwconv f16-dwconv qs8-dwconv f32-dwconv2d-chw f16-dwconv2d-chw )
CONV_BENCHES=( f32-conv-hwc2chw f16-conv-hwc2chw f32-conv-hwc )

shapes_for() {
  case "$1" in
    mnk)    printf '%s\n' "${SHAPES_MNK[@]}" ;;
    igemm)  printf '%s\n' "${SHAPES_IGEMM[@]}" ;;
    dwconv) printf '%s\n' "${SHAPES_DWCONV[@]}" ;;
    conv)   printf '%s\n' "${SHAPES_CONV[@]}" ;;
  esac
}

# ---- native runners -----------------------------------------------------------
run_native_builtin() {
  local b="$1" exe="$NATIVE_DIR/$b-bench"
  [[ -x "$exe" ]] || { log "SKIP native builtin $b (no binary)"; return; }
  local out="$NATIVE_OUT/native_${b}.json"
  log "native builtin $b"
  "$exe" --benchmark_format=json --benchmark_min_time="$MIN_TIME" \
         --benchmark_out="$out" --benchmark_out_format=json \
         >/dev/null 2>>"$LOG" || log "  WARN native $b exit $?"
}

run_native_shape() {
  local b="$1" kind="$2" exe="$NATIVE_DIR/$b-bench"
  [[ -x "$exe" ]] || { log "SKIP native shape $b (no binary)"; return; }
  local idx=0 shape
  while IFS= read -r shape; do
    idx=$((idx+1))
    # shellcheck disable=SC2086
    local n; n="$("$exe" $shape --benchmark_list_tests 2>/dev/null | grep -c '/')"
    if [[ "${n:-0}" -eq 0 ]]; then
      log "  native $b shape[$idx] '$shape' -> 0 tests, skip"
      continue
    fi
    local safe="${shape// /x}"
    local out="$NATIVE_OUT/native_${b}_${safe}.json"
    log "native shape $b [$idx] '$shape' ($n tests)"
    # shellcheck disable=SC2086
    "$exe" $shape --benchmark_format=json --benchmark_min_time="$MIN_TIME" \
           --benchmark_out="$out" --benchmark_out_format=json \
           >/dev/null 2>>"$LOG" || log "  WARN native $b exit $?"
  done < <(shapes_for "$kind")
}

# ---- wasm runners (d8, capture stdout, strip preamble) ------------------------
run_wasm_builtin() {
  local b="$1" js="$b-bench.js"
  [[ -f "$WASM_DIR/$js" ]] || { log "SKIP wasm builtin $b (no js)"; return; }
  local out="$WASM_OUT/wasm_${b}.json"
  log "wasm builtin $b"
  ( cd "$WASM_DIR" && "$D8" --experimental-wasm-fp16 "$js" -- \
        --benchmark_format=json --benchmark_min_time="$MIN_TIME" ) 2>>"$LOG" \
    | python3 "$EXTRACT" > "$out" || { log "  WARN wasm $b no JSON"; rm -f "$out"; }
}

run_wasm_shape() {
  local b="$1" kind="$2" js="$b-bench.js"
  [[ -f "$WASM_DIR/$js" ]] || { log "SKIP wasm shape $b (no js)"; return; }
  local idx=0 shape
  while IFS= read -r shape; do
    idx=$((idx+1))
    local safe="${shape// /x}"
    local out="$WASM_OUT/wasm_${b}_${safe}.json"
    log "wasm shape $b [$idx] '$shape'"
    # shellcheck disable=SC2086
    ( cd "$WASM_DIR" && "$D8" --experimental-wasm-fp16 "$js" -- $shape \
          --benchmark_format=json --benchmark_min_time="$MIN_TIME" ) 2>>"$LOG" \
      | python3 "$EXTRACT" > "$out" \
      || { log "  wasm $b shape[$idx] no JSON, skip"; rm -f "$out"; }
    [[ -s "$out" ]] || rm -f "$out"
  done < <(shapes_for "$kind")
}

# ---- drive --------------------------------------------------------------------
log "START prod-remaining stamp=$STAMP out=$OUT"

log "=== NATIVE built-in ==="
for b in "${BUILTIN[@]}"; do run_native_builtin "$b"; done
log "=== NATIVE shape ==="
for b in "${MNK_BENCHES[@]}";    do run_native_shape "$b" mnk;    done
for b in "${IGEMM_BENCHES[@]}";  do run_native_shape "$b" igemm;  done
for b in "${DWCONV_BENCHES[@]}"; do run_native_shape "$b" dwconv; done
for b in "${CONV_BENCHES[@]}";   do run_native_shape "$b" conv;   done

log "=== WASM built-in ==="
for b in "${BUILTIN[@]}"; do run_wasm_builtin "$b"; done
log "=== WASM shape ==="
for b in "${MNK_BENCHES[@]}";    do run_wasm_shape "$b" mnk;    done
for b in "${IGEMM_BENCHES[@]}";  do run_wasm_shape "$b" igemm;  done
for b in "${DWCONV_BENCHES[@]}"; do run_wasm_shape "$b" dwconv; done
for b in "${CONV_BENCHES[@]}";   do run_wasm_shape "$b" conv;   done

log "DONE prod-remaining stamp=$STAMP"
log "native JSONs: $(ls "$NATIVE_OUT" | wc -l)  wasm JSONs: $(ls "$WASM_OUT" | wc -l)"
touch "$OUT/.complete"
