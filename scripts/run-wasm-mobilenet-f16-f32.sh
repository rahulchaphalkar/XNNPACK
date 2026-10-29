#!/usr/bin/env bash
#
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Runs the WebAssembly subgraph MobileNet benchmark under d8 and prints an
# FP32 vs FP16 comparison table for every MobileNet variant.
#
# Usage: scripts/run-wasm-mobilenet-f16-f32.sh --d8 /path/to/d8 [options]
#   --d8 PATH          d8 binary (or set D8 env var)
#   --repeat N         benchmark repetitions per variant (default 5)
#   --threads N        --num_threads passed to the benchmark (default 1)
#   --min-time S       --benchmark_min_time per run, e.g. 1s (default 1s)
#   --filter REGEX     benchmark name filter (default 'FP(16|32)MobileNet')
#   --out DIR          results directory (default build/bench-results/wasm-mobilenet/<timestamp>)
#   --build-dir DIR    directory containing bench/subgraph/subgraph-mobilenet-bench.js (default build/wasm)
#   --d8-flags "..."   extra d8 flags (default '--experimental-wasm-fp16'; set D8_FLAGS env to override)

set -euo pipefail

D8="${D8:-}"
REPEAT=5
THREADS=1
MIN_TIME=1s
FILTER='FP(16|32)MobileNet'
OUT=""
BUILD_DIR="${BUILD_DIR:-build/wasm}"
D8_FLAGS="${D8_FLAGS:---experimental-wasm-fp16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --d8) D8="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --min-time) MIN_TIME="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --d8-flags) D8_FLAGS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${D8}" || ! -x "${D8}" ]]; then
  echo "d8 not found; pass --d8 /path/to/d8 or set D8" >&2
  exit 1
fi

BENCH_DIR="${BUILD_DIR}/bench/subgraph"
if [[ ! -f "${BENCH_DIR}/subgraph-mobilenet-bench.js" ]]; then
  echo "benchmark not built; run scripts/build-wasm-mobilenet.sh first" >&2
  exit 1
fi

OUT="${OUT:-build/bench-results/wasm-mobilenet/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
D8="$(cd "$(dirname "${D8}")" && pwd)/$(basename "${D8}")"

{
  echo "date: $(date -Iseconds)"
  echo "host: $(uname -a)"
  echo "git: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "d8: ${D8}"
  echo "d8 version: $("${D8}" -e 'print(version())' 2>/dev/null || true)"
  echo "d8 flags: ${D8_FLAGS}"
  echo "repeat: ${REPEAT} threads: ${THREADS} min_time: ${MIN_TIME} filter: ${FILTER}"
} | tee "${OUT}/manifest.txt"

# d8 resolves the sibling .wasm relative to the cwd, so run from the bench dir.
# JSON goes to stdout; V8/benchmark warnings go to stderr.
# shellcheck disable=SC2086
(cd "${BENCH_DIR}" && "${D8}" ${D8_FLAGS} subgraph-mobilenet-bench.js -- \
    --benchmark_filter="${FILTER}" \
    --benchmark_repetitions="${REPEAT}" \
    --benchmark_report_aggregates_only=false \
    --benchmark_min_time="${MIN_TIME}" \
    --num_threads="${THREADS}" \
    --benchmark_format=json) \
  > "${OUT}/results.json" 2> "${OUT}/stderr.txt"

python3 - "${OUT}/results.json" "${OUT}/summary.csv" <<'PY'
import json, statistics, sys

raw = open(sys.argv[1]).read()
data = json.loads(raw[raw.index('{'):])

runs = {}
for b in data["benchmarks"]:
    if b.get("run_type") != "iteration":
        continue
    name = b["name"].split("/")[0]
    runs.setdefault(name, []).append(b["real_time"])
errors = {b["name"].split("/")[0]: b["error_message"]
          for b in data["benchmarks"] if "error_message" in b}

models = sorted({n[4:] for n in runs} | {n[4:] for n in errors})
rows = []
for m in models:
    f32 = runs.get("FP32" + m)
    f16 = runs.get("FP16" + m)
    med32 = statistics.median(f32) if f32 else None
    med16 = statistics.median(f16) if f16 else None
    ratio = med32 / med16 if med32 and med16 else None
    rows.append((m, med32, med16, ratio, len(f32 or []), len(f16 or [])))

hdr = f"{'Model':<20}{'FP32 us':>12}{'FP16 us':>12}{'FP32/FP16':>11}{'n32':>5}{'n16':>5}"
print()
print(hdr)
print("-" * len(hdr))
fmt = lambda v, p: f"{v:.{p}f}" if v is not None else "ERROR"
for m, a, b, r, n32, n16 in rows:
    print(f"{m:<20}{fmt(a,1):>12}{fmt(b,1):>12}{fmt(r,3):>11}{n32:>5}{n16:>5}")
for name, msg in sorted(errors.items()):
    print(f"  {name}: {msg}")

with open(sys.argv[2], "w") as f:
    f.write("model,fp32_median_us,fp16_median_us,fp32_over_fp16,n_fp32,n_fp16\n")
    for m, a, b, r, n32, n16 in rows:
        f.write(f"{m},{a or ''},{b or ''},{r or ''},{n32},{n16}\n")
PY

echo
echo "Results: ${OUT}"
