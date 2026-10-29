# WebAssembly MobileNet FP32 vs FP16 benchmarking

Build the subgraph MobileNet benchmark (`bench/subgraph/mobilenet.cc`) for
WebAssembly with relaxed-SIMD FP16 enabled, run it under `d8`, and compare the
FP32 and FP16 variants of MobileNetV1, V2, V3-Large and V3-Small.

## Prerequisites

- Emscripten SDK (`emcmake` on `PATH`, or `EMSDK=/path/to/emsdk`).
- A `d8` built from a V8 with WebAssembly FP16 support; pass
  `--experimental-wasm-fp16` (default in the run script).
- Branch `wasm-fp16-mobilenet` = `f16-raddstoreexpminusmax` (WASM f16 softmax
  kernel, needed by V2/V3) + the 7 commits of PR #11185 (`gemm-config-fix`:
  WASM `f16_gemm`/`igemm` dispatch, kernel rename to `wasmrelaxedsimdfp16_splat`,
  scalar fallback) + a 2-line `src/xnnpack/hardware-config.h` change that marks
  WASM FP16 builds as FP16-compatible. Without the last one `xnn_create_runtime`
  rejects `XNN_FLAG_FORCE_FP16_INFERENCE` in `src/subgraph.c`, and on plain
  `master` every `FP16MobileNet*` fails with `failed to create runtime`.

## Build

```bash
scripts/build-wasm-mobilenet.sh            # -> build/wasm/bench/subgraph/subgraph-mobilenet-bench.{js,wasm}
```

Key CMake settings (see the script): `-DXNNPACK_ENABLE_WASMRELAXEDSIMDFP16=ON`
(default is OFF), `XNNPACK_BUILD_TESTS=ON` (benchmarks need gtest), and linker
flags `-sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=256MB -sENVIRONMENT=shell`.
Only the `subgraph-mobilenet-bench` target is built.

## Run and compare

```bash
scripts/run-wasm-mobilenet-f16-f32.sh --d8 /path/to/d8 --repeat 5 --threads 1
```

Prints a table of median wall time per model with the `FP32/FP16` ratio
(> 1 means FP16 is faster) and writes `results.json`, `summary.csv`,
`manifest.txt` (d8 version, git rev, flags) and `stderr.txt` under
`build/bench-results/wasm-mobilenet/<timestamp>/`.

Useful options: `--min-time 2s` (longer runs), `--filter 'FP(16|32)MobileNetV2'`
(single model), `--d8-flags "--experimental-wasm-fp16 --no-liftoff"`,
`--build-dir`/`--out` to point at other directories.

## Manual run

```bash
cd build/wasm/bench/subgraph
/path/to/d8 --experimental-wasm-fp16 subgraph-mobilenet-bench.js -- \
  --benchmark_filter='FP(16|32)MobileNet' --num_threads=1 --benchmark_min_time=1s
```

Run from the benchmark directory so `d8` finds the sibling `.wasm`. Use
`--benchmark_format=json` and capture stdout; `--benchmark_out=` does not
persist from the `d8` shell filesystem.

## Notes

- FP16 inference is selected with `XNN_FLAG_FORCE_FP16_INFERENCE`; the FP16
  variants reuse the FP32 model definitions and convert at runtime creation.
- If any `FP16MobileNet*` reports `failed to create runtime`, an f16 config is
  missing for an op on this build. Check `xnn_is_f16_compatible_config()` in
  `src/xnnpack/hardware-config.h` and the `XNN_ARCH_WASMRELAXEDSIMDFP16`
  branches in `src/configs/*-config.c`; new dispatch entries require
  `python3 tools/update-microkernels.py` to move kernels into the prod list.
- Under a `d8` without hardware FP16 lowering, FP16 runs are expected to be
  slower than FP32 (the local smoke test showed ~2.5-3x slower).
