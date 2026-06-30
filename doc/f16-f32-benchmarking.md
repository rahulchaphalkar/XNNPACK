# F16 vs F32 Benchmarking (Native + WASM)

This guide provides a reproducible starting point for f16 vs f32 microkernel performance comparisons in XNNPACK.

## Scope of the starter harness

The script starts with ops that have clear f32/f16 benchmark pairs:

- gemm
- gemm_minmax
- igemm

You can extend this list later for additional ops once pair parity is confirmed.

## Build

Native:

```bash
./scripts/build-local.sh
```

WASM:

```bash
./scripts/build-wasm.sh
```

If you need to force relaxed SIMD FP16 support at configure time:

```bash
./scripts/build-wasm.sh -DXNNPACK_ENABLE_WASMRELAXEDSIMDFP16=ON
```

## Run comparisons

Native only:

```bash
./scripts/bench-compare-f16-f32.sh --backend native --num-threads 1 --min-iters 50 --repeats 3
```

WASM only (d8):

```bash
./scripts/bench-compare-f16-f32.sh --backend wasm --wasm-runtime d8 --num-threads 1 --min-iters 50 --repeats 3
```

Both backends:

```bash
./scripts/bench-compare-f16-f32.sh --backend both --wasm-runtime d8
```

Custom op subset:

```bash
./scripts/bench-compare-f16-f32.sh --backend native --ops gemm,igemm
```

Add custom shape triples (M N K):

```bash
./scripts/bench-compare-f16-f32.sh --backend native --shape "4096 256 256" --shape "1024 1024 1024"
```

Dry run to inspect exact commands:

```bash
./scripts/bench-compare-f16-f32.sh --backend both --wasm-runtime d8 --dry-run
```

## Output layout

Results are written under:

- `build/bench-results/f16-f32/<timestamp>/manifest.csv`
- `build/bench-results/f16-f32/<timestamp>/*.json`
- `build/bench-results/f16-f32/<timestamp>/*.txt`

`manifest.csv` records one row per run and includes backend, op, precision, shape, repeat, and output paths.

## Summarize speedups

Use the summary helper to compute median f32/f16 ratios per backend, op, and shape.

```bash
python3 ./scripts/summarize-f16-f32-results.py \
	--manifest build/bench-results/f16-f32/<timestamp>/manifest.csv \
	--metric real_time \
	--out build/bench-results/f16-f32/<timestamp>/summary.csv
```

The script reports:

- Median f32 and f16 times across repeats.
- `speedup_f32_over_f16` as `f32_median / f16_median`.
- The fastest kernel name observed for each precision.

## Notes for fair comparisons

- Use the same shape set and repeat count for f16 and f32.
- Keep `--num-threads` fixed across all runs.
- Start with single-thread (`--num-threads 1`) when comparing native with wasm runtime behavior.
- Run enough repeats (at least 3) and compare medians, not one-off values.
- Treat missing pairs as coverage gaps, not performance regressions.
