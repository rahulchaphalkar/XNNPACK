#!/usr/bin/env python3
"""Summarize f32 vs f16 benchmark runs produced by bench-compare-f16-f32.sh."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.csv produced by bench-compare-f16-f32.sh",
    )
    parser.add_argument(
        "--metric",
        choices=["real_time", "cpu_time"],
        default="real_time",
        help="Google Benchmark metric to compare (default: real_time)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output CSV path for summary rows",
    )
    parser.add_argument(
        "--pairing",
        choices=["best", "same-name", "same-name-gen", "same-gen"],
        default="best",
        help=(
            "How to pair f32 vs f16 kernels: 'best' compares the fastest kernel "
            "of each precision per shape; 'same-name' compares kernels that share "
            "the same MRxNR/tile token per shape; 'same-name-gen' compares kernels "
            "that share BOTH the same tile token AND the same hardware generation "
            "(e.g. f32 fma3 vs f16 avx2 at 256-bit), emitting one row per shared "
            "(tile, generation) so 256-bit f32/f16 pairs appear even when f32 also "
            "has a faster avx512 kernel; 'same-gen' compares the fastest kernel of "
            "each precision within the same hardware generation REGARDLESS of tile "
            "(best for PROD kernels, where f16 and f32 deliberately use different "
            "tile shapes, e.g. f16 8x32 avx512fp16 vs f32 7x16 avx512f) (default: best)"
        ),
    )
    return parser.parse_args()


def is_aggregate_name(name: str) -> bool:
    aggregate_suffixes = ("_mean", "_median", "_stddev", "_cv")
    return name.endswith(aggregate_suffixes)


def iter_entries_from_json(json_path: Path, metric: str):
    """Yield (benchmark_name, value) for every usable entry in a JSON file."""
    data = json.loads(json_path.read_text())
    for entry in data.get("benchmarks", []):
        name = str(entry.get("name", ""))
        if is_aggregate_name(name):
            continue
        if metric not in entry:
            continue
        # Skip benchmarks that were skipped/errored (e.g. unsupported on this
        # arch): Google Benchmark emits them with iterations == 0 and time 0.
        if float(entry.get("iterations", 0)) <= 0:
            continue
        value = float(entry[metric])
        if value <= 0:
            continue
        yield name, value


def iter_entries_from_text(text_path: Path, metric: str):
    """Yield (benchmark_name, value) for every usable line in a text file."""
    # Example line format from Google Benchmark text output:
    # name   5899832 ns   5899903 ns   119 FLOPS=...
    line_re = re.compile(
        r"^(?P<name>\S+)\s+(?P<real>[0-9]+(?:\.[0-9]+)?)\s+ns\s+"
        r"(?P<cpu>[0-9]+(?:\.[0-9]+)?)\s+ns\s+(?P<iters>[0-9]+)\b"
    )
    for line in text_path.read_text().splitlines():
        match = line_re.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        if is_aggregate_name(name):
            continue
        if int(match.group("iters")) <= 0:
            continue
        value = float(match.group("real" if metric == "real_time" else "cpu"))
        if value <= 0:
            continue
        yield name, value


def entries_for(json_path: Path, txt_path: Path, metric: str) -> list[tuple[str, float]]:
    """Return all usable (name, value) entries, preferring JSON over text."""
    if json_path.is_file():
        return list(iter_entries_from_json(json_path, metric))
    if txt_path and txt_path.is_file():
        return list(iter_entries_from_text(txt_path, metric))
    raise FileNotFoundError(
        f"Missing both JSON and text benchmark outputs: json={json_path}, txt={txt_path}"
    )


def tile_token(name: str) -> str:
    """Extract the microkernel tile descriptor from a benchmark name.

    The Google Benchmark name looks like
    ``f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast/M:49/N:1024/...``.
    The tile descriptor (``7x64``, ``3p8c``, ``1x8c8`` ...) is the last
    ``_``-separated token before the ISA/variant marked by ``__``.
    """
    base = name.split("/", 1)[0]
    head = base.split("__", 1)[0]
    return head.rsplit("_", 1)[-1]


def tile_sort_key(tile: str) -> tuple[int, int, str]:
    nums = re.findall(r"\d+", tile)
    if not nums:
        return (10**9, 10**9, tile)
    if len(nums) == 1:
        return (int(nums[0]), 0, tile)
    return (int(nums[0]), int(nums[1]), tile)


def isa_token(name: str) -> str:
    """Extract the ISA/variant token (everything after ``__``) from a name.

    ``f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast/M:...`` -> ``avx512fp16_broadcast``.
    Returns an empty string when there is no ``__`` marker.
    """
    base = name.split("/", 1)[0]
    return base.split("__", 1)[1] if "__" in base else ""


# Order matters: the first substring that matches wins. ``avx512`` must be
# checked before the 256-bit tokens (``avx256`` contains ``avx2``).
_GENERATION_RULES = (
    ("avx512", "avx512"),   # 512-bit: avx512f / avx512fp16 / avx512skx / avx512vnni(+gfni)
    ("avx256", "avx2"),     # 256-bit VNNI (avx256vnni / avx256vnnigfni)
    ("avxvnni", "avx2"),    # 256-bit VNNI (avxvnni)
    ("avx2", "avx2"),       # 256-bit: avx2 / avx2_madd
    ("fma3", "avx2"),       # 256-bit: FMA3 is the AVX2-generation f32 kernel
    ("avx", "avx2"),        # 256-bit: plain AVX (no FMA)
    ("wasmrelaxedsimd", "wasmsimd"),  # 128-bit wasm SIMD128 (relaxed superset)
    ("wasmsimd", "wasmsimd"),         # 128-bit wasm SIMD128
    ("sse", "sse"),         # 128-bit SSE family
    ("neonfp16", "neonfp16"),
    ("neon", "neon"),
    ("rvv", "rvv"),
    ("hvx", "hvx"),
    ("scalar", "scalar"),
)


def isa_generation(isa: str) -> str:
    """Map an ISA token to a hardware-generation class.

    Groups kernels by the register/ISA generation so that same-generation f32
    and f16 kernels can be paired (e.g. f32 ``fma3`` and f16 ``avx2`` are both
    the 256-bit AVX2 generation). Returns ``other`` for unrecognised tokens.
    """
    s = isa.lower()
    for needle, gen in _GENERATION_RULES:
        if needle in s:
            return gen
    return "other"



def shape_sort_key(shape: str) -> tuple[int, int, int]:
    parts = shape.strip().split()
    if len(parts) != 3:
        return (10**9, 10**9, 10**9)
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return (10**9, 10**9, 10**9)


def read_manifest(manifest_path: Path) -> list[tuple[dict, Path, Path]]:
    """Parse the manifest into (row, resolved_json_path, resolved_txt_path)."""
    rows: list[tuple[dict, Path, Path]] = []
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "backend",
            "op",
            "precision",
            "shape",
            "repeat",
            "json_path",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        for row in reader:
            json_path = Path(row["json_path"]).expanduser()
            if str(json_path) and not json_path.is_absolute():
                json_path = (manifest_path.parent / json_path).resolve()

            txt_path = Path(row.get("stdout_path", "")).expanduser()
            if str(txt_path) and not txt_path.is_absolute():
                txt_path = (manifest_path.parent / txt_path).resolve()

            rows.append((row, json_path, txt_path))
    return rows


BEST_HEADER = [
    "backend",
    "op",
    "shape",
    "metric",
    "f32_median",
    "f16_median",
    "speedup_f32_over_f16",
    "f32_samples",
    "f16_samples",
    "f32_fastest_kernel",
    "f16_fastest_kernel",
]

SAME_NAME_HEADER = [
    "backend",
    "op",
    "shape",
    "tile",
    "metric",
    "f32_median",
    "f16_median",
    "speedup_f32_over_f16",
    "f32_samples",
    "f16_samples",
    "f32_kernel",
    "f16_kernel",
]

SAME_NAME_GEN_HEADER = [
    "backend",
    "op",
    "shape",
    "tile",
    "gen",
    "metric",
    "f32_median",
    "f16_median",
    "speedup_f32_over_f16",
    "f32_samples",
    "f16_samples",
    "f32_isa",
    "f16_isa",
    "f32_kernel",
    "f16_kernel",
]

SAME_GEN_HEADER = [
    "backend",
    "op",
    "shape",
    "gen",
    "metric",
    "f32_median",
    "f16_median",
    "speedup_f32_over_f16",
    "f32_samples",
    "f16_samples",
    "f32_tile",
    "f16_tile",
    "f32_isa",
    "f16_isa",
    "f32_kernel",
    "f16_kernel",
]


def summarize_best(rows: list[tuple[dict, Path, Path]], metric: str) -> list[dict]:
    """Compare the single fastest kernel of each precision per (backend, op, shape)."""
    run_groups: dict[tuple[str, str, str, str], list[tuple[float, str]]] = defaultdict(list)
    for row, json_path, txt_path in rows:
        best_value = math.inf
        best_name = ""
        for name, value in entries_for(json_path, txt_path, metric):
            if value < best_value:
                best_value = value
                best_name = name
        if not math.isfinite(best_value):
            raise ValueError(f"No usable benchmark values for row: {row}")
        key = (row["backend"], row["op"], row["shape"], row["precision"])
        run_groups[key].append((best_value, best_name))

    out: list[dict] = []
    index_keys = {(k[0], k[1], k[2]) for k in run_groups.keys()}
    for backend, op, shape in sorted(index_keys, key=lambda x: (x[0], x[1], shape_sort_key(x[2]))):
        f32 = run_groups.get((backend, op, shape, "f32"), [])
        f16 = run_groups.get((backend, op, shape, "f16"), [])
        f32_samples = [v for v, _ in f32]
        f16_samples = [v for v, _ in f16]
        if not f32_samples or not f16_samples:
            continue

        f32_med = median(f32_samples)
        f16_med = median(f16_samples)
        speedup = f32_med / f16_med if f16_med > 0 else math.nan

        out.append(
            {
                "backend": backend,
                "op": op,
                "shape": shape,
                "metric": metric,
                "f32_median": f"{f32_med:.4f}",
                "f16_median": f"{f16_med:.4f}",
                "speedup_f32_over_f16": f"{speedup:.4f}",
                "f32_samples": str(len(f32_samples)),
                "f16_samples": str(len(f16_samples)),
                "f32_fastest_kernel": min(f32, key=lambda x: x[0])[1],
                "f16_fastest_kernel": min(f16, key=lambda x: x[0])[1],
            }
        )
    return out


def summarize_same_name(rows: list[tuple[dict, Path, Path]], metric: str) -> list[dict]:
    """Compare kernels sharing the same tile token per (backend, op, shape).

    For each precision and tile, the fastest microkernel carrying that tile
    token is selected (its median across repeats), then f32 and f16 are paired
    on identical tile tokens. Tiles present in only one precision are skipped.
    """
    # key (backend, op, shape, precision, tile) -> kernel_name -> per-file values
    samples: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, json_path, txt_path in rows:
        # Within a single output file, reduce duplicate names (e.g. parameter
        # sweeps) to that file's best value for the (tile, kernel) pair.
        per_file: dict[tuple[str, str], float] = {}
        for name, value in entries_for(json_path, txt_path, metric):
            kernel = name.split("/", 1)[0]
            slot = (tile_token(name), kernel)
            if slot not in per_file or value < per_file[slot]:
                per_file[slot] = value
        for (tile, kernel), value in per_file.items():
            key = (row["backend"], row["op"], row["shape"], row["precision"], tile)
            samples[key][kernel].append(value)

    # Reduce each (backend, op, shape, precision, tile) to its fastest kernel.
    best: dict[tuple, tuple[float, str, int]] = {}
    for key, kernels in samples.items():
        best_med = math.inf
        best_kernel = ""
        best_n = 0
        for kernel, vals in kernels.items():
            med = median(vals)
            if med < best_med:
                best_med, best_kernel, best_n = med, kernel, len(vals)
        best[key] = (best_med, best_kernel, best_n)

    index = {(b, o, s, t) for (b, o, s, _p, t) in best.keys()}
    out: list[dict] = []
    for backend, op, shape, tile in sorted(
        index, key=lambda x: (x[0], x[1], shape_sort_key(x[2]), tile_sort_key(x[3]))
    ):
        f32 = best.get((backend, op, shape, "f32", tile))
        f16 = best.get((backend, op, shape, "f16", tile))
        if not f32 or not f16:
            continue
        f32_med, f32_kernel, f32_n = f32
        f16_med, f16_kernel, f16_n = f16
        speedup = f32_med / f16_med if f16_med > 0 else math.nan

        out.append(
            {
                "backend": backend,
                "op": op,
                "shape": shape,
                "tile": tile,
                "metric": metric,
                "f32_median": f"{f32_med:.4f}",
                "f16_median": f"{f16_med:.4f}",
                "speedup_f32_over_f16": f"{speedup:.4f}",
                "f32_samples": str(f32_n),
                "f16_samples": str(f16_n),
                "f32_kernel": f32_kernel,
                "f16_kernel": f16_kernel,
            }
        )
    return out


def summarize_same_name_gen(rows: list[tuple[dict, Path, Path]], metric: str) -> list[dict]:
    """Compare kernels sharing the same tile AND hardware generation per shape.

    For each precision, tile, and hardware generation (avx512 / avx2-256 / sse /
    wasmsimd / neon / scalar ...), the fastest microkernel in that bucket is
    selected (its median across repeats). f32 and f16 are then paired on every
    shared (tile, generation), so a 256-bit ``f32 fma3`` vs ``f16 avx2`` row is
    emitted even when f32 also has a faster avx512 kernel at that tile. Buckets
    present in only one precision are skipped.
    """
    # key (backend, op, shape, precision, tile, gen) -> kernel_name -> per-file values
    samples: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, json_path, txt_path in rows:
        # Within a single output file, reduce duplicate names to that file's best.
        per_file: dict[tuple[str, str, str], float] = {}
        for name, value in entries_for(json_path, txt_path, metric):
            kernel = name.split("/", 1)[0]
            slot = (tile_token(name), isa_generation(isa_token(name)), kernel)
            if slot not in per_file or value < per_file[slot]:
                per_file[slot] = value
        for (tile, gen, kernel), value in per_file.items():
            key = (row["backend"], row["op"], row["shape"], row["precision"], tile, gen)
            samples[key][kernel].append(value)

    # Reduce each (backend, op, shape, precision, tile, gen) to its fastest kernel.
    best: dict[tuple, tuple[float, str, int]] = {}
    for key, kernels in samples.items():
        best_med = math.inf
        best_kernel = ""
        best_n = 0
        for kernel, vals in kernels.items():
            med = median(vals)
            if med < best_med:
                best_med, best_kernel, best_n = med, kernel, len(vals)
        best[key] = (best_med, best_kernel, best_n)

    index = {(b, o, s, t, g) for (b, o, s, _p, t, g) in best.keys()}
    out: list[dict] = []
    for backend, op, shape, tile, gen in sorted(
        index, key=lambda x: (x[0], x[1], shape_sort_key(x[2]), tile_sort_key(x[3]), x[4])
    ):
        f32 = best.get((backend, op, shape, "f32", tile, gen))
        f16 = best.get((backend, op, shape, "f16", tile, gen))
        if not f32 or not f16:
            continue
        f32_med, f32_kernel, f32_n = f32
        f16_med, f16_kernel, f16_n = f16
        speedup = f32_med / f16_med if f16_med > 0 else math.nan

        out.append(
            {
                "backend": backend,
                "op": op,
                "shape": shape,
                "tile": tile,
                "gen": gen,
                "metric": metric,
                "f32_median": f"{f32_med:.4f}",
                "f16_median": f"{f16_med:.4f}",
                "speedup_f32_over_f16": f"{speedup:.4f}",
                "f32_samples": str(f32_n),
                "f16_samples": str(f16_n),
                "f32_isa": isa_token(f32_kernel),
                "f16_isa": isa_token(f16_kernel),
                "f32_kernel": f32_kernel,
                "f16_kernel": f16_kernel,
            }
        )
    return out


def summarize_same_gen(rows: list[tuple[dict, Path, Path]], metric: str) -> list[dict]:
    """Compare the fastest kernel of each precision within the same generation.

    Buckets kernels by (backend, op, shape, precision, hardware generation) and
    picks the fastest kernel per bucket (its median across repeats), IGNORING
    the tile token. f32 and f16 are paired on every shared generation.

    This is the right comparison for production-selected (PROD) kernels, where
    the f16 and f32 kernels for the same op+generation are deliberately shaped
    differently (e.g. f16 gemm uses 8x32 ``avx512fp16`` while f32 gemm uses 7x16
    ``avx512f``). Same-tile pairing would discard these because the tiles differ;
    same-gen keeps them because they are the same hardware generation and are the
    kernels actually dispatched in production. The paired tiles are reported in
    the ``f32_tile`` / ``f16_tile`` columns.
    """
    # key (backend, op, shape, precision, gen) -> kernel_name -> per-file values
    samples: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, json_path, txt_path in rows:
        per_file: dict[tuple[str, str], float] = {}
        for name, value in entries_for(json_path, txt_path, metric):
            kernel = name.split("/", 1)[0]
            slot = (isa_generation(isa_token(name)), kernel)
            if slot not in per_file or value < per_file[slot]:
                per_file[slot] = value
        for (gen, kernel), value in per_file.items():
            key = (row["backend"], row["op"], row["shape"], row["precision"], gen)
            samples[key][kernel].append(value)

    # Reduce each (backend, op, shape, precision, gen) to its fastest kernel.
    best: dict[tuple, tuple[float, str, int]] = {}
    for key, kernels in samples.items():
        best_med = math.inf
        best_kernel = ""
        best_n = 0
        for kernel, vals in kernels.items():
            med = median(vals)
            if med < best_med:
                best_med, best_kernel, best_n = med, kernel, len(vals)
        best[key] = (best_med, best_kernel, best_n)

    index = {(b, o, s, g) for (b, o, s, _p, g) in best.keys()}
    out: list[dict] = []
    for backend, op, shape, gen in sorted(
        index, key=lambda x: (x[0], x[1], shape_sort_key(x[2]), x[3])
    ):
        f32 = best.get((backend, op, shape, "f32", gen))
        f16 = best.get((backend, op, shape, "f16", gen))
        if not f32 or not f16:
            continue
        f32_med, f32_kernel, f32_n = f32
        f16_med, f16_kernel, f16_n = f16
        speedup = f32_med / f16_med if f16_med > 0 else math.nan

        out.append(
            {
                "backend": backend,
                "op": op,
                "shape": shape,
                "gen": gen,
                "metric": metric,
                "f32_median": f"{f32_med:.4f}",
                "f16_median": f"{f16_med:.4f}",
                "speedup_f32_over_f16": f"{speedup:.4f}",
                "f32_samples": str(f32_n),
                "f16_samples": str(f16_n),
                "f32_tile": tile_token(f32_kernel),
                "f16_tile": tile_token(f16_kernel),
                "f32_isa": isa_token(f32_kernel),
                "f16_isa": isa_token(f16_kernel),
                "f32_kernel": f32_kernel,
                "f16_kernel": f16_kernel,
            }
        )
    return out


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)

    if args.pairing == "same-gen":
        summary_rows = summarize_same_gen(rows, args.metric)
        header = SAME_GEN_HEADER
    elif args.pairing == "same-name-gen":
        summary_rows = summarize_same_name_gen(rows, args.metric)
        header = SAME_NAME_GEN_HEADER
    elif args.pairing == "same-name":
        summary_rows = summarize_same_name(rows, args.metric)
        header = SAME_NAME_HEADER
    else:
        summary_rows = summarize_best(rows, args.metric)
        header = BEST_HEADER

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote summary CSV: {out_path}", file=sys.stderr)

    writer = csv.DictWriter(sys.stdout, fieldnames=header)
    writer.writeheader()
    writer.writerows(summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
