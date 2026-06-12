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
    return parser.parse_args()


def is_aggregate_name(name: str) -> bool:
    aggregate_suffixes = ("_mean", "_median", "_stddev", "_cv")
    return name.endswith(aggregate_suffixes)


def best_time_from_json(json_path: Path, metric: str) -> tuple[float, str]:
    data = json.loads(json_path.read_text())
    entries = data.get("benchmarks", [])

    best_value = math.inf
    best_name = ""
    for entry in entries:
        name = str(entry.get("name", ""))
        if is_aggregate_name(name):
            continue
        if metric not in entry:
            continue
        value = float(entry[metric])
        if value < best_value:
            best_value = value
            best_name = name

    if not math.isfinite(best_value):
        raise ValueError(f"No usable benchmark values in {json_path}")

    return best_value, best_name


def best_time_from_text(text_path: Path, metric: str) -> tuple[float, str]:
    # Example line format from Google Benchmark text output:
    # name   5899832 ns   5899903 ns   119 FLOPS=...
    line_re = re.compile(
        r"^(?P<name>\S+)\s+(?P<real>[0-9]+(?:\.[0-9]+)?)\s+ns\s+"
        r"(?P<cpu>[0-9]+(?:\.[0-9]+)?)\s+ns\s+(?P<iters>[0-9]+)\b"
    )

    best_value = math.inf
    best_name = ""
    for line in text_path.read_text().splitlines():
        match = line_re.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        if is_aggregate_name(name):
            continue
        value = float(match.group("real" if metric == "real_time" else "cpu"))
        if value < best_value:
            best_value = value
            best_name = name

    if not math.isfinite(best_value):
        raise ValueError(f"No usable benchmark values in text output: {text_path}")

    return best_value, best_name


def shape_sort_key(shape: str) -> tuple[int, int, int]:
    parts = shape.strip().split()
    if len(parts) != 3:
        return (10**9, 10**9, 10**9)
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return (10**9, 10**9, 10**9)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    run_groups: dict[tuple[str, str, str, str], list[tuple[float, str]]] = defaultdict(list)

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
            if not json_path.is_absolute():
                json_path = (manifest_path.parent / json_path).resolve()

            txt_path = Path(row.get("stdout_path", "")).expanduser()
            if txt_path and not txt_path.is_absolute():
                txt_path = (manifest_path.parent / txt_path).resolve()

            if json_path.is_file():
                best_time, best_kernel = best_time_from_json(json_path, args.metric)
            elif txt_path.is_file():
                best_time, best_kernel = best_time_from_text(txt_path, args.metric)
            else:
                raise FileNotFoundError(
                    f"Missing both JSON and text benchmark outputs for row: json={json_path}, txt={txt_path}"
                )

            key = (row["backend"], row["op"], row["shape"], row["precision"])
            run_groups[key].append((best_time, best_kernel))

    rows = []
    index_keys = {(k[0], k[1], k[2]) for k in run_groups.keys()}
    for backend, op, shape in sorted(index_keys, key=lambda x: (x[0], x[1], shape_sort_key(x[2]))):
        f32_samples = [v for v, _ in run_groups.get((backend, op, shape, "f32"), [])]
        f16_samples = [v for v, _ in run_groups.get((backend, op, shape, "f16"), [])]
        if not f32_samples or not f16_samples:
            continue

        f32_med = median(f32_samples)
        f16_med = median(f16_samples)
        speedup = f32_med / f16_med if f16_med > 0 else math.nan

        f32_best_kernel = min(run_groups[(backend, op, shape, "f32")], key=lambda x: x[0])[1]
        f16_best_kernel = min(run_groups[(backend, op, shape, "f16")], key=lambda x: x[0])[1]

        rows.append(
            {
                "backend": backend,
                "op": op,
                "shape": shape,
                "f32_median": f"{f32_med:.4f}",
                "f16_median": f"{f16_med:.4f}",
                "speedup_f32_over_f16": f"{speedup:.4f}",
                "metric": args.metric,
                "f32_samples": str(len(f32_samples)),
                "f16_samples": str(len(f16_samples)),
                "f32_fastest_kernel": f32_best_kernel,
                "f16_fastest_kernel": f16_best_kernel,
            }
        )

    header = [
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

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary CSV: {out_path}")

    writer = csv.DictWriter(sys.stdout, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
