#!/usr/bin/env python3
"""Collect RAW PROD microkernel benchmark results into one flat CSV.

Two kinds of inputs are merged:

  * Combined benches (``vunary``, ``vbinary``, ``rminmax``, ``rsum``, ``rdsum``,
    ``rdminmax``) run the *entire* registered set in one JSON each. They have
    built-in problem sizes and embed the full kernel symbol in the benchmark
    name, e.g. ``vbinary/xnn_f16_vadd_ukernel__avx512fp16_u64/N:8192/real_time``.
    We keep only rows whose kernel is PROD (from the .bzl PROD lists).

  * Shape benches (gemm/igemm/dwconv/...) are already PROD-filtered upstream by
    scripts/bench-compare-f16-f32-prod.sh; we read their manifest.csv and copy
    every benchmark row through, tagging it PROD.

Output columns:
  backend,bench,precision,family,isa,gen,tile_or_size,variant,kernel,shape,
  real_time_ns,cpu_time_ns,iterations,source
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "prod_kernel_list", REPO_ROOT / "scripts" / "prod-kernel-list.py"
)
pkl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pkl)


def _canon(s: str) -> str:
    """Canonical form for matching bench symbols against source stems.

    Both the benchmark symbol (``xnn_f16_vadd_ukernel__avx512fp16_u64``) and the
    source stem (``f16-vadd-avx512fp16-u64``) reduce to the same string
    ``f16_vadd_avx512fp16_u64``.

    The ``minmax`` token is dropped: reduction sources encode it in the path
    (``f32-rdsum-7p7x-minmax-wasmsimd-u16``) while the emitted symbol omits it
    (``..._7p7x__wasmsimd_u16``). Stripping it from both sides keeps them equal,
    and is harmless for families that carry ``minmax`` in both (e.g. rminmax).
    """
    s = s.replace("-", "_")
    if s.startswith("xnn_"):
        s = s[len("xnn_"):]
    s = s.replace("_ukernel__", "_").replace("_ukernel_", "_")
    s = s.replace("_minmax_", "_")
    if s.endswith("_minmax"):
        s = s[: -len("_minmax")]
    s = re.sub(r"_+", "_", s)
    return s


def _norm_symbol(sym: str) -> str:
    """Normalize a benchmark kernel symbol (see :func:`_canon`)."""
    return _canon(sym)


def prod_stems(backend: str) -> list[tuple[str, dict]]:
    """Return [(normalized_stem, prod_row)] for the requested backend.

    Longest stems first so exact/most-specific matches win during prefix
    matching (a vbinary stem ``f16_vadd_avx512fp16_u64`` before a hypothetical
    shorter one).
    """
    isas = pkl.backend_isas(backend)
    out: list[tuple[str, dict]] = []
    for r in pkl.collect(REPO_ROOT / "gen", isas):
        stem = _canon(pkl.stem_of(r["source"]))
        out.append((stem, r))
    out.sort(key=lambda kv: len(kv[0]), reverse=True)
    return out


def match_prod(norm: str, stems: list[tuple[str, dict]]) -> dict | None:
    """Return the PROD row whose stem matches this normalized symbol.

    A source file may generate several unroll variants (e.g.
    ``f32-vabs-avx512f.c`` -> ``..._u16/_u32/_u48``); such stems are a prefix
    of the symbol. Others encode the unroll in the filename and match exactly.
    The ``_`` boundary after the stem keeps ``avx512f`` from matching
    ``avx512fp16`` and prevents cross-family bleed.
    """
    for stem, row in stems:
        if norm == stem or norm.startswith(stem + "_"):
            return row
    return None


def _kernel_symbol_from_name(name: str) -> str | None:
    """Extract the kernel symbol from a benchmark name.

    Combined benches embed an ``xnn_`` symbol segment, e.g.
    ``vbinary/xnn_f16_vadd_ukernel__avx512fp16_u64/N:8192/real_time``.
    Shape/gemm-style benches instead put the (non-``xnn_``) symbol first, e.g.
    ``f32_gemm_minmax_ukernel_7x16__avx512f_broadcast/M:.../real_time``.
    """
    parts = name.split("/")
    for part in parts:
        # Some benches prefix the symbol (e.g. "BM_xnn_f32_dwconv_...").
        idx = part.find("xnn_")
        if idx >= 0:
            return part[idx:]
    # Fall back to the first segment when it looks like a kernel symbol
    # (contains a ukernel/ISA marker) rather than a family tag like "vbinary".
    first = parts[0] if parts else ""
    if "ukernel" in first or "__" in first:
        return first
    return None


def _shape_from_name(name: str) -> str:
    """Return the size/shape suffix of a benchmark name (after the symbol)."""
    segs = name.split("/")
    keep = [s for s in segs[1:] if not s.startswith("xnn_") and s != "real_time"
            and s != "cpu_time" and s != "manual_time"]
    return " ".join(keep)


def _bench_label(stem: str) -> str:
    """Derive a clean bench label from a JSON file stem.

    ``native_vbinary`` -> ``vbinary``; ``native_f32-gemm_49x1024x1024`` ->
    ``f32-gemm`` (the trailing ``_<shape>`` token is dropped, the shape lives in
    its own column).
    """
    b = stem.split("_", 1)[-1]
    return re.sub(r"_[0-9]+(x[0-9]+)+$", "", b)


def collect_combined(json_paths: list[Path], backend: str,
                     stems: list[tuple[str, dict]],
                     writer: csv.writer) -> tuple[int, int]:
    kept = seen = 0
    for jp in json_paths:
        bench = _bench_label(jp.stem)  # native_vbinary -> vbinary
        try:
            data = json.loads(jp.read_text())
        except json.JSONDecodeError:
            print(f"WARN: skipping malformed JSON {jp}", file=sys.stderr)
            continue
        for b in data.get("benchmarks", []):
            if b.get("run_type") == "aggregate":
                continue
            name = b.get("name", "")
            sym = _kernel_symbol_from_name(name)
            if sym is None:
                continue
            seen += 1
            key = _norm_symbol(sym)
            row = match_prod(key, stems)
            if row is None:
                continue
            kept += 1
            stem = _canon(pkl.stem_of(row["source"]))
            variant = key[len(stem):].lstrip("_") or (row["variant"] or "")
            writer.writerow([
                backend, bench, row["precision"], row["family"], row["isa"],
                row["gen"], row["tile"] or "", variant, sym,
                _shape_from_name(name),
                f"{b.get('real_time', '')}", f"{b.get('cpu_time', '')}",
                b.get("iterations", ""), row["source"],
            ])
    return kept, seen


def collect_shape(manifest: Path, backend: str, writer: csv.writer) -> int:
    kept = 0
    with manifest.open() as f:
        for m in csv.DictReader(f):
            jp = Path(m["json_path"])
            if not jp.exists():
                continue
            data = json.loads(jp.read_text())
            for b in data.get("benchmarks", []):
                if b.get("run_type") == "aggregate":
                    continue
                name = b.get("name", "")
                sym = name.split("/")[0]
                # tile is the token before "__", isa after.
                isa = sym.split("__")[-1] if "__" in sym else ""
                tile = ""
                mt = re.search(r"_(\d+x\d+[a-z0-9]*|\d+p\d+c)__", sym)
                if mt:
                    tile = mt.group(1)
                writer.writerow([
                    backend, m["op"], m["precision"], m["op"], isa,
                    pkl.isa_generation(isa), tile, "", sym, m["shape"],
                    f"{b.get('real_time', '')}", f"{b.get('cpu_time', '')}",
                    b.get("iterations", ""), "",
                ])
                kept += 1
    return kept


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["x86", "wasm"], default="x86",
                   help="Which PROD ISA set to match against (default: x86).")
    p.add_argument("--backend-tag", default="native",
                   help="Value written in the 'backend' column (default: native).")
    p.add_argument("--combined-dir", type=Path, required=True,
                   help="Directory holding native_<bench>.json combined outputs.")
    p.add_argument("--shape-manifest", type=Path,
                   help="manifest.csv from bench-compare-f16-f32-prod.sh (optional).")
    p.add_argument("--all-benches", action="store_true",
                   help="Process every *_*.json in --combined-dir (not just the "
                        "six combined families). Kernel symbols are PROD-matched "
                        "the same way; benches without a ukernel symbol are "
                        "skipped.")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    args = p.parse_args(argv)

    stems = prod_stems(args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "bench", "precision", "family", "isa", "gen",
                    "tile_or_size", "variant", "kernel", "shape",
                    "real_time_ns", "cpu_time_ns", "iterations", "source"])
        combined = sorted(args.combined_dir.glob("*_*.json"))
        if not args.all_benches:
            combined = [c for c in combined if c.stem.split("_", 1)[-1] in
                        ("vunary", "vbinary", "rminmax", "rsum", "rdsum",
                         "rdminmax")]
        ckept, cseen = collect_combined(combined, args.backend_tag, stems, w)
        skept = 0
        if args.shape_manifest and args.shape_manifest.exists():
            skept = collect_shape(args.shape_manifest, args.backend_tag, w)

    print(f"combined: kept {ckept} PROD of {cseen} rows; shape: {skept} rows")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
