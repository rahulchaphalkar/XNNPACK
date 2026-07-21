#!/usr/bin/env python3
"""Match f32 vs f16 PROD kernel results from the raw combined CSV.

Reads the flat ``prod-raw-combined.csv`` produced by ``prod-collect-raw.py`` and
emits, for every ``(backend, family, shape)`` where BOTH precisions ran, the
fastest f16 and fastest f32 kernel plus the f16-over-f32 speedup. Two kinds of
comparison are produced (see ``--mode``):

  * ``best``   -- absolute best of each precision across ALL generations. If f16
                  tops out at avx2 while f32 reaches avx512, this pairs the
                  avx512 f32 kernel against the avx2 f16 kernel (``f32_gen`` and
                  ``f16_gen`` may differ). Answers "fastest f32 I can ship vs
                  fastest f16 I can ship".
  * ``iso-gen``-- fastest of each precision WITHIN one generation bucket, emitted
                  only for gens where both precisions exist (``f32_gen`` ==
                  ``f16_gen``). Answers "at the same hardware generation, how do
                  the precisions compare".

f16 and f32 PROD kernels deliberately use different tile shapes / unrolls, so we
compare the fastest kernel of each precision (ignoring tile).

The ``f32acc-`` prefix on f16 kernels (e.g. ``f16-f32acc-vexp``) is stripped so
they pair with the corresponding f32 family (``f32-vexp``). Type-conversion
families (``*vcvt``) are excluded: they convert between types rather than
recomputing the same op at a different precision.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def norm_family(family: str) -> str:
    """Collapse a raw family name to its precision-agnostic form."""
    f = family
    if f.startswith("f32acc-"):
        f = f[len("f32acc-"):]
    return f


def is_conversion(family: str) -> bool:
    return "vcvt" in family


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True,
                    help="prod-raw-combined.csv from prod-collect-raw.py")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output matched CSV path.")
    ap.add_argument("--metric", choices=["real_time_ns", "cpu_time_ns"],
                    default="real_time_ns", help="Timing column to compare.")
    ap.add_argument("--mode", choices=["best", "iso-gen", "both"],
                    default="both",
                    help="best: absolute fastest f32 vs f16 across all gens "
                         "(gens may differ). iso-gen: fastest per precision "
                         "within each shared gen. both (default): emit both, "
                         "tagged in the 'comparison' column.")
    ap.add_argument("--include-conversions", action="store_true",
                    help="Also pair *vcvt conversion families (off by default).")
    args = ap.parse_args()

    # per_gen[(backend, nfamily, shape, gen)][prec] = list[(ns, kernel)]
    # all_gen[(backend, nfamily, shape)][prec]      = list[(ns, kernel, gen)]
    per_gen: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    all_gen: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with args.raw.open() as f:
        for row in csv.DictReader(f):
            fam = row["family"]
            if not args.include_conversions and is_conversion(fam):
                continue
            prec = row["precision"]
            if prec not in ("f16", "f32"):
                continue
            try:
                ns = float(row[args.metric])
            except (ValueError, KeyError):
                continue
            if ns <= 0:
                continue
            nfam = norm_family(fam)
            gen = row["gen"]
            shape = row["shape"]
            kernel = row["kernel"]
            per_gen[(row["backend"], nfam, shape, gen)][prec].append((ns, kernel))
            all_gen[(row["backend"], nfam, shape)][prec].append((ns, kernel, gen))

    def make_row(comparison, backend, fam, gen, shape,
                 f32_ns, f32_kernel, f32_gen, f16_ns, f16_kernel, f16_gen):
        return {
            "comparison": comparison,
            "backend": backend,
            "family": fam,
            "gen": gen,
            "shape": shape,
            "f32_gen": f32_gen,
            "f32_kernel": f32_kernel,
            "f32_ns": f"{f32_ns:.3f}",
            "f16_gen": f16_gen,
            "f16_kernel": f16_kernel,
            "f16_ns": f"{f16_ns:.3f}",
            "speedup_f16_over_f32": f"{f32_ns / f16_ns:.4f}",
        }

    rows = []
    if args.mode in ("best", "both"):
        for (backend, fam, shape), byprec in all_gen.items():
            if "f16" not in byprec or "f32" not in byprec:
                continue
            f16_ns, f16_kernel, f16_gen = min(byprec["f16"])
            f32_ns, f32_kernel, f32_gen = min(byprec["f32"])
            rows.append(make_row(
                "best", backend, fam, "best", shape,
                f32_ns, f32_kernel, f32_gen, f16_ns, f16_kernel, f16_gen))

    if args.mode in ("iso-gen", "both"):
        for (backend, fam, shape, gen), byprec in per_gen.items():
            if "f16" not in byprec or "f32" not in byprec:
                continue
            f16_ns, f16_kernel = min(byprec["f16"])
            f32_ns, f32_kernel = min(byprec["f32"])
            rows.append(make_row(
                "iso-gen", backend, fam, gen, shape,
                f32_ns, f32_kernel, gen, f16_ns, f16_kernel, gen))

    rows.sort(key=lambda r: (r["comparison"], r["backend"], r["family"],
                             r["gen"], r["shape"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "comparison", "backend", "family", "gen", "shape",
            "f32_gen", "f32_kernel", "f32_ns",
            "f16_gen", "f16_kernel", "f16_ns", "speedup_f16_over_f32"])
        w.writeheader()
        w.writerows(rows)

    # Report families that only ever ran in a single precision (never paired).
    paired_fams = {(r["backend"], r["family"]) for r in rows}
    fam_precs: dict[tuple, set] = defaultdict(set)
    for (backend, fam, _shape), byprec in all_gen.items():
        fam_precs[(backend, fam)] |= set(byprec)

    import statistics
    print(f"matched pairs: {len(rows)}  -> {args.out}")
    for comp in ("best", "iso-gen"):
        sub = [r for r in rows if r["comparison"] == comp]
        if not sub:
            continue
        sp = sorted(float(r["speedup_f16_over_f32"]) for r in sub)
        cross = sum(1 for r in sub if r["f32_gen"] != r["f16_gen"])
        extra = f"  cross-gen={cross}" if comp == "best" else ""
        print(f"  {comp:8} n={len(sub):4}  speedup f16/f32  min={sp[0]:.2f}  "
              f"median={statistics.median(sp):.2f}  max={sp[-1]:.2f}{extra}")

    unpaired = {k: v for k, v in fam_precs.items()
                if k not in paired_fams and len(v) == 1}
    if unpaired:
        print(f"\nfamilies present in ONE precision only (no pair): {len(unpaired)}")
        for (backend, fam), precs in sorted(unpaired.items()):
            print(f"  {backend:6} {fam:24} {'/'.join(sorted(precs))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
