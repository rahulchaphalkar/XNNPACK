#!/usr/bin/env python3
"""Generate an Excel inventory of XNNPACK x64 and portable microkernels."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


X64_C_BUCKETS = (
    "sse",
    "sse2",
    "sse2fma",
    "ssse3",
    "sse41",
    "avx",
    "f16c",
    "fma3",
    "avx2",
    "avxvnni",
    "avxvnniint8",
    "avx256skx",
    "avx256vnni",
    "avx256vnnigfni",
    "avx512f",
    "avx512skx",
    "avx512vbmi",
    "avx512vnni",
    "avx512vnnigfni",
    "avx512amx",
    "avx512fp16",
    "avx512bf16",
)
PORTABLE_C_BUCKETS = ("scalar",)
WASM_C_BUCKETS = ("wasmsimd", "wasmrelaxedsimd")
ASM_BUCKETS = ("amd64", "wasm32", "wasmsimd32", "wasmrelaxedsimd32")


@dataclass(frozen=True)
class GeneratorConstants:
    isas: frozenset[str]
    isa_map: dict[str, str]
    architectures: frozenset[str]
    symbol_pattern: str


@dataclass(frozen=True)
class ManifestEntry:
    source: str
    status: str
    manifest_bucket: str
    language: str


@dataclass(frozen=True)
class KernelRow:
    symbol: str
    family: str
    logical_operation: str
    source_directory: str
    source: str
    language: str
    generated: bool
    status: str
    target_class: str
    target_architecture: str
    manifest_bucket: str
    kernel_isa: str


@dataclass(frozen=True)
class ArchitectureInfo:
    name: str
    kind: str
    target_class: str
    target_architecture: str
    manifest_bucket: str
    capability_tier: str
    vector_width_bits: int | str
    fp16_mode: str
    build_condition: str
    notes: str


def _literal_collection(node: ast.AST) -> object:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset" and len(node.args) == 1:
            return frozenset(ast.literal_eval(node.args[0]))
    return ast.literal_eval(node)


def load_generator_constants(update_script: Path) -> GeneratorConstants:
    tree = ast.parse(update_script.read_text(encoding="utf-8"), filename=str(update_script))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in {"_ISA_LIST", "_ISA_MAP", "_ARCH_LIST"}:
            values[target.id] = _literal_collection(node.value)
        elif target.id == "_MICROKERNEL_NAME_REGEX":
            if not isinstance(node.value, ast.Call) or not node.value.args:
                raise ValueError("unexpected microkernel regex definition")
            values[target.id] = ast.literal_eval(node.value.args[0])

    required = {"_ISA_LIST", "_ISA_MAP", "_ARCH_LIST", "_MICROKERNEL_NAME_REGEX"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing generator constants: {', '.join(sorted(missing))}")
    return GeneratorConstants(
        isas=values["_ISA_LIST"],
        isa_map=values["_ISA_MAP"],
        architectures=values["_ARCH_LIST"],
        symbol_pattern=values["_MICROKERNEL_NAME_REGEX"],
    )


def load_build_conditions(build_params: Path) -> dict[str, str]:
    tree = ast.parse(build_params.read_text(encoding="utf-8"), filename=str(build_params))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "XNNPACK_PARAMS_FOR_ARCH":
            continue
        if not isinstance(node.value, ast.Dict):
            raise ValueError("XNNPACK_PARAMS_FOR_ARCH is not a dictionary")
        conditions: dict[str, str] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            name = ast.literal_eval(key_node)
            condition = "always"
            if isinstance(value_node, ast.Call):
                for keyword in value_node.keywords:
                    if keyword.arg == "cond":
                        condition = ast.literal_eval(keyword.value)
                        break
            conditions[name] = condition
        return conditions
    raise ValueError(f"XNNPACK_PARAMS_FOR_ARCH not found in {build_params}")


def parse_manifest(path: Path, bucket: str) -> list[ManifestEntry]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    entries: list[ManifestEntry] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.List):
            continue
        if target.id.startswith("NON_PROD_"):
            status = "NON_PROD"
        elif target.id.startswith("PROD_"):
            status = "PROD"
        else:
            continue
        language = "Assembly" if "_ASM_" in target.id else "C"
        for element in node.value.elts:
            source = ast.literal_eval(element)
            if not isinstance(source, str):
                raise ValueError(f"non-string source in {path}: {source!r}")
            entries.append(ManifestEntry(source, status, bucket, language))
    return entries


def selected_manifest_entries(repo_root: Path) -> list[ManifestEntry]:
    gen_dir = repo_root / "gen"
    entries: list[ManifestEntry] = []
    for bucket in X64_C_BUCKETS + PORTABLE_C_BUCKETS + WASM_C_BUCKETS + ASM_BUCKETS:
        manifest = gen_dir / f"{bucket}_microkernels.bzl"
        if not manifest.exists():
            raise FileNotFoundError(f"selected manifest does not exist: {manifest}")
        entries.extend(parse_manifest(manifest, bucket))
    return entries


def kernel_isa(source: str, constants: GeneratorConstants) -> str:
    stem = Path(source).stem
    components = stem.split("-")
    candidates = constants.isas | constants.architectures
    matches = [component for component in components if component in candidates]
    isa_matches = [component for component in matches if component in constants.isas]
    if isa_matches:
        return isa_matches[0]
    if matches:
        return matches[-1]
    raise ValueError(f"could not determine ISA from {source}")


def target_fields(bucket: str) -> tuple[str, str]:
    if bucket in X64_C_BUCKETS or bucket == "amd64":
        return "x64", "x86_64"
    if bucket in PORTABLE_C_BUCKETS:
        return "portable", "architecture-neutral"
    if bucket in WASM_C_BUCKETS or bucket in ASM_BUCKETS:
        return "WebAssembly", "wasm32"
    raise ValueError(f"unclassified manifest bucket: {bucket}")


def source_family(source: str) -> tuple[str, str]:
    parts = Path(source).parts
    if len(parts) < 3 or parts[0] != "src":
        raise ValueError(f"unexpected microkernel source path: {source}")
    return parts[1], str(Path(*parts[:2]))


def extract_symbols(source_path: Path, pattern: re.Pattern[str]) -> list[str]:
    content = source_path.read_text(encoding="utf-8")
    return sorted(set(pattern.findall(content)))


def logical_operation(symbol: str) -> str:
    prefix, separator, _ = symbol.partition("_ukernel")
    if not separator or not prefix.startswith("xnn_"):
        raise ValueError(f"could not determine logical operation from {symbol}")
    return prefix.removeprefix("xnn_").replace("_", "-")


def collect_kernel_rows(repo_root: Path) -> list[KernelRow]:
    constants = load_generator_constants(repo_root / "tools" / "update-microkernels.py")
    pattern = re.compile(constants.symbol_pattern)
    rows: list[KernelRow] = []
    for entry in selected_manifest_entries(repo_root):
        family, source_directory = source_family(entry.source)
        source_path = repo_root / entry.source
        symbols = extract_symbols(source_path, pattern)
        if not symbols:
            symbols = [""]
        target_class, target_architecture = target_fields(entry.manifest_bucket)
        for symbol in symbols:
            rows.append(
                KernelRow(
                    symbol=symbol,
                    family=family,
                    logical_operation=logical_operation(symbol) if symbol else "",
                    source_directory=source_directory,
                    source=entry.source,
                    language=entry.language,
                    generated="/gen/" in f"/{entry.source}",
                    status=entry.status,
                    target_class=target_class,
                    target_architecture=target_architecture,
                    manifest_bucket=entry.manifest_bucket,
                    kernel_isa=kernel_isa(entry.source, constants),
                )
            )
    rows.sort(
        key=lambda row: (
            row.family,
            row.logical_operation,
            row.target_class,
            row.kernel_isa,
            row.status,
            row.source,
            row.symbol,
        )
    )
    return rows


def unique_source_rows(rows: Iterable[KernelRow]) -> dict[str, KernelRow]:
    sources: dict[str, KernelRow] = {}
    for row in rows:
        previous = sources.setdefault(row.source, row)
        comparable = (
            row.family,
            row.language,
            row.status,
            row.target_class,
            row.target_architecture,
            row.manifest_bucket,
            row.kernel_isa,
        )
        previous_comparable = (
            previous.family,
            previous.language,
            previous.status,
            previous.target_class,
            previous.target_architecture,
            previous.manifest_bucket,
            previous.kernel_isa,
        )
        if comparable != previous_comparable:
            raise ValueError(f"inconsistent classifications for {row.source}")
    return sources


def architecture_names(constants: GeneratorConstants) -> list[str]:
    return list(X64_C_BUCKETS) + [
        "amd64",
        "scalar",
        "wasm",
        "wasm32",
        "wasmsimd",
        "wasmsimd32",
        "wasmrelaxedsimd",
        "wasmrelaxedsimd32",
    ] + sorted(constants.isa_map)


def _capability_metadata(name: str) -> tuple[str, int | str, str, str]:
    if name == "scalar":
        return "Portable scalar", "", "Software/conversion fallback", "Architecture-neutral C fallback"
    if name in {"sse", "sse2", "sse2fma", "ssse3", "sse41"}:
        return "SSE", 128, "None native", "Applicable to x64; some manifests are shared with x86-32"
    if name == "avx":
        return "AVX", 256, "None native", "AVX without AVX2, F16C, or FMA"
    if name == "f16c":
        return "AVX/F16C", 256, "F16/F32 conversion", "FP16 storage conversion; arithmetic remains FP32"
    if name == "fma3":
        return "AVX/FMA3", 256, "F32 accumulation", "Common f32 256-bit GEMM naming tier"
    if name == "avx2":
        return "AVX2", 256, "F16 conversion/F32 accumulation", "Common f16 256-bit kernel naming tier"
    if name in {"avxvnni", "avxvnniint8"}:
        return "AVX-VNNI", 256, "F16 conversion/F32 accumulation", "256-bit VNNI integer acceleration"
    if name.startswith("avx256"):
        return "AVX-512 extensions, 256-bit vectors", 256, "F16 conversion/F32 accumulation", "Uses AVX-512 extensions with 256-bit vector operations"
    if name == "avx512f":
        return "AVX-512 Foundation", 512, "None native", "AVX-512 Foundation"
    if name == "avx512skx":
        return "AVX-512 Skylake-X", 512, "F16 conversion/F32 accumulation", "Adds CD, BW, DQ, and VL"
    if name == "avx512vbmi":
        return "AVX-512 VBMI", 512, "F16 conversion/F32 accumulation", "Adds Vector Byte Manipulation Instructions"
    if name in {"avx512vnni", "avx512vnnigfni"}:
        return "AVX-512 VNNI", 512, "F16 conversion/F32 accumulation", "Integer dot-product acceleration"
    if name == "avx512amx":
        return "AVX-512 + AMX", 512, "F16 conversion/F32 accumulation", "Includes AMX tile and INT8 instructions"
    if name == "avx512fp16":
        return "AVX-512 FP16", 512, "Native FP16", "Direct FP16 vector arithmetic"
    if name == "avx512bf16":
        return "AVX-512 BF16", 512, "No native FP16", "Native BF16 dot-product arithmetic"
    if name in {"wasm", "wasm32"}:
        return "WebAssembly scalar", "", "Software/conversion fallback", "Base wasm32 build target"
    if name in {"wasmsimd", "wasmsimd32"}:
        return "WebAssembly SIMD", 128, "None native", "Standard WebAssembly SIMD"
    if name in {"wasmrelaxedsimd", "wasmrelaxedsimd32"}:
        return "WebAssembly Relaxed SIMD", 128, "Direct FP16 (build-gated) and FP32-accumulation variants", "Built with relaxed SIMD and FP16 enabled"
    if name == "wasmrelaxedsimdfp16":
        return "WebAssembly Relaxed SIMD FP16", 128, "Direct FP16", "Dedicated direct-FP16 kernel tag"
    if name == "wasmblendvps":
        return "WebAssembly Relaxed SIMD", 128, "None native", "Relaxed-SIMD blend variant"
    if name == "wasmpshufb":
        return "WebAssembly Relaxed SIMD", 128, "None native", "Relaxed-SIMD byte-shuffle variant"
    if name in {"wasmsdot", "wasmusdot"}:
        return "WebAssembly Relaxed SIMD dot product", 128, "None native", "Signed/unsigned relaxed-SIMD dot-product variant"
    if name == "amd64":
        return "AMD64 assembly", "varies", "Varies by exact ISA", "Architecture bucket; exact ISA is parsed from each assembly source"
    raise ValueError(f"missing capability metadata for {name}")


def architecture_infos(
    constants: GeneratorConstants, build_conditions: dict[str, str]
) -> list[ArchitectureInfo]:
    infos: list[ArchitectureInfo] = []
    bucket_names = {"amd64", "wasm", "wasm32", "wasmsimd32", "wasmrelaxedsimd32"}
    for name in architecture_names(constants):
        if name == "amd64":
            target_class, target_architecture = "x64", "x86_64"
        elif name == "scalar":
            target_class, target_architecture = "portable", "architecture-neutral"
        elif name.startswith("wasm"):
            target_class, target_architecture = "WebAssembly", "wasm32"
        else:
            target_class, target_architecture = "x64", "x86_64"
        manifest_bucket = constants.isa_map.get(name, name)
        tier, width, fp16_mode, notes = _capability_metadata(name)
        condition_name = constants.isa_map.get(name, name)
        infos.append(
            ArchitectureInfo(
                name=name,
                kind="Architecture/build bucket" if name in bucket_names else "Exact kernel ISA",
                target_class=target_class,
                target_architecture=target_architecture,
                manifest_bucket=manifest_bucket,
                capability_tier=tier,
                vector_width_bits=width,
                fp16_mode=fp16_mode,
                build_condition=build_conditions.get(condition_name, "inherited from manifest bucket"),
                notes=notes,
            )
        )
    return infos


def build_family_support(rows: list[KernelRow], isa_names: list[str]) -> list[dict[str, object]]:
    source_rows = unique_source_rows(rows)
    by_family: dict[str, list[KernelRow]] = defaultdict(list)
    for row in source_rows.values():
        by_family[row.family].append(row)
    symbol_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.symbol:
            symbol_counts[row.family] += 1

    result: list[dict[str, object]] = []
    for family in sorted(by_family):
        family_sources = by_family[family]
        record: dict[str, object] = {
            "Family": family,
            "Source Directory": family_sources[0].source_directory,
            "Source Count": len(family_sources),
            "PROD Sources": sum(row.status == "PROD" for row in family_sources),
            "NON_PROD Sources": sum(row.status == "NON_PROD" for row in family_sources),
            "Kernel Symbols": symbol_counts[family],
            "Logical Operations": len(
                {row.logical_operation for row in rows if row.family == family}
            ),
        }
        for isa in isa_names:
            record[isa] = sum(row.kernel_isa == isa for row in family_sources)
        result.append(record)
    return result


def build_operation_support(
    rows: list[KernelRow], isa_names: list[str]
) -> list[dict[str, object]]:
    source_rows: dict[tuple[str, str], KernelRow] = {}
    for row in rows:
        key = (row.source, row.logical_operation)
        previous = source_rows.setdefault(key, row)
        if previous.family != row.family:
            raise ValueError(
                f"logical operation {row.logical_operation} spans inconsistent families"
            )

    grouped_sources: dict[tuple[str, str], list[KernelRow]] = defaultdict(list)
    for row in source_rows.values():
        grouped_sources[(row.family, row.logical_operation)].append(row)
    symbol_counts = Counter((row.family, row.logical_operation) for row in rows if row.symbol)

    result: list[dict[str, object]] = []
    for family, operation in sorted(grouped_sources):
        operation_sources = grouped_sources[(family, operation)]
        record: dict[str, object] = {
            "Logical Operation": operation,
            "Source Family": family,
            "Source Directory": operation_sources[0].source_directory,
            "Source Count": len(operation_sources),
            "PROD Sources": sum(row.status == "PROD" for row in operation_sources),
            "NON_PROD Sources": sum(
                row.status == "NON_PROD" for row in operation_sources
            ),
            "Kernel Symbols": symbol_counts[(family, operation)],
        }
        for isa in isa_names:
            record[isa] = sum(row.kernel_isa == isa for row in operation_sources)
        result.append(record)
    return result


def kernel_fp16_path(row: KernelRow) -> str:
    family_tokens = set(row.family.split("-"))
    if "f16" not in family_tokens:
        return "Not an FP16 kernel"
    if "f32acc" in row.source or "_f32acc_" in row.symbol:
        return "FP16 I/O, FP32 accumulation"
    if row.kernel_isa == "avx512fp16":
        return "Native/direct FP16"
    if row.kernel_isa in {"wasmrelaxedsimd", "wasmrelaxedsimdfp16"}:
        return "Direct FP16 (build-gated)"
    if row.kernel_isa == "scalar":
        return "Scalar/software FP16"
    return "FP16 conversion or scalar arithmetic"


def build_architecture_rows(
    rows: list[KernelRow], infos: list[ArchitectureInfo]
) -> list[dict[str, object]]:
    source_rows = list(unique_source_rows(rows).values())
    result: list[dict[str, object]] = []
    for info in infos:
        if info.kind == "Architecture/build bucket":
            matching_sources = [row for row in source_rows if row.manifest_bucket == info.name]
            matching_symbols = [row for row in rows if row.manifest_bucket == info.name and row.symbol]
        else:
            matching_sources = [row for row in source_rows if row.kernel_isa == info.name]
            matching_symbols = [row for row in rows if row.kernel_isa == info.name and row.symbol]
        result.append(
            {
                "Name": info.name,
                "Kind": info.kind,
                "Target Class": info.target_class,
                "Target Architecture": info.target_architecture,
                "Manifest Bucket": info.manifest_bucket,
                "Capability Tier": info.capability_tier,
                "Vector Width (bits)": info.vector_width_bits,
                "FP16 Mode": info.fp16_mode,
                "Build Condition": info.build_condition,
                "Source Count": len(matching_sources),
                "PROD Sources": sum(row.status == "PROD" for row in matching_sources),
                "Kernel Symbols": len(matching_symbols),
                "Notes": info.notes,
            }
        )
    return result


def validate_inventory(
    repo_root: Path, entries: list[ManifestEntry], rows: list[KernelRow]
) -> None:
    errors: list[str] = []
    entry_sources = [entry.source for entry in entries]
    if len(entry_sources) != len(set(entry_sources)):
        errors.append("a source occurs in more than one selected manifest/status list")
    row_sources = set(unique_source_rows(rows))
    if set(entry_sources) != row_sources:
        errors.append("detailed rows do not reconcile with selected manifest sources")
    missing_sources = [source for source in entry_sources if not (repo_root / source).is_file()]
    if missing_sources:
        errors.append(f"{len(missing_sources)} manifest sources do not exist")
    blank_symbols = {row.source for row in rows if not row.symbol}
    if blank_symbols:
        errors.append(f"{len(blank_symbols)} sources contain no recognized microkernel symbol")
    allowed_classes = {"x64", "portable", "WebAssembly"}
    unexpected_classes = {row.target_class for row in rows} - allowed_classes
    if unexpected_classes:
        errors.append(f"unexpected target classes: {sorted(unexpected_classes)}")
    representatives = {
        "f32-gemm": any(row.family == "f32-gemm" for row in rows),
        "f16-dwconv": any(row.family == "f16-dwconv" for row in rows),
        "avx512fp16": any(row.kernel_isa == "avx512fp16" for row in rows),
        "wasmrelaxedsimdfp16": any(
            row.kernel_isa == "wasmrelaxedsimdfp16" for row in rows
        ),
    }
    absent = [name for name, present in representatives.items() if not present]
    if absent:
        errors.append(f"representative inventory rows are absent: {', '.join(absent)}")
    if errors:
        raise ValueError("inventory validation failed:\n- " + "\n- ".join(errors))


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _add_table_sheet(
    workbook: object,
    title: str,
    headers: list[str],
    records: list[dict[str, object]],
    table_name: str,
) -> object:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    worksheet = workbook.create_sheet(title)
    worksheet.sheet_view.showGridLines = False
    worksheet.freeze_panes = "A2"
    worksheet.append(headers)
    for record in records:
        worksheet.append([record.get(header, "") for header in headers])

    header_fill = PatternFill("solid", fgColor="173F5F")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(vertical="center")
    worksheet.row_dimensions[1].height = 26

    end_column = get_column_letter(len(headers))
    table = Table(displayName=table_name, ref=f"A1:{end_column}{len(records) + 1}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    worksheet.add_table(table)

    for column_index, header in enumerate(headers, start=1):
        values = [header] + [str(record.get(header, "")) for record in records[:500]]
        width = min(max(max(map(len, values)) + 2, 10), 55)
        if header in {"Source", "Notes", "Value"}:
            width = min(max(width, 32), 80)
        worksheet.column_dimensions[get_column_letter(column_index)].width = width
    return worksheet


def write_workbook(repo_root: Path, output: Path, rows: list[KernelRow]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    constants = load_generator_constants(repo_root / "tools" / "update-microkernels.py")
    build_conditions = load_build_conditions(repo_root / "build_params.bzl")
    infos = architecture_infos(constants, build_conditions)
    exact_isas = [info.name for info in infos if info.kind == "Exact kernel ISA"]

    kernel_headers = [
        "Kernel Symbol",
        "Family",
        "Logical Operation",
        "Source Directory",
        "Source",
        "Language",
        "Generated",
        "Status",
        "Target Class",
        "Target Architecture",
        "Manifest Bucket",
        "Exact ISA",
        "Capability Tier",
        "Vector Width (bits)",
        "ISA FP16 Capability",
        "Kernel FP16 Path",
    ]
    info_by_name = {info.name: info for info in infos}
    kernel_records: list[dict[str, object]] = []
    for row in rows:
        info = info_by_name[row.kernel_isa]
        kernel_records.append(
            {
                "Kernel Symbol": row.symbol,
                "Family": row.family,
                "Logical Operation": row.logical_operation,
                "Source Directory": row.source_directory,
                "Source": row.source,
                "Language": row.language,
                "Generated": "Yes" if row.generated else "No",
                "Status": row.status,
                "Target Class": row.target_class,
                "Target Architecture": row.target_architecture,
                "Manifest Bucket": row.manifest_bucket,
                "Exact ISA": row.kernel_isa,
                "Capability Tier": info.capability_tier,
                "Vector Width (bits)": info.vector_width_bits,
                "ISA FP16 Capability": info.fp16_mode,
                "Kernel FP16 Path": kernel_fp16_path(row),
            }
        )

    family_records = build_family_support(rows, exact_isas)
    family_headers = [
        "Family",
        "Source Directory",
        "Source Count",
        "PROD Sources",
        "NON_PROD Sources",
        "Kernel Symbols",
        "Logical Operations",
    ] + exact_isas
    operation_records = build_operation_support(rows, exact_isas)
    operation_headers = [
        "Logical Operation",
        "Source Family",
        "Source Directory",
        "Source Count",
        "PROD Sources",
        "NON_PROD Sources",
        "Kernel Symbols",
    ] + exact_isas
    architecture_records = build_architecture_rows(rows, infos)
    architecture_headers = list(architecture_records[0])
    about_records = [
        {"Property": "Repository revision", "Value": _git_revision(repo_root)},
        {"Property": "Generated at", "Value": datetime.now(timezone.utc).isoformat()},
        {"Property": "Scope", "Value": "x64-applicable C, AMD64 assembly, scalar portable C, and all WebAssembly variants"},
        {"Property": "Excluded", "Value": "ARM/AArch64, RISC-V/RVV, Hexagon/HVX, SME, templates, headers, and non-microkernel sources"},
        {"Property": "Inventory unit", "Value": "One Kernels row per microkernel symbol; operation and family source counts use unique compiled source files"},
        {"Property": "Logical Operation", "Value": "Exact symbol prefix before _ukernel, normalized with hyphens; exposes operations hidden inside umbrella source directories"},
        {"Property": "PROD", "Value": "Source referenced by a runtime config and listed in a generated PROD manifest"},
        {"Property": "NON_PROD", "Value": "Existing compiled microkernel source not currently selected by a runtime config"},
        {"Property": "Manifest Bucket", "Value": "Generated build-list grouping; mapped WebAssembly aliases share the wasmrelaxedsimd bucket"},
        {"Property": "Exact ISA", "Value": "Raw ISA token parsed from the kernel source filename"},
        {"Property": "x64 note", "Value": "C manifests are generally shared by x86 and x86-64; this inventory records their applicability to x64"},
        {"Property": "Primary source", "Value": "gen/*_microkernels.bzl"},
        {"Property": "Manifest generator", "Value": "tools/update-microkernels.py"},
        {"Property": "Build metadata", "Value": "build_params.bzl"},
    ]

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.creator = "XNNPACK kernel inventory generator"
    kernels_sheet = _add_table_sheet(
        workbook, "Kernels", kernel_headers, kernel_records, "KernelsTable"
    )
    _add_table_sheet(
        workbook,
        "Operation Support",
        operation_headers,
        operation_records,
        "OperationSupportTable",
    )
    _add_table_sheet(
        workbook, "Family Support", family_headers, family_records, "FamilySupportTable"
    )
    _add_table_sheet(
        workbook,
        "Architectures",
        architecture_headers,
        architecture_records,
        "ArchitecturesTable",
    )
    _add_table_sheet(
        workbook, "About", ["Property", "Value"], about_records, "AboutTable"
    )

    status_column = kernel_headers.index("Status") + 1
    prod_fill = PatternFill("solid", fgColor="D9EAD3")
    non_prod_fill = PatternFill("solid", fgColor="EAF2F8")
    for row_index, record in enumerate(kernel_records, start=2):
        status_cell = kernels_sheet.cell(row=row_index, column=status_column)
        status_cell.fill = prod_fill if record["Status"] == "PROD" else non_prod_fill
        if record["Status"] == "PROD":
            status_cell.font = Font(color="274E13", bold=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)


def validate_workbook(output: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(output, read_only=False, data_only=False)
    expected = {
        "Kernels": "KernelsTable",
        "Operation Support": "OperationSupportTable",
        "Family Support": "FamilySupportTable",
        "Architectures": "ArchitecturesTable",
        "About": "AboutTable",
    }
    if workbook.sheetnames != list(expected):
        raise ValueError(f"unexpected workbook sheets: {workbook.sheetnames}")
    for sheet_name, table_name in expected.items():
        if table_name not in workbook[sheet_name].tables:
            raise ValueError(f"{sheet_name} is missing Excel table {table_name}")
    hyperlink_count = sum(
        cell.hyperlink is not None
        for worksheet in workbook.worksheets
        for row in worksheet.iter_rows()
        for cell in row
    )
    if hyperlink_count:
        raise ValueError(f"workbook contains {hyperlink_count} nonportable hyperlinks")
    workbook.close()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        help="Workbook path (default: <repo>/build/xnnpack-kernel-inventory-x64-portable.xlsx)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate source inventory without writing a workbook",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    entries = selected_manifest_entries(repo_root)
    rows = collect_kernel_rows(repo_root)
    validate_inventory(repo_root, entries, rows)
    source_count = len(unique_source_rows(rows))
    if args.check:
        print(f"Validated {len(rows)} kernel symbols in {source_count} source files")
        return 0
    output = args.output or repo_root / "build" / "xnnpack-kernel-inventory-x64-portable.xlsx"
    output = output.resolve()
    write_workbook(repo_root, output, rows)
    validate_workbook(output)
    print(f"Wrote {len(rows)} kernel symbols from {source_count} source files to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))