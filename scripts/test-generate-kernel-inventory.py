#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate-kernel-inventory.py")
SPEC = importlib.util.spec_from_file_location("generate_kernel_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


class KernelInventoryTest(unittest.TestCase):
    def test_logical_operation_uses_complete_symbol_prefix(self) -> None:
        cases = {
            "xnn_f16_vadd_ukernel__avx512fp16_u32": "f16-vadd",
            "xnn_f32_gemm_minmax_ukernel_7x16__avx512f": "f32-gemm-minmax",
            "xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx2": "qd8-f32-qc8w-gemm-minmax",
            "xnn_x8_packw_gemm_goi_ukernel_x8c4__scalar": "x8-packw-gemm-goi",
        }
        for symbol, expected in cases.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(inventory.logical_operation(symbol), expected)

    def test_parse_manifest_preserves_status_and_language(self) -> None:
        content = """
PROD_AVX2_MICROKERNEL_SRCS = [
    "src/f16-gemm/gen/f16-gemm-1x8-minmax-avx2.c",
]
NON_PROD_AVX2_MICROKERNEL_SRCS = [
    "src/f16-gemm/gen/f16-gemm-2x8-minmax-avx2.c",
]
ALL_AVX2_MICROKERNEL_SRCS = PROD_AVX2_MICROKERNEL_SRCS + NON_PROD_AVX2_MICROKERNEL_SRCS
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "avx2_microkernels.bzl"
            path.write_text(content, encoding="utf-8")
            entries = inventory.parse_manifest(path, "avx2")

        self.assertEqual([entry.status for entry in entries], ["PROD", "NON_PROD"])
        self.assertEqual([entry.language for entry in entries], ["C", "C"])

    def test_parse_assembly_manifest(self) -> None:
        content = """
PROD_AMD64_ASM_MICROKERNEL_SRCS = [
    "src/f32-gemm/gen/f32-gemm-6x16-minmax-asm-amd64-fma3-broadcast.S",
]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amd64_microkernels.bzl"
            path.write_text(content, encoding="utf-8")
            entries = inventory.parse_manifest(path, "amd64")

        self.assertEqual(entries[0].language, "Assembly")

    def test_kernel_isa_keeps_exact_long_and_mapped_tags(self) -> None:
        constants = inventory.GeneratorConstants(
            isas=frozenset({
                "avx",
                "avx2",
                "avx256vnni",
                "avx512f",
                "avx512fp16",
                "wasmrelaxedsimd",
                "wasmrelaxedsimdfp16",
                "wasmsdot",
            }),
            isa_map={"wasmsdot": "wasmrelaxedsimd"},
            architectures=frozenset({"amd64"}),
            symbol_pattern="",
        )
        cases = {
            "src/f16-gemm/gen/f16-gemm-8x32-minmax-avx512fp16.c": "avx512fp16",
            "src/qs8-gemm/gen/qs8-gemm-1x16-avx256vnni.c": "avx256vnni",
            "src/f16-dwconv/gen/f16-dwconv-9p8c-wasmrelaxedsimdfp16.c": "wasmrelaxedsimdfp16",
            "src/qs8-gemm/gen/qs8-gemm-1x8-wasmsdot.c": "wasmsdot",
            "src/f32-gemm/gen/f32-gemm-6x16-asm-amd64-avx512f.S": "avx512f",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(inventory.kernel_isa(source, constants), expected)

    def test_extract_symbols_deduplicates_multiple_definitions(self) -> None:
        content = """
void xnn_f32_gemm_ukernel_1x4__scalar(void) {}
void xnn_f32_gemm_ukernel_2x4__scalar(void) {}
void xnn_f32_gemm_ukernel_1x4__scalar(void);
"""
        pattern = __import__("re").compile(
            r"\bxnn_(?:[a-z0-9]+(?:_[a-z0-9]+)*)_ukernel(?:_[a-z0-9]+)*__(?:[a-z0-9]+(?:_[a-z0-9]+)*)\b"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.c"
            path.write_text(content, encoding="utf-8")
            symbols = inventory.extract_symbols(path, pattern)

        self.assertEqual(
            symbols,
            [
                "xnn_f32_gemm_ukernel_1x4__scalar",
                "xnn_f32_gemm_ukernel_2x4__scalar",
            ],
        )

    def test_family_support_counts_unique_sources_and_all_symbols(self) -> None:
        common = {
            "family": "f32-gemm",
            "logical_operation": "f32-gemm-minmax",
            "source_directory": "src/f32-gemm",
            "source": "src/f32-gemm/gen/f32-gemm-scalar.c",
            "language": "C",
            "generated": True,
            "status": "PROD",
            "target_class": "portable",
            "target_architecture": "architecture-neutral",
            "manifest_bucket": "scalar",
            "kernel_isa": "scalar",
        }
        rows = [
            inventory.KernelRow(symbol="xnn_f32_gemm_ukernel_1x4__scalar", **common),
            inventory.KernelRow(symbol="xnn_f32_gemm_ukernel_2x4__scalar", **common),
        ]

        support = inventory.build_family_support(rows, ["scalar", "avx2"])

        self.assertEqual(len(support), 1)
        self.assertEqual(support[0]["Source Count"], 1)
        self.assertEqual(support[0]["Kernel Symbols"], 2)
        self.assertEqual(support[0]["Logical Operations"], 1)
        self.assertEqual(support[0]["scalar"], 1)
        self.assertEqual(support[0]["avx2"], 0)

    def test_operation_support_splits_umbrella_family(self) -> None:
        common = {
            "family": "f16-vbinary",
            "source_directory": "src/f16-vbinary",
            "language": "C",
            "generated": True,
            "status": "PROD",
            "target_class": "x64",
            "target_architecture": "x86_64",
            "manifest_bucket": "avx512fp16",
            "kernel_isa": "avx512fp16",
        }
        rows = [
            inventory.KernelRow(
                symbol="xnn_f16_vadd_ukernel__avx512fp16_u32",
                logical_operation="f16-vadd",
                source="src/f16-vbinary/gen/f16-vadd-avx512fp16-u32.c",
                **common,
            ),
            inventory.KernelRow(
                symbol="xnn_f16_vmul_ukernel__avx512fp16_u32",
                logical_operation="f16-vmul",
                source="src/f16-vbinary/gen/f16-vmul-avx512fp16-u32.c",
                **common,
            ),
        ]

        support = inventory.build_operation_support(rows, ["avx512fp16"])

        self.assertEqual(
            [row["Logical Operation"] for row in support], ["f16-vadd", "f16-vmul"]
        )
        self.assertTrue(all(row["avx512fp16"] == 1 for row in support))

    def test_wasm_fp16_path_distinguishes_direct_and_f32acc(self) -> None:
        common = {
            "family": "f16-dwconv",
            "logical_operation": "f16-dwconv-minmax",
            "source_directory": "src/f16-dwconv",
            "language": "C",
            "generated": True,
            "status": "PROD",
            "target_class": "WebAssembly",
            "target_architecture": "wasm32",
            "manifest_bucket": "wasmrelaxedsimd",
            "kernel_isa": "wasmrelaxedsimd",
        }
        direct = inventory.KernelRow(
            symbol="xnn_f16_dwconv_ukernel_9p8c__wasmrelaxedsimd",
            source="src/f16-dwconv/gen/f16-dwconv-9p8c-wasmrelaxedsimd.c",
            **common,
        )
        f32acc = inventory.KernelRow(
            symbol="xnn_f16_f32acc_dwconv_ukernel_9p8c__wasmrelaxedsimd",
            source="src/f16-dwconv/gen/f16-f32acc-dwconv-9p8c-wasmrelaxedsimd.c",
            **common,
        )

        self.assertEqual(
            inventory.kernel_fp16_path(direct), "Direct FP16 (build-gated)"
        )
        self.assertEqual(
            inventory.kernel_fp16_path(f32acc), "FP16 I/O, FP32 accumulation"
        )


if __name__ == "__main__":
    unittest.main()