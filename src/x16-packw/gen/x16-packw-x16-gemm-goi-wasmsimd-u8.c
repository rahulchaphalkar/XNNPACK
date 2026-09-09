// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"


void xnn_x16_packw_gemm_goi_ukernel_x16__wasmsimd_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* weights,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 16
    const uint16_t* w0 = weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const v128_t vb0 = wasm_v128_load(b + 0);
        const v128_t vb8 = wasm_v128_load(b + 8);
        b += 16;
        wasm_v128_store(packed_w + 0, vb0);
        wasm_v128_store(packed_w + 8, vb8);
      } else {
        const v128_t vzero = wasm_i16x8_splat(0);
        wasm_v128_store(packed_w + 0, vzero);
        wasm_v128_store(packed_w + 8, vzero);
      }
      packed_w += 16;

      const uint16_t* w1 = w0 + kc;
      const uint16_t* w2 = w1 + kc;
      const uint16_t* w3 = w2 + kc;
      const uint16_t* w4 = w3 + kc;
      const uint16_t* w5 = w4 + kc;
      const uint16_t* w6 = w5 + kc;
      const uint16_t* w7 = w6 + kc;
      const uint16_t* w8 = w7 + kc;
      const uint16_t* w9 = w8 + kc;
      const uint16_t* w10 = w9 + kc;
      const uint16_t* w11 = w10 + kc;
      const uint16_t* w12 = w11 + kc;
      const uint16_t* w13 = w12 + kc;
      const uint16_t* w14 = w13 + kc;
      const uint16_t* w15 = w14 + kc;

      // KC main loop multiple of 8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const v128_t v3_0 = wasm_v128_load(w0);
        w0 += 8;
        const v128_t v3_1 = wasm_v128_load(w1);
        w1 += 8;
        const v128_t v3_2 = wasm_v128_load(w2);
        w2 += 8;
        const v128_t v3_3 = wasm_v128_load(w3);
        w3 += 8;
        const v128_t v3_4 = wasm_v128_load(w4);
        w4 += 8;
        const v128_t v3_5 = wasm_v128_load(w5);
        w5 += 8;
        const v128_t v3_6 = wasm_v128_load(w6);
        w6 += 8;
        const v128_t v3_7 = wasm_v128_load(w7);
        w7 += 8;
        const v128_t v3_8 = wasm_v128_load(w8);
        w8 += 8;
        const v128_t v3_9 = wasm_v128_load(w9);
        w9 += 8;
        const v128_t v3_10 = wasm_v128_load(w10);
        w10 += 8;
        const v128_t v3_11 = wasm_v128_load(w11);
        w11 += 8;
        const v128_t v3_12 = wasm_v128_load(w12);
        w12 += 8;
        const v128_t v3_13 = wasm_v128_load(w13);
        w13 += 8;
        const v128_t v3_14 = wasm_v128_load(w14);
        w14 += 8;
        const v128_t v3_15 = wasm_v128_load(w15);
        w15 += 8;

        const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_1 = wasm_v16x8_shuffle(v3_0, v3_4, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_3 = wasm_v16x8_shuffle(v3_1, v3_5, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_5 = wasm_v16x8_shuffle(v3_2, v3_6, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_7 = wasm_v16x8_shuffle(v3_3, v3_7, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_1 = wasm_v16x8_shuffle(v2_0, v2_4, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_2 = wasm_v16x8_shuffle(v2_1, v2_5, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_3 = wasm_v16x8_shuffle(v2_1, v2_5, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_5 = wasm_v16x8_shuffle(v2_2, v2_6, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_6 = wasm_v16x8_shuffle(v2_3, v2_7, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_7 = wasm_v16x8_shuffle(v2_3, v2_7, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w2 = wasm_v16x8_shuffle(v1_1, v1_5, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w3 = wasm_v16x8_shuffle(v1_1, v1_5, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w4 = wasm_v16x8_shuffle(v1_2, v1_6, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w5 = wasm_v16x8_shuffle(v1_2, v1_6, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w6 = wasm_v16x8_shuffle(v1_3, v1_7, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w7 = wasm_v16x8_shuffle(v1_3, v1_7, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_9 = wasm_v16x8_shuffle(v3_8, v3_12, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_11 = wasm_v16x8_shuffle(v3_9, v3_13, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_13 = wasm_v16x8_shuffle(v3_10, v3_14, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_15 = wasm_v16x8_shuffle(v3_11, v3_15, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_9 = wasm_v16x8_shuffle(v2_8, v2_12, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_10 = wasm_v16x8_shuffle(v2_9, v2_13, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_11 = wasm_v16x8_shuffle(v2_9, v2_13, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_13 = wasm_v16x8_shuffle(v2_10, v2_14, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_14 = wasm_v16x8_shuffle(v2_11, v2_15, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_15 = wasm_v16x8_shuffle(v2_11, v2_15, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w9 = wasm_v16x8_shuffle(v1_8, v1_12, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w10 = wasm_v16x8_shuffle(v1_9, v1_13, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w11 = wasm_v16x8_shuffle(v1_9, v1_13, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w12 = wasm_v16x8_shuffle(v1_10, v1_14, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w13 = wasm_v16x8_shuffle(v1_10, v1_14, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w14 = wasm_v16x8_shuffle(v1_11, v1_15, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w15 = wasm_v16x8_shuffle(v1_11, v1_15, 4, 12, 5, 13, 6, 14, 7, 15);

        wasm_v128_store(packed_w + 0, w0);
        wasm_v128_store(packed_w + 8, w8);
        wasm_v128_store(packed_w + 16, w1);
        wasm_v128_store(packed_w + 24, w9);
        wasm_v128_store(packed_w + 32, w2);
        wasm_v128_store(packed_w + 40, w10);
        wasm_v128_store(packed_w + 48, w3);
        wasm_v128_store(packed_w + 56, w11);
        wasm_v128_store(packed_w + 64, w4);
        wasm_v128_store(packed_w + 72, w12);
        wasm_v128_store(packed_w + 80, w5);
        wasm_v128_store(packed_w + 88, w13);
        wasm_v128_store(packed_w + 96, w6);
        wasm_v128_store(packed_w + 104, w14);
        wasm_v128_store(packed_w + 112, w7);
        wasm_v128_store(packed_w + 120, w15);
        packed_w += 128;
      }

      // KC remainder (1..7)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 7);

        if (k & 4) {
          const v128_t v3_0 = wasm_v128_load64_zero(w0);
          w0 += 4;
          const v128_t v3_1 = wasm_v128_load64_zero(w1);
          w1 += 4;
          const v128_t v3_2 = wasm_v128_load64_zero(w2);
          w2 += 4;
          const v128_t v3_3 = wasm_v128_load64_zero(w3);
          w3 += 4;
          const v128_t v3_4 = wasm_v128_load64_zero(w4);
          w4 += 4;
          const v128_t v3_5 = wasm_v128_load64_zero(w5);
          w5 += 4;
          const v128_t v3_6 = wasm_v128_load64_zero(w6);
          w6 += 4;
          const v128_t v3_7 = wasm_v128_load64_zero(w7);
          w7 += 4;
          const v128_t v3_8 = wasm_v128_load64_zero(w8);
          w8 += 4;
          const v128_t v3_9 = wasm_v128_load64_zero(w9);
          w9 += 4;
          const v128_t v3_10 = wasm_v128_load64_zero(w10);
          w10 += 4;
          const v128_t v3_11 = wasm_v128_load64_zero(w11);
          w11 += 4;
          const v128_t v3_12 = wasm_v128_load64_zero(w12);
          w12 += 4;
          const v128_t v3_13 = wasm_v128_load64_zero(w13);
          w13 += 4;
          const v128_t v3_14 = wasm_v128_load64_zero(w14);
          w14 += 4;
          const v128_t v3_15 = wasm_v128_load64_zero(w15);
          w15 += 4;

          const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_1 = wasm_v16x8_shuffle(v3_0, v3_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_3 = wasm_v16x8_shuffle(v3_1, v3_5, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_5 = wasm_v16x8_shuffle(v3_2, v3_6, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_7 = wasm_v16x8_shuffle(v3_3, v3_7, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_1 = wasm_v16x8_shuffle(v2_0, v2_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_2 = wasm_v16x8_shuffle(v2_1, v2_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_3 = wasm_v16x8_shuffle(v2_1, v2_5, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_5 = wasm_v16x8_shuffle(v2_2, v2_6, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_6 = wasm_v16x8_shuffle(v2_3, v2_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_7 = wasm_v16x8_shuffle(v2_3, v2_7, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w2 = wasm_v16x8_shuffle(v1_1, v1_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w3 = wasm_v16x8_shuffle(v1_1, v1_5, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_9 = wasm_v16x8_shuffle(v3_8, v3_12, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_11 = wasm_v16x8_shuffle(v3_9, v3_13, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_13 = wasm_v16x8_shuffle(v3_10, v3_14, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_15 = wasm_v16x8_shuffle(v3_11, v3_15, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_9 = wasm_v16x8_shuffle(v2_8, v2_12, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_10 = wasm_v16x8_shuffle(v2_9, v2_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_11 = wasm_v16x8_shuffle(v2_9, v2_13, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_13 = wasm_v16x8_shuffle(v2_10, v2_14, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_14 = wasm_v16x8_shuffle(v2_11, v2_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_15 = wasm_v16x8_shuffle(v2_11, v2_15, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w9 = wasm_v16x8_shuffle(v1_8, v1_12, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w10 = wasm_v16x8_shuffle(v1_9, v1_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w11 = wasm_v16x8_shuffle(v1_9, v1_13, 4, 12, 5, 13, 6, 14, 7, 15);

          wasm_v128_store(packed_w + 0, w0);
          wasm_v128_store(packed_w + 8, w8);
          wasm_v128_store(packed_w + 16, w1);
          wasm_v128_store(packed_w + 24, w9);
          wasm_v128_store(packed_w + 32, w2);
          wasm_v128_store(packed_w + 40, w10);
          wasm_v128_store(packed_w + 48, w3);
          wasm_v128_store(packed_w + 56, w11);
          packed_w += 64;
        }

        if (k & 2) {
          const v128_t v3_0 = wasm_v128_load32_zero(w0);
          w0 += 2;
          const v128_t v3_1 = wasm_v128_load32_zero(w1);
          w1 += 2;
          const v128_t v3_2 = wasm_v128_load32_zero(w2);
          w2 += 2;
          const v128_t v3_3 = wasm_v128_load32_zero(w3);
          w3 += 2;
          const v128_t v3_4 = wasm_v128_load32_zero(w4);
          w4 += 2;
          const v128_t v3_5 = wasm_v128_load32_zero(w5);
          w5 += 2;
          const v128_t v3_6 = wasm_v128_load32_zero(w6);
          w6 += 2;
          const v128_t v3_7 = wasm_v128_load32_zero(w7);
          w7 += 2;
          const v128_t v3_8 = wasm_v128_load32_zero(w8);
          w8 += 2;
          const v128_t v3_9 = wasm_v128_load32_zero(w9);
          w9 += 2;
          const v128_t v3_10 = wasm_v128_load32_zero(w10);
          w10 += 2;
          const v128_t v3_11 = wasm_v128_load32_zero(w11);
          w11 += 2;
          const v128_t v3_12 = wasm_v128_load32_zero(w12);
          w12 += 2;
          const v128_t v3_13 = wasm_v128_load32_zero(w13);
          w13 += 2;
          const v128_t v3_14 = wasm_v128_load32_zero(w14);
          w14 += 2;
          const v128_t v3_15 = wasm_v128_load32_zero(w15);
          w15 += 2;

          const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w9 = wasm_v16x8_shuffle(v1_8, v1_12, 4, 12, 5, 13, 6, 14, 7, 15);

          wasm_v128_store(packed_w + 0, w0);
          wasm_v128_store(packed_w + 8, w8);
          wasm_v128_store(packed_w + 16, w1);
          wasm_v128_store(packed_w + 24, w9);
          packed_w += 32;
        }

        if (k & 1) {
          const v128_t v3_0 = wasm_v128_load16_lane(w0, wasm_i16x8_splat(0), 0);
          w0 += 1;
          const v128_t v3_1 = wasm_v128_load16_lane(w1, wasm_i16x8_splat(0), 0);
          w1 += 1;
          const v128_t v3_2 = wasm_v128_load16_lane(w2, wasm_i16x8_splat(0), 0);
          w2 += 1;
          const v128_t v3_3 = wasm_v128_load16_lane(w3, wasm_i16x8_splat(0), 0);
          w3 += 1;
          const v128_t v3_4 = wasm_v128_load16_lane(w4, wasm_i16x8_splat(0), 0);
          w4 += 1;
          const v128_t v3_5 = wasm_v128_load16_lane(w5, wasm_i16x8_splat(0), 0);
          w5 += 1;
          const v128_t v3_6 = wasm_v128_load16_lane(w6, wasm_i16x8_splat(0), 0);
          w6 += 1;
          const v128_t v3_7 = wasm_v128_load16_lane(w7, wasm_i16x8_splat(0), 0);
          w7 += 1;
          const v128_t v3_8 = wasm_v128_load16_lane(w8, wasm_i16x8_splat(0), 0);
          w8 += 1;
          const v128_t v3_9 = wasm_v128_load16_lane(w9, wasm_i16x8_splat(0), 0);
          w9 += 1;
          const v128_t v3_10 = wasm_v128_load16_lane(w10, wasm_i16x8_splat(0), 0);
          w10 += 1;
          const v128_t v3_11 = wasm_v128_load16_lane(w11, wasm_i16x8_splat(0), 0);
          w11 += 1;
          const v128_t v3_12 = wasm_v128_load16_lane(w12, wasm_i16x8_splat(0), 0);
          w12 += 1;
          const v128_t v3_13 = wasm_v128_load16_lane(w13, wasm_i16x8_splat(0), 0);
          w13 += 1;
          const v128_t v3_14 = wasm_v128_load16_lane(w14, wasm_i16x8_splat(0), 0);
          w14 += 1;
          const v128_t v3_15 = wasm_v128_load16_lane(w15, wasm_i16x8_splat(0), 0);
          w15 += 1;

          const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);

          wasm_v128_store(packed_w + 0, w0);
          wasm_v128_store(packed_w + 8, w8);
          packed_w += 16;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++ = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const v128_t vzero = wasm_i16x8_splat(0);
        wasm_v128_store(packed_w + 0, vzero);
        wasm_v128_store(packed_w + 8, vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
      const uint16_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint16_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint16_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint16_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint16_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint16_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint16_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint16_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint16_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint16_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const uint16_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const uint16_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const uint16_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const uint16_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      // KC main loop multiple of 8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const v128_t v3_0 = wasm_v128_load(w0);
        w0 += 8;
        const v128_t v3_1 = wasm_v128_load(w1);
        w1 += 8;
        const v128_t v3_2 = wasm_v128_load(w2);
        w2 += 8;
        const v128_t v3_3 = wasm_v128_load(w3);
        w3 += 8;
        const v128_t v3_4 = wasm_v128_load(w4);
        w4 += 8;
        const v128_t v3_5 = wasm_v128_load(w5);
        w5 += 8;
        const v128_t v3_6 = wasm_v128_load(w6);
        w6 += 8;
        const v128_t v3_7 = wasm_v128_load(w7);
        w7 += 8;
        const v128_t v3_8 = wasm_v128_load(w8);
        w8 += 8;
        const v128_t v3_9 = wasm_v128_load(w9);
        w9 += 8;
        const v128_t v3_10 = wasm_v128_load(w10);
        w10 += 8;
        const v128_t v3_11 = wasm_v128_load(w11);
        w11 += 8;
        const v128_t v3_12 = wasm_v128_load(w12);
        w12 += 8;
        const v128_t v3_13 = wasm_v128_load(w13);
        w13 += 8;
        const v128_t v3_14 = wasm_v128_load(w14);
        w14 += 8;
        const v128_t v3_15 = wasm_i16x8_splat(0);

        const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_1 = wasm_v16x8_shuffle(v3_0, v3_4, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_3 = wasm_v16x8_shuffle(v3_1, v3_5, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_5 = wasm_v16x8_shuffle(v3_2, v3_6, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_7 = wasm_v16x8_shuffle(v3_3, v3_7, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_1 = wasm_v16x8_shuffle(v2_0, v2_4, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_2 = wasm_v16x8_shuffle(v2_1, v2_5, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_3 = wasm_v16x8_shuffle(v2_1, v2_5, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_5 = wasm_v16x8_shuffle(v2_2, v2_6, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_6 = wasm_v16x8_shuffle(v2_3, v2_7, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_7 = wasm_v16x8_shuffle(v2_3, v2_7, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w2 = wasm_v16x8_shuffle(v1_1, v1_5, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w3 = wasm_v16x8_shuffle(v1_1, v1_5, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w4 = wasm_v16x8_shuffle(v1_2, v1_6, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w5 = wasm_v16x8_shuffle(v1_2, v1_6, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w6 = wasm_v16x8_shuffle(v1_3, v1_7, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w7 = wasm_v16x8_shuffle(v1_3, v1_7, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_9 = wasm_v16x8_shuffle(v3_8, v3_12, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_11 = wasm_v16x8_shuffle(v3_9, v3_13, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_13 = wasm_v16x8_shuffle(v3_10, v3_14, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v2_15 = wasm_v16x8_shuffle(v3_11, v3_15, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_9 = wasm_v16x8_shuffle(v2_8, v2_12, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_10 = wasm_v16x8_shuffle(v2_9, v2_13, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_11 = wasm_v16x8_shuffle(v2_9, v2_13, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_13 = wasm_v16x8_shuffle(v2_10, v2_14, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t v1_14 = wasm_v16x8_shuffle(v2_11, v2_15, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t v1_15 = wasm_v16x8_shuffle(v2_11, v2_15, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w9 = wasm_v16x8_shuffle(v1_8, v1_12, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w10 = wasm_v16x8_shuffle(v1_9, v1_13, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w11 = wasm_v16x8_shuffle(v1_9, v1_13, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w12 = wasm_v16x8_shuffle(v1_10, v1_14, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w13 = wasm_v16x8_shuffle(v1_10, v1_14, 4, 12, 5, 13, 6, 14, 7, 15);
        const v128_t w14 = wasm_v16x8_shuffle(v1_11, v1_15, 0, 8, 1, 9, 2, 10, 3, 11);
        const v128_t w15 = wasm_v16x8_shuffle(v1_11, v1_15, 4, 12, 5, 13, 6, 14, 7, 15);

        wasm_v128_store(packed_w + 0, w0);
        wasm_v128_store(packed_w + 8, w8);
        wasm_v128_store(packed_w + 16, w1);
        wasm_v128_store(packed_w + 24, w9);
        wasm_v128_store(packed_w + 32, w2);
        wasm_v128_store(packed_w + 40, w10);
        wasm_v128_store(packed_w + 48, w3);
        wasm_v128_store(packed_w + 56, w11);
        wasm_v128_store(packed_w + 64, w4);
        wasm_v128_store(packed_w + 72, w12);
        wasm_v128_store(packed_w + 80, w5);
        wasm_v128_store(packed_w + 88, w13);
        wasm_v128_store(packed_w + 96, w6);
        wasm_v128_store(packed_w + 104, w14);
        wasm_v128_store(packed_w + 112, w7);
        wasm_v128_store(packed_w + 120, w15);
        packed_w += 128;
      }

      // KC remainder (1..7)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 7);

        if (k & 4) {
          const v128_t v3_0 = wasm_v128_load64_zero(w0);
          w0 += 4;
          const v128_t v3_1 = wasm_v128_load64_zero(w1);
          w1 += 4;
          const v128_t v3_2 = wasm_v128_load64_zero(w2);
          w2 += 4;
          const v128_t v3_3 = wasm_v128_load64_zero(w3);
          w3 += 4;
          const v128_t v3_4 = wasm_v128_load64_zero(w4);
          w4 += 4;
          const v128_t v3_5 = wasm_v128_load64_zero(w5);
          w5 += 4;
          const v128_t v3_6 = wasm_v128_load64_zero(w6);
          w6 += 4;
          const v128_t v3_7 = wasm_v128_load64_zero(w7);
          w7 += 4;
          const v128_t v3_8 = wasm_v128_load64_zero(w8);
          w8 += 4;
          const v128_t v3_9 = wasm_v128_load64_zero(w9);
          w9 += 4;
          const v128_t v3_10 = wasm_v128_load64_zero(w10);
          w10 += 4;
          const v128_t v3_11 = wasm_v128_load64_zero(w11);
          w11 += 4;
          const v128_t v3_12 = wasm_v128_load64_zero(w12);
          w12 += 4;
          const v128_t v3_13 = wasm_v128_load64_zero(w13);
          w13 += 4;
          const v128_t v3_14 = wasm_v128_load64_zero(w14);
          w14 += 4;
          const v128_t v3_15 = wasm_i16x8_splat(0);

          const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_1 = wasm_v16x8_shuffle(v3_0, v3_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_3 = wasm_v16x8_shuffle(v3_1, v3_5, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_5 = wasm_v16x8_shuffle(v3_2, v3_6, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_7 = wasm_v16x8_shuffle(v3_3, v3_7, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_1 = wasm_v16x8_shuffle(v2_0, v2_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_2 = wasm_v16x8_shuffle(v2_1, v2_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_3 = wasm_v16x8_shuffle(v2_1, v2_5, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_5 = wasm_v16x8_shuffle(v2_2, v2_6, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_6 = wasm_v16x8_shuffle(v2_3, v2_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_7 = wasm_v16x8_shuffle(v2_3, v2_7, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w2 = wasm_v16x8_shuffle(v1_1, v1_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w3 = wasm_v16x8_shuffle(v1_1, v1_5, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_9 = wasm_v16x8_shuffle(v3_8, v3_12, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_11 = wasm_v16x8_shuffle(v3_9, v3_13, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_13 = wasm_v16x8_shuffle(v3_10, v3_14, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_15 = wasm_v16x8_shuffle(v3_11, v3_15, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_9 = wasm_v16x8_shuffle(v2_8, v2_12, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_10 = wasm_v16x8_shuffle(v2_9, v2_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_11 = wasm_v16x8_shuffle(v2_9, v2_13, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_13 = wasm_v16x8_shuffle(v2_10, v2_14, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v1_14 = wasm_v16x8_shuffle(v2_11, v2_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_15 = wasm_v16x8_shuffle(v2_11, v2_15, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w9 = wasm_v16x8_shuffle(v1_8, v1_12, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t w10 = wasm_v16x8_shuffle(v1_9, v1_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w11 = wasm_v16x8_shuffle(v1_9, v1_13, 4, 12, 5, 13, 6, 14, 7, 15);

          wasm_v128_store(packed_w + 0, w0);
          wasm_v128_store(packed_w + 8, w8);
          wasm_v128_store(packed_w + 16, w1);
          wasm_v128_store(packed_w + 24, w9);
          wasm_v128_store(packed_w + 32, w2);
          wasm_v128_store(packed_w + 40, w10);
          wasm_v128_store(packed_w + 48, w3);
          wasm_v128_store(packed_w + 56, w11);
          packed_w += 64;
        }

        if (k & 2) {
          const v128_t v3_0 = wasm_v128_load32_zero(w0);
          w0 += 2;
          const v128_t v3_1 = wasm_v128_load32_zero(w1);
          w1 += 2;
          const v128_t v3_2 = wasm_v128_load32_zero(w2);
          w2 += 2;
          const v128_t v3_3 = wasm_v128_load32_zero(w3);
          w3 += 2;
          const v128_t v3_4 = wasm_v128_load32_zero(w4);
          w4 += 2;
          const v128_t v3_5 = wasm_v128_load32_zero(w5);
          w5 += 2;
          const v128_t v3_6 = wasm_v128_load32_zero(w6);
          w6 += 2;
          const v128_t v3_7 = wasm_v128_load32_zero(w7);
          w7 += 2;
          const v128_t v3_8 = wasm_v128_load32_zero(w8);
          w8 += 2;
          const v128_t v3_9 = wasm_v128_load32_zero(w9);
          w9 += 2;
          const v128_t v3_10 = wasm_v128_load32_zero(w10);
          w10 += 2;
          const v128_t v3_11 = wasm_v128_load32_zero(w11);
          w11 += 2;
          const v128_t v3_12 = wasm_v128_load32_zero(w12);
          w12 += 2;
          const v128_t v3_13 = wasm_v128_load32_zero(w13);
          w13 += 2;
          const v128_t v3_14 = wasm_v128_load32_zero(w14);
          w14 += 2;
          const v128_t v3_15 = wasm_i16x8_splat(0);

          const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w1 = wasm_v16x8_shuffle(v1_0, v1_4, 4, 12, 5, 13, 6, 14, 7, 15);
          const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w9 = wasm_v16x8_shuffle(v1_8, v1_12, 4, 12, 5, 13, 6, 14, 7, 15);

          wasm_v128_store(packed_w + 0, w0);
          wasm_v128_store(packed_w + 8, w8);
          wasm_v128_store(packed_w + 16, w1);
          wasm_v128_store(packed_w + 24, w9);
          packed_w += 32;
        }

        if (k & 1) {
          const v128_t v3_0 = wasm_v128_load16_lane(w0, wasm_i16x8_splat(0), 0);
          w0 += 1;
          const v128_t v3_1 = wasm_v128_load16_lane(w1, wasm_i16x8_splat(0), 0);
          w1 += 1;
          const v128_t v3_2 = wasm_v128_load16_lane(w2, wasm_i16x8_splat(0), 0);
          w2 += 1;
          const v128_t v3_3 = wasm_v128_load16_lane(w3, wasm_i16x8_splat(0), 0);
          w3 += 1;
          const v128_t v3_4 = wasm_v128_load16_lane(w4, wasm_i16x8_splat(0), 0);
          w4 += 1;
          const v128_t v3_5 = wasm_v128_load16_lane(w5, wasm_i16x8_splat(0), 0);
          w5 += 1;
          const v128_t v3_6 = wasm_v128_load16_lane(w6, wasm_i16x8_splat(0), 0);
          w6 += 1;
          const v128_t v3_7 = wasm_v128_load16_lane(w7, wasm_i16x8_splat(0), 0);
          w7 += 1;
          const v128_t v3_8 = wasm_v128_load16_lane(w8, wasm_i16x8_splat(0), 0);
          w8 += 1;
          const v128_t v3_9 = wasm_v128_load16_lane(w9, wasm_i16x8_splat(0), 0);
          w9 += 1;
          const v128_t v3_10 = wasm_v128_load16_lane(w10, wasm_i16x8_splat(0), 0);
          w10 += 1;
          const v128_t v3_11 = wasm_v128_load16_lane(w11, wasm_i16x8_splat(0), 0);
          w11 += 1;
          const v128_t v3_12 = wasm_v128_load16_lane(w12, wasm_i16x8_splat(0), 0);
          w12 += 1;
          const v128_t v3_13 = wasm_v128_load16_lane(w13, wasm_i16x8_splat(0), 0);
          w13 += 1;
          const v128_t v3_14 = wasm_v128_load16_lane(w14, wasm_i16x8_splat(0), 0);
          w14 += 1;
          const v128_t v3_15 = wasm_i16x8_splat(0);

          const v128_t v2_0 = wasm_v16x8_shuffle(v3_0, v3_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_2 = wasm_v16x8_shuffle(v3_1, v3_5, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_4 = wasm_v16x8_shuffle(v3_2, v3_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_6 = wasm_v16x8_shuffle(v3_3, v3_7, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_0 = wasm_v16x8_shuffle(v2_0, v2_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_4 = wasm_v16x8_shuffle(v2_2, v2_6, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w0 = wasm_v16x8_shuffle(v1_0, v1_4, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_8 = wasm_v16x8_shuffle(v3_8, v3_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_10 = wasm_v16x8_shuffle(v3_9, v3_13, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_12 = wasm_v16x8_shuffle(v3_10, v3_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v2_14 = wasm_v16x8_shuffle(v3_11, v3_15, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_8 = wasm_v16x8_shuffle(v2_8, v2_12, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t v1_12 = wasm_v16x8_shuffle(v2_10, v2_14, 0, 8, 1, 9, 2, 10, 3, 11);
          const v128_t w8 = wasm_v16x8_shuffle(v1_8, v1_12, 0, 8, 1, 9, 2, 10, 3, 11);

          wasm_v128_store(packed_w + 0, w0);
          wasm_v128_store(packed_w + 8, w8);
          packed_w += 16;
        }
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
