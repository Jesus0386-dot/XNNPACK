// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/wasmsimd-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrsqrt_ukernel__wasmsimd_rsqrt_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(input);
    const v128_t vx4567 = wasm_v128_load(input + 4);
    const v128_t vx89AB = wasm_v128_load(input + 8);
    const v128_t vxCDEF = wasm_v128_load(input + 12);
    input += 16;

    const v128_t vt0123 = wasm_f32x4_sqrt(vx0123);
    const v128_t vt4567 = wasm_f32x4_sqrt(vx4567);
    const v128_t vt89AB = wasm_f32x4_sqrt(vx89AB);
    const v128_t vtCDEF = wasm_f32x4_sqrt(vxCDEF);
    const v128_t vy0123 = wasm_f32x4_div(vone, vt0123);
    const v128_t vy4567 = wasm_f32x4_div(vone, vt4567);
    const v128_t vy89AB = wasm_f32x4_div(vone, vt89AB);
    const v128_t vyCDEF = wasm_f32x4_div(vone, vtCDEF);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    wasm_v128_store(output + 12, vyCDEF);
    output += 16;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;
    const v128_t vt = wasm_f32x4_sqrt(vx);
    const v128_t vy = wasm_f32x4_div(vone, vt);
    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);
    v128_t vt = wasm_f32x4_sqrt(vx);
    v128_t vy = wasm_f32x4_div(vone, vt);
    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vy, 0);
    }
  }
}
