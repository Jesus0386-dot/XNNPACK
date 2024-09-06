// Auto-generated file. Do not edit!
//   Template: src/qu8-rdsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_qu8_rdsum_ukernel_7p7x__neon_c64(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const uint8_t* i0 = input;
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
    const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
    const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
    const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

    uint32x4_t vacc0123 = vdupq_n_u32(0);
    uint32x4_t vacc4567 = vdupq_n_u32(0);
    uint32x4_t vacc89AB = vdupq_n_u32(0);
    uint32x4_t vaccCDEF = vdupq_n_u32(0);
    uint32x4_t vaccGHIJ = vdupq_n_u32(0);
    uint32x4_t vaccKLMN = vdupq_n_u32(0);
    uint32x4_t vaccOPQR = vdupq_n_u32(0);
    uint32x4_t vaccSTUV = vdupq_n_u32(0);
    uint32x4_t vaccWXYZ = vdupq_n_u32(0);
    uint32x4_t vaccabcd = vdupq_n_u32(0);
    uint32x4_t vaccerfg = vdupq_n_u32(0);
    uint32x4_t vacchijl = vdupq_n_u32(0);
    uint32x4_t vaccmnop = vdupq_n_u32(0);
    uint32x4_t vaccqrst = vdupq_n_u32(0);
    uint32x4_t vaccuvqx = vdupq_n_u32(0);
    uint32x4_t vaccyz01 = vdupq_n_u32(0);

    // 256 uint8s may be summed into an uint16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      uint16x8_t vacc16_01234567 = vmovq_n_u16(0);
      uint16x8_t vacc16_89ABCDEF = vmovq_n_u16(0);
      uint16x8_t vacc16_GHIJKLMN = vmovq_n_u16(0);
      uint16x8_t vacc16_OPQRSTUV = vmovq_n_u16(0);
      uint16x8_t vacc16_WXYZabcd = vmovq_n_u16(0);
      uint16x8_t vacc16_erfghijl = vmovq_n_u16(0);
      uint16x8_t vacc16_mnopqrst = vmovq_n_u16(0);
      uint16x8_t vacc16_uvqxyz01 = vmovq_n_u16(0);
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        uint8x8_t vin01234567;
        uint8x8_t vin89ABCDEF;
        uint8x8_t vinGHIJKLMN;
        uint8x8_t vinOPQRSTUV;
        uint8x8_t vinWXYZabcd;
        uint8x8_t vinerfghijl;
        uint8x8_t vinmnopqrst;
        uint8x8_t vinuvqxyz01;
        vin01234567 = vld1_u8(&i0[0]);
        vin89ABCDEF = vld1_u8(&i0[8]);
        vinGHIJKLMN = vld1_u8(&i0[16]);
        vinOPQRSTUV = vld1_u8(&i0[24]);
        vinWXYZabcd = vld1_u8(&i0[32]);
        vinerfghijl = vld1_u8(&i0[40]);
        vinmnopqrst = vld1_u8(&i0[48]);
        vinuvqxyz01 = vld1_u8(&i0[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        vin01234567 = vld1_u8(&i1[0]);
        vin89ABCDEF = vld1_u8(&i1[8]);
        vinGHIJKLMN = vld1_u8(&i1[16]);
        vinOPQRSTUV = vld1_u8(&i1[24]);
        vinWXYZabcd = vld1_u8(&i1[32]);
        vinerfghijl = vld1_u8(&i1[40]);
        vinmnopqrst = vld1_u8(&i1[48]);
        vinuvqxyz01 = vld1_u8(&i1[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        vin01234567 = vld1_u8(&i2[0]);
        vin89ABCDEF = vld1_u8(&i2[8]);
        vinGHIJKLMN = vld1_u8(&i2[16]);
        vinOPQRSTUV = vld1_u8(&i2[24]);
        vinWXYZabcd = vld1_u8(&i2[32]);
        vinerfghijl = vld1_u8(&i2[40]);
        vinmnopqrst = vld1_u8(&i2[48]);
        vinuvqxyz01 = vld1_u8(&i2[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        vin01234567 = vld1_u8(&i3[0]);
        vin89ABCDEF = vld1_u8(&i3[8]);
        vinGHIJKLMN = vld1_u8(&i3[16]);
        vinOPQRSTUV = vld1_u8(&i3[24]);
        vinWXYZabcd = vld1_u8(&i3[32]);
        vinerfghijl = vld1_u8(&i3[40]);
        vinmnopqrst = vld1_u8(&i3[48]);
        vinuvqxyz01 = vld1_u8(&i3[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        vin01234567 = vld1_u8(&i4[0]);
        vin89ABCDEF = vld1_u8(&i4[8]);
        vinGHIJKLMN = vld1_u8(&i4[16]);
        vinOPQRSTUV = vld1_u8(&i4[24]);
        vinWXYZabcd = vld1_u8(&i4[32]);
        vinerfghijl = vld1_u8(&i4[40]);
        vinmnopqrst = vld1_u8(&i4[48]);
        vinuvqxyz01 = vld1_u8(&i4[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        vin01234567 = vld1_u8(&i5[0]);
        vin89ABCDEF = vld1_u8(&i5[8]);
        vinGHIJKLMN = vld1_u8(&i5[16]);
        vinOPQRSTUV = vld1_u8(&i5[24]);
        vinWXYZabcd = vld1_u8(&i5[32]);
        vinerfghijl = vld1_u8(&i5[40]);
        vinmnopqrst = vld1_u8(&i5[48]);
        vinuvqxyz01 = vld1_u8(&i5[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        vin01234567 = vld1_u8(&i6[0]);
        vin89ABCDEF = vld1_u8(&i6[8]);
        vinGHIJKLMN = vld1_u8(&i6[16]);
        vinOPQRSTUV = vld1_u8(&i6[24]);
        vinWXYZabcd = vld1_u8(&i6[32]);
        vinerfghijl = vld1_u8(&i6[40]);
        vinmnopqrst = vld1_u8(&i6[48]);
        vinuvqxyz01 = vld1_u8(&i6[56]);
        vacc16_01234567 = vaddw_u8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_u8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_u8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_u8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vacc16_WXYZabcd = vaddw_u8(vacc16_WXYZabcd, vinWXYZabcd);
        vacc16_erfghijl = vaddw_u8(vacc16_erfghijl, vinerfghijl);
        vacc16_mnopqrst = vaddw_u8(vacc16_mnopqrst, vinmnopqrst);
        vacc16_uvqxyz01 = vaddw_u8(vacc16_uvqxyz01, vinuvqxyz01);
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0123 = vaddw_u16(vacc0123, vget_low_u16(vacc16_01234567));
      vacc4567 = vaddw_u16(vacc4567, vget_high_u16(vacc16_01234567));
      vacc89AB = vaddw_u16(vacc89AB, vget_low_u16(vacc16_89ABCDEF));
      vaccCDEF = vaddw_u16(vaccCDEF, vget_high_u16(vacc16_89ABCDEF));
      vaccGHIJ = vaddw_u16(vaccGHIJ, vget_low_u16(vacc16_GHIJKLMN));
      vaccKLMN = vaddw_u16(vaccKLMN, vget_high_u16(vacc16_GHIJKLMN));
      vaccOPQR = vaddw_u16(vaccOPQR, vget_low_u16(vacc16_OPQRSTUV));
      vaccSTUV = vaddw_u16(vaccSTUV, vget_high_u16(vacc16_OPQRSTUV));
      vaccWXYZ = vaddw_u16(vaccWXYZ, vget_low_u16(vacc16_WXYZabcd));
      vaccabcd = vaddw_u16(vaccabcd, vget_high_u16(vacc16_WXYZabcd));
      vaccerfg = vaddw_u16(vaccerfg, vget_low_u16(vacc16_erfghijl));
      vacchijl = vaddw_u16(vacchijl, vget_high_u16(vacc16_erfghijl));
      vaccmnop = vaddw_u16(vaccmnop, vget_low_u16(vacc16_mnopqrst));
      vaccqrst = vaddw_u16(vaccqrst, vget_high_u16(vacc16_mnopqrst));
      vaccuvqx = vaddw_u16(vaccuvqx, vget_low_u16(vacc16_uvqxyz01));
      vaccyz01 = vaddw_u16(vaccyz01, vget_high_u16(vacc16_uvqxyz01));
      r = doz(r, 252);
    }

    const uint32_t* o = output;
    uint32x4_t vo0123 = vld1q_u32(o); o += 4;
    uint32x4_t vo4567 = vld1q_u32(o); o += 4;
    uint32x4_t vo89AB = vld1q_u32(o); o += 4;
    uint32x4_t voCDEF = vld1q_u32(o); o += 4;
    uint32x4_t voGHIJ = vld1q_u32(o); o += 4;
    uint32x4_t voKLMN = vld1q_u32(o); o += 4;
    uint32x4_t voOPQR = vld1q_u32(o); o += 4;
    uint32x4_t voSTUV = vld1q_u32(o); o += 4;
    uint32x4_t voWXYZ = vld1q_u32(o); o += 4;
    uint32x4_t voabcd = vld1q_u32(o); o += 4;
    uint32x4_t voerfg = vld1q_u32(o); o += 4;
    uint32x4_t vohijl = vld1q_u32(o); o += 4;
    uint32x4_t vomnop = vld1q_u32(o); o += 4;
    uint32x4_t voqrst = vld1q_u32(o); o += 4;
    uint32x4_t vouvqx = vld1q_u32(o); o += 4;
    uint32x4_t voyz01 = vld1q_u32(o); o += 4;
    vacc0123 = vaddq_u32(vo0123, vacc0123);
    vacc4567 = vaddq_u32(vo4567, vacc4567);
    vacc89AB = vaddq_u32(vo89AB, vacc89AB);
    vaccCDEF = vaddq_u32(voCDEF, vaccCDEF);
    vaccGHIJ = vaddq_u32(voGHIJ, vaccGHIJ);
    vaccKLMN = vaddq_u32(voKLMN, vaccKLMN);
    vaccOPQR = vaddq_u32(voOPQR, vaccOPQR);
    vaccSTUV = vaddq_u32(voSTUV, vaccSTUV);
    vaccWXYZ = vaddq_u32(voWXYZ, vaccWXYZ);
    vaccabcd = vaddq_u32(voabcd, vaccabcd);
    vaccerfg = vaddq_u32(voerfg, vaccerfg);
    vacchijl = vaddq_u32(vohijl, vacchijl);
    vaccmnop = vaddq_u32(vomnop, vaccmnop);
    vaccqrst = vaddq_u32(voqrst, vaccqrst);
    vaccuvqx = vaddq_u32(vouvqx, vaccuvqx);
    vaccyz01 = vaddq_u32(voyz01, vaccyz01);
    vst1q_u32(output, vacc0123); output += 4;
    vst1q_u32(output, vacc4567); output += 4;
    vst1q_u32(output, vacc89AB); output += 4;
    vst1q_u32(output, vaccCDEF); output += 4;
    vst1q_u32(output, vaccGHIJ); output += 4;
    vst1q_u32(output, vaccKLMN); output += 4;
    vst1q_u32(output, vaccOPQR); output += 4;
    vst1q_u32(output, vaccSTUV); output += 4;
    vst1q_u32(output, vaccWXYZ); output += 4;
    vst1q_u32(output, vaccabcd); output += 4;
    vst1q_u32(output, vaccerfg); output += 4;
    vst1q_u32(output, vacchijl); output += 4;
    vst1q_u32(output, vaccmnop); output += 4;
    vst1q_u32(output, vaccqrst); output += 4;
    vst1q_u32(output, vaccuvqx); output += 4;
    vst1q_u32(output, vaccyz01); output += 4;

    input = (const uint8_t*) ((uintptr_t) input + 64 * sizeof(uint8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 uint8s may be summed into an uint16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const uint8_t* i0 = input;
      const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
      const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
      const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
      const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
      const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
      const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

      uint32x4_t vacc0 = vdupq_n_u32(0);
      uint32x4_t vacc1 = vdupq_n_u32(0);

      for (; num_batches > 0; --num_batches) {
        uint16x8_t vacc16 = vmovq_n_u16(0);
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          uint8x8_t vin0 = vld1_u8(&i0[0]);
          uint8x8_t vin1 = vld1_u8(&i1[0]);
          uint8x8_t vin2 = vld1_u8(&i2[0]);
          uint8x8_t vin3 = vld1_u8(&i3[0]);
          uint8x8_t vin4 = vld1_u8(&i4[0]);
          uint8x8_t vin5 = vld1_u8(&i5[0]);
          uint8x8_t vin6 = vld1_u8(&i6[0]);
          vacc16 = vaddw_u8(vacc16, vin0);
          vacc16 = vaddw_u8(vacc16, vin1);
          vacc16 = vaddw_u8(vacc16, vin2);
          vacc16 = vaddw_u8(vacc16, vin3);
          vacc16 = vaddw_u8(vacc16, vin4);
          vacc16 = vaddw_u8(vacc16, vin5);
          vacc16 = vaddw_u8(vacc16, vin6);
          i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = vaddw_u16(vacc0, vget_low_u16(vacc16));
        vacc1 = vaddw_u16(vacc1, vget_high_u16(vacc16));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        uint32x4_t vo0 = vld1q_u32(output);
        uint32x4_t vo1 = vld1q_u32(output + 4);
        vo0 = vaddq_u32(vo0, vacc0);
        vo1 = vaddq_u32(vo1, vacc1);
        vst1q_u32(output, vo0); output += 4;
        vst1q_u32(output, vo1); output += 4;
        channels -= 8;
        input = (const uint8_t*) ((uintptr_t) input + 8 * sizeof(uint8_t));
      } else {
        if (channels & 4) {
          uint32x4_t vo = vld1q_u32(output);
          vo = vaddq_u32(vo, vacc0);
          vst1q_u32(output, vo); output += 4;
          vacc0 = vacc1;
        }
        if (channels & 2) {
          uint32x2_t vo = vld1_u32(output);
          vo = vadd_u32(vo, vget_low_u32(vacc0));
          vst1_u32(output, vo); output += 2;
          vacc0 = vextq_u32(vacc0, vacc0, 2);
        }
        if (channels & 1) {
          *output += vgetq_lane_u32(vacc0, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
