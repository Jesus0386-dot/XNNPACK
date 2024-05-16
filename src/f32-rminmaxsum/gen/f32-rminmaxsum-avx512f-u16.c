// Auto-generated file. Do not edit!
//   Template: src/f32-rminmaxsum/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>


void xnn_f32_rminmaxsum_ukernel__avx512f_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512 vmin0 = _mm512_set1_ps(*input);
  __m512 vmax0 = vmin0;
  __m512 vsum0 = _mm512_setzero_ps();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vt = _mm512_loadu_ps(input);
    input += 16;

    vmin0 = _mm512_min_ps(vmin0, vt);
    vmax0 = _mm512_max_ps(vmax0, vt);
    vsum0 = _mm512_add_ps(vsum0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_maskz_loadu_ps(vmask, input);

    vmin0 = _mm512_mask_min_ps(vmin0, vmask, vmin0, vt);
    vmax0 = _mm512_mask_max_ps(vmax0, vmask, vmax0, vt);
    vsum0 = _mm512_mask_add_ps(vsum0, vmask, vsum0, vt);
  }
  __m256 vmin256 = _mm256_min_ps(_mm512_castps512_ps256(vmin0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmin0), 1)));
  __m256 vmax256 = _mm256_max_ps(_mm512_castps512_ps256(vmax0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmax0), 1)));
  __m256 vsum256 = _mm256_add_ps(_mm512_castps512_ps256(vsum0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vsum0), 1)));
  __m128 vmin = _mm_min_ps(_mm256_castps256_ps128(vmin256), _mm256_extractf128_ps(vmin256, 1));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax256), _mm256_extractf128_ps(vmax256, 1));
  __m128 vsum = _mm_add_ps(_mm256_castps256_ps128(vsum256), _mm256_extractf128_ps(vsum256, 1));
  vmin = _mm_min_ps(vmin, _mm_movehl_ps(vmin, vmin));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
  vmin = _mm_min_ss(vmin, _mm_movehdup_ps(vmin));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  vsum = _mm_add_ss(vsum, _mm_movehdup_ps(vsum));
  _mm_store_ss(output, vmin);
  _mm_store_ss(output + 1, vmax);
  _mm_store_ss(output + 2, vsum);
}
