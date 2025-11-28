#pragma once
#include "vdb/vector/distance/space_header.hh"

namespace vdb {

/*
 * L2 Distance (float32)
 *
 * @param pVect1v: pointer to the first vector
 * @param pVect2v: pointer to the second vector
 * @param params_ptr: pointer to the size of the vector (quantity only)
 *
 * quantity:
 * @param size_t qty: quantity of the vector
 */
static float L2Sqr(const void *pVect1v, const void *pVect2v,
                   const void *params_ptr) {
  const float *pVect1 = static_cast<const float *>(pVect1v);
  const float *pVect2 = static_cast<const float *>(pVect2v);
  size_t qty = *static_cast<const size_t *>(params_ptr);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float t = *pVect1 - *pVect2;
    pVect1++;
    pVect2++;
    res += t * t;
  }
  return (res);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v,
                                  const void *params_ptr) {
  const float *pVect1 = static_cast<const float *>(pVect1v);
  const float *pVect2 = static_cast<const float *>(pVect2v);
  size_t qty = *static_cast<const size_t *>(params_ptr);
  float PORTABLE_ALIGN64 TmpRes[16];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m512 diff, v1, v2;
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    diff = _mm512_sub_ps(v1, v2);
    // sum = _mm512_fmadd_ps(diff, diff, sum);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
  }

  _mm512_store_ps(TmpRes, sum);
  float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] +
              TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] +
              TmpRes[15];

  return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v,
                               const void *params_ptr) {
  const float *pVect1 = static_cast<const float *>(pVect1v);
  const float *pVect2 = static_cast<const float *>(pVect2v);
  size_t qty = *static_cast<const size_t *>(params_ptr);
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] +
         TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v,
                               const void *params_ptr) {
  const float *pVect1 = static_cast<const float *>(pVect1v);
  const float *pVect2 = static_cast<const float *>(pVect2v);
  size_t qty = *static_cast<const size_t *>(params_ptr);
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  _mm_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC_FLOAT32 L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

static float L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v,
                                     const void *params_ptr) {
  size_t qty = *static_cast<const size_t *>(params_ptr);
  size_t qty16 = qty >> 4 << 4;
  float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
  const float *pVect1 = static_cast<const float *>(pVect1v) + qty16;
  const float *pVect2 = static_cast<const float *>(pVect2v) + qty16;

  size_t qty_left = qty - qty16;
  float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
  return (res + res_tail);
}
#endif

#if defined(USE_SSE)
static float L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v,
                           const void *params_ptr) {
  float PORTABLE_ALIGN32 TmpRes[8];
  const float *pVect1 = static_cast<const float *>(pVect1v);
  const float *pVect2 = static_cast<const float *>(pVect2v);
  size_t qty = *static_cast<const size_t *>(params_ptr);

  size_t qty4 = qty >> 2;

  const float *pEnd1 = pVect1 + (qty4 << 2);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }
  _mm_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v,
                                    const void *params_ptr) {
  size_t qty = *static_cast<const size_t *>(params_ptr);
  size_t qty4 = qty >> 2 << 2;

  float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
  size_t qty_left = qty - qty4;

  const float *pVect1 = static_cast<const float *>(pVect1v) + qty4;
  const float *pVect2 = static_cast<const float *>(pVect2v) + qty4;
  float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

  return (res + res_tail);
}
#endif

class L2SpaceFloat32 : public SpaceInterface {
  DISTFUNC_FLOAT32 fstdistfunc_;
  size_t data_size_;
  size_t dim_;

 public:
  L2SpaceFloat32(size_t dim) {
    fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
    if (AVX512Capable())
      L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
    else if (AVXCapable())
      L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#elif defined(USE_AVX)
    if (AVXCapable()) L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#endif

    if (dim % 16 == 0)
      fstdistfunc_ = L2SqrSIMD16Ext;
    else if (dim % 4 == 0)
      fstdistfunc_ = L2SqrSIMD4Ext;
    else if (dim > 16)
      fstdistfunc_ = L2SqrSIMD16ExtResiduals;
    else if (dim > 4)
      fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
    dim_ = dim;
    data_size_ = dim * sizeof(float);
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC_FLOAT32 get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &dim_; }

  ~L2SpaceFloat32() {}
};
}  // namespace vdb
