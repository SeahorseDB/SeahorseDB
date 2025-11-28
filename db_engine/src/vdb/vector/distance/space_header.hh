#pragma once
#include <cstdint>
#include <cstddef>

#include "vdb/vector/common/simd_header.hh"
#include "vdb/vector/common/params.hh"

/*
 * Distance function type
 *
 * @param pVect1: pointer to the first vector
 * @param pVect2: pointer to the second vector
 * @param param_ptr: pointer to the parameters
 */
using DISTFUNC_FLOAT32 = float (*)(const void *, const void *, const void *);

namespace vdb {
enum class DistanceSpace : uint8_t {
  kIP = 0,
  kL2 = 1,
  kCosine = 2,
  kL2WithNorm = 3,
  kMax = 4
};

class SpaceInterface {
 public:
  // virtual void search(void *);
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC_FLOAT32 get_dist_func() = 0;

  virtual void *get_dist_func_param() = 0;

  virtual ~SpaceInterface() {}
};

static inline float ApplyNorm(DistanceSpace space, float distance,
                              const float *norm_1, const float *norm_2) {
  if (space == DistanceSpace::kCosine) {
    auto inner_product = 1.0f - distance;
    inner_product /= *norm_1 * *norm_2;
    return 1.0f - inner_product;
  } else if (space == DistanceSpace::kL2WithNorm) {
    return distance / (*norm_1 * *norm_2);
  }
  return distance;
}

static inline float ApplyNormBitwise(DistanceSpace space, float distance,
                                     const float *norm_1, const float *norm_2) {
  if (space == DistanceSpace::kCosine || space == DistanceSpace::kIP) {
    auto inner_product = 1.0f - distance;
    inner_product /= *norm_1 * *norm_2;
    return 1.0f - inner_product;
  } else if (space == DistanceSpace::kL2WithNorm) {
    return distance / (*norm_1 * *norm_2);
  }
  return distance;
}
}  // namespace vdb
