#pragma once

#include <cstdint>
#include <cstddef>

namespace vdb {

/* params of undefined case
 * size_t dim: dimension of the vector */

/* params of quantity and sign transition points case
 * size_t qty: quantity of the vector
 * uint8_t *sign_transition_points: pointer to the sign transition points */
struct QuantityAndSignTransitionParams {
  size_t qty;
  const uint8_t *sign_transition_points;
};

/* params of quantity and norm case
 * size_t qty: quantity of the vector
 * float norm_v1: norm of the first vector
 * float norm_v2: norm of the second vector */
struct QuantityAndNormParams {
  size_t qty;
  float norm_v1;
  float norm_v2;
};

/* params of quantity and object case
 * size_t qty: quantity of the vector
 * void *object: pointer to the object */
struct QuantityAndObjectParams {
  size_t qty;
  const void *object;
};
}  // namespace vdb