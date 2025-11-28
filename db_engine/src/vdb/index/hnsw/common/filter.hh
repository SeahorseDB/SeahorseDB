#pragma once

#include <stdint.h>

namespace vdb {
namespace hnsw {
typedef uint64_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
class BaseFilterFunctor {
 public:
  virtual bool operator()([[maybe_unused]] labeltype id) { return true; }
  virtual ~BaseFilterFunctor() {};
};

}  // namespace hnsw
}  // namespace vdb