#pragma once

#include <atomic>

namespace vdb {

template <typename T>
class AtomicCounterGuard {
 private:
  std::atomic<T>& counter_;

 public:
  AtomicCounterGuard(std::atomic<T>& counter) : counter_(counter) {
    ++counter_;
  }
  ~AtomicCounterGuard() { --counter_; }
};
}  // namespace vdb