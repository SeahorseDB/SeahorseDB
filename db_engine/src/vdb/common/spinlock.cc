#include <atomic>

#include "vdb/common/spinlock.hh"

namespace vdb {
void Spinlock::Lock() {
  bool expected = false;
  while (!lock_flag_.compare_exchange_weak(expected, true,
                                           std::memory_order_acquire)) {
    expected = false;
  }
}

void Spinlock::Unlock() { lock_flag_.store(false, std::memory_order_release); }

bool Spinlock::IsLocked() const {
  return lock_flag_.load(std::memory_order_acquire);
}
}  // namespace vdb
