#pragma once

#include <atomic>

namespace vdb {

class Spinlock {
 public:
  Spinlock() = default;
  ~Spinlock() = default;

  void Lock();
  void Unlock();
  bool IsLocked() const;

 private:
  std::atomic<bool> lock_flag_ = {false};
};
}  // namespace vdb
