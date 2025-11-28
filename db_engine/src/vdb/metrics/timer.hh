#pragma once

#include <chrono>
#include <string>

namespace vdb {
namespace metrics {
class Timer {
 public:
  explicit Timer(bool start = true);
  void Start();
  void Stop();

  static int64_t GetCurrentTime();
  int64_t GetElapsedNanoTime() const;
  std::string ElapsedTimeToString() const;

 private:
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point end_time_;
};

std::string TimeToString(const int64_t nano_second);
}  // namespace metrics
}  // namespace vdb
