#include <sstream>
#include <iostream>
#include <iomanip>

#include "vdb/metrics/timer.hh"

namespace vdb {
namespace metrics {
Timer::Timer(bool start) {
  if (start) Start();
}

void Timer::Start() { start_time_ = std::chrono::steady_clock::now(); }

void Timer::Stop() { end_time_ = std::chrono::steady_clock::now(); }

int64_t Timer::GetCurrentTime() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t Timer::GetElapsedNanoTime() const {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ -
                                                              start_time_)
      .count();
}

std::string Timer::ElapsedTimeToString() const {
  return TimeToString(GetElapsedNanoTime());
}

std::string TimeToString(const int64_t time_in_nano) {
  constexpr int64_t hour_in_nano = 3600LL * 1000LL * 1000LL * 1000LL;
  constexpr int64_t minute_in_nano = 60LL * 1000LL * 1000LL * 1000LL;
  constexpr int64_t second_in_nano = 1000LL * 1000LL * 1000LL;
  constexpr int64_t milli_in_nano = 1000LL * 1000LL;
  constexpr int64_t micro_in_nano = 1000LL;
  std::stringstream ss;
  int64_t nano_second, micro_second, milli_second, second, minute, hour;
  if ((micro_second = (time_in_nano / micro_in_nano)) == 0) {
    /* shows in nano second */
    ss << time_in_nano << " ns";
  } else if ((milli_second = (time_in_nano / milli_in_nano)) == 0) {
    /* shows in micro second */
    nano_second = time_in_nano - (micro_second * 1000LL);
    ss << micro_second << "." << std::setfill('0') << std::setw(3)
       << nano_second << " us";
  } else if ((second = (time_in_nano / second_in_nano)) == 0) {
    /* shows in milli second */
    micro_second -= (milli_second * 1000LL);
    ss << milli_second << "." << std::setfill('0') << std::setw(3)
       << micro_second << " ms";
  } else if ((minute = (time_in_nano / minute_in_nano)) == 0) {
    /* shows in second */
    milli_second -= (second * 1000LL);
    ss << second << "." << std::setfill('0') << std::setw(3) << milli_second
       << " sec";
  } else if ((hour = (time_in_nano / hour_in_nano)) == 0) {
    /* shows in minute */
    second -= (minute * 60LL);
    ss << minute << " min ";
    ss << second << " sec";
  } else {
    /* shows in hour */
    minute -= (hour * 60LL);
    ss << hour << " hour ";
    ss << minute << " min";
  }
  return ss.str();
}

}  // namespace metrics
}  // namespace vdb
