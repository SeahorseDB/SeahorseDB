#pragma once
#include <string>
#include <sstream>

#include "vdb/metrics/metric.hh"
namespace vdb {
namespace metrics {
class StatInfo {
 public:
  std::string ToString() const {
    std::stringstream ss;
    ss << "Label= " << label << std::endl;
    ss << "Desc=  " << desc << std::endl;
    return ss.str();
  }
  const std::string label;
  const std::string desc;
};

class MetricInfo {
 public:
  std::string ToString() const {
    std::stringstream ss;
    ss << "Name: " << name << std::endl;
    ss << "Count Stat:" << std::endl;
    ss << count.ToString() << std::endl;
    ss << "Time Stat:" << std::endl;
    ss << time.ToString() << std::endl;
    ss << "Size Stat:" << std::endl;
    ss << size.ToString() << std::endl;
    return ss.str();
  }
  const std::string name;
  StatInfo count;
  StatInfo time;
  StatInfo size;
};

extern MetricInfo metric_info_array[kMetricIndexMax];
}  // namespace metrics
}  // namespace vdb
