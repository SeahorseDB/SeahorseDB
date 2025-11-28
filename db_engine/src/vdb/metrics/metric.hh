#pragma once

#include <sstream>
#include <string_view>
#include <string>

namespace vdb {
namespace metrics {

struct Metric {
  uint64_t count;
  uint64_t time_min;
  uint64_t time_max;
  uint64_t time;
  uint64_t size_min;
  uint64_t size_max;
  uint64_t size;
};

enum MetricIndex {
#define MetricMacro(metric_name, count_label, count_desc, time_label, \
                    time_desc, size_label, size_desc, for_debug)      \
  metric_name,
#include "input/metric.input"
#undef MetricMacro
  kMetricIndexMax
};
}  // namespace metrics
}  // namespace vdb
