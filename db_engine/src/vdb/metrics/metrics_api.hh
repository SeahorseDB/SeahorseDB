#pragma once

#include <memory>

#include "vdb/metrics/metric.hh"

namespace vdb {
namespace metrics {

enum TimerAction { kStart, kEnd, kMax };

void CollectCount(MetricIndex metric_index);
void CollectValue(MetricIndex metric_index, size_t value);

/* collect size value without incresing count */
void CollectOnlyValue(MetricIndex metric_index, size_t value);
/* collect time directly in nano second */
void CollectTime(MetricIndex metric_index, int64_t time);
/* collect time by timer class (start, end) */
void CollectDuration(MetricIndex metric_index, TimerAction action);

inline void CollectDurationStart(MetricIndex metric_index) {
  CollectDuration(metric_index, kStart);
}

inline void CollectDurationEnd(MetricIndex metric_index) {
  CollectDuration(metric_index, kEnd);
}

void CreateMetricsCollection();

class ScopedDurationMetric {
 private:
  MetricIndex metric_index_;

 public:
  explicit ScopedDurationMetric(MetricIndex metric_index)
      : metric_index_(metric_index) {
    CollectDurationStart(metric_index_);
  }

  ~ScopedDurationMetric() { CollectDurationEnd(metric_index_); }

  ScopedDurationMetric(const ScopedDurationMetric&) = delete;
  ScopedDurationMetric& operator=(const ScopedDurationMetric&) = delete;
  ScopedDurationMetric(ScopedDurationMetric&&) = delete;
  ScopedDurationMetric& operator=(ScopedDurationMetric&&) = delete;
};

}  // namespace metrics
}  // namespace vdb
