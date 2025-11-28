#pragma once
#include <memory>
#include "vdb/metrics/metric.hh"

namespace vdb {
namespace metrics {
class Metrics {
 public:
  Metrics() {
    memset(metrics_, 0, sizeof(Metric) * kMetricIndexMax);
    used_ = false;
  }
  Metrics(const Metrics &other) {
    memcpy(metrics_, other.metrics_, sizeof(Metric) * kMetricIndexMax);
    used_ = other.used_;
  }
  Metrics &operator=(const Metrics &other) {
    if (this == &other) return *this;
    memcpy(metrics_, other.metrics_, sizeof(Metric) * kMetricIndexMax);
    used_ = other.used_;
    return *this;
  }

  inline bool Used() const { return used_; }
  /* collect metrics */
  void AddCount(size_t metric_index);
  void AddSize(size_t metric_index, size_t value);
  void AddTime(size_t metric_index, size_t elapsed_time);

  void Add(const Metrics &other);
  void Subtract(const Metrics &other);

  std::shared_ptr<Metrics> GetDifference(const Metrics &other);

  void Clear() {
    if (Used()) {
      memset(metrics_, 0, sizeof(Metric) * kMetricIndexMax);
      used_ = false;
    }
  }

  std::string ToString(bool brief = true) const;
  std::string ToMarkdownTable(std::string topic) const;

  Metric metrics_[kMetricIndexMax];
  bool used_;
};

}  // namespace metrics
}  // namespace vdb
