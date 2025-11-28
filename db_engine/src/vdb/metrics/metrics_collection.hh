#pragma once

#include "vdb/common/memory_allocator.hh"
#include "vdb/metrics/metric.hh"
#include "vdb/metrics/metrics.hh"
#include <atomic>
#include <memory>

namespace vdb {
namespace metrics {

class MetricsCollection {
 public:
  MetricsCollection();
  MetricsCollection(const MetricsCollection &other);
  MetricsCollection &operator=(const MetricsCollection &other);

  void AllocateMetricsToThread();
  Metrics *GetMetrics(size_t id = SIZE_MAX);

  static std::shared_ptr<Metrics> GetDifferenceMetrics(
      const MetricsCollection &before, const MetricsCollection &after);

  bool IsMetricsUsed(int64_t i);
  void ClearMetrics();
  std::string ToString(bool brief = true);

 private:
  vdb::vector<Metrics> thread_metrics_;
  vdb::vector<std::weak_ptr<bool>> thread_metrics_status_;
  std::atomic_uint64_t thread_metrics_cursor_;
};

extern thread_local size_t thread_metrics_id;
}  // namespace metrics
}  // namespace vdb
