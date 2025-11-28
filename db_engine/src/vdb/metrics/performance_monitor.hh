#pragma once
#include "vdb/metrics/metrics_collection.hh"
#include <memory>

namespace vdb {
namespace metrics {
class PerformanceMonitor;
extern std::shared_ptr<PerformanceMonitor> performance_monitor;

class PerformanceMonitor {
 public:
  PerformanceMonitor();

  MetricsCollection *GetCollection();
  MetricsCollection *GetSavedCollection();
  void ResetCollection();
  void SaveCollection();
  std::shared_ptr<Metrics> EstimatePerformance();
  std::string GetPerformanceReport();

  MetricsCollection saved_collection_;
  MetricsCollection current_collection_;
};

}  // namespace metrics
}  // namespace vdb
