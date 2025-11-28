#include "vdb/common/memory_allocator.hh"
#include "vdb/metrics/performance_monitor.hh"

namespace vdb {
namespace metrics {
std::shared_ptr<PerformanceMonitor> performance_monitor = nullptr;

PerformanceMonitor::PerformanceMonitor()
    : saved_collection_{}, current_collection_{} {}

MetricsCollection* PerformanceMonitor::GetCollection() {
  return &current_collection_;
}

MetricsCollection* PerformanceMonitor::GetSavedCollection() {
  return &saved_collection_;
}

void PerformanceMonitor::ResetCollection() {
  current_collection_.ClearMetrics();
  saved_collection_.ClearMetrics();
}

void PerformanceMonitor::SaveCollection() {
  saved_collection_ = current_collection_;
}

std::shared_ptr<Metrics> PerformanceMonitor::EstimatePerformance() {
  return MetricsCollection::GetDifferenceMetrics(saved_collection_,
                                                 current_collection_);
}

std::string PerformanceMonitor::GetPerformanceReport() {
  // return EstimatePerformance()->ToString(false);
  return EstimatePerformance()->ToMarkdownTable("count") + "\n" +
         EstimatePerformance()->ToMarkdownTable("time") + "\n" +
         EstimatePerformance()->ToMarkdownTable("size");
}

}  // namespace metrics
}  // namespace vdb
