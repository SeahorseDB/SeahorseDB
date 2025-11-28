#include "vdb/metrics/metrics_collection.hh"
#include <iostream>

namespace vdb {
namespace metrics {

thread_local size_t thread_metrics_id = SIZE_MAX;
thread_local std::shared_ptr<bool> thread_life = std::make_shared<bool>(true);
constexpr int kNumberOfThreadMetricsArray = 1024;

MetricsCollection::MetricsCollection()
    : thread_metrics_{kNumberOfThreadMetricsArray},
      thread_metrics_status_{kNumberOfThreadMetricsArray},
      thread_metrics_cursor_{0} {}

MetricsCollection::MetricsCollection(const MetricsCollection &other)
    : thread_metrics_{other.thread_metrics_},
      thread_metrics_status_{other.thread_metrics_status_},
      thread_metrics_cursor_{other.thread_metrics_cursor_.load()} {}

MetricsCollection &MetricsCollection::operator=(
    const MetricsCollection &other) {
  if (this == &other) return *this;
  thread_metrics_ = other.thread_metrics_;
  thread_metrics_status_ = other.thread_metrics_status_;
  thread_metrics_cursor_ = other.thread_metrics_cursor_.load();
  return *this;
}

std::shared_ptr<Metrics> MetricsCollection::GetDifferenceMetrics(
    const MetricsCollection &before, const MetricsCollection &after) {
  auto difference = vdb::make_shared<Metrics>();
  for (size_t i = 0; i < after.thread_metrics_.size(); i++) {
    const Metrics &metric_collection = after.thread_metrics_[i];
    difference->Add(metric_collection);
  }

  for (size_t i = 0; i < before.thread_metrics_.size(); i++) {
    const Metrics &metric_collection = before.thread_metrics_[i];
    difference->Subtract(metric_collection);
  }
  return difference;
}

bool MetricsCollection::IsMetricsUsed(int64_t i) {
  return !thread_metrics_status_[i].expired();
}

void MetricsCollection::AllocateMetricsToThread() {
  if (thread_metrics_id != SIZE_MAX) return;
  do {
    size_t logical_thread_id = thread_metrics_cursor_.fetch_add(1);
    thread_metrics_id = logical_thread_id % kNumberOfThreadMetricsArray;
    if (IsMetricsUsed(thread_metrics_id)) continue;
  } while (false);
  thread_metrics_status_[thread_metrics_id] = thread_life;
}

Metrics *MetricsCollection::GetMetrics(size_t id) {
  if (id == SIZE_MAX) {
    if (thread_metrics_id == SIZE_MAX) {
      AllocateMetricsToThread();
    }
    id = thread_metrics_id;
  }
  return &thread_metrics_[id];
}

void MetricsCollection::ClearMetrics() {
  for (auto &metrics : thread_metrics_) {
    metrics.Clear();
  }
}
std::string MetricsCollection::ToString(bool brief) {
  std::stringstream ss;
  for (size_t i = 0; i < thread_metrics_.size(); i++) {
    auto &metrics = thread_metrics_[i];
    if (IsMetricsUsed(i)) {
      ss << metrics.ToString(brief) << std::endl;
    }
  }
  ss << std::endl;
  return ss.str();
}
}  // namespace metrics
}  // namespace vdb
