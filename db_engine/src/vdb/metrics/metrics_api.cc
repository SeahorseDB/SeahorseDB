#include "vdb/common/memory_allocator.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/server_configuration.hh"
#include "vdb/metrics/metrics_api.hh"
#include "vdb/metrics/metric_info.hh"
#include "vdb/metrics/performance_monitor.hh"
#include "vdb/metrics/timer.hh"
namespace vdb {
namespace metrics {

void CollectCount(MetricIndex metric_index) {
  if (performance_monitor == nullptr) {
    return;
  }
  auto metrics = performance_monitor->GetCollection()->GetMetrics();
  metrics->AddCount(metric_index);
}

void CollectValue(MetricIndex metric_index, size_t value) {
  if (performance_monitor == nullptr) {
    return;
  }
  auto metrics = performance_monitor->GetCollection()->GetMetrics();
  metrics->AddSize(metric_index, value);
  metrics->AddCount(metric_index);
}

void CollectOnlyValue(MetricIndex metric_index, size_t value) {
  if (performance_monitor == nullptr) {
    return;
  }
  auto metrics = performance_monitor->GetCollection()->GetMetrics();
  metrics->AddSize(metric_index, value);
}

void CollectTime(MetricIndex metric_index, int64_t time) {
  if (performance_monitor == nullptr) {
    return;
  }
  auto metrics = performance_monitor->GetCollection()->GetMetrics();
  metrics->AddTime(metric_index, time);
  metrics->AddCount(metric_index);
}

void CollectDuration(MetricIndex metric_index, TimerAction action) {
  if (performance_monitor == nullptr) {
    return;
  }
  constexpr size_t timer_count = 10;
  static thread_local Timer timers[timer_count];
  static thread_local MetricIndex metric_indexes[timer_count];
  static thread_local int32_t used_timer_count = 0;

  auto metrics = performance_monitor->GetCollection()->GetMetrics();
  if (action == TimerAction::kStart) {
    auto &current_timer = timers[used_timer_count];
    current_timer.Start();
    metric_indexes[used_timer_count] = metric_index;
    ++used_timer_count;
  } else /* TimerAction::kEnd */ {
    if (used_timer_count == 0) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogAlways,
                 "Metric Nested Timer Error: (%s) timer end is called before "
                 "timer starts. "
                 "time is not collected.",
                 metric_info_array[metric_index].name.data());
      return;
    }
    --used_timer_count;
    auto &current_timer = timers[used_timer_count];
    current_timer.Stop();
    MetricIndex saved_index = metric_indexes[used_timer_count];
    if (saved_index != metric_index) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogAlways,
                 "Metric Nested Timer Error: Different Metrics are used when "
                 "starting(%s) and stopping(%s) same timer. "
                 "time is not collected.",
                 metric_info_array[saved_index].name.data(),
                 metric_info_array[metric_index].name.data());
      return;
    }
    metrics->AddTime(metric_index, current_timer.GetElapsedNanoTime());
    metrics->AddCount(metric_index);
  }
}

}  // namespace metrics
}  // namespace vdb
