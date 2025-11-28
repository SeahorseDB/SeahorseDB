#include <iomanip>
#include <memory>
#include <sstream>

#include "vdb/common/memory_allocator.hh"
#include "vdb/metrics/metric_info.hh"
#include "vdb/metrics/metrics.hh"
#include "vdb/metrics/timer.hh"

namespace vdb {
namespace metrics {
void Metrics::AddCount(size_t metric_index) {
  Metric &metric = metrics_[metric_index];
  metric.count++;
  used_ = true;
}

void Metrics::AddSize(size_t metric_index, size_t value) {
  Metric &metric = metrics_[metric_index];
  metric.size += value;
  if (metric.size_min > value || metric.size_min == 0) {
    metric.size_min = value;
  }
  if (metric.size_max < value || metric.size_max == 0) {
    metric.size_max = value;
  }
  used_ = true;
}

void Metrics::AddTime(size_t metric_index, size_t elapsed_time) {
  Metric &metric = metrics_[metric_index];
  metric.time += elapsed_time;
  if (metric.time_min > elapsed_time || metric.time_min == 0) {
    metric.time_min = elapsed_time;
  }
  if (metric.time_max < elapsed_time || metric.time_max == 0) {
    metric.time_max = elapsed_time;
  }
  used_ = true;
}

void Metrics::Add(const Metrics &other) {
  if (!other.Used()) return;
  for (size_t i = 0; i < kMetricIndexMax; i++) {
    Metric &metric = metrics_[i];
    const Metric &other_metric = other.metrics_[i];
    metric.count += other_metric.count;
    metric.size += other_metric.size;
    metric.time += other_metric.time;
    if ((other_metric.size_min != 0 &&
         metric.size_min > other_metric.size_min) ||
        metric.size_min == 0) {
      metric.size_min = other_metric.size_min;
    }
    if ((other_metric.size_max != 0 &&
         metric.size_max < other_metric.size_max) ||
        metric.size_max == 0) {
      metric.size_max = other_metric.size_max;
    }
    if ((other_metric.time_min != 0 &&
         metric.time_min > other_metric.time_min) ||
        metric.time_min == 0) {
      metric.time_min = other_metric.time_min;
    }
    if ((other_metric.time_max != 0 &&
         metric.time_max < other_metric.time_max) ||
        metric.time_max == 0) {
      metric.time_max = other_metric.time_max;
    }
  }
  used_ = true;
}

void Metrics::Subtract(const Metrics &other) {
  if (!other.Used()) return;
  for (size_t i = 0; i < kMetricIndexMax; i++) {
    Metric &metric = metrics_[i];
    const Metric &other_metric = other.metrics_[i];
    metric.count -= other_metric.count;
    metric.size -= other_metric.size;
    metric.time -= other_metric.time;
  }
  used_ = true;
}

std::shared_ptr<Metrics> Metrics::GetDifference(const Metrics &other) {
  if (!Used()) {
    /* TODO shows system log */
    return vdb::make_shared<Metrics>();
  }
  auto difference = vdb::make_shared<Metrics>(*this);
  if (!other.Used()) {
    return difference;
  } else {
    difference->Subtract(other);
    return difference;
  }
}

std::string Metrics::ToString(bool brief) const {
  std::stringstream ss;
  if (!Used()) {
    /* unused metrics shows nothing. */
    return "";
  }
  for (size_t i = 0; i < kMetricIndexMax; i++) {
    const Metric &metric = metrics_[i];
    const MetricInfo &metric_info = metric_info_array[i];
    if (metric.count == 0) continue;

    if (brief) {
      ss << std::setw(20) << std::left << metric_info.name << ": ";
      ss << "[Count = " << metric.count << "] ";
      ss << "[Size  = " << metric.size << " <Avg = " << std::setprecision(3)
         << (double)metric.size / metric.count << ">] ";
      ss << "[Time  = " << TimeToString(metric.time)
         << " <Avg = " << TimeToString(metric.time / metric.count) << ">]";
      ss << std::endl;
    } else {
      ss << std::setw(20) << std::left << metric_info.name << ": " << std::endl;
      ss << "  [Count = " << metric.count << " (" << metric_info.count.label
         << ")]" << std::endl;
      ss << "  [Size  = " << metric.size << " <Avg = " << std::setprecision(3)
         << (double)metric.size / metric.count << "> ("
         << metric_info.size.label << ")]" << std::endl;
      ss << "  [Time = " << TimeToString(metric.time)
         << " <Avg = " << TimeToString(metric.time / metric.count) << "> ("
         << metric_info.time.label << ")]" << std::endl;
    }
  }
  return ss.str();
}

std::string Metrics::ToMarkdownTable(std::string topic) const {
  std::stringstream ss;
  ss << "## " << topic << std::endl;

  if (topic == "count") {
    ss << "| Metric                                            | Count    |"
       << std::endl;
    ss << "|:--------------------------------------------------|---------:|"
       << std::endl;
    for (size_t i = 0; i < kMetricIndexMax; i++) {
      const Metric &metric = metrics_[i];
      if (metric.count == 0) continue;
      ss << "| " << std::left << std::setw(50)
         << metric_info_array[i].name  // 49 -> 50
         << "| " << std::right << std::setw(8) << metric.count << " |"
         << std::endl;
    }
  } else if (topic == "time") {
    ss << "| Metric                                            | Count    | "
          "Avg Time         | Total Time       | Min Time        | Max Time    "
          "    |"
       << std::endl;
    ss << "|:--------------------------------------------------|---------:|:--"
          "---------------|:-----------------|:---------------|:---------------"
          "|"
       << std::endl;
    for (size_t i = 0; i < kMetricIndexMax; i++) {
      const Metric &metric = metrics_[i];
      if (metric.count == 0) continue;
      if (metric.time == 0) continue;
      std::string avg_time = TimeToString(metric.time / metric.count);
      std::string total_time = TimeToString(metric.time);
      std::string min_time = TimeToString(metric.time_min);
      std::string max_time = TimeToString(metric.time_max);
      ss << "| " << std::left << std::setw(50) << metric_info_array[i].name
         << "| " << std::right << std::setw(8) << metric.count << " | "
         << std::left << std::setw(16) << avg_time << " | " << std::left
         << std::setw(16) << total_time << " | " << std::left << std::setw(16)
         << min_time << " | " << std::left << std::setw(16) << max_time << " |"
         << std::endl;
    }
  } else if (topic == "size") {
    ss << "| Metric                                            | Count    | "
          "Avg Size   | Total Size     | Min Size    | Max Size    |"
       << std::endl;
    ss << "|:--------------------------------------------------|---------:|---"
          "--------:|---------------:|------------:|------------:|"
       << std::endl;
    for (size_t i = 0; i < kMetricIndexMax; i++) {
      const Metric &metric = metrics_[i];
      if (metric.count == 0) continue;
      if (metric.size == 0) continue;
      ss << "| " << std::left << std::setw(50) << metric_info_array[i].name
         << "| " << std::right << std::setw(8) << metric.count << " | "
         << std::right << std::setw(10) << std::fixed << std::setprecision(2)
         << (double)metric.size / metric.count << " | " << std::right
         << std::setw(14) << metric.size << " | " << std::right << std::setw(12)
         << metric.size_min << " | " << std::right << std::setw(12)
         << metric.size_max << " |" << std::endl;
    }
  }

  return ss.str();
}
}  // namespace metrics
}  // namespace vdb
