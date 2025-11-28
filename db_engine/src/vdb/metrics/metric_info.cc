#include <vdb/metrics/metric_info.hh>

namespace vdb {
namespace metrics {
/* automatically generated code
 * Don't Edit it. */
MetricInfo metric_info_array[kMetricIndexMax] = {
#define MetricMacro(metric_name, count_label, count_desc, time_label, \
                    time_desc, size_label, size_desc, for_debug)      \
  {/* metric name */ #metric_name,                                    \
   /* count label, desc */ {count_label, count_desc},                 \
   /* time  label, desc */ {time_label, time_desc},                   \
   /* size  label, desc */ {size_label, size_desc}},
#include "input/metric.input"
#undef MetricMacro
};
}  // namespace metrics
}  // namespace vdb
