#include <gtest/gtest.h>
#include <iterator>
#include <memory>

#include "vdb/metrics/metrics_api.hh"
#include "vdb/metrics/metric_info.hh"
#include "vdb/metrics/metrics_collection.hh"
#include "vdb/metrics/performance_monitor.hh"
#include "vdb/metrics/timer.hh"
#include "vdb/tests/util_for_test.hh"
#include "vdb/common/system_log.hh"

using namespace vdb::tests;

namespace vdb {

class GlobalEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    server.vdb_active_set_size_limit = 10000;
    server.allow_bg_index_thread = false;
    // std::cout << "Global setup before any test case runs." << std::endl;
    /* disable redis server log */
#ifdef _DEBUG_GTEST
    server.verbosity = LL_DEBUG;
#else
    server.verbosity = LL_NOTHING;
#endif
    server.logfile = empty_string.data();
    AllocatePerformanceMonitor();
  }

  void TearDown() override {
    // std::cout << "Global teardown after all test cases have run." <<
    // std::endl;
  }
};

class MetricTestSuite : public ::testing::Test {
 protected:
  void SetUp() override {
    AllocateTableDictionary();
    metrics::performance_monitor->ResetCollection();
    metrics::thread_metrics_id = SIZE_MAX;
  }

  void TearDown() override { DeallocateTableDictionary(); }
};

class TimerTest : public MetricTestSuite {};
class MetricsTest : public MetricTestSuite {};
class MetricsCollectionTest : public MetricTestSuite {};
class PerformanceMonitorTest : public MetricTestSuite {};

TEST_F(TimerTest, SimpleTimer) {
  metrics::Timer auto_start;
  sleep(1);
  auto_start.Stop();
  metrics::Timer non_auto_start(false);
  non_auto_start.Start();
  sleep(1);
  non_auto_start.Stop();

  ASSERT_GT(auto_start.GetElapsedNanoTime(), 1000LL * 1000LL * 1000LL);
  ASSERT_LT(auto_start.GetElapsedNanoTime(), 2 * 1000LL * 1000LL * 1000LL);
  ASSERT_GT(non_auto_start.GetElapsedNanoTime(), 1000LL * 1000LL * 1000LL);
  ASSERT_LT(non_auto_start.GetElapsedNanoTime(), 2 * 1000LL * 1000LL * 1000LL);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "%s",
             auto_start.ElapsedTimeToString().data());
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "%s",
             non_auto_start.ElapsedTimeToString().data());
}

TEST_F(MetricsTest, MetricInfoTest) {
  for (auto metric_info : metrics::metric_info_array) {
    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "\n%s",
               metric_info.ToString().data());
  }
}

TEST_F(MetricsTest, InitialMetricsTest) {
  metrics::Metrics metrics;
  for (int i = 0; i < metrics::kMetricIndexMax; i++) {
    for (int j = 0; j < 100000; j++) {
      metrics.AddCount(i);
    }
    metrics.AddSize(i, 100000);
    metrics.AddTime(i, 1000000000000LL);
  }
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "\nbrief\n%s\ndetail\n%s",
             metrics.ToString().data(), metrics.ToString(false).data());
}

TEST_F(MetricsCollectionTest, CollectionTest) {
  auto collection = vdb::make_shared<metrics::MetricsCollection>();
  collection->AllocateMetricsToThread();
  collection->GetMetrics()->AddCount(metrics::MetricIndex::CollectCountOnly);
  collection->GetMetrics()->AddSize(metrics::MetricIndex::IndexAddPoint, 10);
  collection->GetMetrics()->AddCount(metrics::MetricIndex::IndexAddPoint);
  collection->GetMetrics()->AddCount(metrics::MetricIndex::CollectSizeOnly);
  collection->GetMetrics()->AddSize(metrics::MetricIndex::CollectSizeOnly, 5);
  metrics::Timer tmr;
  sleep(1);
  tmr.Stop();
  collection->GetMetrics()->AddCount(metrics::MetricIndex::CollectTimeOnly);
  collection->GetMetrics()->AddTime(metrics::MetricIndex::CollectTimeOnly,
                                    tmr.GetElapsedNanoTime());
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "\nbrief\n%s",
             collection->ToString().c_str());
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "\ndetail\n%s",
             collection->ToString(false).c_str());
}

TEST_F(PerformanceMonitorTest, SimpleTest) {
  metrics::CollectCount(metrics::MetricIndex::CollectCountOnly);
  metrics::CollectValue(metrics::MetricIndex::IndexAddPoint, 10);
  metrics::CollectValue(metrics::MetricIndex::CollectSizeOnly, 5);
  metrics::CollectDuration(metrics::MetricIndex::CollectTimeOnly,
                           metrics::TimerAction::kStart);
  sleep(1);
  metrics::CollectDuration(metrics::MetricIndex::CollectTimeOnly,
                           metrics::TimerAction::kEnd);
  metrics::CollectDurationStart(metrics::MetricIndex::CollectTimeOnly);
  sleep(1);
  metrics::CollectDurationEnd(metrics::MetricIndex::CollectTimeOnly);
  auto current_collection = metrics::performance_monitor->GetCollection();
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "\nbrief\n%s",
             current_collection->ToString().c_str());
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "\ndetail\n%s",
             current_collection->ToString(false).c_str());
}

}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::GlobalEnvironment);
  return RUN_ALL_TESTS();
}
