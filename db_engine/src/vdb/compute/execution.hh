#pragma once

#include <memory>
#include <string_view>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <tuple>
#include <thread>
#include <atomic>
#include <functional>
#include <algorithm>
#include <limits>
#include <chrono>

#include <arrow/acero/api.h>
#include <arrow/util/async_generator.h>
#include <arrow/util/vector.h>
#include <arrow/util/async_generator_fwd.h>

#include "vdb/data/table.hh"
#include "vdb/data/expression.hh"
#include "vdb/data/label_info.hh"
#include "vdb/index/hnsw/common/filter.hh"
#include "vdb/common/util.hh"

namespace vdb {
// Tracks InactiveSets to prevent premature deallocation during async execution
class InactiveSetCleanupTracker {
 public:
  InactiveSetCleanupTracker() = default;

  void Track(std::shared_ptr<class InactiveSet> iset) {
    yielded_.emplace_back(std::move(iset));
  }

  std::shared_ptr<class InactiveSet> Get(size_t index) const {
    if (index < yielded_.size()) {
      return yielded_[index];
    }
    return nullptr;
  }

  size_t Size() const { return yielded_.size(); }
  bool Empty() const { return yielded_.empty(); }

  void Clear() { yielded_.clear(); }

 private:
  std::vector<std::shared_ptr<class InactiveSet>> yielded_;
};

arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> SegmentGenerator(
    std::vector<std::shared_ptr<Segment>> segments,
    const std::vector<int>& column_indices = {});

std::tuple<arrow::AsyncGenerator<std::optional<arrow::ExecBatch>>,
           std::shared_ptr<InactiveSetCleanupTracker>>
SegmentGeneratorWithTracking(std::vector<std::shared_ptr<Segment>> segments,
                             const std::vector<int>& column_indices);

arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeTable(
    std::shared_ptr<arrow::Table> table);

arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeRecordBatch(
    std::shared_ptr<arrow::RecordBatch> record_batch);

arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> RecordBatchGenerator(
    std::shared_ptr<arrow::RecordBatch> rb);

arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> RecordBatchesGenerator(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches);

class StreamingSinkConsumer : public arrow::acero::SinkNodeConsumer {
 public:
  // Processing modes
  enum class Mode {
    ACCUMULATE,  // Store all batches (for queries returning results)
    STREAMING    // Process batches immediately without storing (for full scans)
  };

  explicit StreamingSinkConsumer(Mode mode = Mode::ACCUMULATE)
      : mode_(mode), streaming_processor_(nullptr) {}

  explicit StreamingSinkConsumer(
      std::function<arrow::Status(const arrow::RecordBatch&)> processor)
      : mode_(Mode::STREAMING), streaming_processor_(std::move(processor)) {}

  arrow::Status Init(const std::shared_ptr<arrow::Schema>& schema,
                     arrow::acero::BackpressureControl* backpressure_control,
                     arrow::acero::ExecPlan* plan) override {
    (void)backpressure_control;  // Unused: no backpressure control needed
    (void)plan;                  // Unused: plan managed externally
    schema_ = schema;
    return arrow::Status::OK();
  }

  arrow::Status Consume(arrow::ExecBatch batch) override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (batch.length > 0) {
      auto rb_result = batch.ToRecordBatch(schema_);
      if (!rb_result.ok()) {
        return rb_result.status();
      }

      auto rb = rb_result.ValueUnsafe();

      if (mode_ == Mode::STREAMING && streaming_processor_) {
        ARROW_RETURN_NOT_OK(streaming_processor_(*rb));
      } else {
        batches_.push_back(rb);
      }
    }

    return arrow::Status::OK();
  }

  arrow::Future<> Finish() override {
    return arrow::Future<>::MakeFinished(arrow::Status::OK());
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> GetBatches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return batches_;
  }

  arrow::Result<std::shared_ptr<arrow::Table>> GetTable() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (batches_.empty()) {
      if (schema_) {
        std::vector<std::shared_ptr<arrow::ChunkedArray>> empty_columns;
        for (int i = 0; i < schema_->num_fields(); i++) {
          auto empty_array = arrow::MakeEmptyArray(schema_->field(i)->type());
          if (!empty_array.ok()) {
            return empty_array.status();
          }
          empty_columns.push_back(
              std::make_shared<arrow::ChunkedArray>(empty_array.ValueUnsafe()));
        }
        return arrow::Table::Make(schema_, empty_columns);
      }
      return arrow::Status::Invalid("No batches received");
    }
    return arrow::Table::FromRecordBatches(batches_);
  }

  void ClearBatches() {
    std::lock_guard<std::mutex> lock(mutex_);
    batches_.clear();
    batches_.shrink_to_fit();
  }

  Mode GetMode() const { return mode_; }

 private:
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  std::shared_ptr<arrow::Schema> schema_;
  Mode mode_;
  std::function<arrow::Status(const arrow::RecordBatch&)> streaming_processor_;
};

class FilterFunctor {
 public:
  FilterFunctor(std::shared_ptr<vdb::expression::Expression> expression,
                std::shared_ptr<vdb::Segment> segment)
      : pred_(
            std::dynamic_pointer_cast<vdb::expression::Predicate>(expression)),
        segment_(segment) {
    if (pred_) {
      auto schema = segment_->GetSchema();
      for (const auto& col_name : pred_->GetReferencedColumns()) {
        int col_idx = schema->GetFieldIndex(col_name);
        if (col_idx != -1) {
          filter_column_indices_.push_back(col_idx);
        }
      }
      filter_column_indices_.push_back(schema->num_fields() - 1);
    }
  }

  virtual ~FilterFunctor() = default;

  bool operator()(uint64_t label) {
    auto set_id = LabelInfo::GetSetNumber(label);
    auto record_number = LabelInfo::GetRecordNumber(label);

    auto result =
        segment_->GetRecordbatchWithSet(set_id, filter_column_indices_);
    if (!result.ok()) {
      return false;
    }

    auto [rb, set] = result.ValueUnsafe();
    if (!rb) {
      return false;
    }

    auto eval_result = pred_->EvaluateRecord(rb, record_number);
    if (!eval_result.ok()) {
      return false;  // Evaluation failed, filter out this record
    }
    return eval_result.ValueUnsafe();
  }

 private:
  std::shared_ptr<vdb::expression::Predicate> pred_;
  std::shared_ptr<vdb::Segment> segment_;
  std::vector<int> filter_column_indices_;
};

// Adapter class to make VDB's FilterFunctor compatible with
// hnswlib::BaseFilterFunctor
class AnnFilterFunctor : public vdb::hnsw::BaseFilterFunctor {
 public:
  explicit AnnFilterFunctor(std::shared_ptr<FilterFunctor> filter)
      : filter_(filter) {}

  bool operator()(vdb::hnsw::labeltype id) override {
    return filter_ ? (*filter_)(static_cast<uint64_t>(id)) : true;
  }

 private:
  std::shared_ptr<FilterFunctor> filter_;
};

// Iterative result set that uses RecordBatchReader directly
// This avoids materializing the entire Table in memory
class IterativeResultSet {
 public:
  explicit IterativeResultSet(
      std::shared_ptr<arrow::RecordBatchReader> reader,
      size_t limit = std::numeric_limits<size_t>::max(),
      std::shared_ptr<InactiveSetCleanupTracker> cleanup_tracker = nullptr,
      std::shared_ptr<class Table> vdb_table = nullptr,
      bool include_internal_columns = false);

  ~IterativeResultSet();

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> Next();

  bool HasNext() const;

  bool GetIncludeInternalColumns() const { return include_internal_columns_; }

  // TTL management for automatic cleanup of inactive scans
  void UpdateAccessTime() {
    last_access_time_ = std::chrono::steady_clock::now();
  }

  bool IsExpired(std::chrono::seconds ttl) const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_access_time_);
    return elapsed >= ttl;
  }

  std::chrono::steady_clock::time_point GetLastAccessTime() const {
    return last_access_time_;
  }

 private:
  std::shared_ptr<arrow::RecordBatchReader> reader_;
  std::shared_ptr<arrow::RecordBatch> next_record_batch_;
  bool finished_;
  size_t limit_;
  size_t rows_read_;
  std::shared_ptr<InactiveSetCleanupTracker> cleanup_tracker_;
  std::deque<std::shared_ptr<class InactiveSet>> cached_isets_;
  size_t total_batch_count_;
  std::chrono::steady_clock::time_point last_access_time_ =
      std::chrono::steady_clock::now();
  std::shared_ptr<class Table> vdb_table_;
  bool include_internal_columns_;
};

constexpr const size_t kUnlimited = 0;

/**
 * VectorSearchExecutor Class Hierarchy and Relationships
 *
 *                ┌─────────────────────────────────┐
 *                │      VectorSearchExecutor       │
 *                │         (Abstract)              │
 *                └──────────────┬──────────────────┘
 *                               │
 *                               ▼
 *                   ┌───────────────────┐
 *                   │   AnnExecutor     │
 *                   │   (Abstract)      │
 *                   └─────────┬─────────┘
 *                             │
 *                     ┌───────┼───────┐
 *                     │               │
 *                     ▼               ▼
 *              ┌─────────────────┐ ┌─────────────────────┐
 *              │ AnnBatchExecutor│ │ AnnEmptyExecutor    │
 *              │                 │ │                     │
 *              └─────────────────┘ └─────────────────────┘
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                              Utility Classes │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *    IndexSearchExecutorUtils
 *    ├── RunIndexSearch<QueryType, SearchFunction>()
 *    ├── RunIndexSearchSingleThread()
 *    └── RunIndexSearchMultipleThreads()
 *
 *    ResultSetBuilderUtils
 *    ├── AggregateLabelsBySegmentAndSet()
 *    ├── AddDistanceColumn()
 *    ├── BuildResultSet()
 *    └── CreateAnnFilterFunctor() / CreateFilterFunctor()
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                              Data Structures │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *    IndexSearchList
 *    ├── vector<IndexSearchElement> index_search_results_
 *    ├── Set() / Empty()
 *    └── SortAscending() / SortDescending()
 *
 *    IndexSearchElement
 *    ├── float value_      (distance)
 *    ├── uint64_t label_
 *    └── int64_t segment_idx_
 *
 *    ProjectionInfo
 *    ├── bool is_select_star
 *    ├── vector<string> projection_list
 *    ├── vector<int> table_select_column_indices
 *    ├── bool has_index_column
 *    └── vector<string> index_column_names
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                               Filter Classes │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *    FilterFunctor ──adapter──> AnnFilterFunctor
 *    │                          (hnswlib::BaseFilterFunctor)
 *    ├── Expression pred_
 *    ├── Segment segment_
 *    └── operator()(uint64_t label)
 *
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                               Execution Flow │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 *    1. Builder.Build() ──> Executor
 *    2. Executor.Run() ──>
 *       ├── RunIndexSearch() ──uses──> IndexSearchExecutorUtils
 *       ├── BuildResultSet() ──uses──> ResultSetBuilderUtils
 *       └── SerializeTable() ──> Buffer
 *
 *    Threading Strategy:
 *    ├── Query Level: Parallel processing of multiple query vectors
 *    ├── Segment Level: Parallel processing within each query
 *    └── Result Level: Sequential aggregation from all threads
 */

class PlanExecutor {
 public:
  PlanExecutor() = default;

  static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
  ReplaceEmbeddingColumn(std::shared_ptr<arrow::RecordBatch> batch,
                         std::shared_ptr<vdb::Table> vdb_table);

  static arrow::Result<std::shared_ptr<arrow::Buffer>> ExecutePlanToBuffer(
      std::shared_ptr<arrow::Schema> schema,
      arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> generator,
      std::optional<std::string_view> maybe_projection,
      std::shared_ptr<vdb::expression::Expression> filter,
      std::shared_ptr<vdb::Table> vdb_table, size_t limit = kUnlimited,
      bool include_internal_columns = false);

  static arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ExecutePlan(
      std::shared_ptr<arrow::Schema> schema,
      arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> generator,
      std::optional<std::string_view> maybe_projection,
      std::shared_ptr<vdb::expression::Expression> filter,
      bool include_internal_columns = false);
};

// ScanRegistry - Centralized scan lifecycle management
// Manages active scans without client-specific tracking
// Client tracking remains in vdb_api.cc to maintain layer separation
class ScanRegistry {
 public:
  static ScanRegistry& GetInstance();

  // Scan lifecycle management
  arrow::Status RegisterScan(const std::string& uuid,
                             std::shared_ptr<IterativeResultSet> result_set);

  arrow::Status UnregisterScan(const std::string& uuid);

  std::shared_ptr<IterativeResultSet> GetScan(const std::string& uuid);

  // TTL-based cleanup (uses IterativeResultSet::IsExpired)
  void CleanupExpiredScans(std::chrono::seconds ttl);

  // Get list of expired scan UUIDs (for logging/debugging)
  std::vector<std::string> GetExpiredScanUUIDs(std::chrono::seconds ttl);

  // Statistics
  size_t GetActiveScanCount() const;

 private:
  ScanRegistry() = default;
  ~ScanRegistry() = default;
  ScanRegistry(const ScanRegistry&) = delete;
  ScanRegistry& operator=(const ScanRegistry&) = delete;

  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<IterativeResultSet>> scans_;
};

struct IndexSearchElement {
  explicit IndexSearchElement() : value_(0), label_(0), segment_idx_(-1) {}

  IndexSearchElement(float v, uint64_t l, int64_t i)
      : value_(v), label_(l), segment_idx_(i) {}

  IndexSearchElement(const IndexSearchElement& other)
      : value_(other.value_),
        label_(other.label_),
        segment_idx_(other.segment_idx_) {}

  IndexSearchElement& operator=(const IndexSearchElement& other) {
    if (this != &other) {
      value_ = other.value_;
      label_ = other.label_;
      segment_idx_ = other.segment_idx_;
    }
    return *this;
  }

  IndexSearchElement(IndexSearchElement&& other) noexcept
      : value_(other.value_),
        label_(other.label_),
        segment_idx_(other.segment_idx_) {}

  IndexSearchElement& operator=(IndexSearchElement&& other) noexcept {
    if (this != &other) {
      value_ = other.value_;
      label_ = other.label_;
      segment_idx_ = other.segment_idx_;
    }
    return *this;
  }

  bool operator<(const IndexSearchElement& other) const {
    return value_ < other.value_;
  }

  bool operator>(const IndexSearchElement& other) const {
    return value_ > other.value_;
  }

  float value_;  // distance for dense vectors
  uint64_t label_;
  int64_t segment_idx_;
};

struct IndexSearchList {
  IndexSearchList() = delete;

  IndexSearchList(size_t segment_count, size_t search_size)
      : index_search_results_(segment_count * search_size,
                              IndexSearchElement()) {}

  void Set(uint64_t idx, const IndexSearchElement& result) {
    index_search_results_[idx] = result;
    has_any_.store(true, std::memory_order_relaxed);
  }

  bool Empty() const { return !has_any_.load(std::memory_order_relaxed); }

  void SortAscending();

  void SortDescending();

  std::vector<IndexSearchElement> index_search_results_;
  std::atomic<bool> has_any_{false};
};

// Base class for all vector search executors
class VectorSearchExecutor {
 public:
  struct ProjectionInfo {
    /*
     * Whether the projection is "*".
     */
    bool is_select_star = false;

    /*
     * The names of columns to be projected finally.
     * The order of the names is the same as the order of the projection.
     * These include the distance column.
     */
    std::vector<std::string> projection_list;

    /*
     * The indices of the columns to be taken from the SeahorseDB table.
     * It doesn't include the distance column. It is used for pruning
     * columns that are not in the projection.
     */
    std::vector<int> table_select_column_indices;

    /*
     * Whether the index column is included in the projection.
     */
    bool has_index_column = false;

    /*
     * The name of the index column.
     */
    std::vector<std::string> index_column_names;
  };

  virtual ~VectorSearchExecutor() = default;

  virtual arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> Run() = 0;
};

// Utility class for building ProjectionInfo
class ProjectionInfoBuilder {
 public:
  static arrow::Result<std::shared_ptr<VectorSearchExecutor::ProjectionInfo>>
  BuildProjectionInfo(std::string_view projection,
                      std::shared_ptr<vdb::vector<IndexInfo>> index_infos,
                      std::shared_ptr<arrow::Schema> table_schema,
                      const std::vector<std::string>&
                          result_column_names,  // e.g., {"distance"}
                      std::function<bool(const IndexInfo&)> index_filter) {
    auto projection_info =
        std::make_shared<VectorSearchExecutor::ProjectionInfo>();

    std::vector<uint64_t> index_column_ids;
    for (const auto& index_info : *index_infos) {
      if (index_filter(index_info)) {
        index_column_ids.push_back(index_info.GetColumnId());
      }
    }

    // Handle wildcard or empty projection case
    if (projection == "*" || projection.empty()) {
      return BuildAllColumnsProjectionInfo(table_schema, index_column_ids,
                                           result_column_names);
    }

    // Parse projection expressions
    ARROW_ASSIGN_OR_RAISE(
        auto projection_exprs,
        vdb::expression::Expression::ParseSimpleProjectionList(projection,
                                                               table_schema));

    if (projection_exprs.empty()) {
      return BuildAllColumnsProjectionInfo(table_schema, index_column_ids,
                                           result_column_names);
    }

    // Process specific column projections
    for (const auto& expr : projection_exprs) {
      const auto& field = table_schema->GetFieldByName(expr->ToString());
      if (field) {
        int idx = table_schema->GetFieldIndex(field->name());
        projection_info->projection_list.push_back(field->name());
        if (std::find(result_column_names.begin(), result_column_names.end(),
                      field->name()) == result_column_names.end()) {
          projection_info->table_select_column_indices.push_back(idx);
        }

        if (std::find(index_column_ids.begin(), index_column_ids.end(), idx) !=
            index_column_ids.end()) {
          projection_info->has_index_column = true;
          projection_info->index_column_names.push_back(field->name());
        }
      }
    }

    return projection_info;
  }

 private:
  static arrow::Result<std::shared_ptr<VectorSearchExecutor::ProjectionInfo>>
  BuildAllColumnsProjectionInfo(
      std::shared_ptr<arrow::Schema> table_schema,
      std::vector<uint64_t> index_column_ids,
      const std::vector<std::string>& result_column_names) {
    auto projection_info =
        std::make_shared<VectorSearchExecutor::ProjectionInfo>();
    projection_info->is_select_star = true;

    for (int i = 0; i < table_schema->num_fields(); i++) {
      const auto& field = table_schema->field(i);

      // Skip invisible columns (internal columns)
      if (vdb::IsInternalColumn(field->name())) {
        continue;
      }

      projection_info->projection_list.push_back(field->name());
      if (std::find(result_column_names.begin(), result_column_names.end(),
                    field->name()) == result_column_names.end()) {
        projection_info->table_select_column_indices.push_back(i);
      }

      if (std::find(index_column_ids.begin(), index_column_ids.end(), i) !=
          index_column_ids.end()) {
        projection_info->has_index_column = true;
        projection_info->index_column_names.push_back(field->name());
      }
    }

    return projection_info;
  }
};

// Utility class for index search operations
class IndexSearchExecutorUtils {
 public:
  // Template function for thread count-based index search dispatch
  template <typename QueryType, typename SearchFunction>
  static arrow::Result<std::shared_ptr<IndexSearchList>> RunIndexSearch(
      const QueryType& query, size_t query_idx, size_t num_queries,
      size_t num_segments, size_t result_size,
      std::optional<size_t> num_index_handler_threads,
      SearchFunction search_function, bool sort_ascending = true);

  // Template function for single thread index search
  template <typename QueryType, typename SearchFunction>
  static arrow::Result<std::shared_ptr<IndexSearchList>>
  RunIndexSearchSingleThread(const QueryType& query, size_t query_idx,
                             size_t num_queries, size_t num_segments,
                             size_t result_size, SearchFunction search_function,
                             bool sort_ascending = true);

  // Template function for multiple thread index search
  template <typename QueryType, typename SearchFunction>
  static arrow::Result<std::shared_ptr<IndexSearchList>>
  RunIndexSearchMultipleThreads(const QueryType& query, size_t query_idx,
                                size_t num_queries, size_t num_segments,
                                size_t result_size, size_t num_threads,
                                SearchFunction search_function,
                                bool sort_ascending = true);
};

// Forward declarations for result set building utilities
class ResultSetBuilderUtils {
 public:
  using SegmentSetIdPair = std::tuple<int64_t, uint32_t>;
  using LabelValuePair = std::pair<uint64_t, float>;

  // Core aggregation function for IndexSearchList
  static std::shared_ptr<
      std::map<SegmentSetIdPair, std::vector<LabelValuePair>>>
  AggregateLabelsBySegmentAndSet(
      std::shared_ptr<IndexSearchList> index_search_results, size_t top_k_size);

  // Value column function
  static arrow::Result<std::shared_ptr<arrow::RecordBatch>> AddValueColumn(
      std::shared_ptr<arrow::RecordBatch> rb, std::vector<float> values,
      const std::string& column_name);

  // Embedding column replacement
  static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
  ReplaceEmbeddingColumn(std::shared_ptr<arrow::RecordBatch> rb,
                         std::shared_ptr<Segment> segment, uint32_t set_id,
                         const std::vector<LabelValuePair>& labels,
                         const std::vector<std::string>& index_column_names,
                         std::shared_ptr<arrow::Schema> table_schema);

  // Result set building from labels for single value (distance)
  static arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
  BuildResultSetFromLabels(
      const std::map<SegmentSetIdPair, std::vector<LabelValuePair>>&
          aggregated_labels,
      const std::vector<std::shared_ptr<Segment>>& segments,
      std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info,
      std::shared_ptr<arrow::Schema> table_schema,
      const std::string& value_column_name);

  // Complete result set building for IndexSearchList
  static arrow::Result<std::shared_ptr<arrow::Table>> BuildResultSet(
      std::shared_ptr<IndexSearchList> index_search_results, size_t top_k_size,
      const std::vector<std::shared_ptr<Segment>>& segments,
      std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info,
      std::shared_ptr<arrow::Schema> table_schema,
      const std::string& value_column_name);

  // Filter functor creation utilities
  static std::shared_ptr<vdb::AnnFilterFunctor> CreateAnnFilterFunctor(
      const std::shared_ptr<Segment>& segment,
      std::shared_ptr<vdb::expression::Expression> filter,
      bool can_use_in_filter);

  static std::shared_ptr<vdb::FilterFunctor> CreateFilterFunctor(
      const std::shared_ptr<Segment>& segment,
      std::shared_ptr<vdb::expression::Expression> filter,
      bool can_use_in_filter);
};

class AnnExecutor : public VectorSearchExecutor {
 public:
  virtual ~AnnExecutor() = default;

  virtual arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> Run() = 0;
};

/**
 * AnnBatchExecutor Execution Flow
 *
 * Single Thread Execution:
 *
 * Run()
 *   └── RunVectorSearch() [for each query]
 *       ├── RunIndexSearch()
 *       │   └── RunIndexSearchSingleThread()
 *       │       └── SearchIndexFromSegment() [sequential for each segment]
 *       ├── BuildResultSet()
 *       │   ├── AggregateLabelsBySegmentAndSet()
 *       │   └── BuildResultSetFromLabels()
 *       │       ├── AddDistanceColumn()
 *       │       ├── ReplaceEmbeddingColumn() [if needed]
 *       │       └── Projection as defined in ProjectionInfo
 *       └── SerializeTable() [convert to buffer]
 *
 * Multiple Thread Execution:
 *
 * Run()
 *   ├── Multiple Threads
 *   │   └── RunVectorSearch() [parallel for each query vector]
 *   │       ├── RunIndexSearch()
 *   │       │   └── RunIndexSearchMultipleThreads()
 *   │       │       └── SearchIndexFromSegment() [parallel for each segment]
 *   │       ├── BuildResultSet()
 *   │       │   ├── AggregateLabelsBySegmentAndSet()
 *   │       │   └── BuildResultSetFromLabels()
 *   │       │       ├── AddDistanceColumn()
 *   │       │   ├── ReplaceEmbeddingColumn() [if needed]
 *   │       │   └── Projection as defined in ProjectionInfo
 *   │       └── SerializeTable() [convert to buffer]
 *   └── Results will be stored in a buffer list order by query vector
 *
 * Threading Strategy:
 * - Query Vector Level: Parallel processing of multiple query vectors
 * - Segment Level: Parallel processing of segments within each query vector
 * - Result Level: Sequential aggregation of results from all threads
 *   TODO: parallel processing of building result set of each Segment-Set pair
 */

class AnnBatchExecutor : public AnnExecutor, public PlanExecutor {
 public:
  AnnBatchExecutor() = delete;

  AnnBatchExecutor(
      uint64_t ann_column_id, size_t top_k_size, size_t search_size,
      size_t ef_search, std::shared_ptr<arrow::Schema> schema_with_dist,
      std::vector<std::shared_ptr<Segment>> segments,
      std::shared_ptr<vdb::vector<IndexInfo>> index_infos,
      std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info,
      std::shared_ptr<vdb::expression::Expression> filter,
      std::optional<size_t> num_query_threads,
      std::optional<size_t> num_index_handler_threads,
      const std::vector<float*>& query_vectors, bool can_use_in_filter)
      : PlanExecutor(),
        ann_column_id_(ann_column_id),
        top_k_size_(top_k_size),
        search_size_(search_size),
        ef_search_(ef_search),
        table_schema_with_dist_(schema_with_dist),
        index_infos_(index_infos),
        segments_(segments),
        projection_info_(projection_info),
        filter_(filter),
        num_query_threads_(num_query_threads),
        num_index_handler_threads_(num_index_handler_threads),
        query_vectors_(query_vectors),
        can_use_in_filter_(can_use_in_filter) {}

  ~AnnBatchExecutor() override = default;

  /**
   * Run vector search for a batch of query vectors.
   *
   * This method performs the complete vector search process for a batch of
   * query vectors It chooses parallel execution strategy based on the number of
   * query vectors and the number of threads.
   *
   * @return Serialized buffer list corresponding to search results of each
   * query vector
   */
  arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> Run() override;

  /*
   * Search Nearest Neighbors using multiple threads
   *
   * One segment has one vector index and a table has multiple segments.
   * It means that it needs to search all indexes in multiple segments to get
   * the top k nearest neighbors.
   *
   *
   * There are two ways to do this:
   * 1. Search from multiple segments in a single thread:
   * RunIndexSearchSingleThread
   *    - Search sequentially one by one in a single thread, which is simple
   *    and easy to understand.
   * 2. Search from multiple segments in multiple threads:
   * RunIndexSearchMultipleThreads
   *    - Search multiple segments in multiple threads, which is faster.
   *
   * RunIndexSearch is wrapper of RunIndexSearchSingleThread and
   * RunIndexSearchMultipleThreads.
   *
   * @param raw_query Pointer to the raw query vector data
   * @param query_idx Index of the current query in the batch
   * @param num_queries Total number of queries in the batch
   * @return Serialized buffer containing the search results
   */
  arrow::Result<std::shared_ptr<IndexSearchList>> RunIndexSearch(
      const float* raw_query, size_t query_idx, size_t num_queries);

  /*
   * Build result set from index search results.
   *
   * The function is abstracted interface for building result set from index
   * search results which includes aggregation of labels and building result
   * set from aggregated labels.
   *
   * @param index_search_results Pointer to the index search results
   * @return Result set
   */
  arrow::Result<std::shared_ptr<arrow::Table>> BuildResultSet(
      std::shared_ptr<IndexSearchList> index_search_results);

 protected:
  /**
   * Run vector search for a single query vector.
   *
   * This method performs the complete vector search process for a single query
   * vector. It represents the vector search for a single query vector on a
   * single thread:
   *  1. Searches across all segments (using either single or multiple threads)
   *  2. Builds the result set from the search results
   *  3. Serializes the record batch to a buffer as Arrow IPC format
   *
   * @param raw_query Pointer to the raw query vector data
   * @param query_idx Index of the current query in the batch
   * @param num_queries Total number of queries in the batch
   * @return Serialized buffer containing the search results
   */
  arrow::Result<std::shared_ptr<arrow::Buffer>> RunVectorSearch(
      float* raw_query, size_t query_idx, size_t num_queries);

  /*
   * Search index from a segment.
   *
   * @param i Index of the segment
   * @param raw_query Pointer to the raw query vector data
   * @param index_search_results Pointer to return the index search results
   */
  void SearchIndexFromSegment(
      size_t segment_idx, const float* raw_query,
      std::shared_ptr<IndexSearchList> index_search_results);

  /*
   * Create filter functor. It is used to in-filter of HNSW index search
   *
   * The reason why it needs segment is for evaluating the filter expression
   * with the value in the segment.
   *
   * @param segment Segment
   * @return Filter functor
   */
  std::shared_ptr<vdb::AnnFilterFunctor> CreateFilterFunctor(
      const std::shared_ptr<Segment>& segment);

 private:
  uint64_t ann_column_id_;
  size_t top_k_size_;
  size_t search_size_;
  size_t ef_search_;
  std::shared_ptr<arrow::Schema> table_schema_with_dist_;
  std::shared_ptr<vdb::vector<IndexInfo>> index_infos_;
  std::vector<std::shared_ptr<Segment>> segments_;
  std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info_;
  std::shared_ptr<vdb::expression::Expression> filter_;
  std::optional<size_t> num_query_threads_;
  std::optional<size_t> num_index_handler_threads_;
  std::vector<float*> query_vectors_;
  bool can_use_in_filter_;
};

class AnnEmptyExecutor : public AnnExecutor {
 public:
  AnnEmptyExecutor(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info,
      size_t num_query_vectors)
      : table_schema_with_dist_(schema),
        projection_info_(projection_info),
        num_query_vectors_(num_query_vectors) {}

  ~AnnEmptyExecutor() override = default;

  arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> Run() override;

 private:
  std::shared_ptr<arrow::Schema> table_schema_with_dist_;
  std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info_;
  size_t num_query_vectors_;
};

class AnnExecutorBuilder {
 public:
  AnnExecutorBuilder()
      : ann_column_id_(std::numeric_limits<uint64_t>::max()),
        top_k_size_(0),
        search_size_(0),
        ef_search_(0),
        table_schema_(nullptr),
        index_infos_(nullptr),
        segments_(),
        projection_(),
        filter_(),
        num_max_concurrency_(std::nullopt),
        queries_(),
        query_(nullptr) {}

  AnnExecutorBuilder& SetAnnColumnId(uint64_t ann_column_id) {
    this->ann_column_id_ = ann_column_id;
    return *this;
  }

  AnnExecutorBuilder& SetTopKSize(size_t top_k_size) {
    this->top_k_size_ = top_k_size;
    return *this;
  }

  AnnExecutorBuilder& SetSearchSize(size_t search_size) {
    this->search_size_ = search_size;
    return *this;
  }

  AnnExecutorBuilder& SetEfSearch(size_t ef_search) {
    this->ef_search_ = ef_search;
    return *this;
  }

  AnnExecutorBuilder& SetTableSchema(std::shared_ptr<arrow::Schema> schema) {
    this->table_schema_ = schema;
    return *this;
  }

  AnnExecutorBuilder& SetIndexInfos(
      std::shared_ptr<vdb::vector<IndexInfo>> index_infos) {
    this->index_infos_ = index_infos;
    return *this;
  }

  AnnExecutorBuilder& SetSegments(
      std::vector<std::shared_ptr<Segment>> segments) {
    this->segments_ = segments;
    return *this;
  }

  AnnExecutorBuilder& SetProjection(std::string_view projection) {
    this->projection_ = projection;
    return *this;
  }

  AnnExecutorBuilder& SetFilter(
      std::shared_ptr<vdb::expression::Expression> filter) {
    this->filter_ = filter;
    return *this;
  }

  AnnExecutorBuilder& SetQueryRecordBatches(
      std::vector<std::shared_ptr<arrow::RecordBatch>> queries) {
    this->queries_ = queries;
    return *this;
  }

  AnnExecutorBuilder& SetQueryVector(const float* query) {
    this->query_ = query;
    return *this;
  }

  AnnExecutorBuilder& SetNumMaxConcurrency(size_t num_max_concurrency) {
    this->num_max_concurrency_ = num_max_concurrency;
    return *this;
  }

  bool IsSingleQuery() const;
  bool IsBatchQuery() const;
  arrow::Result<std::vector<float*>> GetQueryRawVectorsFromRecordBatches(
      const std::vector<std::shared_ptr<arrow::RecordBatch>>& rbs) const;

  using num_query_threads_t = size_t;
  using num_index_handler_threads_t = size_t;
  std::tuple<num_query_threads_t, num_index_handler_threads_t> GetNumThreads(
      const std::vector<float*>& query_vectors) const;

  arrow::Result<std::shared_ptr<AnnExecutor>> Build() const;

 private:
  uint64_t ann_column_id_;
  size_t top_k_size_;
  size_t search_size_;
  size_t ef_search_;
  std::shared_ptr<arrow::Schema> table_schema_;
  std::shared_ptr<vdb::vector<IndexInfo>> index_infos_;
  std::vector<std::shared_ptr<Segment>> segments_;
  std::optional<std::string_view> projection_;
  std::shared_ptr<vdb::expression::Expression> filter_;
  std::optional<size_t> num_max_concurrency_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> queries_;
  const float* query_;
};

}  // namespace vdb
