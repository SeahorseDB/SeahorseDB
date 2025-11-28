#include <optional>
#include <string_view>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <tuple>
#include <thread>
#include <atomic>
#include <future>
#include <algorithm>
#include <limits>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/acero/query_context.h>

#include "vdb/compute/execution.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/server_configuration.hh"
#include "vdb/data/expression.hh"
#include "vdb/data/label_info.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/vdb.hh"

namespace vdb {

// Streaming generator that loads InactiveSets on-demand for memory efficiency
// - Loads selected columns via GetColumns() (zero-copy mmap)
// - Tracks InactiveSets via InactiveSetCleanupTracker for GC
// - Releases segment references after processing
// - Single-threaded, used by Arrow Acero execution pipeline
struct InactiveSetGenerator {
  std::shared_ptr<std::vector<std::shared_ptr<Segment>>> segments;
  std::vector<int> column_indices;
  size_t segment_idx;
  size_t set_idx;
  bool active_set_processed;
  std::shared_ptr<InactiveSetCleanupTracker> cleanup_tracker;

  InactiveSetGenerator(std::vector<std::shared_ptr<Segment>> segs,
                       std::vector<int> col_indices)
      : segments(std::make_shared<std::vector<std::shared_ptr<Segment>>>(
            std::move(segs))),
        column_indices(std::move(col_indices)),
        segment_idx(0),
        set_idx(0),
        active_set_processed(false),
        cleanup_tracker(std::make_shared<InactiveSetCleanupTracker>()) {}

  arrow::Future<std::optional<arrow::ExecBatch>> operator()() {
    while (segment_idx < segments->size()) {
      auto &segment = (*segments)[segment_idx];

      auto &inactive_sets = segment->InactiveSets();
      if (set_idx < inactive_sets.size()) {
        auto iset = inactive_sets[set_idx];
        set_idx++;

        arrow::Result<std::shared_ptr<arrow::RecordBatch>> rb_result;
        if (column_indices.empty()) {
          rb_result =
              arrow::Result<std::shared_ptr<arrow::RecordBatch>>(iset->GetRb());
        } else {
          rb_result = iset->GetColumns(column_indices);
        }

        if (!rb_result.ok()) {
          return arrow::Future<std::optional<arrow::ExecBatch>>::MakeFinished(
              rb_result.status());
        }

        auto rb = rb_result.ValueUnsafe();
        if (rb && rb->num_rows() > 0) {
          cleanup_tracker->Track(iset);

          auto exec_batch = std::make_optional(arrow::ExecBatch(*rb));

          return arrow::Future<std::optional<arrow::ExecBatch>>::MakeFinished(
              std::move(exec_batch));
        }
        continue;
      }

      if (!active_set_processed) {
        active_set_processed = true;
        auto active_rb = segment->ActiveSetRecordBatch();

        if (active_rb && active_rb->num_rows() > 0) {
          if (!column_indices.empty()) {
            auto selected_result = active_rb->SelectColumns(column_indices);
            if (!selected_result.ok()) {
              return arrow::Future<std::optional<arrow::ExecBatch>>::
                  MakeFinished(selected_result.status());
            }
            active_rb = selected_result.ValueUnsafe();
          }

          return arrow::Future<std::optional<arrow::ExecBatch>>::MakeFinished(
              std::make_optional(arrow::ExecBatch(*active_rb)));
        }
      }

      if (segment_idx > 0) {
        (*segments)[segment_idx - 1].reset();
      }

      segment_idx++;
      set_idx = 0;
      active_set_processed = false;
    }

    return arrow::Future<std::optional<arrow::ExecBatch>>::MakeFinished(
        std::nullopt);
  }
};

arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> SegmentGenerator(
    std::vector<std::shared_ptr<Segment>> segments,
    const std::vector<int> &column_indices) {
  return InactiveSetGenerator(std::move(segments), column_indices);
}

std::tuple<arrow::AsyncGenerator<std::optional<arrow::ExecBatch>>,
           std::shared_ptr<InactiveSetCleanupTracker>>
SegmentGeneratorWithTracking(std::vector<std::shared_ptr<Segment>> segments,
                             const std::vector<int> &column_indices) {
  InactiveSetGenerator gen(std::move(segments), column_indices);
  auto cleanup_tracker = gen.cleanup_tracker;
  return std::make_tuple(std::move(gen), cleanup_tracker);
}

arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> RecordBatchGenerator(
    std::shared_ptr<arrow::RecordBatch> rb) {
  std::vector<arrow::ExecBatch> batches;
  batches.push_back(arrow::ExecBatch(*rb));
  auto opt_batches = arrow::internal::MapVector(
      [](arrow::ExecBatch batch) {
        return std::make_optional(std::move(batch));
      },
      std::move(batches));
  return arrow::MakeVectorGenerator(std::move(opt_batches));
}

arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> RecordBatchesGenerator(
    const std::vector<std::shared_ptr<arrow::RecordBatch>> &record_batches) {
  std::vector<arrow::ExecBatch> batches;
  for (const auto &rb : record_batches) {
    if (rb && rb->num_rows() > 0) {
      batches.push_back(arrow::ExecBatch(*rb));
    }
  }
  auto opt_batches = arrow::internal::MapVector(
      [](arrow::ExecBatch batch) {
        return std::make_optional(std::move(batch));
      },
      std::move(batches));
  return arrow::MakeVectorGenerator(std::move(opt_batches));
}

arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeTable(
    std::shared_ptr<arrow::Table> table) {
  // Create an in-memory output stream
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::BufferOutputStream::Create());

  // Create an IPC stream writer
  ARROW_ASSIGN_OR_RAISE(auto writer,
                        arrow::ipc::MakeStreamWriter(out, table->schema()));

  if (table) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Serializing table with %lu rows", table->num_rows());
    ARROW_RETURN_NOT_OK(writer->WriteTable(*table));
  }

  ARROW_RETURN_NOT_OK(writer->Close());
  return out->Finish();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeRecordBatch(
    std::shared_ptr<arrow::RecordBatch> record_batch) {
  // Create an in-memory output stream
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::BufferOutputStream::Create());

  // Create an IPC stream writer
  ARROW_ASSIGN_OR_RAISE(
      auto writer, arrow::ipc::MakeStreamWriter(out, record_batch->schema()));

  if (record_batch) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Serializing record batch with %lu rows",
               record_batch->num_rows());
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*record_batch));
  }

  ARROW_RETURN_NOT_OK(writer->Close());
  return out->Finish();
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
_GenerateEmptyResults(
    std::shared_ptr<arrow::Schema> schema_with_dist_or_score,
    const std::shared_ptr<VectorSearchExecutor::ProjectionInfo>
        &projection_info,
    size_t num_query_vectors) {
  std::vector<std::shared_ptr<arrow::Field>> final_fields;
  for (const auto &name : projection_info->projection_list) {
    auto f = schema_with_dist_or_score->GetFieldByName(name);
    if (f) {
      final_fields.push_back(f);
    }
  }
  auto result_schema = arrow::schema(final_fields);

  ARROW_ASSIGN_OR_RAISE(auto rb, arrow::RecordBatch::MakeEmpty(result_schema));
  ARROW_ASSIGN_OR_RAISE(auto buffer, SerializeRecordBatch(rb));

  std::vector<std::shared_ptr<arrow::Buffer>> results;
  for (size_t i = 0; i < num_query_vectors; i++) {
    results.push_back(buffer);
  }
  return results;
}

IterativeResultSet::IterativeResultSet(
    std::shared_ptr<arrow::RecordBatchReader> reader, size_t limit,
    std::shared_ptr<InactiveSetCleanupTracker> cleanup_tracker,
    std::shared_ptr<vdb::Table> vdb_table, bool include_internal_columns)
    : reader_(std::move(reader)),
      next_record_batch_(nullptr),
      finished_(false),
      limit_(limit),
      rows_read_(0),
      cleanup_tracker_(std::move(cleanup_tracker)),
      total_batch_count_(0),
      vdb_table_(std::move(vdb_table)),
      include_internal_columns_(include_internal_columns) {}

IterativeResultSet::~IterativeResultSet() {
  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "IterativeResultSet::~IterativeResultSet: Destructor called, "
             "rows_read=%lld, limit=%lld, finished=%d",
             rows_read_, limit_, finished_);

  // Close reader and release Arrow ExecPlan resources
  if (reader_) {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "IterativeResultSet::~IterativeResultSet: Closing reader");

    auto status = reader_->Close();
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                 "Failed to close RecordBatchReader: %s",
                 status.ToString().c_str());
    }

    // Release reader to break reference chain to segments
    reader_.reset();

    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "IterativeResultSet::~IterativeResultSet: Reader released");
  }

  // Release cached batch
  if (next_record_batch_) {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "IterativeResultSet::~IterativeResultSet: Releasing cached "
               "batch (rows=%lld)",
               next_record_batch_->num_rows());
    next_record_batch_.reset();
  }

  // Release all InactiveSet references before GC
  cached_isets_.clear();

  if (cleanup_tracker_) {
    cleanup_tracker_->Clear();
    cleanup_tracker_.reset();
  }

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "IterativeResultSet::~IterativeResultSet: Destructor completed");
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> IterativeResultSet::Next() {
  auto temp = next_record_batch_;

  // Replace embedding columns if table is provided
  if (temp && vdb_table_) {
    ARROW_ASSIGN_OR_RAISE(
        temp, PlanExecutor::ReplaceEmbeddingColumn(temp, vdb_table_));
  }

  if (temp) {
    rows_read_ += temp->num_rows();

    // Only track InactiveSet batches, not ActiveSet batches
    if (cleanup_tracker_ && !cleanup_tracker_->Empty() &&
        total_batch_count_ < cleanup_tracker_->Size()) {
      // Get InactiveSet by index (returns nullptr if expired/released)
      if (auto iset = cleanup_tracker_->Get(total_batch_count_)) {
        cached_isets_.push_back(iset);

        // Keep only 3 most recent InactiveSets (current, prefetch, and one for
        // safety)
        while (cached_isets_.size() > 3) {
          auto iset_to_gc = cached_isets_.front();
          cached_isets_.pop_front();
        }
      }
      // If iset is nullptr, it was already GC'd (normal for ActiveSet batches)
    }

    total_batch_count_++;
  }

  if (finished_) {
    next_record_batch_ = nullptr;
    return temp;
  }

  if (limit_ != 0 && rows_read_ >= limit_) {
    next_record_batch_ = nullptr;
    finished_ = true;
    return temp;
  }

  ARROW_ASSIGN_OR_RAISE(auto next_batch, reader_->Next());

  if (next_batch == nullptr) {
    finished_ = true;
    next_record_batch_ = nullptr;
  } else if (limit_ != 0 && rows_read_ + next_batch->num_rows() > limit_) {
    int64_t remaining = limit_ - rows_read_;
    next_record_batch_ = next_batch->Slice(0, remaining);
    finished_ = true;
  } else {
    next_record_batch_ = std::move(next_batch);
  }

  return temp;
}

bool IterativeResultSet::HasNext() const {
  return next_record_batch_ != nullptr;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
PlanExecutor::ReplaceEmbeddingColumn(std::shared_ptr<arrow::RecordBatch> batch,
                                     std::shared_ptr<vdb::Table> vdb_table) {
  if (batch == nullptr || batch->num_rows() == 0) {
    return batch;
  }

  auto result_schema = batch->schema();
  auto extended_schema = vdb_table->GetExtendedSchema();
  auto index_infos = vdb_table->GetIndexInfos();

  auto columns = batch->columns();
  auto fields = result_schema->fields();
  bool any_replaced = false;

  for (auto &index_info : *index_infos) {
    auto ann_column_id = index_info.GetColumnId();
    auto &ann_column_name = extended_schema->field(ann_column_id)->name();

    // Find column in result
    int64_t column_index = -1;
    for (int64_t i = 0; i < result_schema->num_fields(); ++i) {
      if (result_schema->field(i)->name() == ann_column_name) {
        column_index = i;
        break;
      }
    }

    if (column_index >= 0) {
      auto label_array = columns[column_index];
      auto embedding_store = vdb_table->GetEmbeddingStore(ann_column_id);

      auto label_uint64_array =
          std::reinterpret_pointer_cast<arrow::UInt64Array>(label_array);
      auto labels = label_uint64_array->raw_values();
      auto count = label_uint64_array->length();

      std::shared_ptr<arrow::Array> embedding_array;
      ARROW_ASSIGN_OR_RAISE(embedding_array,
                            embedding_store->ReadToArray(labels, count));

      columns[column_index] = embedding_array;
      fields[column_index] = extended_schema->field(ann_column_id);
      any_replaced = true;
    }
  }

  if (any_replaced) {
    auto updated_schema = arrow::schema(fields);
    return arrow::RecordBatch::Make(updated_schema, batch->num_rows(), columns);
  }

  return batch;
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>>
PlanExecutor::ExecutePlan(
    std::shared_ptr<arrow::Schema> schema,
    arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> generator,
    std::optional<std::string_view> maybe_projection,
    std::shared_ptr<vdb::expression::Expression> filter,
    bool include_internal_columns) {
  std::vector<arrow::acero::Declaration> decls;
  decls.emplace_back(arrow::acero::Declaration(
      "source",
      arrow::acero::SourceNodeOptions(schema->RemoveMetadata(), generator)));

  if (filter) {
    SYSTEM_LOG(vdb::LogTopic::Executor, LogLevel::kLogDebug,
               "Filter is applied. (include_internal_columns: %s)",
               include_internal_columns ? "true" : "false");
    ARROW_ASSIGN_OR_RAISE(auto bound_filter, filter->BuildArrowExpression());
    decls.emplace_back(arrow::acero::Declaration(
        "filter", arrow::acero::FilterNodeOptions(bound_filter)));
  } else {
    SYSTEM_LOG(vdb::LogTopic::Executor, LogLevel::kLogDebug,
               "No filter is applied. (include_internal_columns: %s)",
               include_internal_columns ? "true" : "false");
  }

  std::string_view projection = maybe_projection.value_or("*");

  ARROW_ASSIGN_OR_RAISE(auto projection_exprs,
                        vdb::expression::Expression::ParseSimpleProjectionList(
                            projection, schema, include_internal_columns));

  if (!projection_exprs.empty()) {
    std::vector<arrow::compute::Expression> exprs;
    for (const auto &proj : projection_exprs) {
      ARROW_ASSIGN_OR_RAISE(auto expr, proj->BuildArrowExpression());
      exprs.emplace_back(expr);
    }
    decls.emplace_back(arrow::acero::Declaration{
        "project", arrow::acero::ProjectNodeOptions(exprs)});
  }

  arrow::acero::Declaration decl = arrow::acero::Declaration::Sequence(decls);

  ARROW_ASSIGN_OR_RAISE(
      auto reader,
      arrow::acero::DeclarationToReader(decl, false, &vdb::arrow_pool,
                                        arrow::compute::GetFunctionRegistry()));

  return reader;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> PlanExecutor::ExecutePlanToBuffer(
    std::shared_ptr<arrow::Schema> schema,
    arrow::AsyncGenerator<std::optional<arrow::ExecBatch>> generator,
    std::optional<std::string_view> maybe_projection,
    std::shared_ptr<vdb::expression::Expression> filter,
    std::shared_ptr<vdb::Table> vdb_table, size_t limit,
    bool include_internal_columns) {
  ARROW_ASSIGN_OR_RAISE(auto reader,
                        ExecutePlan(schema, generator, maybe_projection, filter,
                                    include_internal_columns));

  auto result_schema = reader->schema();
  if (!result_schema) {
    return arrow::Status::Invalid("ExecutePlanToBuffer: reader schema is null");
  }

  std::vector<std::shared_ptr<arrow::Field>> output_fields;

  auto extended_schema = vdb_table->GetExtendedSchema();
  auto index_infos = vdb_table->GetIndexInfos();

  std::unordered_map<std::string, std::shared_ptr<arrow::Field>>
      embedding_fields;
  for (auto &index_info : *index_infos) {
    auto ann_column_id = index_info.GetColumnId();
    auto &ann_column_name = extended_schema->field(ann_column_id)->name();
    embedding_fields[ann_column_name] = extended_schema->field(ann_column_id);
  }

  for (int i = 0; i < result_schema->num_fields(); ++i) {
    auto field = result_schema->field(i);
    // For regular scans, skip internal columns. For internal scans, include
    // them.
    if (!include_internal_columns && IsInternalColumn(field->name())) {
      continue;
    }
    auto it = embedding_fields.find(field->name());
    if (it != embedding_fields.end()) {
      output_fields.push_back(it->second);
    } else {
      output_fields.push_back(field);
    }
  }

  auto output_schema = arrow::schema(output_fields, result_schema->metadata());

  ARROW_ASSIGN_OR_RAISE(auto output_stream,
                        arrow::io::BufferOutputStream::Create());
  ARROW_ASSIGN_OR_RAISE(
      auto writer, arrow::ipc::MakeStreamWriter(output_stream, output_schema));

  int64_t total_rows = 0;
  [[maybe_unused]] int batch_count = 0;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->Next());
    if (batch == nullptr) break;

    if (batch->num_rows() == 0) {
      continue;
    }

    batch_count++;

    if (!include_internal_columns) {
      ARROW_ASSIGN_OR_RAISE(batch, vdb::FilterInternalColumns(batch));
    }

    ARROW_ASSIGN_OR_RAISE(batch, ReplaceEmbeddingColumn(batch, vdb_table));

    if (batch->num_columns() != output_schema->num_fields()) {
      return arrow::Status::Invalid(
          "ExecutePlanToBuffer: batch column count mismatch: batch=",
          batch->num_columns(), " output_schema=", output_schema->num_fields());
    }

    for (int i = 0; i < batch->num_columns(); ++i) {
      auto batch_col_type = batch->column(i)->type();
      auto output_field_type = output_schema->field(i)->type();
      if (!batch_col_type->Equals(output_field_type)) {
        return arrow::Status::Invalid(
            "ExecutePlanToBuffer: batch column type mismatch at "
            "index ",
            i);
      }
    }

    batch = arrow::RecordBatch::Make(output_schema, batch->num_rows(),
                                     batch->columns());

    if (limit != kUnlimited &&
        total_rows + batch->num_rows() > static_cast<int64_t>(limit)) {
      int64_t remaining = static_cast<int64_t>(limit) - total_rows;
      batch = batch->Slice(0, remaining);
      ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
      total_rows += batch->num_rows();
      break;
    }

    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    total_rows += batch->num_rows();
  }

  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_ASSIGN_OR_RAISE(auto result_buffer, output_stream->Finish());

  return result_buffer;
}

arrow::Result<std::shared_ptr<arrow::Table>> AnnBatchExecutor::BuildResultSet(
    std::shared_ptr<IndexSearchList> index_search_results) {
  return ResultSetBuilderUtils::BuildResultSet(
      index_search_results, top_k_size_, segments_, projection_info_,
      table_schema_with_dist_, "distance");
}

arrow::Result<std::shared_ptr<arrow::Buffer>> AnnBatchExecutor::RunVectorSearch(
    float *raw_query, size_t query_idx, size_t num_queries) {
  ARROW_ASSIGN_OR_RAISE(auto search_results,
                        RunIndexSearch(raw_query, query_idx, num_queries));
  if (search_results->Empty()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "No ANN results found for query %lu/%lu", query_idx + 1,
               num_queries);
    ARROW_ASSIGN_OR_RAISE(
        auto buffers,
        _GenerateEmptyResults(table_schema_with_dist_, projection_info_, 1));
    return buffers[0];
  }

  ARROW_ASSIGN_OR_RAISE(auto resultset, BuildResultSet(search_results));
  ARROW_ASSIGN_OR_RAISE(auto buffer, SerializeTable(resultset));
  return buffer;
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
AnnBatchExecutor::Run() {
  std::vector<arrow::Result<std::shared_ptr<arrow::Buffer>>> thread_results(
      query_vectors_.size());

  std::vector<std::thread> ann_threads;
  std::atomic<size_t> query_vector_idx = 0;
  auto num_threads = num_query_threads_.value_or(
      std::min(static_cast<size_t>(std::thread::hardware_concurrency()),
               query_vectors_.size()));

  if (num_threads == 0) {
    return arrow::Status::Invalid("Num threads is 0");
  }

  std::vector<std::shared_ptr<arrow::Buffer>> results;
  if (num_threads == 1) {
    // no need to use thread
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Start of ann executor with single thread, num queries: %lu",
               query_vectors_.size());
    for (size_t i = 0; i < query_vectors_.size(); i++) {
      auto raw_query = query_vectors_[i];

      ARROW_ASSIGN_OR_RAISE(
          auto buffer, RunVectorSearch(raw_query, i, query_vectors_.size()));
      results.push_back(std::move(buffer));
    }
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "End of ann executor with single thread, num queries: %lu",
               query_vectors_.size());
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Start of ann executor with multiple threads, num queries: %lu, "
               "num threads: %lu",
               query_vectors_.size(), num_threads);

    for (size_t tid = 0; tid < num_threads; tid++) {
      ann_threads.emplace_back([&]() {
        size_t i = query_vector_idx.fetch_add(1);
        while (i < query_vectors_.size()) {
          SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
                     "Thread %lu/%lu is processing for query vector %lu", i,
                     ann_threads.size(), i);
          auto raw_query = query_vectors_[i];
          thread_results[i] =
              RunVectorSearch(raw_query, i, query_vectors_.size());
          i = query_vector_idx.fetch_add(1);
        }
      });
    }

    std::for_each(ann_threads.begin(), ann_threads.end(),
                  [](std::thread &t) { t.join(); });
    for (auto &result : thread_results) {
      ARROW_ASSIGN_OR_RAISE(auto buffer, result);
      results.push_back(std::move(buffer));
    }

    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "End of ann executor with multiple threads, num queries: %lu, "
               "num threads: %lu",
               query_vectors_.size(), num_threads);
  }

  return results;
}

void IndexSearchList::SortAscending() {
  std::sort(index_search_results_.begin(), index_search_results_.end(),
            std::less<IndexSearchElement>());
}

void IndexSearchList::SortDescending() {
  std::sort(index_search_results_.begin(), index_search_results_.end(),
            std::greater<IndexSearchElement>());
}

arrow::Result<std::shared_ptr<IndexSearchList>>
AnnBatchExecutor::RunIndexSearch(const float *raw_query, size_t query_idx,
                                 size_t num_queries) {
  auto search_function = [this](size_t segment_idx, const float *query,
                                std::shared_ptr<IndexSearchList> results) {
    SearchIndexFromSegment(segment_idx, query, results);
  };

  return IndexSearchExecutorUtils::RunIndexSearch(
      raw_query, query_idx, num_queries, segments_.size(), search_size_,
      num_index_handler_threads_, search_function, true);  // sort ascending
}

void AnnBatchExecutor::SearchIndexFromSegment(
    size_t segment_idx, const float *raw_query,
    std::shared_ptr<IndexSearchList> ann_results) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "Start vector search for segment number %lu", segment_idx);

  const size_t offset = segment_idx * search_size_;
  auto index_handler = segments_[segment_idx]->IndexHandler();
  auto filter_functor = CreateFilterFunctor(segments_[segment_idx]);
  auto sub_result = index_handler->SearchKnn(
      raw_query, search_size_, ann_column_id_, filter_functor.get());

  for (size_t j = 0; j < sub_result->size(); j++) {
    const auto &[distance, label] = (*sub_result)[j];
    ann_results->Set(offset + j,
                     IndexSearchElement(distance, label, segment_idx));
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "End of vector search for segment number %lu", segment_idx);
}

std::shared_ptr<vdb::AnnFilterFunctor> AnnBatchExecutor::CreateFilterFunctor(
    const std::shared_ptr<Segment> &segment) {
  return ResultSetBuilderUtils::CreateAnnFilterFunctor(segment, filter_,
                                                       can_use_in_filter_);
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>>
AnnEmptyExecutor::Run() {
  return _GenerateEmptyResults(table_schema_with_dist_, projection_info_,
                               num_query_vectors_);
}

bool AnnExecutorBuilder::IsSingleQuery() const {
  return queries_.empty() && query_ != nullptr;
}

bool AnnExecutorBuilder::IsBatchQuery() const {
  return !queries_.empty() && query_ == nullptr;
}

arrow::Result<std::vector<float *>>
AnnExecutorBuilder::GetQueryRawVectorsFromRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>> &rbs) const {
  std::vector<float *> query_vectors;
  for (auto &rb : rbs) {
    auto query_list_array = std::static_pointer_cast<arrow::FixedSizeListArray>(
        rb->GetColumnByName("query"));
    auto dim = query_list_array->list_type()->list_size();
    for (size_t i = 0; i < static_cast<size_t>(query_list_array->length());
         i++) {
      auto raw_query = const_cast<float *>(
          query_list_array->values()->data()->GetValues<float>(1, dim * i));
      query_vectors.push_back(raw_query);
    }
  }
  return query_vectors;
}

std::tuple<AnnExecutorBuilder::num_query_threads_t,
           AnnExecutorBuilder::num_index_handler_threads_t>
AnnExecutorBuilder::GetNumThreads(
    const std::vector<float *> &query_vectors) const {
  // The number of threads should be less than max concurrency
  // It is determined by the number of query vectors and the
  // number of segments.
  size_t num_query_threads = 0;
  size_t num_index_handler_threads = 0;
  size_t num_max_concurrency = num_max_concurrency_.value_or(
      static_cast<size_t>(std::thread::hardware_concurrency()));
  if (!num_max_concurrency_.has_value()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Num concurrency is not set, using default value (hardware "
               "concurrency): %lu",
               num_max_concurrency);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Num concurrency is set: %lu", num_max_concurrency);
  }

  size_t num_query_vectors = query_vectors.size();
  if (num_query_vectors < num_max_concurrency) {
    num_query_threads = num_query_vectors;
  } else {
    num_query_threads = num_max_concurrency;
  }

  if (num_query_threads * segments_.size() > num_max_concurrency) {
    size_t capacity = std::max(
        1ul,
        static_cast<size_t>(std::floor(
            num_max_concurrency / static_cast<double>(num_query_threads))));
    num_index_handler_threads = std::min(segments_.size(), capacity);
  } else {
    num_index_handler_threads = segments_.size();
  }

  SYSTEM_LOG(
      vdb::LogTopic::Unknown, LogLevel::kLogDebug,
      "Determined num threads for AnnExecutor - Num concurrency: %lu, "
      "Num query vectors: %lu, Num segments: %lu, Num query threads: %lu, "
      "Num index handler threads: %lu",
      num_max_concurrency, num_query_vectors, segments_.size(),
      num_query_threads, num_index_handler_threads);
  return {num_query_threads, num_index_handler_threads};
}

arrow::Result<std::shared_ptr<AnnExecutor>> AnnExecutorBuilder::Build() const {
  if (!queries_.empty() && query_ != nullptr) {
    return arrow::Status::Invalid("Both queries and query are set");
  }
  if (num_max_concurrency_.has_value() && num_max_concurrency_.value() == 0) {
    return arrow::Status::Invalid("Num concurrency for AnnExecutor is 0");
  }
  if (table_schema_ == nullptr) {
    return arrow::Status::Invalid("Schema is empty");
  }
  if (search_size_ == 0) {
    return arrow::Status::Invalid("Search size is 0");
  }
  if (ef_search_ == 0) {
    return arrow::Status::Invalid("Ef search is 0");
  }

  std::vector<float *> query_vectors;
  if (IsBatchQuery()) {
    ARROW_ASSIGN_OR_RAISE(query_vectors,
                          GetQueryRawVectorsFromRecordBatches(queries_));
  } else {
    float *raw_query = const_cast<float *>(query_);
    query_vectors.push_back(raw_query);
  }

  ARROW_ASSIGN_OR_RAISE(
      auto table_schema_with_dist,
      table_schema_->AddField(
          table_schema_->fields().size(),
          std::make_shared<arrow::Field>("distance", arrow::float32())));

  std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info;
  auto projection_string = projection_.value_or("*");
  ARROW_ASSIGN_OR_RAISE(
      projection_info,
      ProjectionInfoBuilder::BuildProjectionInfo(
          projection_string, index_infos_, table_schema_with_dist, {"distance"},
          [](const IndexInfo &index_info) {
            return index_info.IsDenseVectorIndex();
          }));

  if (segments_.empty()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Building empty executor due to table is empty");
    return std::make_shared<AnnEmptyExecutor>(
        table_schema_with_dist, projection_info, query_vectors.size());
  }

  auto [num_query_threads, num_index_handler_threads] =
      GetNumThreads(query_vectors);

  bool can_use_in_filter =
      filter_ && vdb::ServerConfiguration::GetEnableInFilter();

  return std::make_shared<AnnBatchExecutor>(
      ann_column_id_, top_k_size_, search_size_, ef_search_,
      table_schema_with_dist, segments_, index_infos_, projection_info, filter_,
      num_query_threads, num_index_handler_threads, query_vectors,
      can_use_in_filter);
}

// ResultSetBuilderUtils implementations - no more templates!
std::shared_ptr<std::map<ResultSetBuilderUtils::SegmentSetIdPair,
                         std::vector<ResultSetBuilderUtils::LabelValuePair>>>
ResultSetBuilderUtils::AggregateLabelsBySegmentAndSet(
    std::shared_ptr<IndexSearchList> index_search_results, size_t top_k_size) {
  auto result = std::make_shared<
      std::map<SegmentSetIdPair, std::vector<LabelValuePair>>>();

  size_t result_cnt = 0;
  for (const auto &element : index_search_results->index_search_results_) {
    if (element.segment_idx_ == -1) {
      continue;  // this is empty slot go to next element
    }

    if (result_cnt == top_k_size) break;

    auto set_id = vdb::LabelInfo::GetSetNumber(element.label_);
    (*result)[std::make_tuple(element.segment_idx_, set_id)].push_back(
        std::make_pair(element.label_, element.value_));

    result_cnt++;
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ResultSetBuilderUtils::AddValueColumn(std::shared_ptr<arrow::RecordBatch> rb,
                                      std::vector<float> values,
                                      const std::string &column_name) {
  auto builder = std::make_unique<arrow::FloatBuilder>();
  ARROW_RETURN_NOT_OK(builder->AppendValues(values));

  std::shared_ptr<arrow::Array> column;
  ARROW_RETURN_NOT_OK(builder->Finish(&column));
  return rb->AddColumn(
      rb->schema()->fields().size(),
      std::make_shared<arrow::Field>(column_name, arrow::float32()), column);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ResultSetBuilderUtils::ReplaceEmbeddingColumn(
    std::shared_ptr<arrow::RecordBatch> rb, std::shared_ptr<Segment> segment,
    uint32_t set_id, const std::vector<LabelValuePair> &labels,
    const std::vector<std::string> &index_column_names,
    std::shared_ptr<arrow::Schema> table_schema) {
  std::vector<uint64_t> label_list;
  for (const auto &label_value : labels) {
    label_list.push_back(label_value.first);
  }

  std::shared_ptr<arrow::RecordBatch> result_rb;
  std::shared_ptr<arrow::RecordBatch> temp_rb = rb;
  for (const auto &index_column_name : index_column_names) {
    auto embedding_field_idx =
        temp_rb->schema()->GetFieldIndex(index_column_name);
    ARROW_ASSIGN_OR_RAISE(auto emb_arr,
                          segment->IndexHandler()->GetEmbeddingArray(
                              embedding_field_idx, set_id, label_list));

    auto embedding_field_def = table_schema->GetFieldByName(index_column_name);

    ARROW_ASSIGN_OR_RAISE(
        result_rb,
        temp_rb->SetColumn(embedding_field_idx, embedding_field_def, emb_arr));
    temp_rb = result_rb;
  }
  return result_rb;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
ResultSetBuilderUtils::BuildResultSetFromLabels(
    const std::map<SegmentSetIdPair, std::vector<LabelValuePair>>
        &aggregated_labels,
    const std::vector<std::shared_ptr<Segment>> &segments,
    std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info,
    std::shared_ptr<arrow::Schema> table_schema,
    const std::string &value_column_name) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;

  for (const auto &[key, labels] : aggregated_labels) {
    const auto &[segment_idx, set_id] = key;
    auto segment = segments[segment_idx];

    std::vector<float> values;
    std::vector<uint32_t> record_ids;
    for (const auto &label_value : labels) {
      uint64_t label = label_value.first;
      float value = label_value.second;
      auto record_number = vdb::LabelInfo::GetRecordNumber(label);
      values.push_back(value);
      record_ids.push_back(record_number);
    }

    // Keep InactiveSet shared_ptr in scope to ensure RecordBatch safety
    std::shared_ptr<arrow::RecordBatch> set_record_batch;
    std::shared_ptr<InactiveSet> set;

    if (!projection_info->is_select_star) {
      // Use selective column loading for lazy loading optimization
      auto result = segment->GetRecordbatchWithSet(
          set_id, projection_info->table_select_column_indices);
      if (!result.ok()) {
        return result.status();
      }
      std::tie(set_record_batch, set) = result.ValueUnsafe();
    } else {
      // Load all columns
      std::tie(set_record_batch, set) = segment->GetRecordbatchWithSet(set_id);
    }

    if (set_record_batch == nullptr) {
      return arrow::Status::Invalid("Failed to get RecordBatch for set_id " +
                                    std::to_string(set_id));
    }

    ARROW_ASSIGN_OR_RAISE(
        auto result,
        arrow::compute::Take(
            set_record_batch,
            arrow::UInt32Array(record_ids.size(),
                               arrow::Buffer::FromVector(record_ids))));

    ARROW_ASSIGN_OR_RAISE(
        auto record_batch,
        AddValueColumn(result.record_batch(), values, value_column_name));

    if (projection_info->has_index_column) {
      ARROW_ASSIGN_OR_RAISE(
          record_batch, ReplaceEmbeddingColumn(
                            record_batch, segment, set_id, labels,
                            projection_info->index_column_names, table_schema));
    }

    record_batches.push_back(record_batch);
  }

  return record_batches;
}

arrow::Result<std::shared_ptr<arrow::Table>>
ResultSetBuilderUtils::BuildResultSet(
    std::shared_ptr<IndexSearchList> index_search_results, size_t top_k_size,
    const std::vector<std::shared_ptr<Segment>> &segments,
    std::shared_ptr<VectorSearchExecutor::ProjectionInfo> projection_info,
    std::shared_ptr<arrow::Schema> table_schema,
    const std::string &value_column_name) {
  auto aggregated_labels =
      AggregateLabelsBySegmentAndSet(index_search_results, top_k_size);

  ARROW_ASSIGN_OR_RAISE(
      auto record_batches,
      BuildResultSetFromLabels(*aggregated_labels, segments, projection_info,
                               table_schema, value_column_name));

  ARROW_ASSIGN_OR_RAISE(auto table,
                        arrow::Table::FromRecordBatches(record_batches));

  // project columns specified in projection
  std::vector<int> projection_indices;
  for (const auto &name : projection_info->projection_list) {
    auto field_idx = table->schema()->GetFieldIndex(name);
    if (field_idx != -1) {
      projection_indices.push_back(field_idx);
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto final_table,
                        table->SelectColumns(projection_indices));

  return final_table->ReplaceSchemaMetadata(nullptr);
}

std::shared_ptr<vdb::AnnFilterFunctor>
ResultSetBuilderUtils::CreateAnnFilterFunctor(
    const std::shared_ptr<Segment> &segment,
    std::shared_ptr<vdb::expression::Expression> filter,
    bool can_use_in_filter) {
  if (can_use_in_filter) {
    auto filter_functor = std::make_shared<vdb::FilterFunctor>(filter, segment);
    return std::make_shared<vdb::AnnFilterFunctor>(filter_functor);
  } else {
    if (filter) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogNotice,
                 "Filter (%s) is set, but 'enable-in-filter' is disabled. "
                 "Therefore, the in-filter feature will not be utilized",
                 filter->ToString().c_str());
    }
  }

  return nullptr;
}

std::shared_ptr<vdb::FilterFunctor> ResultSetBuilderUtils::CreateFilterFunctor(
    const std::shared_ptr<Segment> &segment,
    std::shared_ptr<vdb::expression::Expression> filter,
    bool can_use_in_filter) {
  if (can_use_in_filter) {
    auto filter_functor = std::make_shared<vdb::FilterFunctor>(filter, segment);
    return filter_functor;
  } else {
    if (filter) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogNotice,
                 "Filter (%s) is set, but 'enable-in-filter' is disabled. "
                 "Therefore, the in-filter feature will not be utilized",
                 filter->ToString().c_str());
    }
  }

  return nullptr;
}

// IndexSearchExecutorUtils template implementations
template <typename QueryType, typename SearchFunction>
arrow::Result<std::shared_ptr<IndexSearchList>>
IndexSearchExecutorUtils::RunIndexSearch(
    const QueryType &query, size_t query_idx, size_t num_queries,
    size_t num_segments, size_t result_size,
    std::optional<size_t> num_index_handler_threads,
    SearchFunction search_function, bool sort_ascending) {
  size_t num_threads = num_index_handler_threads.value_or(1);
  if (num_threads == 0) {
    return arrow::Status::Invalid(
        "The number index handler threads should be greater than 0");
  }

  return num_threads == 1
             ? RunIndexSearchSingleThread(query, query_idx, num_queries,
                                          num_segments, result_size,
                                          search_function, sort_ascending)
             : RunIndexSearchMultipleThreads(
                   query, query_idx, num_queries, num_segments, result_size,
                   num_threads, search_function, sort_ascending);
}

template <typename QueryType, typename SearchFunction>
arrow::Result<std::shared_ptr<IndexSearchList>>
IndexSearchExecutorUtils::RunIndexSearchSingleThread(
    const QueryType &query, size_t query_idx, size_t num_queries,
    size_t num_segments, size_t result_size, SearchFunction search_function,
    bool sort_ascending) {
  auto index_search_results =
      std::make_shared<IndexSearchList>(num_segments, result_size);

  for (size_t i = 0; i < num_segments; ++i) {
    search_function(i, query, index_search_results);
  }

  if (sort_ascending) {
    index_search_results->SortAscending();
  } else {
    index_search_results->SortDescending();
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "End of vector search from index handler for query vector %lu/%lu "
             "with single thread",
             query_idx, num_queries);

  return index_search_results;
}

template <typename QueryType, typename SearchFunction>
arrow::Result<std::shared_ptr<IndexSearchList>>
IndexSearchExecutorUtils::RunIndexSearchMultipleThreads(
    const QueryType &query, size_t query_idx, size_t num_queries,
    size_t num_segments, size_t result_size, size_t num_threads,
    SearchFunction search_function, bool sort_ascending) {
  auto index_search_results =
      std::make_shared<IndexSearchList>(num_segments, result_size);
  std::vector<std::thread> index_handler_threads;
  std::atomic<size_t> segment_idx = 0;

  for (size_t tid = 0; tid < num_threads; tid++) {
    index_handler_threads.emplace_back([&segment_idx, &query,
                                        &index_search_results, num_segments,
                                        search_function]() {
      size_t idx = segment_idx.fetch_add(1);
      while (idx < num_segments) {
        search_function(idx, query, index_search_results);
        idx = segment_idx.fetch_add(1);
      }
    });
  }

  std::for_each(index_handler_threads.begin(), index_handler_threads.end(),
                [](std::thread &t) { t.join(); });

  if (sort_ascending) {
    index_search_results->SortAscending();
  } else {
    index_search_results->SortDescending();
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "End of vector search from index handler for query vector %lu/%lu "
             "with multiple threads",
             query_idx, num_queries);

  return index_search_results;
}

// Explicit template instantiations for IndexSearchExecutorUtils
template arrow::Result<std::shared_ptr<IndexSearchList>>
IndexSearchExecutorUtils::RunIndexSearch<
    const float *, std::function<void(size_t, const float *,
                                      std::shared_ptr<IndexSearchList>)>>(
    const float *const &query, size_t query_idx, size_t num_queries,
    size_t num_segments, size_t result_size,
    std::optional<size_t> num_index_handler_threads,
    std::function<void(size_t, const float *, std::shared_ptr<IndexSearchList>)>
        search_function,
    bool sort_ascending);

template arrow::Result<std::shared_ptr<IndexSearchList>>
IndexSearchExecutorUtils::RunIndexSearch<
    const std::string &, std::function<void(size_t, const std::string &,
                                            std::shared_ptr<IndexSearchList>)>>(
    const std::string &query, size_t query_idx, size_t num_queries,
    size_t num_segments, size_t result_size,
    std::optional<size_t> num_index_handler_threads,
    std::function<void(size_t, const std::string &,
                       std::shared_ptr<IndexSearchList>)>
        search_function,
    bool sort_ascending);

// ScanRegistry implementation
ScanRegistry &ScanRegistry::GetInstance() {
  static ScanRegistry instance;
  return instance;
}

arrow::Status ScanRegistry::RegisterScan(
    const std::string &uuid, std::shared_ptr<IterativeResultSet> result_set) {
  if (!result_set) {
    return arrow::Status::Invalid("Cannot register null result set");
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (scans_.find(uuid) != scans_.end()) {
    return arrow::Status::Invalid("Scan UUID already exists: " + uuid);
  }

  scans_[uuid] = result_set;

  SYSTEM_LOG(vdb::LogTopic::Executor, vdb::LogLevel::kLogVerbose,
             "ScanRegistry::RegisterScan: uuid=%s, total_scans=%zu, "
             "include_internal=%s",
             uuid.c_str(), scans_.size(),
             result_set->GetIncludeInternalColumns() ? "true" : "false");

  return arrow::Status::OK();
}

arrow::Status ScanRegistry::UnregisterScan(const std::string &uuid) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = scans_.find(uuid);
  if (it == scans_.end()) {
    SYSTEM_LOG(vdb::LogTopic::Executor, vdb::LogLevel::kLogVerbose,
               "ScanRegistry::UnregisterScan: uuid=%s not found", uuid.c_str());
    return arrow::Status::OK();  // Not an error, already removed
  }

  scans_.erase(it);

  SYSTEM_LOG(vdb::LogTopic::Executor, vdb::LogLevel::kLogVerbose,
             "ScanRegistry::UnregisterScan: uuid=%s, remaining_scans=%zu",
             uuid.c_str(), scans_.size());

  return arrow::Status::OK();
}

std::shared_ptr<IterativeResultSet> ScanRegistry::GetScan(
    const std::string &uuid) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = scans_.find(uuid);
  if (it == scans_.end()) {
    return nullptr;
  }

  // Update access time
  it->second->UpdateAccessTime();

  return it->second;
}

void ScanRegistry::CleanupExpiredScans(std::chrono::seconds ttl) {
  std::vector<std::string> expired_uuids;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = scans_.begin(); it != scans_.end();) {
      if (it->second->IsExpired(ttl)) {
        expired_uuids.push_back(it->first);
        SYSTEM_LOG(vdb::LogTopic::Executor, vdb::LogLevel::kLogVerbose,
                   "ScanRegistry: Found expired scan: %s", it->first.c_str());
        it = scans_.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (!expired_uuids.empty()) {
    SYSTEM_LOG(vdb::LogTopic::Executor, vdb::LogLevel::kLogVerbose,
               "ScanRegistry: Removed %zu expired scan(s), remaining: %zu",
               expired_uuids.size(), GetActiveScanCount());
  }
}

std::vector<std::string> ScanRegistry::GetExpiredScanUUIDs(
    std::chrono::seconds ttl) {
  std::vector<std::string> expired_uuids;
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto &[uuid, result_set] : scans_) {
    if (result_set->IsExpired(ttl)) {
      expired_uuids.push_back(uuid);
    }
  }

  return expired_uuids;
}

size_t ScanRegistry::GetActiveScanCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return scans_.size();
}

}  // namespace vdb
