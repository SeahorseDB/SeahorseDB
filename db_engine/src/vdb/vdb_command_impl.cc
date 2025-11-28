#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <arrow/api.h>
#include <arrow/compare.h>
#include <arrow/json/from_string.h>
#include <arrow/result.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include "vdb/index/hnsw/common/interface.hh"

#include "vdb/data/expression.hh"
#include "vdb/vdb.hh"
#include "vdb/vdb_api.hh"
#include "vdb/data/checker.hh"
#include "vdb/common/fd_manager.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/memory_allocator.hh"
#include "vdb/common/util.hh"
#include "vdb/compute/execution.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/data/segmentation.hh"
#include "vdb/data/table.hh"
#include "vdb/data/statistics_collector.hh"

#ifdef __cplusplus
extern "C" {
#include "server.h"
}
#endif

// TODO - This macro should be replaced with the proper error handling in the
// future.
#define ARROW_IS_OK(ARROW_FUNC)                                 \
  {                                                             \
    auto status = ARROW_FUNC;                                   \
    if (!status.ok())                                           \
      std::cerr << "- Err) " << status.ToString() << std::endl; \
  }

namespace vdb {

// Helper function to check if user's filter contains internal columns
// Returns an error Status if internal columns are found
arrow::Status ValidateFilterForInternalColumns(std::string_view filter) {
  if (filter.empty()) {
    return arrow::Status::OK();
  }

  // Parse filter to extract potential column names
  std::string filter_str = std::string(filter);
  std::vector<std::string> tokens;
  std::string current_token;
  bool in_quotes = false;

  for (char c : filter_str) {
    if (c == '\'') {
      in_quotes = !in_quotes;
      current_token.clear();
    } else if (!in_quotes && (std::isspace(c) || c == '(' || c == ')' ||
                              c == '=' || c == '!' || c == '>' || c == '<' ||
                              c == ',' || c == '&' || c == '|')) {
      if (!current_token.empty()) {
        tokens.push_back(current_token);
        current_token.clear();
      }
    } else if (!in_quotes) {
      current_token += c;
    }
  }
  if (!current_token.empty() && !in_quotes) {
    tokens.push_back(current_token);
  }

  // Check if any token is an internal column name
  for (const auto& token : tokens) {
    // Skip SQL keywords and operators (case-insensitive)
    std::string upper_token = token;
    std::transform(upper_token.begin(), upper_token.end(), upper_token.begin(),
                   ::toupper);

    if (upper_token == "AND" || upper_token == "OR" || upper_token == "NOT" ||
        upper_token == "IN" || upper_token == "IS" || upper_token == "NULL" ||
        upper_token == "TRUE" || upper_token == "FALSE" ||
        upper_token == "LIKE") {
      continue;
    }

    // Check if this token is an internal column name
    if (vdb::IsInternalColumn(token)) {
      return arrow::Status::Invalid("Could not find column '" + token +
                                    "' from table schema");
    }
  }

  return arrow::Status::OK();
}

arrow::Result<std::pair<size_t, size_t>> _TotalRecordCountCommand() {
  auto table_dictionary = GetTableDictionary();
  size_t total_count = 0;
  size_t deleted_count = 0;
  for (auto& [table_name, table] : *table_dictionary) {
    for (auto& [segment_id, segment] : table->GetSegments()) {
      total_count += segment->Size();
      deleted_count += segment->DeletedSize();
    }
  }
  return std::make_pair(total_count, deleted_count);
}

arrow::Result<std::pair<size_t, size_t>> _TotalTableCountCommand() {
  auto table_dictionary = GetTableDictionary();
  size_t total_count = 0;
  size_t unloaded_count = 0;
  total_count = table_dictionary->size();
  /* TODO: count unloaded tables */
  return std::make_pair(total_count, unloaded_count);
}

Status _CreateTableCommand(sds serialized_schema) {
  auto schema_buffer = std::make_shared<arrow::Buffer>(
      reinterpret_cast<uint8_t*>(serialized_schema), sdslen(serialized_schema));
  auto buf_reader = std::make_shared<arrow::io::BufferReader>(schema_buffer);
  auto maybe_schema = arrow::ipc::ReadSchema(buf_reader.get(), nullptr);
  if (!maybe_schema.ok()) {
    return Status::InvalidArgument(
        "CreateTableCommand is Failed: Schema is incorrect. error=" +
        maybe_schema.status().ToString());
  }

  auto schema = maybe_schema.ValueUnsafe();
  auto table_dictionary = GetTableDictionary();
  auto metadata = schema->metadata();
  auto maybe_table_name = metadata->Get("table name");

  if (!maybe_table_name.ok()) {
    return Status::InvalidArgument(
        "CreateTableCommand is Failed: Table name is not provided.");
  }

  auto table_name = maybe_table_name.ValueUnsafe();
  if (std::any_of(table_name.begin(), table_name.end(),
                  [](char c) { return std::isspace(c); }))
    return Status::InvalidArgument(
        "CreateTableCommand is Failed: Invalid table name.");

  auto iter = table_dictionary->find(table_name);

  if (iter != table_dictionary->end()) {
    return Status::AlreadyExists(
        "CreateTableCommand is Failed: Table already exists.");
  }

  TableBuilderOptions options;
  options.SetTableName(table_name).SetSchema(schema);
  TableBuilder builder{options};
  ARROW_ASSIGN_OR_RAISE(auto table, builder.Build());
  table_dictionary->insert({table_name, table});
  return Status::Ok();
}

Status _DropTableCommand(const std::string& table_name, bool unloading) {
  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    if (!unloading) iter->second->SetDropping();
    table_dictionary->erase(iter);
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
               "Table (%s) Dropping is Done.", table_name.data());
    return Status::Ok();
  } else {
    return Status::InvalidArgument(
        "DropTableCommand is Failed: Table does not exists.");
  }
}

Status _InsertCommand(const std::string& table_name,
                      const std::string_view record_view) {
  auto table_dictionary = GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return Status::InvalidArgument(
        "InsertCommand is Failed: Invalid table name.");
  }

  auto table = iter->second;
  return table->AppendRecord(record_view);
}

Status _BatchInsertCommand(const std::string& table_name, sds serialized_rbs) {
  auto table_dictionary = GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return Status::InvalidArgument(
        "TableBatchInsertCommand is Failed: Invalid table name. ");
  }

  auto table = iter->second;
  auto schema = table->GetSchema();
  auto rbs_buffer = std::make_shared<arrow::Buffer>(
      reinterpret_cast<uint8_t*>(serialized_rbs), sdslen(serialized_rbs));
  auto maybe_reader = arrow::ipc::RecordBatchStreamReader::Open(
      std::make_shared<arrow::io::BufferReader>(rbs_buffer));
  if (!maybe_reader.ok()) {
    return Status::InvalidArgument("TableBatchInsertCommand is Failed: " +
                                   maybe_reader.status().ToString());
  }
  auto reader = maybe_reader.ValueUnsafe();
  auto rbs_schema = reader->schema();
  auto status = CheckRecordBatchIsInsertable(rbs_schema, table);
  if (!status.ok()) {
    return arrow::Status::Invalid("TableBatchInsertCommand is Failed: " +
                                  status.ToString());
  }

  auto maybe_rbs = reader->ToRecordBatches();
  if (!maybe_rbs.ok()) {
    return Status::InvalidArgument("TableBatchInsertCommand is Failed: " +
                                   maybe_rbs.status().ToString());
  }
  auto rbs = maybe_rbs.ValueUnsafe();

  return table->AppendRecords(rbs);
}

// Common structure to hold scan preparation results
struct ScanPreparationResult {
  std::shared_ptr<Table> table;
  std::shared_ptr<expression::Predicate> predicate;
  std::vector<std::shared_ptr<Segment>> filtered_segments;
  std::vector<int> column_indices;
  std::shared_ptr<arrow::Schema> exec_schema;
  int64_t limit_size;
};

// Helper function to add a column index if it doesn't already exist
static void AddColumnIfNotExists(std::vector<int>& column_indices,
                                 int col_idx) {
  if (std::find(column_indices.begin(), column_indices.end(), col_idx) ==
      column_indices.end()) {
    column_indices.push_back(col_idx);
  }
}

// Helper function to add all columns including internal ones
static void AddAllColumns(const std::shared_ptr<arrow::Schema>& internal_schema,
                          std::vector<int>& column_indices) {
  for (int i = 0; i < internal_schema->num_fields(); ++i) {
    column_indices.push_back(i);
  }
}

// Helper function to prepare common scan parameters
static arrow::Result<ScanPreparationResult> PrepareScanParameters(
    const std::string& table_name, std::string_view projection,
    std::string_view filter, std::string_view limit,
    bool include_internal_columns) {
  ScanPreparationResult result;

  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "ScanCommand is Failed: Invalid table name: " + table_name);
  }

  // For regular scan, check if user's filter contains internal columns (not
  // allowed)
  if (!include_internal_columns && !filter.empty()) {
    ARROW_RETURN_NOT_OK(ValidateFilterForInternalColumns(filter));
  }

  result.table = iter->second;

  // Extended schema with internal columns(ex. __deleted_flag).
  auto extended_schema = result.table->GetExtendedSchema();
  std::string extended_filter;
  if (include_internal_columns) {
    // Internal scan: use user's filter as-is (don't add __deleted_flag filter)
    extended_filter = std::string(filter);
  } else {
    // Regular scan: automatically filter out deleted records
    if (filter.empty())
      extended_filter = std::string(vdb::kDeletedFlagColumn) + " = false";
    else
      extended_filter = "(" + std::string(filter) + ") AND " +
                        std::string(vdb::kDeletedFlagColumn) + " = false";
  }

  expression::ExpressionBuilder builder(extended_schema);
  ARROW_ASSIGN_OR_RAISE(result.predicate, builder.ParseFilter(extended_filter));

  if (!result.predicate) {
    SYSTEM_LOG(
        vdb::LogTopic::Command, LogLevel::kLogDebug,
        "No filter is applied for table \"%s\" (include_internal_columns: %s)",
        table_name.c_str(), include_internal_columns ? "true" : "false");
  } else {
    SYSTEM_LOG(vdb::LogTopic::Command | vdb::LogTopic::Table,
               LogLevel::kLogDebug,
               "Filter \"%s\" is applied for table \"%s\" "
               "(include_internal_columns: %s)",
               extended_filter.data(), table_name.c_str(),
               include_internal_columns ? "true" : "false");
  }

  ARROW_ASSIGN_OR_RAISE(
      result.filtered_segments,
      result.table->GetFilteredSegments(result.predicate, extended_schema));

  ARROW_ASSIGN_OR_RAISE(result.limit_size, vdb::stoi64(limit));

  auto internal_schema = result.table->GetInternalSchema();
  bool is_select_star = (projection == "*" || projection.empty());

  // TODO: Currently only __deleted_flag exists as an internal column, so we add
  // a single column at the end. In the future, multiple internal columns may be
  // added. When generalizing internal column handling, this case should be
  // considered. For example, implement and call a function that adds all
  // internal columns.
  if (!is_select_star) {
    std::string proj_str(projection);
    std::stringstream ss(proj_str);
    std::string col_name;

    while (std::getline(ss, col_name, ',')) {
      col_name.erase(0, col_name.find_first_not_of(" \t"));
      col_name.erase(col_name.find_last_not_of(" \t") + 1);

      int col_idx = internal_schema->GetFieldIndex(col_name);
      if (col_idx != -1) {
        result.column_indices.push_back(col_idx);
      }
    }

    // Add columns referenced by filter predicate (except internal columns for
    // regular scan)
    if (result.predicate) {
      auto filter_cols = result.predicate->GetReferencedColumns();
      for (const auto& filter_col_name : filter_cols) {
        // For regular scan, skip internal columns - they will be added at the
        // end For internal scan, include them (user explicitly requested)
        if (!include_internal_columns &&
            vdb::IsInternalColumn(filter_col_name)) {
          continue;
        }
        int col_idx = internal_schema->GetFieldIndex(filter_col_name);
        if (col_idx != -1) {
          AddColumnIfNotExists(result.column_indices, col_idx);
        }
      }
    }

    // For regular scan, always add __deleted_flag at the END (must be last for
    // proper removal). For internal scan, __deleted_flag is only added if not
    // already included (since user may or may not explicitly project it).
    if (!include_internal_columns) {
      // Regular scan: ensure __deleted_flag is present
      // TODO: When multiple internal columns are added in the future, consider
      // whether all internal columns should be added, or only specific ones
      // needed for filtering (e.g., only __deleted_flag for deletion
      // filtering).
      int delete_flag_idx = internal_schema->num_fields() - 1;
      AddColumnIfNotExists(result.column_indices, delete_flag_idx);
    }

    std::vector<std::shared_ptr<arrow::Field>> selected_fields;
    selected_fields.reserve(result.column_indices.size());
    for (int col_idx : result.column_indices) {
      selected_fields.push_back(internal_schema->field(col_idx));
    }
    result.exec_schema = arrow::schema(selected_fields);
  } else {
    // Full scan: use all column indices
    // For now, both regular and internal scans include all columns
    // (user columns + __deleted_flag), as __deleted_flag is the only
    // internal column.
    // TODO: When multiple internal columns are added in the future:
    // - Regular scan should include only user columns + __deleted_flag
    // - Internal scan should include all columns (including all internal ones)
    AddAllColumns(internal_schema, result.column_indices);
    result.exec_schema = internal_schema;
  }

  return result;
}

// Scan implementation that returns RecordBatchReader for streaming processing
arrow::Result<std::tuple<std::shared_ptr<arrow::RecordBatchReader>,
                         std::shared_ptr<vdb::InactiveSetCleanupTracker>>>
ScanImpl(const std::string& table_name, std::string_view projection,
         std::string_view filter, std::string_view limit,
         bool include_internal_columns) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "ScanImpl is called for table \"%s\", projection \"%s\", "
             "filter \"%s\", limit \"%s\", include_internal_columns \"%s\"",
             table_name.c_str(), projection.data(), filter.data(), limit.data(),
             include_internal_columns ? "true" : "false");

  ARROW_ASSIGN_OR_RAISE(
      auto prep, PrepareScanParameters(table_name, projection, filter, limit,
                                       include_internal_columns));

  auto [generator, cleanup_tracker] = vdb::SegmentGeneratorWithTracking(
      std::move(prep.filtered_segments), prep.column_indices);

  ARROW_ASSIGN_OR_RAISE(
      auto reader, PlanExecutor::ExecutePlan(
                       prep.exec_schema, generator, std::optional{projection},
                       prep.predicate, include_internal_columns));

  return std::make_tuple(reader, cleanup_tracker);
}

arrow::Result<std::shared_ptr<arrow::Buffer>> _ScanCommand(
    const std::string& table_name, std::string_view projection,
    std::string_view filter, std::string_view limit,
    bool include_internal_columns) {
  SYSTEM_LOG(
      vdb::LogTopic::Unknown, LogLevel::kLogDebug,
      "_ScanCommand is called for table \"%s\", projection \"%s\", filter "
      "\"%s\", limit \"%s\", include_internal_columns \"%s\"",
      table_name.c_str(), projection.data(), filter.data(), limit.data(),
      include_internal_columns ? "true" : "false");

  // Prepare scan parameters (get segments, schema, filter expression, etc.)
  ARROW_ASSIGN_OR_RAISE(
      auto prep, PrepareScanParameters(table_name, projection, filter, limit,
                                       include_internal_columns));

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "Scan starting: %zu segments total",
             prep.filtered_segments.size());

  // Create generator for all segments
  auto generator =
      SegmentGenerator(prep.filtered_segments, prep.column_indices);

  // Execute plan with streaming and get serialized buffer
  // This handles: batch processing, projection, embedding replacement, limit,
  // and IPC serialization
  ARROW_ASSIGN_OR_RAISE(
      auto result_buffer,
      PlanExecutor::ExecutePlanToBuffer(
          prep.exec_schema, generator, std::optional{projection},
          prep.predicate, prep.table, prep.limit_size,
          include_internal_columns));

  return result_buffer;
}

arrow::Result<std::tuple<std::shared_ptr<arrow::Buffer>, bool>>
_ScanOpenCommand(std::string_view uuid, const std::string& table_name,
                 std::string_view projection, std::string_view filter,
                 std::string_view limit, bool include_internal_columns) {
  std::string uuid_str(uuid);

  // Check if scan already exists
  if (ScanRegistry::GetInstance().GetScan(uuid_str)) {
    return arrow::Status::Invalid(
        "ScanOpenCommand is Failed: Could not store result set with given "
        "uuid because it already exists: " +
        uuid_str);
  }

  // Parse limit for StreamingResultSet
  ARROW_ASSIGN_OR_RAISE(auto limit_size, vdb::stoi64(limit));

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "_ScanOpenCommand: table=%s, limit_str='%s', limit_size=%ld",
             table_name.c_str(), std::string(limit).c_str(), limit_size);

  // Use streaming to avoid materializing entire table in memory
  ARROW_ASSIGN_OR_RAISE(auto scan_result,
                        ScanImpl(table_name, projection, filter, limit,
                                 include_internal_columns));
  auto reader = std::get<0>(scan_result);
  auto cleanup_tracker = std::get<1>(scan_result);

  // Create StreamingResultSet from reader with limit
  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "ScanOpenCommand: Creating StreamingResultSet for uuid=%s, "
             "limit_size=%ld",
             uuid_str.c_str(), limit_size);

  // Get table for embedding column replacement
  auto table_dict = vdb::GetTableDictionary();
  auto table = table_dict->at(table_name);

  auto result_set = std::make_shared<vdb::IterativeResultSet>(
      reader, limit_size, cleanup_tracker, table, include_internal_columns);

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "ScanOpenCommand: IterativeResultSet created, use_count=%ld",
             result_set.use_count());

  // Prefetch first batch (Next() returns nullptr on first call)
  ARROW_ASSIGN_OR_RAISE(auto first_null, result_set->Next());
  if (first_null != nullptr) {
    return arrow::Status::Invalid(
        "ScanOpenCommand: Unexpected non-null first batch from "
        "IterativeResultSet");
  }

  // Now get the actual first batch
  ARROW_ASSIGN_OR_RAISE(auto prefetched_rb, result_set->Next());
  bool has_next = result_set->HasNext();

  SYSTEM_LOG(
      vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
      "ScanOpenCommand: has_next=%d, prefetched_rb_rows=%ld, use_count=%ld",
      has_next, prefetched_rb ? prefetched_rb->num_rows() : 0,
      result_set.use_count());

  if (has_next) {
    // Register scan in ScanRegistry
    auto status =
        ScanRegistry::GetInstance().RegisterScan(uuid_str, result_set);
    if (!status.ok()) {
      return status;
    }

    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "ScanOpenCommand: Registered in ScanRegistry, use_count=%ld",
               result_set.use_count());
  } else {
    // CRITICAL: If no next batch, explicitly release reader to free memory
    // Even though IterativeResultSet will be destroyed, we need to ensure
    // reader is released immediately to break reference chain to segments
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "ScanOpenCommand: has_next=false, resetting IterativeResultSet "
               "(use_count=%ld)",
               result_set.use_count());

    result_set.reset();
  }

  // Handle empty result (nullptr)
  if (prefetched_rb == nullptr) {
    auto schema = reader->schema();
    if (!include_internal_columns) {
      ARROW_ASSIGN_OR_RAISE(schema,
                            vdb::FilterInternalColumnsFromSchema(schema));
    }
    ARROW_ASSIGN_OR_RAISE(prefetched_rb, arrow::RecordBatch::MakeEmpty(schema));
  } else {
    if (!include_internal_columns) {
      ARROW_ASSIGN_OR_RAISE(prefetched_rb,
                            vdb::FilterInternalColumns(prefetched_rb));
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto serialized_rb,
                        SerializeRecordBatch(prefetched_rb));

  return std::pair(serialized_rb, has_next);
}

arrow::Result<std::pair<std::shared_ptr<arrow::Buffer>, bool>>
_FetchNextCommand(const std::string& uuid) {
  auto result_set = ScanRegistry::GetInstance().GetScan(uuid);
  if (!result_set) {
    return arrow::Status::Invalid(
        "FetchNextCommand is Failed: Could not find result set with given "
        "uuid: " +
        uuid);
  }

  bool include_internal_columns = result_set->GetIncludeInternalColumns();

  // GetScan() already updates access time
  ARROW_ASSIGN_OR_RAISE(auto rb, result_set->Next());
  bool has_next = result_set->HasNext();

  // Remove internal columns(ex. __deleted_flag) column if present
  // Only filter for regular scans, not for internal scans
  if (!include_internal_columns) {
    ARROW_ASSIGN_OR_RAISE(rb, vdb::FilterInternalColumns(rb));
  }

  if (!has_next) {
    auto status = ScanRegistry::GetInstance().UnregisterScan(uuid);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
                 "FetchNextCommand: Failed to unregister scan %s: %s",
                 uuid.c_str(), status.ToString().c_str());
    }

    // Serialize before cleanup request
    ARROW_ASSIGN_OR_RAISE(auto serialized_rb, SerializeRecordBatch(rb));
    rb.reset();

    return std::pair(serialized_rb, has_next);
  }

  ARROW_ASSIGN_OR_RAISE(auto serialized_rb, SerializeRecordBatch(rb));

  return std::pair(serialized_rb, has_next);
}

arrow::Result<std::string> _DebugScanCommand(const std::string& table_name,
                                             std::string_view projection,
                                             std::string_view filter) {
  // ARROW_ASSIGN_OR_RAISE doesn't support structured binding
  ARROW_ASSIGN_OR_RAISE(auto scan_result,
                        ScanImpl(table_name, projection, filter, "0", false));
  auto [reader, cleanup_tracker] = scan_result;

  // Convert RecordBatchReader to Table for debug output
  ARROW_ASSIGN_OR_RAISE(auto filtered_table, reader->ToTable());
  return filtered_table->ToString();
}

std::vector<std::string_view> _ListTableCommand() {
  std::vector<std::string_view> table_list;
  auto table_dictionary = GetTableDictionary();
  for (auto it = table_dictionary->begin(); it != table_dictionary->end();
       it++) {
    table_list.emplace_back(it->first);
  }
  return table_list;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> _DescribeTableCommand(
    const std::string& table_name, bool include_internal_columns) {
  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "DescribeTableCommand is Failed: Table does not exists.");
  }

  // Use GetExtendedSchema if include_internal_columns is true, otherwise
  // GetSchema
  auto schema = include_internal_columns ? iter->second->GetExtendedSchema()
                                         : iter->second->GetSchema();
  return arrow::ipc::SerializeSchema(*schema, &arrow_pool);
}

static bool IsAnnColumn(
    const std::shared_ptr<vdb::vector<IndexInfo>>& index_infos,
    uint64_t ann_column_id) {
  for (auto& index_info : *index_infos) {
    if (index_info.GetColumnId() == ann_column_id &&
        index_info.IsDenseVectorIndex()) {
      return true;
    }
  }

  return false;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> _AnnCommand(
    const std::string& table_name, uint64_t ann_column_id, size_t k,
    const sds query, size_t ef_search, std::string_view projection,
    std::string_view filter) {
  auto table_dictionary = GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid("AnnCommand is Failed: Invalid table name.");
  }

  auto table = iter->second;
  auto schema = table->GetSchema();
  auto index_infos = table->GetIndexInfos();
  bool is_valid_ann_column = IsAnnColumn(index_infos, ann_column_id);

  /* if no ann_column is set, ann search is not supported. */
  if (!is_valid_ann_column) {
    return arrow::Status::Invalid(
        "AnnCommand is Failed: Table does not support ann.");
  }

  auto dimension = table->GetAnnDimension(ann_column_id);

  if (dimension != sdslen(query) / sizeof(float)) {
    return arrow::Status::Invalid(
        "AnnCommand is Failed: Query dimension is not matched.");
  }
  ARROW_ASSIGN_OR_RAISE(
      auto kMaxThreads,
      vdb::stoi64(schema->metadata()
                      ->Get("max_threads")
                      .ValueOr(std::to_string(static_cast<size_t>(
                          std::thread::hardware_concurrency())))));

  auto raw_query = reinterpret_cast<float*>(query);

  expression::ExpressionBuilder builder(schema);
  ARROW_ASSIGN_OR_RAISE(auto predicate, builder.ParseFilter(filter));
  ARROW_ASSIGN_OR_RAISE(auto filtered_segments,
                        table->GetFilteredSegments(predicate, schema));

  auto dist_builder = std::make_unique<arrow::FloatBuilder>();
  std::shared_ptr<arrow::Array> dist_column;

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "Filtered %lu segments from %lu segments.",
             filtered_segments.size(), table->GetSegmentCount());

  ARROW_ASSIGN_OR_RAISE(auto ann_executor,
                        AnnExecutorBuilder()
                            .SetAnnColumnId(ann_column_id)
                            .SetTopKSize(k)
                            .SetSearchSize(GetSearchSize(k, ef_search))
                            .SetEfSearch(ef_search)
                            .SetTableSchema(schema)
                            .SetIndexInfos(index_infos)
                            .SetSegments(filtered_segments)
                            .SetProjection(projection)
                            .SetFilter(predicate)
                            .SetQueryVector(raw_query)
                            .SetNumMaxConcurrency(kMaxThreads)
                            .Build());
  ARROW_ASSIGN_OR_RAISE(auto result_buffers, ann_executor->Run());

  return result_buffers[0];
}

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> _BatchAnnCommand(
    std::string_view table_name, uint64_t ann_column_id, size_t k,
    const sds query_vectors, size_t ef_search, std::string_view projection,
    std::string_view filter) {
  ARROW_ASSIGN_OR_RAISE(auto table, GetTable(std::string(table_name)));
  auto schema = table->GetSchema();
  ARROW_ASSIGN_OR_RAISE(
      auto kMaxThreads,
      vdb::stoi64(schema->metadata()
                      ->Get("max_threads")
                      .ValueOr(std::to_string(static_cast<size_t>(
                          std::thread::hardware_concurrency())))));
  auto index_infos = table->GetIndexInfos();
  bool is_valid_ann_column = IsAnnColumn(index_infos, ann_column_id);

  /* if no ann_column is set, ann search is not supported. */
  if (!is_valid_ann_column) {
    return arrow::Status::Invalid(
        "AnnCommand is Failed: Table does not support ann.");
  }
  ARROW_ASSIGN_OR_RAISE(auto dimension, table->GetAnnDimension(ann_column_id));
  ARROW_ASSIGN_OR_RAISE(auto rbs, MakeRecordBatchesFromSds(query_vectors));
  ARROW_RETURN_NOT_OK(vdb::CheckQueryVectors(rbs, dimension));

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "BatchAnnCommand is called for table %s", table_name.data());

  auto segments = table->GetSegments();
  expression::ExpressionBuilder builder(schema);
  ARROW_ASSIGN_OR_RAISE(auto predicate, builder.ParseFilter(filter));
  ARROW_ASSIGN_OR_RAISE(auto filtered_segments,
                        table->GetFilteredSegments(predicate, schema));

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "Filtered %lu segments from %lu segments.",
             filtered_segments.size(), segments.size());

  ARROW_ASSIGN_OR_RAISE(auto ann_executor,
                        AnnExecutorBuilder()
                            .SetAnnColumnId(ann_column_id)
                            .SetTopKSize(k)
                            .SetSearchSize(GetSearchSize(k, ef_search))
                            .SetEfSearch(ef_search)
                            .SetTableSchema(schema)
                            .SetIndexInfos(index_infos)
                            .SetSegments(filtered_segments)
                            .SetProjection(projection)
                            .SetFilter(predicate)
                            .SetQueryRecordBatches(rbs)
                            .SetNumMaxConcurrency(kMaxThreads)
                            .Build());
  ARROW_ASSIGN_OR_RAISE(auto result_buffers, ann_executor->Run());

  return result_buffers;
}

arrow::Result<uint32_t> _DeleteCommand(const std::string& table_name,
                                       std::string_view filter) {
  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "DeleteCommand is failed: Invalid table name.");
  }

  // Check if user's filter contains internal columns (not allowed)
  if (!filter.empty()) {
    ARROW_RETURN_NOT_OK(ValidateFilterForInternalColumns(filter));
  }

  auto table = iter->second;
  auto schema = table->GetSchema();

  /* Extended schema with __deleted_flag(internal column) */
  auto extended_schema = table->GetInternalSchema();
  extended_schema = extended_schema->WithMetadata(schema->metadata());

  /* Build new filter expression */
  std::string extended_filter;
  if (filter.empty())
    extended_filter = std::string(vdb::kDeletedFlagColumn) + " = false";
  else
    extended_filter = "(" + std::string(filter) + ") AND " +
                      std::string(vdb::kDeletedFlagColumn) + " = false";

  expression::ExpressionBuilder builder(extended_schema);
  ARROW_ASSIGN_OR_RAISE(auto predicate, builder.ParseFilter(extended_filter));

  ARROW_ASSIGN_OR_RAISE(auto segments_to_delete,
                        table->GetFilteredSegments(predicate, schema));

  uint32_t deleted_count = 0;
  for (auto& segment : segments_to_delete) {
    auto status = segment->DeleteRecords(predicate);
    if (!status.ok()) {
      return status.status();
    }
    deleted_count += status.ValueUnsafe();
  }

  return deleted_count;
}

void SplitString(const std::string& s, char delim,
                 std::vector<std::string>* result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    result->push_back(item);
  }
}

arrow::Result<std::map<std::string, std::shared_ptr<arrow::Scalar>>>
_ParseUpdates(std::string_view updates,
              const std::shared_ptr<arrow::Schema>& schema) {
  std::map<std::string, std::string> parsed_map;
  size_t start = 0;
  size_t end = updates.find(',');
  size_t lbraket_pos = updates.find('[');
  size_t rbraket_pos = updates.find(']');

  while (end != std::string_view::npos) {
    if (lbraket_pos != std::string_view::npos && lbraket_pos < end &&
        rbraket_pos > end) {
      end = updates.find(',', end + 1); /* Skip comma in brackets */
      continue;
    }
    std::string_view token = updates.substr(start, end - start);
    size_t pos = token.find('=');
    if (pos != std::string_view::npos) {
      std::string key = std::string(token.substr(0, pos));
      std::string value = std::string(token.substr(pos + 1));

      /* Trim whitespace */
      key.erase(0, key.find_first_not_of(" \n\r\t"));
      key.erase(key.find_last_not_of(" \n\r\t") + 1);
      value.erase(0, value.find_first_not_of(" \n\r\t"));
      value.erase(value.find_last_not_of(" \n\r\t") + 1);

      parsed_map[key] = value;
    }
    start = end + 1;
    end = updates.find(',', start);
    lbraket_pos = updates.find('[', start);
    rbraket_pos = updates.find(']', start);
  }

  /* Handle the last token */
  std::string_view token = updates.substr(start);
  size_t pos = token.find('=');
  if (pos != std::string_view::npos) {
    std::string key = std::string(token.substr(0, pos));
    std::string value = std::string(token.substr(pos + 1));

    /* Trim whitespace */
    key.erase(0, key.find_first_not_of(" \n\r\t"));
    key.erase(key.find_last_not_of(" \n\r\t") + 1);
    value.erase(0, value.find_first_not_of(" \n\r\t"));
    value.erase(value.find_last_not_of(" \n\r\t") + 1);

    parsed_map[key] = value;
  }

  std::map<std::string, std::shared_ptr<arrow::Scalar>> update_map;
  /* Check if the key exists in the schema and check value fits
   * in schema type iterate through the update_map */
  for (auto& [key, value] : parsed_map) {
    auto field = schema->GetFieldByName(key);
    if (field == nullptr) {
      return arrow::Status::Invalid("Update key not found in schema: ", key);
    }
    auto data_type = field->type();
    if (data_type->id() == arrow::Type::STRING ||
        data_type->id() == arrow::Type::LARGE_STRING) {
      if (value.front() == '\'' && value.back() == '\'') {
        value = value.substr(1, value.length() - 2);
      } else {
        return arrow::Status::Invalid(
            "Update value is not a string: ", value,
            ". Please use single quote to wrap the string.");
      }
    }

    if (data_type->id() == arrow::Type::LIST) {
      /* If value type of list is string, change single quote to double quote
       * ex) value = "['a', 'b']" -> value = "["a", "b"]"
       */
      const auto& list_type = static_cast<const arrow::ListType&>(*data_type);
      if (list_type.value_type()->id() == arrow::Type::STRING ||
          list_type.value_type()->id() == arrow::Type::LARGE_STRING) {
        size_t pos = value.find("\'");
        while (pos != std::string::npos) {
          value = value.replace(pos, 1, "\"");
          pos = value.find("\'", pos + 1);
        }
      }

      /* If value type of list is boolean, change True/False or 1/0 to
       * true/false */
      if (list_type.value_type()->id() == arrow::Type::BOOL) {
        /* Change True/False to true/false */
        size_t pos = value.find("True");
        while (pos != std::string::npos) {
          value = value.replace(pos, 4, "true");
          pos = value.find("True", pos + 1);
        }
        pos = value.find("False");
        while (pos != std::string::npos) {
          value = value.replace(pos, 5, "false");
          pos = value.find("False", pos + 1);
        }
        /* Change 1/0 to true/false */
        pos = value.find("1");
        while (pos != std::string::npos) {
          value = value.replace(pos, 1, "true");
          pos = value.find("1", pos + 1);
        }
        pos = value.find("0");
        while (pos != std::string::npos) {
          value = value.replace(pos, 1, "false");
          pos = value.find("0", pos + 1);
        }
      }

      ARROW_ASSIGN_OR_RAISE(auto arr, arrow::json::ArrayFromJSONString(
                                          list_type.value_type(), value));
      auto scalar = std::make_shared<arrow::ListScalar>(arr);
      update_map[key] = scalar;
    } else if (data_type->id() == arrow::Type::FIXED_SIZE_LIST) {
      /* If value type of list is string, change single quote to double quote
       * ex) value = "['a', 'b']" -> value = "["a", "b"]"
       */
      const auto& list_type =
          static_cast<const arrow::FixedSizeListType&>(*data_type);
      if (list_type.value_type()->id() == arrow::Type::STRING ||
          list_type.value_type()->id() == arrow::Type::LARGE_STRING) {
        size_t pos = value.find("\'");
        while (pos != std::string::npos) {
          value = value.replace(pos, 1, "\"");
          pos = value.find("\'", pos + 1);
        }
      }

      /* If value type of list is boolean, change True/False or 1/0 to
       * true/false */
      if (list_type.value_type()->id() == arrow::Type::BOOL) {
        /* Change True/False to true/false */
        size_t pos = value.find("True");
        while (pos != std::string::npos) {
          value = value.replace(pos, 4, "true");
          pos = value.find("True", pos + 1);
        }
        pos = value.find("False");
        while (pos != std::string::npos) {
          value = value.replace(pos, 5, "false");
          pos = value.find("False", pos + 1);
        }
        /* Change 1/0 to true/false */
        pos = value.find("1");
        while (pos != std::string::npos) {
          value = value.replace(pos, 1, "true");
          pos = value.find("1", pos + 1);
        }
        pos = value.find("0");
        while (pos != std::string::npos) {
          value = value.replace(pos, 1, "false");
          pos = value.find("0", pos + 1);
        }
      }

      ARROW_ASSIGN_OR_RAISE(auto arr, arrow::json::ArrayFromJSONString(
                                          list_type.value_type(), value));
      auto scalar = std::make_shared<arrow::FixedSizeListScalar>(arr);
      update_map[key] = scalar;
    } else {
      ARROW_ASSIGN_OR_RAISE(auto scalar,
                            arrow::Scalar::Parse(data_type, value));
      update_map[key] = scalar;
    }
  }

  return update_map;
}

arrow::Result<std::shared_ptr<arrow::Array>> _ReplaceColumnValues(
    const std::shared_ptr<arrow::Array>& column,
    const std::shared_ptr<arrow::Scalar>& update_value) {
  if (column->type()->id() == arrow::Type::LIST) {
    ARROW_ASSIGN_OR_RAISE(auto new_array, arrow::MakeArrayFromScalar(
                                              *update_value, column->length()));
    return new_array;
  } else if (column->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    auto list_type =
        std::static_pointer_cast<arrow::FixedSizeListType>(column->type());

    auto list_scalar =
        std::static_pointer_cast<arrow::FixedSizeListScalar>(update_value);
    auto new_values = list_scalar->value;

    return std::make_shared<arrow::FixedSizeListArray>(
        list_type, column->length(), new_values);
  }

  /* Make mask for updating */
  ARROW_ASSIGN_OR_RAISE(
      auto mask,
      arrow::MakeArrayFromScalar(arrow::BooleanScalar(true), column->length()));

  ARROW_ASSIGN_OR_RAISE(
      auto result, arrow::compute::ReplaceWithMask(column, mask, update_value));
  return result.make_array();
}

arrow::Result<std::shared_ptr<arrow::Array>> _CreateIndexColumnArray(
    const std::shared_ptr<arrow::Field>& field,
    const std::shared_ptr<arrow::Scalar>& update_value, int64_t column_length,
    bool is_dense_vector_column) {
  /* Index columns contain label data instead of actual data (embedding values),
   * and the column data type set in the Schema is also UInt64Array.
   * This is to obtain embedding data via the embedding store using the label.
   * The data coming in at update time has an embedding type, but since the data
   * type cannot be inferred through the Schema, the data type must be inferred
   * from the Index type. */
  if (is_dense_vector_column) {
    auto list_type =
        std::static_pointer_cast<arrow::FixedSizeListType>(field->type());
    auto list_scalar =
        std::static_pointer_cast<arrow::FixedSizeListScalar>(update_value);
    auto new_values = list_scalar->value;

    return std::make_shared<arrow::FixedSizeListArray>(list_type, column_length,
                                                       new_values);
  } else {
    auto string_type =
        std::static_pointer_cast<arrow::StringType>(field->type());
    auto string_scalar =
        std::static_pointer_cast<arrow::StringScalar>(update_value);
    return arrow::MakeArrayFromScalar(*string_scalar, column_length);
  }
}

arrow::Result<std::shared_ptr<arrow::Array>> _ProcessColumnUpdate(
    const std::shared_ptr<arrow::Field>& field,
    const std::shared_ptr<arrow::Array>& column,
    const std::shared_ptr<arrow::Scalar>& update_value, int field_index,
    const std::shared_ptr<vdb::vector<IndexInfo>>& index_infos) {
  // Check if this is an index column
  bool is_index_column = false;
  bool is_dense_vector_column = false;

  for (auto& index_info : *index_infos) {
    if (static_cast<uint64_t>(field_index) == index_info.GetColumnId()) {
      is_index_column = true;
      is_dense_vector_column = index_info.IsDenseVectorIndex();
      break;
    }
  }

  if (is_index_column) {
    return _CreateIndexColumnArray(field, update_value, column->length(),
                                   is_dense_vector_column);
  } else {
    // For non-index columns, use the existing replacement logic
    return _ReplaceColumnValues(column, update_value);
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> _ApplyUpdatesToBatch(
    const std::shared_ptr<arrow::RecordBatch>& rb,
    const std::map<std::string, std::shared_ptr<arrow::Scalar>>& updates,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<vdb::vector<IndexInfo>>& index_infos) {
  /* Create array builders for applying updates to each column. */
  std::vector<std::shared_ptr<arrow::Array>> updated_columns;
  updated_columns.reserve(schema->num_fields());

  /* Iterate through each field and associated array to apply updates. */
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto field = schema->field(i);
    auto column = rb->column(i);

    /* Check if the field name exists in the updates map and apply the update.
     */
    auto it = updates.find(field->name());
    if (it != updates.end()) {
      ARROW_ASSIGN_OR_RAISE(
          auto updated_column,
          _ProcessColumnUpdate(field, column, it->second, i, index_infos));
      updated_columns.push_back(updated_column);
    } else {
      /* If no update is needed, use the original column as is. */
      updated_columns.push_back(column);
    }
  }

  /* Create a new RecordBatch with the updated columns. */
  auto updated_batch =
      arrow::RecordBatch::Make(schema, rb->num_rows(), updated_columns);
  return updated_batch;
}

arrow::Result<std::shared_ptr<arrow::Array>> _ReadEmbeddingDataByIndexType(
    const std::shared_ptr<Segment>& segment, const IndexInfo& index_info,
    const uint64_t* labels, int64_t num_rows) {
  auto column_id = index_info.GetColumnId();

  return segment->IndexHandler()->GetEmbeddingStore(column_id)->ReadToArray(
      labels, num_rows);
}

arrow::Result<uint32_t> _UpdateCommand(const std::string& table_name,
                                       std::string_view updates,
                                       std::string_view filter) {
  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "UpdateCommand is Failed: Invalid table name.");
  }

  // Check if user's filter contains internal columns (not allowed)
  if (!filter.empty()) {
    ARROW_RETURN_NOT_OK(ValidateFilterForInternalColumns(filter));
  }

  auto table = iter->second;
  auto schema = table->GetSchema();

  ARROW_ASSIGN_OR_RAISE(auto updates_map, _ParseUpdates(updates, schema));

  /* Extended schema with __deleted_flag(internal column) */
  auto extended_schema = table->GetInternalSchema();
  extended_schema = extended_schema->WithMetadata(schema->metadata());

  /* Build new filter expression */
  std::string extended_filter;
  if (filter.empty())
    extended_filter = std::string(vdb::kDeletedFlagColumn) + " = false";
  else
    extended_filter = "(" + std::string(filter) + ") AND " +
                      std::string(vdb::kDeletedFlagColumn) + " = false";

  /* Parse the filter to create a predicate */
  expression::ExpressionBuilder builder(extended_schema);
  ARROW_ASSIGN_OR_RAISE(auto predicate, builder.ParseFilter(extended_filter));

  /* Delete the records that match the filter */
  uint32_t update_count = 0;
  ARROW_ASSIGN_OR_RAISE(auto segments_to_delete,
                        table->GetFilteredSegments(predicate, schema));
  int num_segments = segments_to_delete.size();
  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>> deleted_records(
      num_segments);
  for (int idx = 0; idx < num_segments; ++idx) {
    auto& segment = segments_to_delete[idx];
    auto result =
        segment->DeleteRecords(predicate, true, &deleted_records[idx]);
    if (!result.ok()) {
      return result.status();
    }
    update_count += result.ValueUnsafe();
  }

  /* Check if the table has an index.
   * If it has, we need to replace rowid column to embedding column. */
  auto index_infos = table->GetIndexInfos();

  std::vector<std::vector<std::shared_ptr<arrow::RecordBatch>>> updated_records(
      num_segments);
  /* Prepare the updated records for insertion */
  for (int idx = 0; idx < num_segments; ++idx) {
    updated_records[idx].reserve(deleted_records[idx].size());
    for (auto& record_batch : deleted_records[idx]) {
      /* Replace phase: rowid column to embedding column. */
      for (auto& index_info : *index_infos) {
        auto ann_column_id = index_info.GetColumnId();
        auto labels = std::static_pointer_cast<arrow::UInt64Array>(
                          record_batch->column(ann_column_id))
                          ->raw_values();

        ARROW_ASSIGN_OR_RAISE(
            auto emb_arr,
            _ReadEmbeddingDataByIndexType(segments_to_delete[idx], index_info,
                                          labels, record_batch->num_rows()));

        ARROW_ASSIGN_OR_RAISE(
            record_batch,
            record_batch->SetColumn(
                ann_column_id,
                segments_to_delete[idx]->GetSchema()->field(ann_column_id),
                emb_arr));
      }

      /* Apply updates to each record batch */
      ARROW_ASSIGN_OR_RAISE(
          auto updated_batch,
          _ApplyUpdatesToBatch(record_batch, updates_map, schema, index_infos));
      if (updated_batch->num_rows() != 0) {
        updated_records[idx].push_back(std::move(updated_batch));
      }
    }
  }

  for (auto update_record_batches : updated_records) {
    if (!update_record_batches.empty()) {
      auto status = table->AppendRecords(update_record_batches);
      if (!status.ok()) {
        return arrow::Status::Invalid("UpdateCommand is Failed: " +
                                      status.ToString());
      }
    }
  }

  return update_count;
}

// Returns whether the background thread of this table is indexing
// or not.
arrow::Result<bool> _CheckIndexingCommand(const std::string& table_name) {
  auto table_dictionary = GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "CheckIndexingCommand is failed: Invalid table name.");
  }

  auto table = iter->second;
  return table->IsIndexing();
}

arrow::Result<size_t> _CountRecordsCommand(const std::string& table_name) {
  auto table_dictionary = GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "CountRecordsCommand is failed: Invalid table name.");
  }

  auto table = iter->second;
  size_t total_count = 0;
  for (auto& [_, segment] : table->GetSegments()) {
    total_count += segment->ActualSize();
  }
  return total_count;
}

// Returns numbers of total elements and indexed elements in the table with
// `table_name`.
arrow::Result<std::pair<uint64_t, std::vector<IndexedCountInfo>>>
_CountIndexedElementsCommand(const std::string& table_name) {
  auto table_dictionary = GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid(
        "CountIndexedElementsCommand is failed: Invalid table name.");
  }

  auto table = iter->second;
  auto index_infos = table->GetIndexInfos();
  std::vector<IndexedCountInfo> indexed_count_infos;
  for (auto& index_info : *index_infos) {
    IndexedCountInfo indexed_count_info;
    auto column_id = index_info.GetColumnId();
    indexed_count_info.column_name =
        table->GetSchema()->field(column_id)->name();
    indexed_count_info.index_type = index_info.GetIndexType();
    indexed_count_info.indexed_count = 0;
    indexed_count_infos.emplace_back(indexed_count_info);
  }

  size_t total_count = 0;
  for (auto& [_, segment] : table->GetSegments()) {
    int idx = 0;
    for (auto& index_info : *index_infos) {
      auto column_id = index_info.GetColumnId();
      indexed_count_infos[idx++].indexed_count +=
          segment->IndexHandler()->CountIndexedElements(column_id);
    }
    total_count += segment->ActualSize();
  }
  return std::pair{total_count, indexed_count_infos};
}

Status _LoadCommand(const std::string& table_name,
                    const std::string& directory_path) {
  auto actual_directory_path =
      directory_path.empty() ? server.load_data_path : directory_path;

  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    return Status::InvalidArgument("Table already exists: " + table_name);
  }

  auto status =
      LoadTableSnapshot(table_name.c_str(), actual_directory_path.c_str());
  if (!status) {
    return Status::InvalidArgument("Failed to load table snapshot: " +
                                   table_name);
  } else {
    return Status::Ok();
  }
}

Status _UnloadCommand(const std::string& table_name, const std::string& path) {
  // if path is empty, use the default path (server.unload_data_path)
  std::string actual_path = path.empty() ? server.unload_data_path : path;
  bool save_result = false;
  // Make snapshot
  auto prepare_result = PrepareTableSnapshot(table_name.data());
  if (prepare_result) {
    // only save if snapshot is prepared successfully
    save_result = SaveTableSnapshot(table_name.data(), actual_path.data());
  } else {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogDebug,
               "UnloadCommand is failed: Failed to prepare table snapshot.");
  }
  PostTableSnapshot();

  if (!save_result) {
    return Status::InvalidArgument(
        "UnloadCommand is failed: Failed to save table snapshot.");
  }

  // Unload table from memory
  auto drop_result = _DropTableCommand(table_name, true);
  if (!drop_result.ok()) {
    RemoveTableSnapshot(actual_path, table_name);
    return Status::InvalidArgument(
        "UnloadCommand is failed: Failed to drop table.");
  }

  std::string unload_table_path = actual_path + "/" + table_name;
  bool result = SyncAllFilesInDirectory(unload_table_path.c_str());
  if (!result) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Failed to sync unload table directory: %s",
               unload_table_path.c_str());
  }

  return Status::Ok();
}

Status _SaveCommand(const std::string& table_name, const std::string& path) {
  // if path is empty, use the default path (server.save_data_path)
  std::string actual_path = path.empty() ? server.save_data_path : path;
  bool save_result = false;
  auto prepare_result = PrepareTableSnapshot(table_name.data());
  if (prepare_result) {
    // only save if snapshot is prepared successfully
    save_result = SaveTableSnapshot(table_name.data(), actual_path.data());
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "SaveCommand is failed: Failed to prepare table snapshot.");
  }
  PostTableSnapshot();

  if (!save_result) {
    return Status::InvalidArgument(
        "SaveCommand is failed: Failed to save table snapshot.");
  }

  return Status::Ok();
}

arrow::Result<std::string> _SegmentStatisticsCommand(
    const std::optional<std::string>& table_name) {
  auto table_dictionary = GetTableDictionary();
  SegmentStatisticsCollector collector(table_dictionary);

  if (!table_name) {
    auto result = collector.CollectAll();
    if (!result.ok()) {
      return arrow::Status::Invalid(
          "SegmentStatisticsCommand is failed: Failed to collect segment "
          "statistics.");
    }
  } else {
    auto iter = table_dictionary->find(*table_name);
    if (iter == table_dictionary->end()) {
      return arrow::Status::Invalid(
          "SegmentStatisticsCommand is failed: Invalid table name.");
    }
    auto result = collector.CollectByTable(*table_name);
    if (!result.ok()) {
      return arrow::Status::Invalid(
          "SegmentStatisticsCommand is failed: Failed to collect segment "
          "statistics.");
    }
  }

  return arrow::Result<std::string>(collector.ToJsonString());
}

vdb::map<std::string, std::shared_ptr<vdb::Table>>* GetTableDictionary() {
  return static_cast<vdb::map<std::string, std::shared_ptr<vdb::Table>>*>(
      server.table_dictionary);
}

arrow::Result<std::shared_ptr<vdb::Table>> GetTable(
    const std::string& table_name) {
  auto table_dictionary = GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    return arrow::Status::Invalid("Could not find table: " + table_name);
  }
  return iter->second;
}

void RemoveTableSnapshot(const std::string& path,
                         const std::string& table_name) {
  std::string snapshot_path = path + "/" + table_name;
  std::filesystem::remove_all(snapshot_path);
}
}  // namespace vdb
