#pragma once

#include <map>
#include <optional>

#include <arrow/acero/api.h>

#include "vdb/common/memory_allocator.hh"
#include "vdb/common/status.hh"
#include "vdb/compute/execution.hh"
#include "vdb/data/statistics_collector.hh"
#include "vdb/data/table.hh"

// TODO: Memory allocator issue
struct client;

namespace vdb {
struct IndexedCountInfo {
  std::string column_name;
  std::string index_type;
  uint64_t indexed_count;
};

arrow::Result<std::pair<size_t, size_t>> _TotalRecordCountCommand();
arrow::Result<std::pair<size_t, size_t>> _TotalTableCountCommand();

std::vector<std::string_view> _ListTableCommand();

Status _CreateTableCommand(sds serialized_schema);

Status _DropTableCommand(const std::string& table_name, bool unloading = false);

arrow::Result<std::shared_ptr<arrow::Buffer>> _DescribeTableCommand(
    const std::string& table_name, bool include_internal_columns = false);

Status _InsertCommand(const std::string& table_name,
                      const std::string_view record);

Status _BatchInsertCommand(const std::string& table_name, sds serialized_rbs);

arrow::Result<std::string> _DebugScanCommand(const std::string& table_name,
                                             std::string_view proj_list_string,
                                             std::string_view filter_string);

arrow::Result<std::shared_ptr<arrow::Buffer>> _ScanCommand(
    const std::string& table_name, std::string_view proj_list_string,
    std::string_view filter_string, std::string_view limit_string,
    bool include_internal_columns = false);

arrow::Result<std::tuple<std::shared_ptr<arrow::Buffer>, bool>>
_ScanOpenCommand(std::string_view uuid, const std::string& table_name,
                 std::string_view proj_list_string,
                 std::string_view filter_string, std::string_view limit_string,
                 bool include_internal_columns = false);

arrow::Result<std::pair<std::shared_ptr<arrow::Buffer>, bool>>
_FetchNextCommand(const std::string& uuid);

arrow::Result<std::shared_ptr<arrow::Buffer>> _AnnCommand(
    const std::string& table_name, uint64_t ann_column_id, size_t k,
    const sds query, size_t ef_search, std::string_view projection,
    std::string_view filter);

arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> _BatchAnnCommand(
    std::string_view table_name, uint64_t ann_column_id, size_t k,
    const sds query_vectors, size_t ef_search, std::string_view projection,
    std::string_view filter);

arrow::Result<uint32_t> _DeleteCommand(const std::string& table_name,
                                       std::string_view filter);

arrow::Result<std::map<std::string, std::shared_ptr<arrow::Scalar>>>
_ParseUpdates(std::string_view updates,
              const std::shared_ptr<arrow::Schema>& schema);

arrow::Result<std::shared_ptr<arrow::Array>> _ReplaceColumnValues(
    const std::shared_ptr<arrow::Array>& column,
    const std::shared_ptr<arrow::Scalar>& update_value);

arrow::Result<std::shared_ptr<arrow::RecordBatch>> _ApplyUpdatesToBatch(
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const std::map<std::string, std::shared_ptr<arrow::Scalar>>& updates,
    const std::shared_ptr<arrow::Schema>& schema);

arrow::Result<uint32_t> _UpdateCommand(const std::string& table_name,
                                       std::string_view updates,
                                       std::string_view filter);

arrow::Result<bool> _CheckIndexingCommand(const std::string& table_name);

arrow::Result<size_t> _CountRecordsCommand(const std::string& table_name);
arrow::Result<std::pair<uint64_t, std::vector<IndexedCountInfo>>>
_CountIndexedElementsCommand(const std::string& table_name);

Status _LoadCommand(const std::string& table_name,
                    const std::string& directory_path);

Status _SaveCommand(const std::string& table_name, const std::string& path);

Status _UnloadCommand(const std::string& table_name, const std::string& path);

arrow::Result<std::string> _SegmentStatisticsCommand(
    const std::optional<std::string>& table_name);

vdb::map<std::string, std::shared_ptr<Table>>* GetTableDictionary();
arrow::Result<std::shared_ptr<Table>> GetTable(const std::string& table_name);

arrow::Result<std::shared_ptr<arrow::Table>> _BuildAnnResultTable(
    const std::vector<std::tuple<float, size_t, Segment*>>& ann_result,
    std::shared_ptr<arrow::Schema> schema, size_t k,
    std::string_view projection);

void RemoveTableSnapshot(const std::string& path,
                         const std::string& table_name);
}  // namespace vdb
