#pragma once

#include <arrow/api.h>
#include <memory>
#include <arrow/status.h>

#include "vdb/data/table.hh"

namespace vdb {

arrow::Status CheckQueryVectors(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& rbs,
    size_t dimension);

arrow::Status CheckRecordBatchIsInsertable(
    const std::shared_ptr<arrow::Schema>& rbs_schema,
    const std::shared_ptr<vdb::Table>& table);

arrow::Status CheckRecordBatchIsInsertable(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches,
    const std::shared_ptr<vdb::Table>& table);

}  // namespace vdb
