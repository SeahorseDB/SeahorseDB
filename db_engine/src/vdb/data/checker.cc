#include "vdb/data/checker.hh"

#include <arrow/type.h>
#include <arrow/array.h>

#include <sstream>

namespace vdb {

arrow::Status CheckQueryVectors(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& rbs,
    size_t dimension) {
  for (auto& rb : rbs) {
    auto rb_schema = rb->schema();
    auto query_field = rb_schema->GetFieldByName("query");
    if (query_field == nullptr) {
      return arrow::Status::Invalid(
          "BatchAnnCommand is Failed: Query vectors column should be named "
          "'query'.");
    }

    if (query_field->type()->id() != arrow::FixedSizeListType::type_id) {
      return arrow::Status::Invalid(
          "BatchAnnCommand is Failed: Query vectors should be a fixed size "
          "list data type.");
    }

    if (std::static_pointer_cast<arrow::FixedSizeListType>(query_field->type())
            ->value_type()
            ->id() != arrow::FloatType::type_id) {
      return arrow::Status::Invalid(
          "BatchAnnCommand is Failed: Query vectors should be a fixed size "
          "list of float32.");
    }

    if (static_cast<int32_t>(dimension) !=
        std::static_pointer_cast<arrow::FixedSizeListType>(query_field->type())
            ->list_size()) {
      return arrow::Status::Invalid(
          "BatchAnnCommand is Failed: Query vectors should be a same dimension "
          "as the column of the table. dimension of table is " +
          std::to_string(dimension) + ".");
    }
  }
  return arrow::Status::OK();
}

arrow::Status CheckRecordBatchIsInsertable(
    const std::shared_ptr<arrow::Schema>& rbs_schema,
    const std::shared_ptr<vdb::Table>& table) {
  auto table_schema = table->GetSchema();
  if (rbs_schema->num_fields() != table_schema->num_fields()) {
    return arrow::Status::Invalid(
        "Column size is different from table schema.");
  }

  for (int i = 0; i < rbs_schema->num_fields(); i++) {
    auto rbs_field = rbs_schema->field(i);
    auto table_field = table_schema->field(i);
    if (rbs_field->type()->id() != table_field->type()->id()) {
      std::stringstream error_message;
      error_message
          << "Column type is different from table schema. field index: " << i
          << ", table field name: " << table_field->name()
          << ", table field type: " << table_field->type()->ToString()
          << ", rbs field name: " << rbs_field->name()
          << ", rbs field type: " << rbs_field->type()->ToString();
      return arrow::Status::Invalid(error_message.str());
    }

    if (arrow::is_list(rbs_field->type()->id())) {
      auto rbs_list_type =
          std::static_pointer_cast<arrow::ListType>(rbs_field->type());
      auto table_list_type =
          std::static_pointer_cast<arrow::ListType>(table_field->type());
      if (rbs_list_type->value_type()->id() !=
          table_list_type->value_type()->id()) {
        std::stringstream error_message;
        error_message
            << "List value type is different from table schema. field index: "
            << i << ", table field name: " << table_field->name()
            << ", table field value type: "
            << table_list_type->value_type()->ToString()
            << ", rbs field name: " << rbs_field->name()
            << ", rbs field value type: "
            << rbs_list_type->value_type()->ToString();
        return arrow::Status::Invalid(error_message.str());
      }
    }

    if (rbs_field->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
      auto rbs_fixed_size_list_type =
          std::static_pointer_cast<arrow::FixedSizeListType>(rbs_field->type());
      auto table_fixed_size_list_type =
          std::static_pointer_cast<arrow::FixedSizeListType>(
              table_field->type());
      if (rbs_fixed_size_list_type->list_size() !=
          table_fixed_size_list_type->list_size()) {
        std::stringstream error_message;
        error_message << "Fixed size list size is different from table schema. "
                         "field index: "
                      << i << ", table field name: " << table_field->name()
                      << ", table field type: "
                      << table_field->type()->ToString()
                      << ", rbs field name: " << rbs_field->name()
                      << ", rbs field type: " << rbs_field->type()->ToString();
        return arrow::Status::Invalid(error_message.str());
      }
    }
  }

  return arrow::Status::OK();
}

arrow::Status CheckRecordBatchIsInsertable(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches,
    const std::shared_ptr<vdb::Table>& table) {
  if (record_batches.empty()) {
    return arrow::Status::Invalid("Empty record batches");
  }

  // First, check schema compatibility
  auto rbs_schema = record_batches[0]->schema();
  auto status = CheckRecordBatchIsInsertable(rbs_schema, table);
  if (!status.ok()) {
    return status;
  }

  // Then, check nullable constraints with actual data
  auto table_schema = table->GetSchema();
  for (auto& rb : record_batches) {
    for (int64_t i = 0; i < rb->num_columns(); ++i) {
      if (i < table_schema->num_fields()) {
        auto table_field = table_schema->field(i);
        auto input_arr = rb->column(i);
        if (table_field->nullable() == false) {
          if (input_arr->null_count() > 0) {
            return arrow::Status::Invalid(
                "Could not insert recordbatch into segment. Field '" +
                table_field->name() +
                "' is not nullable, but the recordbatch contains null values.");
          }
        }
      }
    }
  }

  return arrow::Status::OK();
}
}  // namespace vdb
