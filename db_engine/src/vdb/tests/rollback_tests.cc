#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sched.h>
#include <unordered_map>
#include <cstdint>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/type_fwd.h>

#include <gtest/gtest.h>

#include "vdb/common/memory_allocator.hh"
#include "vdb/common/fd_manager.hh"
#include "vdb/tests/util_for_test.hh"
#include "vdb/vdb.hh"
#include "vdb/vdb_api.hh"
#include "vdb/tests/base_environment.hh"

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/RollbackTestSuite";

// Parameters for indexed table rollback tests
struct RollbackTestParams {
  std::string index_type;

  RollbackTestParams(const std::string &index_type) : index_type(index_type) {}

  std::string ToString() const { return index_type; }
};

const std::vector<std::string> ROLLBACK_INDEX_TYPES = {"Hnsw"};

std::vector<RollbackTestParams> GenerateRollbackTestParams() {
  std::vector<RollbackTestParams> params;
  for (const auto &index_type : ROLLBACK_INDEX_TYPES) {
    params.emplace_back(index_type);
  }
  return params;
}
}  // namespace vdb

using namespace vdb::tests;

namespace vdb {
namespace {

class RollbackTestBase {
 protected:
  void SetUpRollbackTest() {
    // Disable memory availability check to bypass pre-check
    server.hidden_check_memory_availability = 0;
    // Reduce active set size limit for easier testing
    server.vdb_active_set_size_limit = 100;
    // Enable OOM testing
    server.hidden_enable_oom_testing = 1;
  }

  void CreateTestTable(const std::string &table_name) {
    // Use correct schema format and disable segmentation to avoid ID conflicts
    std::string schema_string =
        "id int32 not null, name String, feature Fixed_Size_List[3, Float32]";
    auto status = CreateTableForTest(table_name, schema_string,
                                     true);  // true = without_segment_id
    ASSERT_TRUE(status.ok()) << "Failed to create table: " << status.ToString();
  }

  void VerifyRecordCount(const std::string &table_name, int expected_count) {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "id", "", "0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    int actual_rows = table->num_rows();
    EXPECT_EQ(actual_rows, expected_count)
        << "Table " << table_name
        << " has unexpected row count. Expected: " << expected_count
        << ", Actual: " << actual_rows;
  }

  int GetTableRowCount(const std::string &table_name) {
    auto serialized_table_result = vdb::_ScanCommand(table_name, "id", "", "0");
    if (!serialized_table_result.ok()) {
      return -1;
    }

    auto table_result =
        DeserializeToTableFrom(serialized_table_result.ValueOrDie());
    if (!table_result.ok()) {
      return -1;
    }

    auto table = table_result.ValueOrDie();
    int row_count = table->num_rows();
    return row_count;
  }

  void VerifyDataPreservation(
      const std::string &table_name,
      const std::shared_ptr<arrow::Table> &initial_table,
      const std::shared_ptr<arrow::Table> &current_table, int verify_count) {
    // Validate input parameters
    ASSERT_TRUE(initial_table != nullptr)
        << "Initial table is null for table: " << table_name;
    ASSERT_TRUE(current_table != nullptr)
        << "Current table is null for table: " << table_name;

    int total_rows =
        std::min(verify_count, static_cast<int>(initial_table->num_rows()));
    int matched_rows = 0;

    // Compare each initial row with current table data
    for (int i = 0; i < total_rows; ++i) {
      // Get initial row ID (assuming first column is ID)
      auto initial_id_col = initial_table->column(0);
      auto initial_id_array = initial_id_col->chunk(0);
      auto initial_id_scalar = initial_id_array->GetScalar(i).ValueOrDie();

      // Find matching row in current table by ID
      bool found_match = false;
      for (int j = 0; j < current_table->num_rows(); ++j) {
        auto current_id_col = current_table->column(0);

        // Find the correct chunk containing row j
        std::shared_ptr<arrow::Array> current_id_array = nullptr;
        int row_offset = j;

        for (int chunk_idx = 0; chunk_idx < current_id_col->num_chunks();
             ++chunk_idx) {
          auto array = current_id_col->chunk(chunk_idx);
          if (row_offset < array->length()) {
            current_id_array = array;
            break;
          }
          row_offset -= array->length();
        }

        if (!current_id_array) continue;

        auto current_id_scalar =
            current_id_array->GetScalar(row_offset).ValueOrDie();

        // If IDs match, compare all columns (but skip feature column for
        // indexed tables to avoid embedding issues)
        if (initial_id_scalar->Equals(*current_id_scalar)) {
          bool all_columns_match = true;

          // For indexed tables, only compare id and name columns to avoid
          // embedding memory issues
          int cols_to_compare =
              (table_name.find("indexed") != std::string::npos)
                  ? 2
                  : initial_table->num_columns();

          for (int col = 0; col < cols_to_compare; ++col) {
            auto initial_col = initial_table->column(col);
            auto current_col = current_table->column(col);
            auto initial_array = initial_col->chunk(0);

            // Find correct chunk for current table
            std::shared_ptr<arrow::Array> current_array = nullptr;
            int curr_row_offset = j;

            for (int chunk_idx = 0; chunk_idx < current_col->num_chunks();
                 ++chunk_idx) {
              auto array = current_col->chunk(chunk_idx);
              if (curr_row_offset < array->length()) {
                current_array = array;
                break;
              }
              curr_row_offset -= array->length();
            }

            if (!current_array) {
              all_columns_match = false;
              break;
            }

            auto initial_val = initial_array->GetScalar(i).ValueOrDie();
            auto current_val =
                current_array->GetScalar(curr_row_offset).ValueOrDie();

            if (!initial_val->Equals(*current_val)) {
              all_columns_match = false;
              break;
            }
          }

          if (all_columns_match) {
            found_match = true;
            matched_rows++;
            break;
          }
        }
      }

      // Assert that each initial row was found and matched
      ASSERT_TRUE(found_match)
          << "Initial row " << i << " (ID: " << initial_id_scalar->ToString()
          << ") was not found or corrupted in current table: " << table_name;
    }

    // Final verification assertions
    ASSERT_EQ(matched_rows, total_rows)
        << "Data preservation failed for table: " << table_name << ". Expected "
        << total_rows << " rows to be preserved, "
        << "but only " << matched_rows << " rows were found intact.";
  }

  void InsertTestRecordsAsBatch(const std::string &table_name, int count,
                                const std::string &name_prefix = "test",
                                int start_id = 100) {
    auto table_dictionary = vdb::GetTableDictionary();
    auto table = table_dictionary->at(table_name);
    auto schema = table->GetSchema();

    // Create record batch
    arrow::Int32Builder id_builder;
    arrow::StringBuilder name_builder;
    auto value_builder =
        std::make_shared<arrow::FloatBuilder>(arrow::default_memory_pool());
    arrow::FixedSizeListBuilder feature_builder(arrow::default_memory_pool(),
                                                value_builder, 3);

    for (int i = 0; i < count; i++) {
      int record_id = start_id + i;
      ASSERT_OK(id_builder.Append(record_id));
      ASSERT_OK(name_builder.Append(name_prefix + std::to_string(i)));

      ASSERT_OK(feature_builder.Append());
      ASSERT_OK(value_builder->Append(i * 1.5f));
      ASSERT_OK(value_builder->Append(i * 2.5f));
      ASSERT_OK(value_builder->Append(i * 3.5f));
    }

    std::shared_ptr<arrow::Array> id_array, name_array, feature_array;
    ASSERT_OK(id_builder.Finish(&id_array));
    ASSERT_OK(name_builder.Finish(&name_array));
    ASSERT_OK(feature_builder.Finish(&feature_array));

    auto batch = arrow::RecordBatch::Make(
        schema, count, {id_array, name_array, feature_array});
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch};

    ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                         SerializeRecordBatches(schema, batches));
    sds serialized_batch_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_batch->data()),
                  static_cast<size_t>(serialized_batch->size()));

    auto status = vdb::_BatchInsertCommand(table_name, serialized_batch_sds);
    sdsfree(serialized_batch_sds);

    ASSERT_TRUE(status.ok()) << "Failed to batch insert " << count
                             << " records: " << status.ToString();
  }

  void TriggerOOMWithGuaranteedFailure(
      const std::string &table_name, int start_id, int &total_inserted_count,
      const std::string &name_prefix = "oom_test",
      bool is_indexed_table = false) {
    bool oom_occurred = false;
    total_inserted_count = 0;
    const int batch_size = 100;

    // Determine available memory based on whether the table is indexed
    const uint64_t memory_overhead =
        is_indexed_table ? 25600 : 10240;  // 25KB vs 10KB
    const std::string table_type = is_indexed_table ? "indexed table" : "table";
    const auto log_topic =
        is_indexed_table ? vdb::LogTopic::Index : vdb::LogTopic::Table;

    try {
      auto table_dictionary = vdb::GetTableDictionary();
      auto table = table_dictionary->at(table_name);
      auto schema = table->GetSchema();

      while (!oom_occurred) {
        // 1) batch preparation is done without memory limit change
        arrow::Int32Builder id_builder;
        arrow::StringBuilder name_builder;
        auto value_builder =
            std::make_shared<arrow::FloatBuilder>(arrow::default_memory_pool());
        arrow::FixedSizeListBuilder feature_builder(
            arrow::default_memory_pool(), value_builder, 3);

        for (int i = 0; i < batch_size; i++) {
          int record_id = start_id + (total_inserted_count * batch_size) + i;
          if (!id_builder.Append(record_id).ok()) {
            oom_occurred = true;
            break;
          }
          if (!name_builder.Append(name_prefix + std::to_string(record_id))
                   .ok()) {
            oom_occurred = true;
            break;
          }
          if (!feature_builder.Append().ok()) {
            oom_occurred = true;
            break;
          }
          if (!value_builder->Append((record_id) * 1.5f).ok() ||
              !value_builder->Append((record_id) * 2.5f).ok() ||
              !value_builder->Append((record_id) * 3.5f).ok()) {
            oom_occurred = true;
            break;
          }
        }
        if (oom_occurred) break;

        std::shared_ptr<arrow::Array> id_array, name_array, feature_array;
        if (!id_builder.Finish(&id_array).ok() ||
            !name_builder.Finish(&name_array).ok() ||
            !feature_builder.Finish(&feature_array).ok()) {
          oom_occurred = true;
          break;
        }

        auto batch = arrow::RecordBatch::Make(
            schema, batch_size, {id_array, name_array, feature_array});
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch};

        auto serialized_batch_result = SerializeRecordBatches(schema, batches);
        if (!serialized_batch_result.ok()) {
          SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                     "Failed to serialize batch for %s after %d batches",
                     table_type.c_str(), total_inserted_count);
          oom_occurred = true;
          break;
        }

        sds serialized_batch_sds = sdsnewlen(
            reinterpret_cast<const void *>(
                serialized_batch_result.ValueOrDie()->data()),
            static_cast<size_t>(serialized_batch_result.ValueOrDie()->size()));

        // 2) Apply memory limit based on table type
        vdb::Status status;
        try {
          uint64_t saved_limit = server.maxmemory;
          uint64_t low_limit = zmalloc_used_memory() + memory_overhead;
          server.maxmemory = low_limit;

          try {
            status = vdb::_BatchInsertCommand(table_name, serialized_batch_sds);
          } catch (const std::exception &e) {
            oom_occurred = true;
            SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                       "Exception during %s insert after %d batches: %s",
                       table_type.c_str(), total_inserted_count, e.what());
          } catch (...) {
            oom_occurred = true;
            SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                       "Unknown exception during %s insert after %d batches",
                       table_type.c_str(), total_inserted_count);
          }

          server.maxmemory = saved_limit;
        } catch (const std::exception &e) {
          oom_occurred = true;
          SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                     "Limit toggle exception for %s after %d batches: %s",
                     table_type.c_str(), total_inserted_count, e.what());
        }

        sdsfree(serialized_batch_sds);

        if (oom_occurred) break;

        if (status.IsOutOfMemory() || !status.ok()) {
          oom_occurred = true;
          SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                     "OOM via status for %s after %d batches: %s",
                     table_type.c_str(), total_inserted_count,
                     status.ToString().c_str());
          break;
        }

        total_inserted_count++;
        SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                   "Inserted %s batch %d (100 rows), total batches: %d",
                   table_type.c_str(), total_inserted_count,
                   total_inserted_count);
      }

      SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                 "%s trigger result: oom=%d, batches=%d", table_type.c_str(),
                 oom_occurred ? 1 : 0, total_inserted_count);

    } catch (const std::exception &e) {
      oom_occurred = true;
      SYSTEM_LOG(
          log_topic, vdb::LogLevel::kLogNotice,
          "Top-level exception in TriggerOOMWithGuaranteedFailure (%s): %s",
          table_type.c_str(), e.what());
    } catch (...) {
      oom_occurred = true;
      SYSTEM_LOG(log_topic, vdb::LogLevel::kLogNotice,
                 "Unknown exception in TriggerOOMWithGuaranteedFailure (%s)",
                 table_type.c_str());
    }

    EXPECT_GT(total_inserted_count, 0)
        << "Expected to insert some batches before OOM for " << table_type;
    ASSERT_TRUE(oom_occurred)
        << "OOM must occur for " << table_type << " test to be valid. "
        << "Inserted " << total_inserted_count
        << " batches without triggering OOM.";
  }

  void InsertTestRecordsAsBatchWithDenseIndex(
      const std::string &table_name, int start_id, int record_count,
      const std::string &name_prefix = "test") {
    auto table_dictionary = vdb::GetTableDictionary();
    auto table = table_dictionary->at(table_name);
    auto schema = table->GetSchema();

    arrow::Int32Builder id_builder;
    arrow::StringBuilder name_builder;
    auto value_builder =
        std::make_shared<arrow::FloatBuilder>(arrow::default_memory_pool());
    auto list_builder = arrow::FixedSizeListBuilder(
        arrow::default_memory_pool(), value_builder, 3);

    for (int i = 0; i < record_count; i++) {
      ASSERT_OK(id_builder.Append(start_id + i));
      ASSERT_OK(
          name_builder.Append(name_prefix + std::to_string(start_id + i)));

      ASSERT_OK(list_builder.Append());
      ASSERT_OK(value_builder->Append(static_cast<float>(start_id + i) + 0.1f));
      ASSERT_OK(value_builder->Append(static_cast<float>(start_id + i) + 0.2f));
      ASSERT_OK(value_builder->Append(static_cast<float>(start_id + i) + 0.3f));
    }

    std::shared_ptr<arrow::Array> id_array, name_array, feature_array;
    ASSERT_OK(id_builder.Finish(&id_array));
    ASSERT_OK(name_builder.Finish(&name_array));
    ASSERT_OK(list_builder.Finish(&feature_array));

    auto record_batch = arrow::RecordBatch::Make(
        schema, record_count, {id_array, name_array, feature_array});
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {record_batch};

    ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                         SerializeRecordBatches(schema, batches));
    sds serialized_batch_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_batch->data()),
                  static_cast<size_t>(serialized_batch->size()));

    auto status = vdb::_BatchInsertCommand(table_name, serialized_batch_sds);
    sdsfree(serialized_batch_sds);

    ASSERT_TRUE(status.ok())
        << "Failed to insert records: " << status.ToString();
  }
};

class MemoryCheckRollbackTest
    : public BaseTestSuiteWithParam<RollbackTestParams>,
      public RollbackTestBase {
 public:
  void SetUp() override {
    BaseTestSuiteWithParam<RollbackTestParams>::SetUp();
    SetUpRollbackTest();
  }

  void CreateTestTableWithDenseIndex(const std::string &table_name) {
    auto table_dictionary = vdb::GetTableDictionary();

    std::string schema_string =
        "id int32 not null, name String, feature Fixed_Size_List[3, Float32]";
    auto status = CreateTableForTest(table_name, schema_string, true);
    ASSERT_TRUE(status.ok()) << "Failed to create table: " << status.ToString();

    auto table = table_dictionary->at(table_name);

    // Create index_info_str based on GetParam() index type
    std::string index_info_str;
    const std::string &index_type = GetParam().index_type;
    index_info_str =
        R"([{"column_id": "2", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "16"}}])";

    auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"index_info", index_info_str}});
    TableWrapper::AddMetadata(table, add_metadata);
    TableWrapper::AddEmbeddingStore(table, 2);
  }
};

// Separate class for indexless table tests
class IndexlessRollbackTest : public BaseTestSuite, public RollbackTestBase {
 public:
  void SetUp() override {
    BaseTestSuite::SetUp();
    SetUpRollbackTest();
  }
};

TEST_F(IndexlessRollbackTest, IndexlessTableOOMRollback) {
  std::string table_name = "test_indexless_oom_rollback_table";

  // 1. Create indexless table
  CreateTestTable(table_name);

  // 2. Insert initial records (should succeed)
  server.maxmemory = 0;
  int initial_row_count = 5;
  InsertTestRecordsAsBatch(table_name, initial_row_count, "initial");
  VerifyRecordCount(table_name, initial_row_count);

  // save initial data query result
  ASSERT_OK_AND_ASSIGN(auto initial_serialized_table,
                       vdb::_ScanCommand(table_name, "*", "id < 105", "0"));
  ASSERT_OK_AND_ASSIGN(auto initial_table,
                       DeserializeToTableFrom(initial_serialized_table));

  // 3. Trigger OOM with guaranteed failure (repeat until OOM occurs)
  int inserted_count = 0;
  TriggerOOMWithGuaranteedFailure(table_name, 1000, inserted_count, "failed",
                                  false);

  // 4. Verify OOM occurred and rollback was successful
  server.maxmemory = 0;
  int min_expected_rows =
      (inserted_count) * 100 +
      initial_row_count;  // minimum expected count (OOM occured when insert
                          // data into active set)
  int max_expected_rows =
      (inserted_count + 1) * 100;  // maximum expected count (OOM occured when
                                   // change active set to inactive set)

  int actual_rows = GetTableRowCount(table_name);
  bool is_valid_count =
      (actual_rows == min_expected_rows) || (actual_rows == max_expected_rows);

  EXPECT_TRUE(is_valid_count)
      << "Actual rows (" << actual_rows << ") should be either "
      << min_expected_rows << " or " << max_expected_rows << ", but got "
      << actual_rows;

  // 5. Verify initial data is preserved
  ASSERT_OK_AND_ASSIGN(auto current_serialized_table,
                       vdb::_ScanCommand(table_name, "*", "id < 105", "0"));
  ASSERT_OK_AND_ASSIGN(auto current_table,
                       DeserializeToTableFrom(current_serialized_table));

  VerifyDataPreservation(table_name, initial_table, current_table,
                         initial_row_count);

  // 6. Insert additional data
  int current_row_count = GetTableRowCount(table_name);
  InsertTestRecordsAsBatch(table_name, 1000, "after_rollback");
  VerifyRecordCount(table_name,
                    current_row_count + 1000);  // 5 initial + 1000 new
}

// Instantiate the parameterized test for indexed rollback tests
INSTANTIATE_TEST_SUITE_P(
    IndexTypes, MemoryCheckRollbackTest,
    testing::ValuesIn(GenerateRollbackTestParams()),
    [](const testing::TestParamInfo<MemoryCheckRollbackTest::ParamType> &info) {
      return info.param.ToString();
    });

TEST_P(MemoryCheckRollbackTest, IndexedTableOOMRollback) {
  const std::string &index_type = GetParam().index_type;
  std::string table_name = "test_" + index_type + "_oom_rollback_table";

  SYSTEM_LOG(vdb::LogTopic::Index, vdb::LogLevel::kLogNotice,
             "Starting %s indexed table OOM rollback test", index_type.c_str());

  // 1. Create table with dense index (using GetParam())
  CreateTestTableWithDenseIndex(table_name);

  // 2. Insert initial records (should succeed)
  server.maxmemory = 0;
  int initial_row_count = 5;
  InsertTestRecordsAsBatchWithDenseIndex(table_name, 100, initial_row_count,
                                         "initial");
  VerifyRecordCount(table_name, initial_row_count);

  // scan initial data
  ASSERT_OK_AND_ASSIGN(
      auto initial_serialized_table,
      vdb::_ScanCommand(table_name, "id, name", "id < 105", "0"));
  ASSERT_OK_AND_ASSIGN(auto initial_table,
                       DeserializeToTableFrom(initial_serialized_table));

  // 3. Trigger OOM with guaranteed failure (with exception handling)
  int inserted_count = 0;

  try {
    TriggerOOMWithGuaranteedFailure(table_name, 1000, inserted_count, "failed",
                                    true);
  } catch (...) {
    // Catch any exceptions during OOM testing
    SYSTEM_LOG(vdb::LogTopic::Index, vdb::LogLevel::kLogDebug,
               "Exception caught during %s OOM test - this may be expected",
               index_type.c_str());
  }

  // 4. Verify rollback was successful
  server.maxmemory = 0;
  int min_expected_rows = (inserted_count) * 100 + initial_row_count;
  int max_expected_rows = (inserted_count + 1) * 100;
  int actual_rows = GetTableRowCount(table_name);

  bool is_valid_count =
      (actual_rows == min_expected_rows) || (actual_rows == max_expected_rows);
  EXPECT_TRUE(is_valid_count) << index_type << " - Actual rows (" << actual_rows
                              << ") should be either " << min_expected_rows
                              << " or " << max_expected_rows;
  EXPECT_GT(inserted_count, 0)
      << index_type << " - Should have inserted some batches before OOM";

  // scan initial data
  ASSERT_OK_AND_ASSIGN(
      auto current_serialized_table,
      vdb::_ScanCommand(table_name, "id, name", "id < 105", "0"));
  ASSERT_OK_AND_ASSIGN(auto current_table,
                       DeserializeToTableFrom(current_serialized_table));

  // 5. Verify initial data preservation
  VerifyDataPreservation(table_name, initial_table, current_table,
                         initial_row_count);

  // 6. Insert additional data
  int current_row_count = GetTableRowCount(table_name);
  InsertTestRecordsAsBatchWithDenseIndex(table_name, 2000, 300,
                                         "after_rollback");
  int new_row_count = GetTableRowCount(table_name);
  EXPECT_EQ(new_row_count, current_row_count + 300)
      << index_type << " - Should be able to insert after rollback";

  SYSTEM_LOG(vdb::LogTopic::Index, vdb::LogLevel::kLogNotice,
             "Completed %s indexed table OOM rollback test successfully",
             index_type.c_str());
}
}  // namespace
}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
