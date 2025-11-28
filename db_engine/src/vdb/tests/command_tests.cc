#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sched.h>
#include <unordered_map>

#include <arrow/api.h>
#include <arrow/json/from_string.h>
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

using namespace vdb::tests;

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/CommandTestSuite";

struct CommandTestParams {
  std::string index_type;
  std::string space_type;

  CommandTestParams(const std::string &index_type,
                    const std::string &space_type)
      : index_type(index_type), space_type(space_type) {}

  std::string ToString() const {
    std::string result = index_type + "_" + space_type;
    return result;
  }
};

class NoIndexTest : public BaseTestSuite {};
class WithIndexTest : public BaseTestSuiteWithParam<CommandTestParams> {
 public:
  void SetUp() override {
    BaseTestSuiteWithParam<CommandTestParams>::SetUp();
    auto index_type = GetParam().index_type;
    auto space_type = GetParam().space_type;
  }
};

/* Cosine or Inner Product Space is not appropriate for this task.

 * Example:

 * Original query vector
 *   [8.5, 2.5, 3.5]
 * Normalized:
 *   [0.8923, 0.2624, 0.3674]

 * Candidate A: [8.4, 2.2, 2.8]
 * Normalized:
 *   [0.9207, 0.2411, 0.3069]

 * Candidate B: [12.3, 3.4, 5.2]
 * Normalized:
 *   [0.8926, 0.2467, 0.3774]

 * Although [8.4, 2.2, 2.8] is much closer to the query in raw L2 distance,
 * the normalized direction of [12.3, 3.4, 5.2] happens to be more aligned
 * with the query in cosine/inner product space.

 * This is a classic case where vectors that look closer in Euclidean space
 * may have worse alignment in cosine space due to their directional difference.
 * => For this task, we prefer L2 distance over cosine similarity for better
 * geometric intuition. */

const std::vector<std::string> INDEX_TYPES = {"Hnsw"};
const std::vector<std::string> SPACE_TYPES = {"L2Space"};

std::vector<CommandTestParams> GenerateTestParams() {
  std::vector<CommandTestParams> params;
  for (const auto &index_type : INDEX_TYPES) {
    for (const auto &space_type : SPACE_TYPES) {
      if (index_type == "Hnsw") {
        params.emplace_back(index_type, space_type);
      }
    }
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(
    IndexTypes, WithIndexTest, testing::ValuesIn(GenerateTestParams()),
    [](const testing::TestParamInfo<WithIndexTest::ParamType> &info) {
      return info.param.ToString();
    });

class SnapshotTest : public BaseTestSuite {};
class MemoryLimitTest : public BaseTestSuite {
  void SetUp() override {
    server.maxmemory = 0;
    BaseTestSuite::SetUp();
    DeallocatePerformanceMonitor();
    server.vdb_active_set_size_limit = 2;
    server.hidden_enable_oom_testing = 1;
  }
};

TEST_P(WithIndexTest, CreateTableCommand) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  /*
   *std::string table_schema_string =
   *    "ID uint32, Name String, Attributes List[ String ], Feature "
   *    "Fixed_Size_List[ 2 ,   Float32 ]";
   */
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto input_schema = std::make_shared<arrow::Schema>(
      arrow::FieldVector{
          std::make_shared<arrow::Field>("id", arrow::uint32(), false),
          std::make_shared<arrow::Field>("name", arrow::utf8()),
          std::make_shared<arrow::Field>("attribute",
                                         arrow::list(arrow::utf8())),
          std::make_shared<arrow::Field>(
              "feature", arrow::fixed_size_list(arrow::float32(), 8))},
      std::make_shared<arrow::KeyValueMetadata>(
          std::unordered_map<std::string, std::string>{
              {"segmentation_info", segmentation_info_str},
              {"table name", test_table_name},
              {"active_set_size_limit", "10000"},
              {"index_info", index_info_str}}));
  auto maybe_serialized_schema =
      arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
  if (!maybe_serialized_schema.ok()) {
    std::cerr << maybe_serialized_schema.status().ToString() << std::endl;
    ASSERT_TRUE(maybe_serialized_schema.ok());
  }
  auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
  sds input_sds =
      sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                static_cast<size_t>(serialized_schema->size()));
  auto status = vdb::_CreateTableCommand(input_sds);
  sdsfree(input_sds);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  ASSERT_TRUE(table->GetSchema()->Equals(input_schema));
#ifdef _DEBUG_GTEST
  std::cout << table->ToString(true) << std::endl;
#endif
}

TEST_P(WithIndexTest, CreateTableCommandAndSetAnnColumn) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ 1024 ,   Float32 ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);

#ifdef _DEBUG_GTEST
  std::cout << "Table Before Setting Ann Column" << std::endl;
  std::cout << table->GetSchema()->metadata()->ToString() << std::endl;
#endif
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
#ifdef _DEBUG_GTEST
  std::cout << "Adding Metadata(Ann Column, etc.) Into Table" << std::endl;
  std::cout << add_metadata->ToString() << std::endl;
#endif
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

#ifdef _DEBUG_GTEST
  std::cout << "Table After Setting Ann Column" << std::endl;
  std::cout << table->GetSchema()->metadata()->ToString() << std::endl;
#endif
}

// Test that column names with allowed underscore patterns succeed
TEST_F(NoIndexTest, CreateTableWithAllowedUnderscorePatterns) {
  auto table_dictionary = vdb::GetTableDictionary();

  // Case 1: Column name with "__" in the middle (e.g., "normal__column") should
  // succeed
  {
    std::string test_table_name = "test_table_with_middle_underscore";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("normal__column", arrow::utf8()),
            std::make_shared<arrow::Field>("another__field", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should succeed because "__" is in the middle, not at the start
    ASSERT_TRUE(status.ok())
        << "Table creation should succeed with normal__column: "
        << status.ToString();

    // Verify table was created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter != table_dictionary->end())
        << "Table should exist after successful creation";

    // Clean up
    vdb::_DropTableCommand(test_table_name, false);

#ifdef _DEBUG_GTEST
    std::cout << "✓ Column with __ in middle (normal__column): Table created "
                 "successfully"
              << std::endl;
#endif
  }

  // Case 2: Column name starting with single underscore (e.g., "_phone") should
  // succeed
  {
    std::string test_table_name = "test_table_with_single_underscore";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("_phone", arrow::utf8()),
            std::make_shared<arrow::Field>("_email", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should succeed because single underscore is allowed
    ASSERT_TRUE(status.ok())
        << "Table creation should succeed with _phone column: "
        << status.ToString();

    // Verify table was created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter != table_dictionary->end())
        << "Table should exist after successful creation";

    // Clean up
    vdb::_DropTableCommand(test_table_name, false);

#ifdef _DEBUG_GTEST
    std::cout << "✓ Column with single underscore (_phone): Table created "
                 "successfully"
              << std::endl;
#endif
  }

  // Case 3: Column name ending with "__" (e.g., "name__") should succeed
  {
    std::string test_table_name = "test_table_with_ending_underscore";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("name__", arrow::utf8()),
            std::make_shared<arrow::Field>("value__", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should succeed because "__" is at the end, not at the start
    ASSERT_TRUE(status.ok())
        << "Table creation should succeed with name__ column: "
        << status.ToString();

    // Verify table was created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter != table_dictionary->end())
        << "Table should exist after successful creation";

    // Clean up
    vdb::_DropTableCommand(test_table_name, false);

#ifdef _DEBUG_GTEST
    std::cout << "✓ Column ending with __ (name__): Table created successfully"
              << std::endl;
#endif
  }

  // Case 4: Normal column names without underscore (e.g., "name", "email")
  // should succeed
  {
    std::string test_table_name = "test_table_with_normal_columns";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("name", arrow::utf8()),
            std::make_shared<arrow::Field>("email", arrow::utf8()),
            std::make_shared<arrow::Field>("phone", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should succeed - normal column names are always allowed
    ASSERT_TRUE(status.ok())
        << "Table creation should succeed with normal column names: "
        << status.ToString();

    // Verify table was created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter != table_dictionary->end())
        << "Table should exist after successful creation";

    // Clean up
    vdb::_DropTableCommand(test_table_name, false);

#ifdef _DEBUG_GTEST
    std::cout << "✓ Normal columns without underscore (name, email, phone): "
                 "Table created successfully"
              << std::endl;
#endif
  }
}

// Test that column names with reserved internal prefix "__" are rejected
TEST_F(NoIndexTest, CreateTableWithReservedInternalPrefixShouldFail) {
  auto table_dictionary = vdb::GetTableDictionary();

  // Case 1: Column name starting with "__phone" should fail
  {
    std::string test_table_name = "test_table_with_phone_prefix";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("name", arrow::utf8()),
            std::make_shared<arrow::Field>("__phone", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should fail because column name starts with internal prefix "__"
    ASSERT_FALSE(status.ok())
        << "Table creation should fail with __phone column";
    ASSERT_TRUE(status.ToString().find("reserved prefix") != std::string::npos)
        << "Error message should mention reserved prefix";

    // Verify table was not created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter == table_dictionary->end())
        << "Table should not exist after failed creation";

#ifdef _DEBUG_GTEST
    std::cout << "✓ Column __phone correctly rejected: " << status.ToString()
              << std::endl;
#endif
  }

  // Case 2: Multiple columns with "__" prefix should all fail
  {
    std::string test_table_name = "test_table_with_multiple_invalid";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("__deleted_flag", arrow::utf8()),
            std::make_shared<arrow::Field>("__second_invalid", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should fail because at least one column starts with "__"
    ASSERT_FALSE(status.ok())
        << "Table creation should fail with multiple __ prefixed columns";
    ASSERT_TRUE(status.ToString().find("reserved prefix") != std::string::npos);

    // Verify table was not created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter == table_dictionary->end())
        << "Table should not exist after failed creation";

#ifdef _DEBUG_GTEST
    std::cout << "✓ Multiple __ columns correctly rejected: "
              << status.ToString() << std::endl;
#endif
  }

  // Case 3: Column name starting with "___" (triple underscore) should FAIL
  // "__" or more underscores are reserved for internal columns
  {
    std::string test_table_name = "test_table_with_triple_underscore";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("name", arrow::utf8()),
            std::make_shared<arrow::Field>("___phone", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should FAIL because "___" (3+ underscores) is reserved for internal
    // columns Any column name starting with "__" (2 or more underscores) is
    // reserved
    ASSERT_FALSE(status.ok())
        << "Table creation should FAIL with ___phone column (internal prefix)";

    std::string error_message = status.ToString();
    ASSERT_TRUE(error_message.find("cannot start with reserved prefix") !=
                std::string::npos)
        << "Error message should mention reserved prefix, got: "
        << error_message;

    // Verify table was NOT created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter == table_dictionary->end())
        << "Table should NOT exist after failed creation";

#ifdef _DEBUG_GTEST
    std::cout << "✓ Column ___phone correctly rejected: " << error_message
              << std::endl;
#endif
  }

  // Case 4: Column name that is exactly "___" (just 3 underscores) should FAIL
  {
    std::string test_table_name = "test_table_with_only_triple_underscore";

    auto input_schema = std::make_shared<arrow::Schema>(
        arrow::FieldVector{
            std::make_shared<arrow::Field>("id", arrow::uint32(), false),
            std::make_shared<arrow::Field>("name", arrow::utf8()),
            std::make_shared<arrow::Field>("___", arrow::utf8())},
        std::make_shared<arrow::KeyValueMetadata>(
            std::unordered_map<std::string, std::string>{
                {"table name", test_table_name},
                {"active_set_size_limit", "10000"}}));

    auto maybe_serialized_schema =
        arrow::ipc::SerializeSchema(*input_schema, &vdb::arrow_pool);
    ASSERT_TRUE(maybe_serialized_schema.ok());

    auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
    sds input_sds =
        sdsnewlen(reinterpret_cast<const void *>(serialized_schema->data()),
                  static_cast<size_t>(serialized_schema->size()));
    auto status = vdb::_CreateTableCommand(input_sds);
    sdsfree(input_sds);

    // Should FAIL because "___" (3 underscores) is reserved for internal
    // columns
    ASSERT_FALSE(status.ok()) << "Table creation should FAIL with column name "
                                 "'___' (internal prefix)";

    std::string error_message = status.ToString();
    ASSERT_TRUE(error_message.find("cannot start with reserved prefix") !=
                std::string::npos)
        << "Error message should mention reserved prefix, got: "
        << error_message;

    // Verify table was NOT created
    auto iter = table_dictionary->find(test_table_name);
    ASSERT_TRUE(iter == table_dictionary->end())
        << "Table should NOT exist after failed creation";

#ifdef _DEBUG_GTEST
    std::cout << "✓ Column name '___' correctly rejected: " << error_message
              << std::endl;
#endif
  }
}

TEST_F(NoIndexTest, ShowTableCommand) {
  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ 1024 ,   Float32 ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  std::string test_table_name2 = "test_table2";
  std::string test_schema_string2 =
      "ID int16 not null, Name String, Attributes List[ String ]";

  status = CreateTableForTest(test_table_name2, test_schema_string2);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  auto table_dictionary = vdb::GetTableDictionary();

  ASSERT_EQ(table_dictionary->size(), 2);

  auto table_list = vdb::_ListTableCommand();

#ifdef _DEBUG_GTEST
  std::cout << "---- Table Lists ----" << std::endl;
  for (auto &table_name : table_list) {
    std::cout << table_name << std::endl;
  }
#endif
}

TEST_F(NoIndexTest, InsertCommand) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ]";

  std::cout << "test_schema_string: " << test_schema_string << std::endl;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  std::cout << "test_table_name: " << test_table_name << std::endl;
  auto table = table_dictionary->begin()->second;
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
  const char *data_sds = "0\u001eJohn\u001eC\u001dPython\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif

  const char *data_sds2 = "1\u001eJane\u001eLisp\u001dPython";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds2));
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto segments = table->GetSegments();
  EXPECT_EQ(segments.size(), 2);

  size_t cnt = 0;
  for (auto &kv : segments) {
    cnt += kv.second->Size();
  }
  EXPECT_EQ(cnt, 2);

#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
}

TEST_F(NoIndexTest, InsertCommandWithoutSegmentId) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  auto status = CreateTableForTest(
      test_table_name,
      "ID uint32 not null, Name String, Attributes List[ String ]", true);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->begin()->second;
  const char *data_sds = "0\u001eJohn\u001eC\u001dPython\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
  ASSERT_TRUE(status.ok()) << status.ToString();

  const char *data_sds2 = "1\u001eJane\u001eLisp\u001dPython";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds2));
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto segments = table->GetSegments();
  EXPECT_EQ(segments.size(), 1);
  auto segment = segments.begin()->second;
  EXPECT_STREQ(segment->GetId().data(), "_default_");
  EXPECT_EQ(segment->Size(), 2);
}

TEST_P(WithIndexTest, InsertCommandWithAddPoint) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();
  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Feature Fixed_Size_List[ 3 ,   Float32 "
      "]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  const char *data_sds = "0\u001eJohn\u001e12.3\u001d3.4\u001d5.2";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  const char *data_sds2 = "1\u001eJane\u001e11.2\u001d3.0\u001d4.0";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds2));
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto segments = table->GetSegments();
  EXPECT_EQ(segments.size(), 2);

  size_t cnt = 0;
  for (auto &kv : segments) {
    cnt += kv.second->Size();
  }
  EXPECT_EQ(cnt, 2);

#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
}

TEST_P(WithIndexTest, TableBatchInsertCommand) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, Name String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);
  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  for (int i = 0; i < 100; i++) {
    /* TODO: This CANNOT WORKING NOW
     * set has no embedding data, so directly inserting from segment's set IS
     * NOT POSSIBLE exchange input by generated recordbatch */
    ASSERT_OK_AND_ASSIGN(auto rb, GenerateRecordBatch(schema, 100, 3));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));

    serialized_rbs.push_back(serialized_rb);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto target_segments = target_table->GetSegments();

  size_t cnt = 0;
  for (auto &kv : target_segments) {
    auto segment = kv.second;
    auto segment_number = segment->GetSegmentNumber();
    for (uint32_t set_id = 0; set_id <= segment->InactiveSets().size();
         set_id++) {
      std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
      std::shared_ptr<arrow::RecordBatch> rb;
      if (set_id == segment->ActiveSetId()) {
        ASSERT_OK_AND_ASSIGN(rb, segment->GetRecordbatch(set_id));
      } else {
        inactive_set = segment->GetInactiveSet(set_id);
        rb = inactive_set->GetRb();
        ASSERT_NE(rb, nullptr);
      }

      bool ann_column_exists = false;
      bool hidden_column_exists = false;
      for (int64_t column_id = 0; column_id < rb->num_columns(); column_id++) {
        auto &column = rb->columns()[column_id];
        auto &field = rb->schema()->field(column_id);
        if (field->name() == vdb::kDeletedFlagColumn) {
          EXPECT_EQ(field->type()->id(), arrow::Type::BOOL);
          hidden_column_exists = true;
        }
        if (field->name() == "feature") {
          EXPECT_EQ(field->type()->id(), arrow::Type::UINT64);
          auto rowid_array =
              std::static_pointer_cast<arrow::UInt64Array>(column);
          for (int64_t i = 0; i < rb->num_rows(); i++) {
            uint64_t rowid = LabelInfo::Build(segment_number, set_id, i);
            auto value = rowid_array->Value(i);
#ifdef _DEBUG_GTEST
            std::cout << "rowid: " << rowid << ", value: " << value
                      << std::endl;
#endif
            EXPECT_EQ(value, rowid);
          }
          ann_column_exists = true;
        }
      }
      EXPECT_TRUE(ann_column_exists);
      EXPECT_TRUE(hidden_column_exists);
    }
    cnt += segment->Size();
  }
  EXPECT_EQ(cnt, 10000);

#ifdef _DEBUG_GTEST
  std::cout << target_table->ToString() << std::endl;
#endif
}

TEST_P(WithIndexTest, TableBatchInsertCommandWithStringPrimaryKey) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, pk String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);

  auto fields = schema->fields();
  auto pk_field = fields[1];
  auto pk_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{{"primary_key", "true"}});
  fields[1] = pk_field->WithMetadata(pk_metadata);

  schema = std::make_shared<arrow::Schema>(fields, schema->metadata());

  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  for (int i = 0; i < 100; i++) {
    ASSERT_OK_AND_ASSIGN(
        auto rb, GenerateRecordBatchWithPrimaryKey(
                     schema, "test" + std::to_string(i) + ".txt", 0, 100, 3));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));

    serialized_rbs.push_back(serialized_rb);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddMetadataToField(target_table, "pk", pk_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto target_segments = target_table->GetSegments();

  size_t cnt = 0;
  for (auto &kv : target_segments) {
    auto segment = kv.second;
    auto segment_number = segment->GetSegmentNumber();
    for (uint32_t set_id = 0; set_id <= segment->InactiveSets().size();
         set_id++) {
      std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
      std::shared_ptr<arrow::RecordBatch> rb;
      if (set_id == segment->ActiveSetId()) {
        ASSERT_OK_AND_ASSIGN(rb, segment->GetRecordbatch(set_id));
      } else {
        inactive_set = segment->GetInactiveSet(set_id);
        rb = inactive_set->GetRb();
        ASSERT_NE(rb, nullptr);
      }

      bool ann_column_exists = false;
      bool hidden_column_exists = false;
      for (int64_t column_id = 0; column_id < rb->num_columns(); column_id++) {
        auto &column = rb->columns()[column_id];
        auto &field = rb->schema()->field(column_id);
        if (field->name() == vdb::kDeletedFlagColumn) {
          EXPECT_EQ(field->type()->id(), arrow::Type::BOOL);
          hidden_column_exists = true;
        }
        if (field->name() == "feature") {
          EXPECT_EQ(field->type()->id(), arrow::Type::UINT64);
          auto rowid_array =
              std::static_pointer_cast<arrow::UInt64Array>(column);
          for (int64_t i = 0; i < rb->num_rows(); i++) {
            uint64_t rowid = LabelInfo::Build(segment_number, set_id, i);
            auto value = rowid_array->Value(i);
#ifdef _DEBUG_GTEST
            std::cout << "rowid: " << rowid << ", value: " << value
                      << std::endl;
#endif
            EXPECT_EQ(value, rowid);
          }
          ann_column_exists = true;
        }
      }
      EXPECT_TRUE(ann_column_exists);
      EXPECT_TRUE(hidden_column_exists);
    }
    cnt += segment->Size();
  }
  EXPECT_EQ(cnt, 10000);

#ifdef _DEBUG_GTEST
  std::cout << target_table->ToString() << std::endl;
#endif
}

TEST_P(WithIndexTest, TableBatchInsertCommandWithLargeStringPrimaryKey) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, pk Large_String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);

  auto fields = schema->fields();
  auto pk_field = fields[1];
  auto pk_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{{"primary_key", "true"}});
  fields[1] = pk_field->WithMetadata(pk_metadata);

  schema = std::make_shared<arrow::Schema>(fields, schema->metadata());

  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  for (int i = 0; i < 100; i++) {
    ASSERT_OK_AND_ASSIGN(
        auto rb,
        GenerateRecordBatchWithPrimaryKey(
            schema, "test" + std::to_string(i) + ".txt", 0, 100, 3, true));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));

    serialized_rbs.push_back(serialized_rb);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddMetadataToField(target_table, "pk", pk_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto target_segments = target_table->GetSegments();

  size_t cnt = 0;
  for (auto &kv : target_segments) {
    auto segment = kv.second;
    auto segment_number = segment->GetSegmentNumber();
    for (uint32_t set_id = 0; set_id <= segment->InactiveSets().size();
         set_id++) {
      std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
      std::shared_ptr<arrow::RecordBatch> rb;
      if (set_id == segment->ActiveSetId()) {
        ASSERT_OK_AND_ASSIGN(rb, segment->GetRecordbatch(set_id));
      } else {
        inactive_set = segment->GetInactiveSet(set_id);
        rb = inactive_set->GetRb();
        ASSERT_NE(rb, nullptr);
      }

      bool ann_column_exists = false;
      bool hidden_column_exists = false;
      for (int64_t column_id = 0; column_id < rb->num_columns(); column_id++) {
        auto &column = rb->columns()[column_id];
        auto &field = rb->schema()->field(column_id);
        if (field->name() == vdb::kDeletedFlagColumn) {
          EXPECT_EQ(field->type()->id(), arrow::Type::BOOL);
          hidden_column_exists = true;
        }
        if (field->name() == "feature") {
          EXPECT_EQ(field->type()->id(), arrow::Type::UINT64);
          auto rowid_array =
              std::static_pointer_cast<arrow::UInt64Array>(column);
          for (int64_t i = 0; i < rb->num_rows(); i++) {
            uint64_t rowid = LabelInfo::Build(segment_number, set_id, i);
            auto value = rowid_array->Value(i);
#ifdef _DEBUG_GTEST
            std::cout << "rowid: " << rowid << ", value: " << value
                      << std::endl;
#endif
            EXPECT_EQ(value, rowid);
          }
          ann_column_exists = true;
        }
      }
      EXPECT_TRUE(ann_column_exists);
      EXPECT_TRUE(hidden_column_exists);
    }
    cnt += segment->Size();
  }
  EXPECT_EQ(cnt, 10000);

#ifdef _DEBUG_GTEST
  std::cout << target_table->ToString() << std::endl;
#endif
}

TEST_P(WithIndexTest, TableBatchInsertCommandWithPrimaryKeyWithPartialOverlap) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, pk Large_String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);

  auto fields = schema->fields();
  auto pk_field = fields[1];
  auto pk_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{{"primary_key", "true"}});
  fields[1] = pk_field->WithMetadata(pk_metadata);

  schema = std::make_shared<arrow::Schema>(fields, schema->metadata());

  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  for (int i = 0; i < 100; i++) {
    ASSERT_OK_AND_ASSIGN(
        auto rb, GenerateRecordBatchWithPrimaryKey(schema, "test.txt", 150 * i,
                                                   200, 3, true));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));

    serialized_rbs.push_back(serialized_rb);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string, true);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddMetadataToField(target_table, "pk", pk_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto target_segments = target_table->GetSegments();

  size_t cnt = 0;
  for (auto &kv : target_segments) {
    auto segment = kv.second;
    auto segment_number = segment->GetSegmentNumber();
    for (uint32_t set_id = 0; set_id <= segment->InactiveSets().size();
         set_id++) {
      std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
      std::shared_ptr<arrow::RecordBatch> rb;
      if (set_id == segment->ActiveSetId()) {
        ASSERT_OK_AND_ASSIGN(rb, segment->GetRecordbatch(set_id));
      } else {
        inactive_set = segment->GetInactiveSet(set_id);
        rb = inactive_set->GetRb();
        ASSERT_NE(rb, nullptr);
      }

      bool ann_column_exists = false;
      bool hidden_column_exists = false;
      for (int64_t column_id = 0; column_id < rb->num_columns(); column_id++) {
        auto &column = rb->columns()[column_id];
        auto &field = rb->schema()->field(column_id);
        if (field->name() == vdb::kDeletedFlagColumn) {
          EXPECT_EQ(field->type()->id(), arrow::Type::BOOL);
          hidden_column_exists = true;
        }
        if (field->name() == "feature") {
          EXPECT_EQ(field->type()->id(), arrow::Type::UINT64);
          auto rowid_array =
              std::static_pointer_cast<arrow::UInt64Array>(column);
          for (int64_t i = 0; i < rb->num_rows(); i++) {
            uint64_t rowid = LabelInfo::Build(segment_number, set_id, i);
            auto value = rowid_array->Value(i);
#ifdef _DEBUG_GTEST
            std::cout << "rowid: " << rowid << ", value: " << value
                      << std::endl;
#endif
            EXPECT_EQ(value, rowid);
          }
          ann_column_exists = true;
        }
      }
      EXPECT_TRUE(ann_column_exists);
      EXPECT_TRUE(hidden_column_exists);
    }
    cnt += segment->Size();
  }
  EXPECT_EQ(cnt, 15050);

#ifdef _DEBUG_GTEST
  std::cout << target_table->ToString() << std::endl;
#endif
}

TEST_P(WithIndexTest, TableBatchInsertCommandWithoutSegmentId) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, Name String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);
  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  for (int i = 0; i < 100; i++) {
    /* TODO: This CANNOT WORKING NOW
     * set has no embedding data, so directly inserting from segment's set IS
     * NOT POSSIBLE exchange input by generated recordbatch */
    ASSERT_OK_AND_ASSIGN(auto rb, GenerateRecordBatch(schema, 100, 3));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));

    serialized_rbs.push_back(serialized_rb);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string, true);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto target_segments = target_table->GetSegments();
  for (auto &kv : target_segments) {
    auto segment = kv.second;
    auto segment_number = segment->GetSegmentNumber();
    for (uint32_t set_id = 0; set_id <= segment->InactiveSets().size();
         set_id++) {
      std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
      std::shared_ptr<arrow::RecordBatch> rb;
      if (set_id == segment->ActiveSetId()) {
        ASSERT_OK_AND_ASSIGN(rb, segment->GetRecordbatch(set_id));
      } else {
        inactive_set = segment->GetInactiveSet(set_id);
        rb = inactive_set->GetRb();
        ASSERT_NE(rb, nullptr);
      }

      bool ann_column_exists = false;
      bool hidden_column_exists = false;
      for (int64_t column_id = 0; column_id < rb->num_columns(); column_id++) {
        auto &column = rb->columns()[column_id];
        auto &field = rb->schema()->field(column_id);
        if (field->name() == vdb::kDeletedFlagColumn) {
          EXPECT_EQ(field->type()->id(), arrow::Type::BOOL);
          hidden_column_exists = true;
        }
        if (field->name() == "feature") {
          EXPECT_EQ(field->type()->id(), arrow::Type::UINT64);
          auto rowid_array =
              std::static_pointer_cast<arrow::UInt64Array>(column);
          for (int64_t i = 0; i < rb->num_rows(); i++) {
            uint64_t rowid = LabelInfo::Build(segment_number, set_id, i);
            auto value = rowid_array->Value(i);
#ifdef _DEBUG_GTEST
            std::cout << "rowid: " << rowid << ", value: " << value
                      << std::endl;
#endif
            EXPECT_EQ(value, rowid);
          }
          ann_column_exists = true;
        }
      }
      EXPECT_TRUE(ann_column_exists);
      EXPECT_TRUE(hidden_column_exists);
    }
  }
  EXPECT_EQ(target_segments.size(), 1);
  auto segment = target_segments.begin()->second;
  EXPECT_STREQ(segment->GetId().data(), "_default_");
}

TEST_P(WithIndexTest, TableBatchInsertCommandWithInvalidSchema) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();
  std::string source_table_name = "test_table";
  std::string schema_string =
      "ID uint64 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(source_table_name, schema_string);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto source_table = table_dictionary->at(source_table_name);

  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(source_table, add_metadata);
  TableWrapper::AddEmbeddingStore(source_table, ann_column_id);

  // Create a record batch with a different schema than the table
  arrow::SchemaBuilder schema_builder;
  ASSERT_OK(schema_builder.AddField(
      arrow::field("ID", arrow::utf8())));  // Changed from uint64 to string
  ASSERT_OK(schema_builder.AddField(arrow::field("Name", arrow::utf8())));
  ASSERT_OK(schema_builder.AddField(
      arrow::field("Feature", arrow::fixed_size_list(arrow::float32(), 3))));
  std::shared_ptr<arrow::Schema> different_schema =
      schema_builder.Finish().ValueOrDie();

  // Create arrays for the record batch
  auto id_arr = std::shared_ptr<arrow::Array>();
  auto name_arr = std::shared_ptr<arrow::Array>();
  auto feature_arr = std::shared_ptr<arrow::Array>();

  arrow::StringBuilder id_builder;
  arrow::StringBuilder name_builder;
  auto value_builder =
      std::make_shared<arrow::FloatBuilder>(arrow::default_memory_pool());
  arrow::FixedSizeListBuilder feature_builder(arrow::default_memory_pool(),
                                              value_builder, 3);

  ARROW_EXPECT_OK(id_builder.Append("0"));
  ARROW_EXPECT_OK(name_builder.Append("John"));
  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(12.3));
  ARROW_EXPECT_OK(value_builder->Append(3.4));
  ARROW_EXPECT_OK(value_builder->Append(5.2));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(name_builder.Finish(&name_arr));
  ARROW_EXPECT_OK(feature_builder.Finish(&feature_arr));

  auto rb = arrow::RecordBatch::Make(different_schema, 1,
                                     {id_arr, name_arr, feature_arr});
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                       SerializeRecordBatches(different_schema, rbs));

  sds serialized_rbs =
      sdsnewlen(reinterpret_cast<const void *>(serialized_rb->data()),
                static_cast<size_t>(serialized_rb->size()));

  // Attempt to insert the record batch with different schema
  status = vdb::_BatchInsertCommand(source_table_name, serialized_rbs);

  // Check that the insertion failed due to schema mismatch
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      status.ToString().find("Column type is different from table schema") !=
      std::string::npos);

  // Clean up
  sdsfree(serialized_rbs);
}

TEST_P(WithIndexTest,
       TableBatchInsertCommandWithNullValuesForNonNullableField) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();
  std::string source_table_name = "test_table";
  std::string schema_string =
      "ID uint64 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(source_table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto source_table = table_dictionary->at(source_table_name);

  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(source_table, add_metadata);
  TableWrapper::AddEmbeddingStore(source_table, ann_column_id);

  // Create a record batch with a different schema than the table
  arrow::SchemaBuilder schema_builder;
  ASSERT_OK(schema_builder.AddField(
      arrow::field("ID", arrow::uint64())));  // Changed from uint64 to string
  ASSERT_OK(schema_builder.AddField(arrow::field("Name", arrow::utf8())));
  ASSERT_OK(schema_builder.AddField(
      arrow::field("Feature", arrow::fixed_size_list(arrow::float32(), 3))));
  std::shared_ptr<arrow::Schema> different_schema =
      schema_builder.Finish().ValueOrDie();

  // Create arrays for the record batch
  auto id_arr = std::shared_ptr<arrow::Array>();
  auto name_arr = std::shared_ptr<arrow::Array>();
  auto feature_arr = std::shared_ptr<arrow::Array>();

  arrow::UInt64Builder id_builder;
  arrow::StringBuilder name_builder;
  auto value_builder =
      std::make_shared<arrow::FloatBuilder>(arrow::default_memory_pool());
  arrow::FixedSizeListBuilder feature_builder(arrow::default_memory_pool(),
                                              value_builder, 3);

  ARROW_EXPECT_OK(id_builder.AppendNull());
  ARROW_EXPECT_OK(name_builder.Append("John"));
  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(12.3));
  ARROW_EXPECT_OK(value_builder->Append(3.4));
  ARROW_EXPECT_OK(value_builder->Append(5.2));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(name_builder.Finish(&name_arr));
  ARROW_EXPECT_OK(feature_builder.Finish(&feature_arr));

  auto rb = arrow::RecordBatch::Make(different_schema, 1,
                                     {id_arr, name_arr, feature_arr});
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                       SerializeRecordBatches(different_schema, rbs));

  sds serialized_rbs =
      sdsnewlen(reinterpret_cast<const void *>(serialized_rb->data()),
                static_cast<size_t>(serialized_rb->size()));

  // Attempt to insert the record batch with different schema
  status = vdb::_BatchInsertCommand(source_table_name, serialized_rbs);

  // Check that the insertion failed due to schema mismatch
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(
      status.ToString().find("Could not insert recordbatch into segment. "
                             "Field 'id' is not nullable, but the recordbatch "
                             "contains null values") != std::string::npos);

  // Clean up
  sdsfree(serialized_rbs);
}

TEST_P(WithIndexTest, DropEmbeddingStoreTest) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, Name String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);
  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  for (int i = 0; i < 100; i++) {
    ASSERT_OK_AND_ASSIGN(auto rb, GenerateRecordBatch(schema, 100, 3));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));

    serialized_rbs.push_back(serialized_rb);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto embedding_store_directory_path =
      target_table->GetEmbeddingStore(ann_column_id)
          ->GetEmbeddingStoreDirectoryPath();

  // Explicitly release the target_table reference to ensure the Table object is
  // destroyed
  target_table = nullptr;
  EXPECT_TRUE(std::filesystem::exists(embedding_store_directory_path));
  status = vdb::_DropTableCommand(target_table_name);
  ASSERT_TRUE(status.ok()) << status.ToString();

  EXPECT_FALSE(std::filesystem::exists(embedding_store_directory_path));
}

TEST_F(NoIndexTest, DeleteCommand) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->begin()->second;
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
  const char *data_sds = "0\u001eJohn\u001eC\u001dPython\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
  ASSERT_TRUE(status.ok()) << status.ToString();
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif

  const char *data_sds2 = "1\u001eJane\u001eLisp\u001dPython";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds2));
  ASSERT_TRUE(status.ok()) << status.ToString();

  const char *data_sds3 = "2\u001eJohn\u001ePython\u001dRuby\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds3));
  ASSERT_TRUE(status.ok()) << status.ToString();

  const char *data_sds4 = "3\u001eMike\u001eC\u001dRuby\u001dLisp";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds4));
  ASSERT_TRUE(status.ok()) << status.ToString();

  ASSERT_OK_AND_ASSIGN(auto count_records,
                       vdb::_CountRecordsCommand(test_table_name));
  EXPECT_EQ(count_records, 4);

  {
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(test_table_name, "*", "id < 4", "0"));
    ASSERT_OK_AND_ASSIGN(auto scan_table,
                         DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0, 1, 2, 3}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John", "Jane", "John", "Mike"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"C", "Python", "Java"}));
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Lisp", "Python"}));
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Python", "Ruby", "Java"}));
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"C", "Ruby", "Lisp"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    ASSERT_OK_AND_ASSIGN(auto sorted_result, SortTable(scan_table));
    EXPECT_TRUE(sorted_result->Equals(*expected_table));
  }

  auto deleted_status = vdb::_DeleteCommand(test_table_name, "id < 2");
  ASSERT_TRUE(deleted_status.ok()) << deleted_status.status().ToString();
  auto deleted_count = deleted_status.ValueUnsafe();
  EXPECT_EQ(deleted_count, 2);

  ASSERT_OK_AND_ASSIGN(count_records,
                       vdb::_CountRecordsCommand(test_table_name));
  EXPECT_EQ(count_records, 2);

  deleted_status = vdb::_DeleteCommand(test_table_name, "id = 3");
  ASSERT_TRUE(deleted_status.ok()) << deleted_status.status().ToString();
  deleted_count = deleted_status.ValueUnsafe();
  EXPECT_EQ(deleted_count, 1);

  ASSERT_OK_AND_ASSIGN(count_records,
                       vdb::_CountRecordsCommand(test_table_name));
  EXPECT_EQ(count_records, 1);

  {
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(test_table_name, "*", "id < 4", "0"));
    ASSERT_OK_AND_ASSIGN(auto scan_table,
                         DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({2}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Python", "Ruby", "Java"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    EXPECT_TRUE(scan_table->Equals(*expected_table));
  }

  auto segments = table->GetSegments();
  EXPECT_EQ(segments.size(), 4);

  size_t cnt = 0;
  for (auto &kv : segments) {
    cnt += kv.second->Size();
  }
  EXPECT_EQ(cnt, 4);

#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
}

TEST_F(NoIndexTest, UpdateCommand) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->begin()->second;
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
  const char *data_sds = "0\u001eJohn\u001eC\u001dPython\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
  ASSERT_TRUE(status.ok()) << status.ToString();

#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif

  const char *data_sds2 = "1\u001eJane\u001eLisp\u001dPython";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds2));
  ASSERT_TRUE(status.ok()) << status.ToString();

  const char *data_sds3 = "2\u001eJohn\u001ePython\u001dRuby\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds3));
  ASSERT_TRUE(status.ok()) << status.ToString();

  ASSERT_OK_AND_ASSIGN(auto count_records,
                       vdb::_CountRecordsCommand(test_table_name));
  EXPECT_EQ(count_records, 3);

  {
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(test_table_name, "*", "id > 1", "0"));
    ASSERT_OK_AND_ASSIGN(auto scan_table,
                         DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({2}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Python", "Ruby", "Java"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    EXPECT_TRUE(scan_table->Equals(*expected_table));
  }

  auto updated_status =
      vdb::_UpdateCommand(test_table_name, "id = 3", "id = 2");
  ASSERT_TRUE(updated_status.ok()) << updated_status.status().ToString();
  auto updated_count = updated_status.ValueUnsafe();
  EXPECT_EQ(updated_count, 1);

  ASSERT_OK_AND_ASSIGN(count_records,
                       vdb::_CountRecordsCommand(test_table_name));
  EXPECT_EQ(count_records, 3);

  {
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(test_table_name, "*", "id > 1", "0"));
    ASSERT_OK_AND_ASSIGN(auto scan_table,
                         DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({3}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Python", "Ruby", "Java"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    EXPECT_TRUE(scan_table->Equals(*expected_table));
  }

  auto segments = table->GetSegments();
  EXPECT_EQ(segments.size(), 4);

  size_t cnt = 0;
  for (auto &kv : segments) {
    cnt += kv.second->Size();
  }
  EXPECT_EQ(cnt, 4);

#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
}

TEST_F(NoIndexTest, TableDescribeCommand) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto maybe_serialized_arrow_schema =
      vdb::_DescribeTableCommand(test_table_name);
  if (!maybe_serialized_arrow_schema.ok()) {
    std::cerr << maybe_serialized_arrow_schema.status().ToString() << std::endl;
    ASSERT_TRUE(maybe_serialized_arrow_schema.ok());
  }
  auto serialized_arrow_schema = maybe_serialized_arrow_schema.ValueUnsafe();
  arrow::ipc::DictionaryMemo in_memo;
  arrow::io::BufferReader buf_reader(serialized_arrow_schema);
  auto maybe_deserialized_arrow_schema = ReadSchema(&buf_reader, &in_memo);
  if (!maybe_deserialized_arrow_schema.ok()) {
    std::cerr << maybe_deserialized_arrow_schema.status().ToString()
              << std::endl;
    ASSERT_TRUE(maybe_deserialized_arrow_schema.ok());
  }
  auto deserialized_arrow_schema =
      maybe_deserialized_arrow_schema.ValueUnsafe();

  EXPECT_EQ(table_dictionary->at(test_table_name)->GetSchema()->ToString(),
            deserialized_arrow_schema->ToString());
#ifdef _DEBUG_GTEST
  std::cout << deserialized_arrow_schema->ToString() << std::endl;
#endif
}

TEST_F(NoIndexTest, TableDescribeInternalCommand) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto maybe_serialized_arrow_schema =
      vdb::_DescribeTableCommand(test_table_name, true);
  if (!maybe_serialized_arrow_schema.ok()) {
    std::cerr << maybe_serialized_arrow_schema.status().ToString() << std::endl;
    ASSERT_TRUE(maybe_serialized_arrow_schema.ok());
  }
  auto serialized_arrow_schema = maybe_serialized_arrow_schema.ValueUnsafe();
  arrow::ipc::DictionaryMemo in_memo;
  arrow::io::BufferReader buf_reader(serialized_arrow_schema);
  auto maybe_deserialized_arrow_schema = ReadSchema(&buf_reader, &in_memo);
  if (!maybe_deserialized_arrow_schema.ok()) {
    std::cerr << maybe_deserialized_arrow_schema.status().ToString()
              << std::endl;
    ASSERT_TRUE(maybe_deserialized_arrow_schema.ok());
  }
  auto deserialized_arrow_schema =
      maybe_deserialized_arrow_schema.ValueUnsafe();

  // Verify that the internal schema matches GetExtendedSchema()
  EXPECT_EQ(
      table_dictionary->at(test_table_name)->GetExtendedSchema()->ToString(),
      deserialized_arrow_schema->ToString());

  // Verify that internal columns are included
  auto field_names = deserialized_arrow_schema->field_names();
  bool has_deleted_flag = false;
  for (const auto &name : field_names) {
    if (name == vdb::kDeletedFlagColumn) {
      has_deleted_flag = true;
      break;
    }
  }
  EXPECT_TRUE(has_deleted_flag)
      << "Internal schema should include " << vdb::kDeletedFlagColumn;

  // Verify that internal schema has more fields than regular schema
  auto regular_schema = table_dictionary->at(test_table_name)->GetSchema();
  EXPECT_GT(deserialized_arrow_schema->num_fields(),
            regular_schema->num_fields())
      << "Internal schema should have more fields than regular schema";

#ifdef _DEBUG_GTEST
  std::cout << "Regular Schema: " << regular_schema->ToString() << std::endl;
  std::cout << "Internal Schema: " << deserialized_arrow_schema->ToString()
            << std::endl;
#endif
}

TEST_F(NoIndexTest, TableDebugScanCommand) {
  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "id uint32 not null, Name String, Attributes List[ String ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  const char *data_sds = "0\u001eJohn\u001eC\u001dPython\u001dJava";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
  ASSERT_TRUE(status.ok()) << status.ToString();

  const char *data_sds2 = "1\u001eJane\u001eLisp\u001dPython";
  status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds2));
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::string filter_string = "id is not null";
  std::string projection = "*";

  auto maybe_result =
      vdb::_DebugScanCommand(test_table_name, projection, filter_string);
  ASSERT_OK(maybe_result) << maybe_result.status().ToString();
#ifdef _DEBUG_GTEST
  auto result = maybe_result.ValueUnsafe();
  std::cout << result << std::endl;
#endif

  filter_string = "id < 1";

  maybe_result =
      vdb::_DebugScanCommand(test_table_name, projection, filter_string);
  ASSERT_OK(maybe_result) << maybe_result.status().ToString();
#ifdef _DEBUG_GTEST
  result = maybe_result.ValueUnsafe();
  std::cout << result << std::endl;
#endif
}

TEST_F(NoIndexTest, ScanCommand) {
  std::string table_name = "test_table";
  auto status = CreateTableForTest(
      table_name, "id uint32 not null, Name String, Attributes List[ String ]");
  ASSERT_TRUE(status.ok()) << status.ToString();

  status = vdb::_InsertCommand(table_name,
                               "0\u001eJohn\u001eC\u001dPython\u001dJava");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eJane\u001eLisp\u001dPython");
  ASSERT_TRUE(status.ok()) << status.ToString();

  {
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, "*", "id is not null", "0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0, 1}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John", "Jane"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"C", "Python", "Java"}));
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Lisp", "Python"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    ASSERT_OK_AND_ASSIGN(auto sorted_result, SortTable(table));
    ASSERT_OK_AND_ASSIGN(auto sorted_expected, SortTable(expected_table));
    EXPECT_TRUE(sorted_result->Equals(*sorted_expected));
  }

  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "id < 1", "0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"C", "Python", "Java"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    EXPECT_TRUE(table->Equals(*expected_table));
  }

  {
    auto s =
        vdb::_InsertCommand(table_name, "2\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "3\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "4\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "5\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "6\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();

    // limit test
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    EXPECT_EQ(table->num_rows(), 7);

    ASSERT_OK_AND_ASSIGN(serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "4"));
    ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
    EXPECT_EQ(table->num_rows(), 4);

    ASSERT_OK_AND_ASSIGN(serialized_table,
                         vdb::_ScanCommand(table_name, "*", "id < 4", "2"));
    ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
    EXPECT_EQ(table->num_rows(), 2);
  }
}

TEST_F(NoIndexTest, ScanOpenAndFetchCommand) {
  std::string table_name = "test_table";
  auto status = CreateTableForTest(
      table_name, "id uint32 not null, Name String, Attributes List[ String ]");
  ASSERT_TRUE(status.ok()) << status.ToString();

  status = vdb::_InsertCommand(table_name,
                               "0\u001eJohn\u001eC\u001dPython\u001dJava");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eJane\u001eLisp\u001dPython");
  ASSERT_TRUE(status.ok()) << status.ToString();

  {
    ASSERT_OK_AND_ASSIGN(auto open_result,
                         vdb::_ScanOpenCommand("id12345", table_name, "*",
                                               "id is not null", "0"));
    auto serialized_table = std::get<0>(open_result);
    ASSERT_OK_AND_ASSIGN(auto rbs,
                         DeserializeToRecordBatchesFrom(serialized_table));
    ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand("id12345"));
    auto serialized_table2 = std::get<0>(fetch_result);
    ASSERT_OK_AND_ASSIGN(auto rbs2,
                         DeserializeToRecordBatchesFrom(serialized_table2));

    auto concat_rbs = rbs;
    concat_rbs.insert(concat_rbs.end(), rbs2.begin(), rbs2.end());
    ASSERT_OK_AND_ASSIGN(auto table,
                         arrow::Table::FromRecordBatches(concat_rbs));

    // check whether the resultset is removed from the ScanRegistry
    auto resultset = vdb::ScanRegistry::GetInstance().GetScan("id12345");
    ASSERT_EQ(resultset, nullptr);

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0, 1}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John", "Jane"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"C", "Python", "Java"}));
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"Lisp", "Python"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    ASSERT_OK_AND_ASSIGN(auto sorted_result, SortTable(table));
    ASSERT_OK_AND_ASSIGN(auto sorted_expected, SortTable(expected_table));
    EXPECT_TRUE(sorted_result->Equals(*sorted_expected))
        << sorted_result->ToString() << "\n"
        << sorted_expected->ToString();
  }

  {
    ASSERT_OK_AND_ASSIGN(
        auto result,
        vdb::_ScanOpenCommand("id12345", table_name, "*", "id < 1", "0"));
    auto serialized_table = std::get<0>(result);
    // auto has_next = std::get<1>(result);
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0}));
    ASSERT_OK(id_builder.Finish(&id_array));

    arrow::StringBuilder name_builder;
    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.AppendValues({"John"}));
    ASSERT_OK(name_builder.Finish(&name_array));

    auto str_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder attributes_builder(arrow::default_memory_pool(),
                                          str_builder);
    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(attributes_builder.Append());
    ASSERT_OK(str_builder->AppendValues({"C", "Python", "Java"}));
    ASSERT_OK(attributes_builder.Finish(&attributes_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false),
                       arrow::field("name", arrow::utf8()),
                       arrow::field("attributes", arrow::list(arrow::utf8()))}),
        {id_array, name_array, attributes_array});

    EXPECT_TRUE(table->Equals(*expected_table));
  }

  {
    auto s =
        vdb::_InsertCommand(table_name, "2\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "3\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "4\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "5\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();
    s = vdb::_InsertCommand(table_name, "6\u001eJane\u001eLisp\u001dPython");
    ASSERT_TRUE(s.ok()) << status.ToString();

    // limit test
    ASSERT_OK_AND_ASSIGN(auto result, vdb::_ScanOpenCommand(
                                          "id12345", table_name, "*", "", "4"));
    auto serialized_table = std::get<0>(result);
    auto has_next = std::get<1>(result);
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    size_t count = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto next_table, vdb::_FetchNextCommand("id12345"));
      serialized_table = std::get<0>(next_table);
      has_next = std::get<1>(next_table);
      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      count += table->num_rows();
    }
    ASSERT_EQ(count, 4);

    // with filter and limit
    ASSERT_OK_AND_ASSIGN(result, vdb::_ScanOpenCommand("id12345", table_name,
                                                       "*", "id < 5", "1"));
    serialized_table = std::get<0>(result);
    ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1);
  }
}

TEST_F(NoIndexTest, ScanOpenFailedByDuplicateUUID) {
  std::string table_name = "test_table";
  auto status = CreateTableForTest(
      table_name, "id uint32 not null, Name String, Attributes List[ String ]");
  ASSERT_TRUE(status.ok()) << status.ToString();

  status = vdb::_InsertCommand(table_name,
                               "0\u001eJohn\u001eC\u001dPython\u001dJava");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eJane\u001eLisp\u001dPython");
  ASSERT_TRUE(status.ok()) << status.ToString();

  {
    ASSERT_OK_AND_ASSIGN(auto open_result,
                         vdb::_ScanOpenCommand("id12345", table_name, "*",
                                               "id is not null", "0"));
    auto serialized_table = std::get<0>(open_result);
    ASSERT_OK_AND_ASSIGN(auto rbs,
                         DeserializeToRecordBatchesFrom(serialized_table));

    // try to open the resultset with the existing uuid
    auto open_result2 = vdb::_ScanOpenCommand("id12345", table_name, "*",
                                              "id is not null", "0");
    ASSERT_TRUE(open_result2.status().IsInvalid());

    ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand("id12345"));
    auto serialized_table2 = std::get<0>(fetch_result);
    ASSERT_OK_AND_ASSIGN(auto rbs2,
                         DeserializeToRecordBatchesFrom(serialized_table2));

    auto concat_rbs = rbs;
    concat_rbs.insert(concat_rbs.end(), rbs2.begin(), rbs2.end());
    ASSERT_OK_AND_ASSIGN(auto table,
                         arrow::Table::FromRecordBatches(concat_rbs));

    // check whether the resultset is removed from the ScanRegistry
    auto resultset = vdb::ScanRegistry::GetInstance().GetScan("id12345");
    ASSERT_EQ(resultset, nullptr);
  }
}
TEST_P(WithIndexTest, ScanCommandWithEmbedding) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  server.vdb_active_set_size_limit = 2;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  ASSERT_OK_AND_ASSIGN(
      auto ids, arrow::json::ArrayFromJSONString(arrow::int32(),
                                                 "[0, 1, 2, 3, 4, 5, 6, 7]"));
  ASSERT_OK_AND_ASSIGN(auto names,
                       arrow::json::ArrayFromJSONString(
                           arrow::utf8(),
                           "[\"John\", \"Jane\", \"Mike\", \"Sarah\", "
                           "\"David\", \"Emily\", \"Tom\", \"Julia\"]"));
  ASSERT_OK_AND_ASSIGN(auto features,
                       arrow::json::ArrayFromJSONString(
                           arrow::fixed_size_list(arrow::float32(), 3),
                           "[[12.3, 3.4, 5.2], [11.2, 3.0, 4.0], [10.5, 2.8, "
                           "3.7], [9.8, 2.6, 3.4], [9.1, 2.4, 3.1], [8.4, 2.2, "
                           "2.8], [7.7, 2.0, 2.5], [7.0, 1.8, 2.2]]"));

  auto schema = table->GetSchema();
  auto batch = arrow::RecordBatch::Make(schema, 8, {ids, names, features});

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch};
  ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                       SerializeRecordBatches(schema, batches));
  sds serialized_batch_sds =
      sdsnewlen(serialized_batch->data(), serialized_batch->size());
  status = vdb::_BatchInsertCommand(table_name, serialized_batch_sds);
  ASSERT_TRUE(status.ok()) << status.ToString();
  sdsfree(serialized_batch_sds);

  {
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, "*", "id is not null", "0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 8);
    for (size_t j = 0; j < 8; j++) {
      ASSERT_OK_AND_ASSIGN(auto id_from_result,
                           GetArrayFromTable(table, "id", j, 1));
      auto id_array =
          std::static_pointer_cast<arrow::UInt32Array>(id_from_result);
      auto id = id_array->Value(0);
      ASSERT_OK_AND_ASSIGN(auto feature_from_result,
                           GetArrayFromTable(table, "feature", j, 1));
      ASSERT_OK_AND_ASSIGN(auto feature_from_input,
                           GetArrayFromRecordBatch(batch, 2, id, 1));
      ASSERT_TRUE(feature_from_result->Equals(feature_from_input))
          << j << "th id: " << id << "\n"
          << "feature_from_result: " << feature_from_result->ToString() << "\n"
          << "feature_from_input: " << feature_from_input->ToString();
    }
  }
}

TEST_P(WithIndexTest, AnnCommand) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
#ifdef _DEBUG_GTEST
  std::cout << "Added Metadata" << std::endl;
  std::cout << add_metadata->ToString() << std::endl;
#endif
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto schema = table->GetSchema();
  status = vdb::_InsertCommand(table_name,
                               "0\u001eJohn\u001e12.3\u001d-3.4\u001d5.2");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name,
                               "1\u001eJane\u001e11.2\u001d-3.0\u001d-4.0");
  ASSERT_TRUE(status.ok()) << status.ToString();

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "name, distance", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();
    ASSERT_EQ(table->num_columns(), 2);
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "distance, name, id", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();
    ASSERT_EQ(table->num_columns(), 3);
    ASSERT_EQ(table->column(0)->type(), arrow::float32());
    ASSERT_EQ(table->column(1)->type(), arrow::utf8());
    ASSERT_EQ(table->column(2)->type(), arrow::uint32());
    ASSERT_EQ(table->schema()->field(0)->name(), "distance");
    ASSERT_EQ(table->schema()->field(1)->name(), "name");
    ASSERT_EQ(table->schema()->field(2)->name(), "id");
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "name", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();
    ASSERT_EQ(table->num_columns(), 1);
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 1;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print 1 point near (12, 3, 5)\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{11.0, -3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 1;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print 1 point near (11, 3, 4)\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{11.0, -3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id < 1"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print points where id < 1\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{11.0, -3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id = 1"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print points where id = 1\n"
                                    << table->ToString();
    ASSERT_OK_AND_ASSIGN(auto id, table->column(0)->GetScalar(0));
    ASSERT_EQ(id->ToString(), "1");
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{11.0, 3.0, 4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 5;
    size_t ef_search = 5;

    // no results by filter
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id < 0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 0) << "print points where id < 0\n"
                                    << table->ToString();
    sdsfree(query);
  }
}

TEST_P(WithIndexTest, AnnCommandWithEmptyTable) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});

  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  {
    std::vector<float> query_vector{12.0, 3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 5;
    size_t ef_search = 5;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 0) << "print no points\n" << table->ToString();
    ASSERT_EQ(table->num_columns(), 4);

    // verify distance column exists and is float32
    auto distance_field = table->schema()->GetFieldByName("distance");
    ASSERT_NE(distance_field, nullptr);
    ASSERT_TRUE(distance_field->type()->Equals(arrow::float32()));

    sdsfree(query);
  }
}

TEST_P(WithIndexTest, AnnCommandWithSmallTable) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});

  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);
  auto schema = table->GetSchema();
  status = vdb::_InsertCommand(table_name,
                               "0\u001eJohn\u001e12.3\u001d3.4\u001d5.2");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name,
                               "1\u001eJane\u001e11.2\u001d3.0\u001d4.0");
  ASSERT_TRUE(status.ok()) << status.ToString();

  {
    std::vector<float> query_vector{12.0, 3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    // table has 2 rows, but top_k is bigger than 2
    size_t k = 5;
    size_t ef_search = 5;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();

    sdsfree(query);
  }
}

TEST_P(WithIndexTest, AnnCommandWithUpdateDelete) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
#ifdef _DEBUG_GTEST
  std::cout << "Added Metadata" << std::endl;
  std::cout << add_metadata->ToString() << std::endl;
#endif
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto schema = table->GetSchema();
  status = vdb::_InsertCommand(table_name,
                               "0\u001eJohn\u001e12.3\u001d3.4\u001d5.2");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name,
                               "1\u001eJane\u001e11.2\u001d3.0\u001d-4.0");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name,
                               "2\u001eMike\u001e10.1\u001d-2.0\u001d-3.6");
  ASSERT_TRUE(status.ok()) << status.ToString();

  {
    std::vector<float> query_vector{12.0, 3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "id", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0, 1}));
    ASSERT_OK(id_builder.Finish(&id_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false)}),
        {id_array});

    ASSERT_OK_AND_ASSIGN(auto sorted_result, SortTable(table));
    EXPECT_TRUE(sorted_result->Equals(*expected_table));
    sdsfree(query);
  }

  auto updated_status =
      vdb::_UpdateCommand(table_name, "feature = [-1.0, -2.0, -3.0]", "id = 1");
  if (!updated_status.ok()) {
    std::cerr << updated_status.status().ToString() << std::endl;
    ASSERT_TRUE(updated_status.ok());
  }
  auto updated_count = updated_status.ValueUnsafe();
  EXPECT_EQ(updated_count, 1);

  {
    std::vector<float> query_vector{12.0, 3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "id", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0, 2}));
    ASSERT_OK(id_builder.Finish(&id_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false)}),
        {id_array});

    ASSERT_OK_AND_ASSIGN(auto sorted_result, SortTable(table));
    EXPECT_TRUE(sorted_result->Equals(*expected_table));
    sdsfree(query);
  }

  auto deleted_status = vdb::_DeleteCommand(table_name, "id = 2");
  if (!deleted_status.ok()) {
    std::cerr << deleted_status.status().ToString() << std::endl;
    ASSERT_TRUE(deleted_status.ok());
  }
  auto deleted_count = deleted_status.ValueUnsafe();
  EXPECT_EQ(deleted_count, 1);

  {
    std::vector<float> query_vector{12.0, 3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "id", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();

    arrow::UInt32Builder id_builder;
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.AppendValues({0, 1}));
    ASSERT_OK(id_builder.Finish(&id_array));

    auto expected_table = arrow::Table::Make(
        arrow::schema({arrow::field("id", arrow::uint32(), false)}),
        {id_array});

    ASSERT_OK_AND_ASSIGN(auto sorted_result, SortTable(table));
    EXPECT_TRUE(sorted_result->Equals(*expected_table));
    sdsfree(query);
  }
}

TEST_P(WithIndexTest, AnnCommandAsync) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  server.allow_bg_index_thread = true;
  server.vdb_active_set_size_limit = 2;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string table_name = "test_table";
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::uint32(), false),
      arrow::field("name", arrow::utf8()),
      arrow::field("feature", arrow::fixed_size_list(arrow::float32(), 3)),
      arrow::field("segment_id", arrow::uint32(), false)};

  std::string segment_type = "value";
  std::string segment_keys = "segment_id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"segmentation_info", segmentation_info_str},
          {"table name", table_name},
          {"active_set_size_limit",
           std::to_string(server.vdb_active_set_size_limit)},
          {"index_info", index_info_str}});
  auto schema = std::make_shared<arrow::Schema>(schema_vector, metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());
  table_dictionary->insert({table_name, table});
  table = table_dictionary->at(table_name);

  auto id_arr = std::shared_ptr<arrow::Array>();
  auto name_arr = std::shared_ptr<arrow::Array>();
  auto feature_arr = std::shared_ptr<arrow::Array>();
  auto segment_id_arr = std::shared_ptr<arrow::Array>();

  arrow::UInt32Builder id_builder;
  arrow::StringBuilder name_builder;
  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder feature_builder(arrow::default_memory_pool(),
                                              value_builder, 3);
  arrow::UInt32Builder segment_id_builder;

  ARROW_EXPECT_OK(id_builder.Append(0));
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(id_builder.Append(2));

  ARROW_EXPECT_OK(name_builder.Append("John"));
  ARROW_EXPECT_OK(name_builder.Append("Jane"));
  ARROW_EXPECT_OK(name_builder.Append("Jack"));

  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(12.3));
  ARROW_EXPECT_OK(value_builder->Append(-3.4));
  ARROW_EXPECT_OK(value_builder->Append(5.2));

  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(-11.2));
  ARROW_EXPECT_OK(value_builder->Append(3.0));
  ARROW_EXPECT_OK(value_builder->Append(-4.0));

  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(-2.8));
  ARROW_EXPECT_OK(value_builder->Append(-1.0));
  ARROW_EXPECT_OK(value_builder->Append(0.5));

  ARROW_EXPECT_OK(segment_id_builder.Append(0));
  ARROW_EXPECT_OK(segment_id_builder.Append(0));
  ARROW_EXPECT_OK(segment_id_builder.Append(0));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(name_builder.Finish(&name_arr));
  ARROW_EXPECT_OK(feature_builder.Finish(&feature_arr));
  ARROW_EXPECT_OK(segment_id_builder.Finish(&segment_id_arr));

  auto rb = arrow::RecordBatch::Make(
      schema, 3, {id_arr, name_arr, feature_arr, segment_id_arr});
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  ASSERT_OK_AND_ASSIGN(auto serialized_rb, SerializeRecordBatches(schema, rbs));

  sds serialized_rb_sds =
      sdsnewlen(reinterpret_cast<const void *>(serialized_rb->data()),
                static_cast<size_t>(serialized_rb->size()));

  auto status = vdb::_BatchInsertCommand(table_name, serialized_rb_sds);

  sdsfree(serialized_rb_sds);

  while (table->IsIndexing()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 2;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 2) << "print all points\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{12.0, -3.0, 5.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 1;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print 1 point near (12, 3, 5)\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    std::vector<float> query_vector{-11.0, 3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 1;
    size_t ef_search = 2;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", ""));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print 1 point near (11, 3, 4)\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    // server.enable_in_filter = false; default is false
    std::vector<float> query_vector{-11.0, 3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 3;
    size_t ef_search = 3;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id = 0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 3) << "Filter is disabled, print all points\n"
                                    << table->ToString();
    sdsfree(query);
  }

  {
    server.enable_in_filter = true;
    std::vector<float> query_vector{-11.0, 3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 3;
    size_t ef_search = 3;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id = 0"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print points where id = 0\n"
                                    << table->ToString();
    ASSERT_OK_AND_ASSIGN(auto id, table->column(0)->GetScalar(0));
    ASSERT_EQ(id->ToString(), "0");
    sdsfree(query);
    server.enable_in_filter = false;
  }

  {
    server.enable_in_filter = true;
    std::vector<float> query_vector{-11.0, 3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 3;
    size_t ef_search = 3;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id > 1"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 1) << "print points where id > 1\n"
                                    << table->ToString();
    ASSERT_OK_AND_ASSIGN(auto id, table->column(0)->GetScalar(0));
    ASSERT_EQ(id->ToString(), "2");
    sdsfree(query);
    server.enable_in_filter = false;
  }

  {
    server.enable_in_filter = true;
    std::vector<float> query_vector{-11.0, 3.0, -4.0};
    sds query =
        sdsnewlen(query_vector.data(), query_vector.size() * sizeof(float));

    size_t k = 1;
    size_t ef_search = 3;

    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_AnnCommand(table_name, ann_column_id, k, query,
                                          ef_search, "*", "id > 2"));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    ASSERT_EQ(table->num_rows(), 0) << "print points where id > 1\n"
                                    << table->ToString();
    sdsfree(query);
    server.enable_in_filter = false;
  }

  table->StopIndexingThread();
  server.allow_bg_index_thread = false;
}

TEST_P(WithIndexTest, BatchAnnCommand) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  server.vdb_active_set_size_limit = 2;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  ASSERT_OK_AND_ASSIGN(
      auto ids, arrow::json::ArrayFromJSONString(arrow::int32(),
                                                 "[0, 1, 2, 3, 4, 5, 6, 7]"));
  ASSERT_OK_AND_ASSIGN(auto names,
                       arrow::json::ArrayFromJSONString(
                           arrow::utf8(),
                           "[\"John\", \"Jane\", \"Mike\", \"Sarah\", "
                           "\"David\", \"Emily\", \"Tom\", \"Julia\"]"));
  ASSERT_OK_AND_ASSIGN(auto features,
                       arrow::json::ArrayFromJSONString(
                           arrow::fixed_size_list(arrow::float32(), 3),
                           "[[12.3, 3.4, 5.2], [11.2, 3.0, 4.0], [10.5, 2.8, "
                           "3.7], [9.8, 2.6, 3.4], [9.1, 2.4, 3.1], [8.4, 2.2, "
                           "2.8], [7.7, 2.0, 2.5], [7.0, 1.8, 2.2]]"));

  auto schema = table->GetSchema();
  auto batch = arrow::RecordBatch::Make(schema, 8, {ids, names, features});

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch};
  ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                       SerializeRecordBatches(schema, batches));
  sds serialized_batch_sds =
      sdsnewlen(serialized_batch->data(), serialized_batch->size());
  status = vdb::_BatchInsertCommand(table_name, serialized_batch_sds);
  ASSERT_TRUE(status.ok()) << status.ToString();
  sdsfree(serialized_batch_sds);

  {
    std::shared_ptr<arrow::FloatBuilder> value_builder =
        std::make_shared<arrow::FloatBuilder>();
    arrow::FixedSizeListBuilder query_builder(arrow::default_memory_pool(),
                                              value_builder, 3);

    std::shared_ptr<arrow::Array> query_array;

    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(12.0));
    ASSERT_OK(value_builder->Append(3.0));
    ASSERT_OK(value_builder->Append(5.0));
    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(6.0));
    ASSERT_OK(value_builder->Append(2.0));
    ASSERT_OK(value_builder->Append(1.0));
    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(8.5));
    ASSERT_OK(value_builder->Append(2.5));
    ASSERT_OK(value_builder->Append(3.5));
    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(10.0));
    ASSERT_OK(value_builder->Append(3.2));
    ASSERT_OK(value_builder->Append(4.5));
    ASSERT_OK(query_builder.Finish(&query_array));

    auto schema = arrow::schema({arrow::field("query", query_array->type())});
    auto query_batch = arrow::RecordBatch::Make(schema, 4, {query_array});

    std::vector<std::shared_ptr<arrow::RecordBatch>> queries = {query_batch};
    ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                         SerializeRecordBatches(schema, queries));
    sds query_sds =
        sdsnewlen(serialized_batch->data(), serialized_batch->size());

    size_t k = 3;
    size_t ef_search = 6;

    ASSERT_OK_AND_ASSIGN(auto serialized_tables,
                         vdb::_BatchAnnCommand(table_name, ann_column_id, k,
                                               query_sds, ef_search, "*", ""));
    sdsfree(query_sds);

    ASSERT_EQ(serialized_tables.size(), 4);
    auto expected_ids = std::vector<std::vector<int32_t>>{
        {0, 1, 2}, {5, 7, 6}, {3, 4, 5}, {1, 2, 3}};
    for (size_t i = 0; i < serialized_tables.size(); i++) {
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_tables[i]));
      ASSERT_EQ(table->num_rows(), 3) << "print all points\n"
                                      << table->ToString();

      std::vector<int32_t> actual_ids;
      for (size_t j = 0; j < 3; j++) {
        ASSERT_OK_AND_ASSIGN(auto id_from_result,
                             GetArrayFromTable(table, "id", j, 1));
        auto id_array =
            std::static_pointer_cast<arrow::UInt32Array>(id_from_result);
        auto id = id_array->Value(0);
        actual_ids.push_back(id);
        ASSERT_OK_AND_ASSIGN(auto feature_from_result,
                             GetArrayFromTable(table, "feature", j, 1));
        ASSERT_OK_AND_ASSIGN(auto feature_from_input,
                             GetArrayFromRecordBatch(batch, 2, id, 1));
        ASSERT_TRUE(feature_from_result->Equals(feature_from_input))
            << "i: " << i << ", j: " << j << ", id: " << id << "\n"
            << "feature_from_result: " << feature_from_result->ToString()
            << "\n"
            << "feature_from_input: " << feature_from_input->ToString();
      }

      std::sort(actual_ids.begin(), actual_ids.end());
      std::sort(expected_ids[i].begin(), expected_ids[i].end());
      for (size_t j = 0; j < 3; j++) {
        ASSERT_EQ(actual_ids[j], expected_ids[i][j]);
      }
    }
  }
}

TEST_P(WithIndexTest, BatchAnnCommandWithNoThread) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  auto table_dictionary = vdb::GetTableDictionary();
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto table = table_dictionary->at(table_name);
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}, {"max_threads", "1"}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  ASSERT_OK_AND_ASSIGN(
      auto ids, arrow::json::ArrayFromJSONString(arrow::int32(),
                                                 "[0, 1, 2, 3, 4, 5, 6, 7]"));
  ASSERT_OK_AND_ASSIGN(auto names,
                       arrow::json::ArrayFromJSONString(
                           arrow::utf8(),
                           "[\"John\", \"Jane\", \"Mike\", \"Sarah\", "
                           "\"David\", \"Emily\", \"Tom\", \"Julia\"]"));
  ASSERT_OK_AND_ASSIGN(auto features,
                       arrow::json::ArrayFromJSONString(
                           arrow::fixed_size_list(arrow::float32(), 3),
                           "[[12.3, 3.4, 5.2], [11.2, 3.0, 4.0], [10.5, 2.8, "
                           "3.7], [9.8, 2.6, 3.4], [9.1, 2.4, 3.1], [8.4, 2.2, "
                           "2.8], [7.7, 2.0, 2.5], [7.0, 1.8, 2.2]]"));

  auto schema = table->GetSchema();
  auto batch = arrow::RecordBatch::Make(schema, 8, {ids, names, features});

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch};
  ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                       SerializeRecordBatches(schema, batches));
  sds serialized_batch_sds =
      sdsnewlen(serialized_batch->data(), serialized_batch->size());
  status = vdb::_BatchInsertCommand(table_name, serialized_batch_sds);
  ASSERT_TRUE(status.ok()) << status.ToString();
  sdsfree(serialized_batch_sds);

  {
    std::shared_ptr<arrow::FloatBuilder> value_builder =
        std::make_shared<arrow::FloatBuilder>();
    arrow::FixedSizeListBuilder query_builder(arrow::default_memory_pool(),
                                              value_builder, 3);

    std::shared_ptr<arrow::Array> query_array;

    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(12.0));
    ASSERT_OK(value_builder->Append(3.0));
    ASSERT_OK(value_builder->Append(5.0));
    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(6.0));
    ASSERT_OK(value_builder->Append(2.0));
    ASSERT_OK(value_builder->Append(1.0));
    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(8.5));
    ASSERT_OK(value_builder->Append(2.5));
    ASSERT_OK(value_builder->Append(3.5));
    ASSERT_OK(query_builder.Append());
    ASSERT_OK(value_builder->Append(10.0));
    ASSERT_OK(value_builder->Append(3.2));
    ASSERT_OK(value_builder->Append(4.5));
    ASSERT_OK(query_builder.Finish(&query_array));

    auto schema = arrow::schema({arrow::field("query", query_array->type())});
    auto query_batch = arrow::RecordBatch::Make(schema, 4, {query_array});

    std::vector<std::shared_ptr<arrow::RecordBatch>> queries = {query_batch};
    ASSERT_OK_AND_ASSIGN(auto serialized_batch,
                         SerializeRecordBatches(schema, queries));
    sds query_sds =
        sdsnewlen(serialized_batch->data(), serialized_batch->size());

    size_t k = 3;
    size_t ef_search = 6;

    ASSERT_OK_AND_ASSIGN(auto serialized_tables,
                         vdb::_BatchAnnCommand(table_name, ann_column_id, k,
                                               query_sds, ef_search, "*", ""));
    sdsfree(query_sds);

    ASSERT_EQ(serialized_tables.size(), 4);
    auto expected_ids = std::vector<std::vector<int32_t>>{
        {0, 1, 2}, {5, 7, 6}, {3, 4, 5}, {1, 2, 3}};
    for (size_t i = 0; i < serialized_tables.size(); i++) {
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_tables[i]));
      ASSERT_EQ(table->num_rows(), 3) << "print all points\n"
                                      << table->ToString();
      auto ids =
          table->GetColumnByName("id")->chunk(0)->data()->GetValues<int32_t>(1);
      std::vector<int32_t> actual_ids(ids, ids + 3);
      std::sort(actual_ids.begin(), actual_ids.end());
      std::sort(expected_ids[i].begin(), expected_ids[i].end());
      for (size_t j = 0; j < 3; j++) {
        ASSERT_EQ(actual_ids[j], expected_ids[i][j]);
      }
    }
  }
}

TEST_P(WithIndexTest, CountIndexedElementsCommand) {
  auto index_type = GetParam().index_type;
  auto space_type = GetParam().space_type;
  server.allow_bg_index_thread = true;
  const size_t kDataSize = 1000;
  const size_t kDim = 1024;
  const std::string table_name{"test_table"};

  auto table_dictionary = vdb::GetTableDictionary();

  // Create vdb::Table with Index
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  size_t ann_column_id = 1;
  size_t ef_construction = 1000;
  size_t M = 32;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_type, ef_construction, M);
  auto schema = arrow::schema(
      {{"id", arrow::int32()},
       {"feature", arrow::fixed_size_list(arrow::float32(), kDim)}},
      std::make_shared<arrow::KeyValueMetadata>(
          std::unordered_map<std::string, std::string>{
              {"segmentation_info", segmentation_info_str},
              {"table name", "test_table"},
              {"active_set_size_limit", "100"},
              {"index_info", index_info_str},
              {"max_threads", "2"}}));
  ASSERT_OK_AND_ASSIGN(
      auto new_schema,
      schema->SetField(0, schema->field(0)->WithNullable(false)));
  vdb::TableBuilderOptions tb_options;
  tb_options.SetTableName(table_name).SetSchema(new_schema);
  vdb::TableBuilder tb{std::move(tb_options)};
  auto maybe_table = tb.Build();
  ASSERT_TRUE(maybe_table.ok());
  auto table = maybe_table.ValueUnsafe();
  table_dictionary->insert({table_name, table});

  // Create arrow::RecordBatch with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0.0, 1.0);
  arrow::Int32Builder id_builder;
  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder feature_builder{arrow::default_memory_pool(),
                                              value_builder, kDim};
  for (size_t i = 0; i < kDataSize; i++) {
    ASSERT_TRUE(id_builder.Append(static_cast<int32_t>(i)).ok());
    ASSERT_TRUE(feature_builder.Append().ok());
    for (size_t d = 0; d < kDim; d++) {
      ASSERT_TRUE(value_builder->Append(dist(gen)).ok());
    }
  }
  std::shared_ptr<arrow::Int32Array> id_arr;
  std::shared_ptr<arrow::FixedSizeListArray> feature_arr;
  ASSERT_TRUE(id_builder.Finish(&id_arr).ok());
  ASSERT_TRUE(feature_builder.Finish(&feature_arr).ok());

  auto rb = arrow::RecordBatch::Make(schema, kDataSize, {id_arr, feature_arr});

  // Insert the record batch in to the table.
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  ASSERT_OK_AND_ASSIGN(auto serialized_rb, SerializeRecordBatches(schema, rbs));
  sds serialized_rb_sds =
      sdsnewlen(reinterpret_cast<const void *>(serialized_rb->data()),
                static_cast<size_t>(serialized_rb->size()));

  ASSERT_TRUE(_BatchInsertCommand(table_name, serialized_rb_sds).ok());
  sdsfree(serialized_rb_sds);

  // Test CountIndexedElementsCommand.
  ASSERT_OK_AND_ASSIGN(auto count_pair,
                       vdb::_CountIndexedElementsCommand(table_name));
  EXPECT_EQ(count_pair.first, kDataSize);
  std::vector<vdb::IndexedCountInfo> indexed_count_infos = count_pair.second;
  for (const auto &indexed_count_info : indexed_count_infos) {
    EXPECT_EQ(indexed_count_info.column_name, "feature");
    EXPECT_EQ(indexed_count_info.index_type, index_type);
    EXPECT_NE(indexed_count_info.indexed_count,
              kDataSize);  // bg threading should be running here.
  }
  while (table->IsIndexing()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_OK_AND_ASSIGN(count_pair,
                       vdb::_CountIndexedElementsCommand(table_name));
  EXPECT_EQ(count_pair.first, kDataSize);
  indexed_count_infos = count_pair.second;
  for (const auto &indexed_count_info : indexed_count_infos) {
    EXPECT_EQ(indexed_count_info.column_name, "feature");
    EXPECT_EQ(indexed_count_info.index_type, index_type);
    EXPECT_EQ(indexed_count_info.indexed_count, kDataSize);
  }
  server.allow_bg_index_thread = false;
}

TEST_F(SnapshotTest, UnloadAndLoadTableSnapshotTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string base_snapshot_path = TestDirectoryPath() + "/test_table.dump.vdb";

  auto unload_status = vdb::_UnloadCommand(table_name, base_snapshot_path);
  ASSERT_TRUE(unload_status.ok());

  /* Unloaded table should not be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it == vdb::GetTableDictionary()->end());

  /* Load table from unloaded snapshot */
  /* TODO: replace LoadVdbSnapshot with LoadTableSnapshot when it is implemented
   */
  auto load_status = vdb::_LoadCommand(table_name, base_snapshot_path.data());
  ASSERT_TRUE(load_status.ok());

  /* Loaded table should be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());
}

TEST_F(SnapshotTest, UnloadAndLoadVdbSnapshotTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string unload_snapshot_path =
      TestDirectoryPath() + "/test_table.unload.vdb";

  auto unload_status = vdb::_UnloadCommand(table_name, unload_snapshot_path);
  ASSERT_TRUE(unload_status.ok());

  std::string base_snapshot_path = TestDirectoryPath() + "/test_table.dump.vdb";

  bool ok = SaveVdbSnapshot(base_snapshot_path.data());
  ASSERT_TRUE(ok);

  /* Unloaded table should not be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it == vdb::GetTableDictionary()->end());

  /* Load table from dumped snapshot */
  auto load_status = LoadVdbSnapshot(base_snapshot_path.data());
  ASSERT_TRUE(load_status);

  /* Unloaded table should not be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it == vdb::GetTableDictionary()->end());
}

TEST_F(SnapshotTest, SaveAndLoadTableSnapshotTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string save_snapshot_path = TestDirectoryPath() + "/test_table.save.vdb";

  auto save_status = vdb::_SaveCommand(table_name, save_snapshot_path);
  ASSERT_TRUE(save_status.ok());

  /* Saved table should be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  DeallocateTableDictionary();
  AllocateTableDictionary();

  /* Load table from saved snapshot */
  auto load_status = vdb::_LoadCommand(table_name, save_snapshot_path.data());
  ASSERT_TRUE(load_status.ok());

  /* Loaded table should be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());
}

TEST_F(SnapshotTest, SaveAndLoadVdbSnapshotTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string save_snapshot_path = TestDirectoryPath() + "/test_table.save.vdb";

  auto save_status = vdb::_SaveCommand(table_name, save_snapshot_path);
  ASSERT_TRUE(save_status.ok());

  std::string base_snapshot_path = TestDirectoryPath() + "/test_table.dump.vdb";

  bool ok = SaveVdbSnapshot(base_snapshot_path.data());
  ASSERT_TRUE(ok);

  /* Saved table should be found in table dictionary */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  DeallocateTableDictionary();
  AllocateTableDictionary();

  /* Load table from dumped snapshot */
  auto load_status = LoadVdbSnapshot(base_snapshot_path.data());
  ASSERT_TRUE(load_status);

  /* Saved table should be included in vdb snapshot */
  it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());
}

TEST_F(SnapshotTest, UnloadToUnPermittedPathTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string unload_snapshot_path =
      TestDirectoryPath() + "/test_table.unload.vdb";

  // create directory with read-only permission
  std::filesystem::create_directory(unload_snapshot_path);
  std::filesystem::permissions(unload_snapshot_path,
                               std::filesystem::perms::owner_read);

  // if the directory is created with write permission or user has root
  // permission, it can be written to
  auto perms = std::filesystem::status(unload_snapshot_path).permissions();
  bool is_read_only = (perms & std::filesystem::perms::owner_write) ==
                      std::filesystem::perms::none;

  bool is_root = (geteuid() == 0);

  if (is_read_only && !is_root) {
    auto unload_status = vdb::_UnloadCommand(table_name, unload_snapshot_path);
    ASSERT_TRUE(!unload_status.ok());

    std::filesystem::remove_all(unload_snapshot_path);

    // make file instead of directory
    std::filesystem::path unload_snapshot_path_file(unload_snapshot_path);
    std::ofstream ofs(unload_snapshot_path_file);
    ofs.close();

    auto unload_status2 = vdb::_UnloadCommand(table_name, unload_snapshot_path);
    ASSERT_TRUE(!unload_status2.ok());
  }
}

TEST_F(SnapshotTest, SaveToUnPermittedPathTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string save_snapshot_path = TestDirectoryPath() + "/test_table.save.vdb";

  // create directory with read-only permission
  std::filesystem::create_directory(save_snapshot_path);
  std::filesystem::permissions(save_snapshot_path,
                               std::filesystem::perms::owner_read);

  // if the directory is created with write permission or user has root
  // permission, it can be written to
  auto perms = std::filesystem::status(save_snapshot_path).permissions();
  bool is_read_only = (perms & std::filesystem::perms::owner_write) ==
                      std::filesystem::perms::none;

  bool is_root = (geteuid() == 0);

  if (is_read_only && !is_root) {
    auto save_status = vdb::_SaveCommand(table_name, save_snapshot_path);
    ASSERT_TRUE(!save_status.ok());

    std::filesystem::remove_all(save_snapshot_path);

    // make file instead of directory
    std::filesystem::path save_snapshot_path_file(save_snapshot_path);
    std::ofstream ofs(save_snapshot_path_file);
    ofs.close();

    auto save_status2 = vdb::_SaveCommand(table_name, save_snapshot_path);
    ASSERT_TRUE(!save_status2.ok());
  }
}

TEST_F(SnapshotTest, SaveAndUnloadToEmptyPathTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string save_data_path = TestDirectoryPath() + "/save_data";
  std::string unload_data_path = TestDirectoryPath() + "/unload_data";

  server.save_data_path = save_data_path.data();
  server.unload_data_path = unload_data_path.data();

  std::string empty_path = "";
  std::string actual_save_path = server.save_data_path;
  std::string actual_unload_path = server.unload_data_path;

  auto save_status = vdb::_SaveCommand(table_name, empty_path);
  ASSERT_TRUE(save_status.ok());

  auto unload_status = vdb::_UnloadCommand(table_name, empty_path);
  ASSERT_TRUE(unload_status.ok());
}

TEST_F(SnapshotTest, SaveLatestTableSnapshotTest) {
  // Create table
  std::string table_name = "test_table";
  std::string schema_string =
      "ID uint32 not null, Name String, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok());

  auto it = vdb::GetTableDictionary()->find(table_name);
  ASSERT_TRUE(it != vdb::GetTableDictionary()->end());

  std::string save_snapshot_path = TestDirectoryPath() + "/test_table.save.vdb";

  auto save_status = vdb::_SaveCommand(table_name, save_snapshot_path);
  ASSERT_TRUE(save_status.ok());

  const char *data_sds = "0\u001eJohn\u001e12.3\u001d3.4\u001d5.2";
  status = vdb::_InsertCommand(table_name, std::string_view(data_sds));
  ASSERT_TRUE(status.ok());

  auto save_status2 = vdb::_SaveCommand(table_name, save_snapshot_path);
  ASSERT_TRUE(save_status2.ok());

  auto drop_status = vdb::_DropTableCommand(table_name);
  ASSERT_TRUE(drop_status.ok());

  /* Load table from saved snapshot */
  auto load_status = vdb::_LoadCommand(table_name, save_snapshot_path.data());
  ASSERT_TRUE(load_status.ok());

  ASSERT_OK_AND_ASSIGN(auto serialized_table,
                       vdb::_ScanCommand(table_name, "*", "", "0"));
  ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
  ASSERT_NE(table->num_rows(), 0);
}

TEST_F(SnapshotTest, UniqueViolationCheckAfterSaveAndLoad) {
  auto table_dictionary = vdb::GetTableDictionary();

  std::string schema_string =
      "ID uint32 not null, pk Large_String, float_col Float32, Feature "
      "Fixed_Size_List[ 3 ,   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(schema_string);

  auto fields = schema->fields();
  auto pk_field = fields[1];
  auto pk_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{{"primary_key", "true"}});
  fields[1] = pk_field->WithMetadata(pk_metadata);

  schema = std::make_shared<arrow::Schema>(fields, schema->metadata());

  server.vdb_active_set_size_limit = 100;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, "Hnsw", "L2Space", ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);

  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs;
  std::vector<std::shared_ptr<arrow::Buffer>> serialized_rbs_after_save;
  for (int i = 0; i < 100; i++) {
    ASSERT_OK_AND_ASSIGN(
        auto rb,
        GenerateRecordBatchWithPrimaryKey(
            schema, "test" + std::to_string(i) + ".txt", 0, 100, 3, true));
    ASSERT_OK_AND_ASSIGN(
        auto rb_after_save,
        GenerateRecordBatchWithPrimaryKey(
            schema, "test" + std::to_string(i) + ".txt", 50, 100, 3, true));
#ifdef _DEBUG_GTEST
    std::cout << rb->ToString() << std::endl;
#endif
    auto options = arrow::ipc::IpcWriteOptions::Defaults();
    options.allow_64bit = true;
    options.memory_pool = &vdb::arrow_pool;
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs_after_save = {
        rb_after_save};
    ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                         SerializeRecordBatches(schema, rbs));
    ASSERT_OK_AND_ASSIGN(auto serialized_rb_after_save,
                         SerializeRecordBatches(schema, rbs_after_save));
    serialized_rbs.push_back(serialized_rb);
    serialized_rbs_after_save.push_back(serialized_rb_after_save);
  }

  /* target table */
  std::string target_table_name = "target_table";
  auto status = CreateTableForTest(target_table_name, schema_string, true);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto target_table = table_dictionary->at(target_table_name);
  TableWrapper::AddMetadata(target_table, add_metadata);
  TableWrapper::AddMetadataToField(target_table, "pk", pk_metadata);
  TableWrapper::AddEmbeddingStore(target_table, ann_column_id);

  for (auto &rb : serialized_rbs) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  std::string save_snapshot_path = TestDirectoryPath() + "/test_table.save.vdb";

  auto save_status = vdb::_SaveCommand(target_table_name, save_snapshot_path);
  ASSERT_TRUE(save_status.ok());

  auto drop_status = vdb::_DropTableCommand(target_table_name);
  ASSERT_TRUE(drop_status.ok());

  /* Load table from saved snapshot */
  auto load_status =
      vdb::_LoadCommand(target_table_name, save_snapshot_path.data());
  ASSERT_TRUE(load_status.ok());

  auto loaded_table = table_dictionary->at(target_table_name);
  EXPECT_EQ(target_table->GetSegment(kDefaultSegmentId)
                ->GetPrimaryKeyIndex()
                ->ToString(),
            loaded_table->GetSegment(kDefaultSegmentId)
                ->GetPrimaryKeyIndex()
                ->ToString());
  for (auto &rb : serialized_rbs_after_save) {
    sds rb_sds = sdsnewlen(reinterpret_cast<const void *>(rb->data()),
                           static_cast<size_t>(rb->size()));
    status = vdb::_BatchInsertCommand(target_table_name, rb_sds);
    ASSERT_TRUE(status.ok()) << status.ToString();
    sdsfree(rb_sds);
  }

  auto target_segments = loaded_table->GetSegments();

  size_t cnt = 0;
  for (auto &kv : target_segments) {
    auto segment = kv.second;
    auto segment_number = segment->GetSegmentNumber();
    for (uint32_t set_id = 0; set_id <= segment->InactiveSets().size();
         set_id++) {
      std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
      std::shared_ptr<arrow::RecordBatch> rb;
      if (set_id == segment->ActiveSetId()) {
        ASSERT_OK_AND_ASSIGN(rb, segment->GetRecordbatch(set_id));
      } else {
        inactive_set = segment->GetInactiveSet(set_id);
        rb = inactive_set->GetRb();
        ASSERT_NE(rb, nullptr);
      }

      bool ann_column_exists = false;
      bool hidden_column_exists = false;
      for (int64_t column_id = 0; column_id < rb->num_columns(); column_id++) {
        auto &column = rb->columns()[column_id];
        auto &field = rb->schema()->field(column_id);
        if (field->name() == vdb::kDeletedFlagColumn) {
          EXPECT_EQ(field->type()->id(), arrow::Type::BOOL);
          hidden_column_exists = true;
        }
        if (field->name() == "feature") {
          EXPECT_EQ(field->type()->id(), arrow::Type::UINT64);
          auto rowid_array =
              std::static_pointer_cast<arrow::UInt64Array>(column);
          for (int64_t i = 0; i < rb->num_rows(); i++) {
            uint64_t rowid = LabelInfo::Build(segment_number, set_id, i);
            auto value = rowid_array->Value(i);
#ifdef _DEBUG_GTEST
            std::cout << "rowid: " << rowid << ", value: " << value
                      << std::endl;
#endif
            EXPECT_EQ(value, rowid);
          }
          ann_column_exists = true;
        }
      }
      EXPECT_TRUE(ann_column_exists);
      EXPECT_TRUE(hidden_column_exists);
    }
    cnt += segment->Size();
  }
  EXPECT_EQ(cnt, 15000);

#ifdef _DEBUG_GTEST
  std::cout << target_table->ToString() << std::endl;
#endif
}

TEST_F(MemoryLimitTest, CreateLargeSegmentFails) {
  server.maxmemory = vdb::initial_vdb_allocated_size + 100000;
  uint32_t dim = 10;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, "Hnsw", "L2Space", ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "100000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  bool throw_catched = false;
  try {
    ASSERT_OK_AND_ASSIGN(auto table, builder.Build());
    std::string dummy_id = "";
    auto org_segment = std::make_shared<vdb::Segment>(table, dummy_id, 0);
    ASSERT_TRUE(false);
  } catch (std::runtime_error &e) {
    throw_catched = true;
  }
  ASSERT_TRUE(throw_catched);
}

TEST_F(MemoryLimitTest, CreateSegmentAndIndexFails) {
  server.maxmemory = vdb::initial_vdb_allocated_size + 90000;
  uint32_t dim = 10;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, "Hnsw", "L2Space", ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "1000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  bool throw_catched = false;
  try {
    ASSERT_OK_AND_ASSIGN(auto table, builder.Build());
    std::string dummy_id = "";
    auto org_segment = std::make_shared<vdb::Segment>(table, dummy_id, 0);
    ASSERT_TRUE(false);
  } catch (std::runtime_error &e) {
    throw_catched = true;
  }
  ASSERT_TRUE(throw_catched);
}

TEST_F(MemoryLimitTest, SingleInsertFails) {
  server.maxmemory = vdb::initial_vdb_allocated_size + 1000;
  auto table_dictionary = vdb::GetTableDictionary();

  std::string test_table_name = "test_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->begin()->second;
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
  const char *data_sds = "0\u001eJohn\u001eC\u001dPython\u001dJava";
  bool failed = false;
  for (int i = 0; i < 1024; i++) {
    try {
      status = vdb::_InsertCommand(test_table_name, std::string_view(data_sds));
    } catch (std::runtime_error &e) {
      status = vdb::Status::OutOfMemory(e.what());
    }
    if (status.IsOutOfMemory()) {
      failed = true;
      break;
    }
  }
  ASSERT_TRUE(failed);
}

TEST_F(MemoryLimitTest, BatchInsertFails) {
  server.maxmemory = vdb::initial_vdb_allocated_size + 100000;
  uint32_t dim = 10;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, "Hnsw", "L2Space", ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "1000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());
  std::string dummy_id = "";
  auto org_segment = std::make_shared<vdb::Segment>(table, dummy_id, 0);

  auto maybe_rb = GenerateRecordBatch(schema, 5000, dim, false);
  ASSERT_TRUE(maybe_rb.ok());
  auto rb = maybe_rb.ValueUnsafe();
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  auto status = org_segment->AppendRecords(rbs);
  ASSERT_TRUE(status.IsOutOfMemory());
}

// Test successful cases of internal column visibility
TEST_F(NoIndexTest, ScanInternalColumnProjectionSuccess) {
  std::string table_name = "test_internal_visibility";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert test data
  status = vdb::_InsertCommand(table_name, "0\u001eAlice\u001e100");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eBob\u001e200");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "2\u001eCharlie\u001e300");
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scan with "*" should NOT show internal columns
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scan should NOT include internal column: "
        << internal_column_name;

    // Should have exactly 3 user-defined columns (id, name, value)
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scan schema (no internal columns):"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scan with "*" SHOULD show internal columns
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column: "
        << internal_column_name;

    // Should have 4 columns: id, name, value, __deleted_flag
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 2 - Internal scan schema (with internal columns):"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Internal scan with ONLY internal column (no visible columns)
  {
    std::string projection = internal_column_name;
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have exactly 1 column: __deleted_flag only
    EXPECT_EQ(schema->num_fields(), 1);
    EXPECT_EQ(schema->field(0)->name(), internal_column_name);
    EXPECT_EQ(table->num_rows(), 3);

    // Verify all values are false (no deleted records)
    auto deleted_flag_column = table->column(0);
    ASSERT_GT(deleted_flag_column->num_chunks(), 0);
    for (int chunk_idx = 0; chunk_idx < deleted_flag_column->num_chunks();
         ++chunk_idx) {
      auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
          deleted_flag_column->chunk(chunk_idx));
      for (int64_t i = 0; i < chunk->length(); i++) {
        EXPECT_FALSE(chunk->Value(i))
            << "All __deleted_flag values should be false (no deletes yet)";
      }
    }

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Internal column ONLY projection:" << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scan with specific internal column + visible column
  {
    std::string projection = "id, " + internal_column_name;
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have exactly 2 columns: id and __deleted_flag
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), internal_column_name);
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 4 - Specific internal column + visible column:"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Regular scan with specific columns (no internal column
  // requested)
  {
    std::string projection = "id, name";
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column) << "Regular scan with specific columns "
                                         "should NOT include internal column";

    // Should have exactly 2 columns: id, name
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Regular scan with specific columns:"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 6: Internal scan with specific regular columns only
  // (internal column NOT explicitly requested)
  // include_internal_columns=true but only visible columns requested
  // Result: Should NOT include internal columns
  {
    std::string projection = "id, name";
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column is NOT present (not explicitly requested)
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Internal scan flag only ALLOWS internal columns, doesn't "
           "auto-add them";

    // Should have exactly 2 columns: id, name (no automatic internal column)
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout
        << "Scenario 6 - Internal scan with specific regular columns only:"
        << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 7: Internal scan with mix of regular and internal columns
  {
    std::string projection = "id, " + internal_column_name + ", name";
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have 3 columns in the requested order: id, __deleted_flag, name
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), internal_column_name);
    EXPECT_EQ(schema->field(2)->name(), "name");
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 7 - Internal scan with mixed columns:" << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }
}

// Test failure cases of internal column projection
TEST_F(NoIndexTest, ScanInternalColumnProjectionFailure) {
  std::string table_name = "test_internal_visibility_fail";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert test data
  status = vdb::_InsertCommand(table_name, "0\u001eAlice\u001e100");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eBob\u001e200");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "2\u001eCharlie\u001e300");
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scan requesting ONLY internal column should fail
  {
    std::string projection = internal_column_name;
    auto result = vdb::_ScanCommand(table_name, projection, "", "0", false);
    EXPECT_FALSE(result.ok())
        << "Regular scan requesting only internal column should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column_name) != std::string::npos)
          << "Error message should mention the internal column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1 - Regular scan with only internal column "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 2: Regular scan with mixed columns (visible + internal) should
  // fail
  {
    std::string projection = "id, " + internal_column_name;
    auto result = vdb::_ScanCommand(table_name, projection, "", "0", false);
    EXPECT_FALSE(result.ok())
        << "Regular scan with mixed columns (visible + internal) should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column_name) != std::string::npos)
          << "Error message should mention the internal column name";

#ifdef _DEBUG_GTEST
      std::cout
          << "Scenario 2 - Regular scan with mixed columns correctly fails:"
          << std::endl;
      std::cout << "  Requested: id, " << internal_column_name << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 3: Empty projection should fail
  {
    auto result = vdb::_ScanCommand(table_name, "", "", "0", false);
    EXPECT_FALSE(result.ok()) << "Empty projection should fail with an error";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Empty projection list") != std::string::npos)
          << "Error message should contain 'Empty projection list', got: "
          << error_msg;

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 3 - Empty projection correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 4: Empty projection with internal scan should also fail
  {
    auto result = vdb::_ScanCommand(table_name, "", "", "0", true);
    EXPECT_FALSE(result.ok()) << "Empty projection should fail even with "
                                 "include_internal_columns=true";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Empty projection list") != std::string::npos)
          << "Error message should contain 'Empty projection list', got: "
          << error_msg;

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 4 - Empty projection with internal scan "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 5: Regular scan with non-existent column should fail
  {
    std::string projection = "id, non_existent_column";
    auto result = vdb::_ScanCommand(table_name, projection, "", "0", false);
    EXPECT_FALSE(result.ok())
        << "Regular scan with non-existent column should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find("non_existent_column") != std::string::npos)
          << "Error message should mention the non-existent column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 5 - Regular scan with non-existent column "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 6: Internal scan with non-existent column should also fail
  {
    std::string projection = "id, non_existent_column";
    auto result = vdb::_ScanCommand(table_name, projection, "", "0", true);
    EXPECT_FALSE(result.ok())
        << "Internal scan with non-existent column should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find("non_existent_column") != std::string::npos)
          << "Error message should mention the non-existent column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 6 - Internal scan with non-existent column "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }
}

// Test internal column projection with delete operations
TEST_F(NoIndexTest, ScanInternalColumnProjectionWithDelete) {
  std::string table_name = "test_delete_visibility";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert test data
  status = vdb::_InsertCommand(table_name, "0\u001eAlice\u001e100");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eBob\u001e200");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "2\u001eCharlie\u001e300");
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1-1: Delete a record and verify behavior difference between
  // regular scan and internal scan
  {
    // Delete the record with id=1 (Bob)
    std::string filter = "id = 1";
    ASSERT_OK_AND_ASSIGN(auto deleted_count,
                         vdb::_DeleteCommand(table_name, filter));
    EXPECT_EQ(deleted_count, 1) << "Should delete exactly 1 record";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1-1 - Deleted record with id=1" << std::endl;
#endif

    // 1-1a: Regular scan should NOT see the deleted record
    {
      ASSERT_OK_AND_ASSIGN(
          auto serialized_table,
          vdb::_ScanCommand(table_name, "id, name", "", "0", false));
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));

      // Should have 2 records (id=0 and id=2, excluding deleted id=1)
      EXPECT_EQ(table->num_rows(), 2);

      // Verify that id=1 is NOT present
      if (table->num_rows() > 0) {
        auto id_col = table->column(0);
        ASSERT_GT(id_col->num_chunks(), 0)
            << "Column should have at least one chunk";

        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks(); ++chunk_idx) {
          auto chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));
          for (int64_t i = 0; i < chunk->length(); i++) {
            auto id_value = chunk->Value(i);
            EXPECT_NE(id_value, 1)
                << "Regular scan should not see deleted record (id=1)";
          }
        }
      }

#ifdef _DEBUG_GTEST
      std::cout << "  Regular scan: " << table->num_rows()
                << " rows (deleted record hidden)" << std::endl;
#endif
    }

    // 1-1b: Internal scan SHOULD see the deleted record with __deleted_flag =
    // true
    {
      std::string projection = "id, name, " + internal_column_name;
      ASSERT_OK_AND_ASSIGN(
          auto serialized_table,
          vdb::_ScanCommand(table_name, projection, "", "0", true));
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));

      // Should have all 3 records (including the deleted one)
      EXPECT_EQ(table->num_rows(), 3);

      // Find the deleted record (id=1) and verify __deleted_flag = true
      if (table->num_rows() > 0) {
        auto id_col = table->column(0);
        auto deleted_flag_col = table->column(2);
        ASSERT_GT(id_col->num_chunks(), 0)
            << "ID column should have at least one chunk";
        ASSERT_GT(deleted_flag_col->num_chunks(), 0)
            << "Deleted flag column should have at least one chunk";

        bool found_deleted_record = false;

        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks(); ++chunk_idx) {
          auto id_chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));
          auto flag_chunk = std::static_pointer_cast<arrow::BooleanArray>(
              deleted_flag_col->chunk(chunk_idx));

          for (int64_t i = 0; i < id_chunk->length(); i++) {
            auto id_value = id_chunk->Value(i);
            auto deleted_flag = flag_chunk->Value(i);

            if (id_value == 1) {
              found_deleted_record = true;
              EXPECT_TRUE(deleted_flag)
                  << "Deleted record (id=1) should have __deleted_flag = true";
            } else {
              EXPECT_FALSE(deleted_flag)
                  << "Active record (id=" << id_value
                  << ") should have __deleted_flag = false";
            }
          }
        }

        EXPECT_TRUE(found_deleted_record)
            << "Internal scan should see deleted record (id=1)";
      }

#ifdef _DEBUG_GTEST
      std::cout << "  Internal scan: " << table->num_rows()
                << " rows (deleted record visible with flag=true)" << std::endl;
#endif
    }
  }

  // Scenario 1-2: Internal scan without filter should show ALL records
  // (active + deleted)
  {
    std::string projection = "id, " + internal_column_name;

    // Internal scan with NO filter
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    // Should have all 3 records (2 active + 1 deleted from previous scenario)
    EXPECT_EQ(table->num_rows(), 3);

    // Count active and deleted records
    int active_count = 0;
    int deleted_count = 0;

    if (table->num_rows() > 0) {
      auto deleted_flag_col = table->column(1);
      ASSERT_GT(deleted_flag_col->num_chunks(), 0)
          << "Column should have at least one chunk";

      for (int chunk_idx = 0; chunk_idx < deleted_flag_col->num_chunks();
           ++chunk_idx) {
        auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
            deleted_flag_col->chunk(chunk_idx));
        for (int64_t i = 0; i < chunk->length(); i++) {
          auto deleted_flag = chunk->Value(i);
          if (deleted_flag) {
            deleted_count++;
          } else {
            active_count++;
          }
        }
      }
    }

    EXPECT_EQ(active_count, 2) << "Should have 2 active records";
    EXPECT_EQ(deleted_count, 1) << "Should have 1 deleted record";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1-2 - Internal scan without filter:" << std::endl;
    std::cout << "  Total records: " << table->num_rows() << std::endl;
    std::cout << "  - Active: " << active_count << std::endl;
    std::cout << "  - Deleted: " << deleted_count << std::endl;
#endif

    // Compare with regular scan (should only show active records)
    ASSERT_OK_AND_ASSIGN(auto serialized_regular,
                         vdb::_ScanCommand(table_name, "id", "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto regular_table,
                         DeserializeToTableFrom(serialized_regular));

    EXPECT_EQ(regular_table->num_rows(), 2)
        << "Regular scan should only show active records";

#ifdef _DEBUG_GTEST
    std::cout << "  Regular scan (for comparison): "
              << regular_table->num_rows() << " rows (active only)"
              << std::endl;
#endif
  }

  // Scenario 1-3: Filter by __deleted_flag to query specific records
  {
    std::string projection = "id, name, " + internal_column_name;

    // 1-3a: Query ONLY deleted records (__deleted_flag = true)
    {
      std::string filter = internal_column_name + " = true";
      ASSERT_OK_AND_ASSIGN(
          auto serialized_table,
          vdb::_ScanCommand(table_name, projection, filter, "0", true));
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));

      // Should have 1 deleted record (id=1)
      EXPECT_EQ(table->num_rows(), 1);

      // Verify it's the correct deleted record
      if (table->num_rows() > 0) {
        auto id_col = table->column(0);
        auto deleted_flag_col = table->column(2);
        ASSERT_GT(id_col->num_chunks(), 0)
            << "ID column should have at least one chunk";
        ASSERT_GT(deleted_flag_col->num_chunks(), 0)
            << "Flag column should have at least one chunk";

        // Find the row across all chunks (data might not be in first chunk)
        bool found_row = false;
        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks() && !found_row;
             ++chunk_idx) {
          auto id_chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));
          auto flag_chunk = std::static_pointer_cast<arrow::BooleanArray>(
              deleted_flag_col->chunk(chunk_idx));

          if (id_chunk->length() > 0) {
            auto id_value = id_chunk->Value(0);
            auto deleted_flag = flag_chunk->Value(0);

            EXPECT_EQ(id_value, 1) << "Should be the deleted record (id=1)";
            EXPECT_TRUE(deleted_flag)
                << "Filtered record should have flag=true";
            found_row = true;
          }
        }
        EXPECT_TRUE(found_row) << "Should find exactly 1 row";
      }

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1-3a - Filter deleted only: " << table->num_rows()
                << " rows (id=1)" << std::endl;
#endif
    }

    // 1-3b: Query ONLY active records (__deleted_flag = false)
    {
      std::string filter = internal_column_name + " = false";
      ASSERT_OK_AND_ASSIGN(
          auto serialized_table,
          vdb::_ScanCommand(table_name, projection, filter, "0", true));
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));

      // Should have 2 active records (id=0, id=2)
      EXPECT_EQ(table->num_rows(), 2);

      // Verify all have __deleted_flag = false
      if (table->num_rows() > 0) {
        auto deleted_flag_col = table->column(2);
        ASSERT_GT(deleted_flag_col->num_chunks(), 0)
            << "Column should have at least one chunk";

        int64_t row_offset = 0;
        for (int chunk_idx = 0; chunk_idx < deleted_flag_col->num_chunks();
             ++chunk_idx) {
          auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
              deleted_flag_col->chunk(chunk_idx));
          for (int64_t i = 0; i < chunk->length(); i++) {
            auto deleted_flag = chunk->Value(i);
            EXPECT_FALSE(deleted_flag)
                << "Filtered record at row " << (row_offset + i)
                << " should have flag=false";
          }
          row_offset += chunk->length();
        }
      }

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1-3b - Filter active only: " << table->num_rows()
                << " rows (id=0,2)" << std::endl;
#endif
    }

    // 1-3c: Complex filter with __deleted_flag
    {
      std::string filter = internal_column_name + " = true AND id = 1";
      ASSERT_OK_AND_ASSIGN(
          auto serialized_table,
          vdb::_ScanCommand(table_name, projection, filter, "0", true));
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));

      // Should have exactly 1 record (deleted AND id=1)
      EXPECT_EQ(table->num_rows(), 1);

      if (table->num_rows() > 0) {
        auto id_col = table->column(0);
        ASSERT_GT(id_col->num_chunks(), 0)
            << "ID column should have at least one chunk";

        // Find the row across all chunks (data might not be in first chunk)
        bool found_row = false;
        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks() && !found_row;
             ++chunk_idx) {
          auto id_chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));

          if (id_chunk->length() > 0) {
            auto id_value = id_chunk->Value(0);
            EXPECT_EQ(id_value, 1);
            found_row = true;
          }
        }
        EXPECT_TRUE(found_row) << "Should find exactly 1 row";
      }

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1-3c - Complex filter (deleted AND id=1): "
                << table->num_rows() << " rows" << std::endl;
#endif
    }
  }
}

// Test successful cases of scanopen internal column projection
TEST_F(NoIndexTest, ScanOpenInternalColumnProjectionSuccess) {
  std::string table_name = "test_scanopen_internal";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert multiple records to simulate large dataset
  for (int i = 0; i < 10; i++) {
    std::string data = std::to_string(i) + "\u001eUser" + std::to_string(i) +
                       "\u001e" + std::to_string(i * 100);
    status = vdb::_InsertCommand(table_name, data);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scanopen with "*" should NOT show internal columns
  {
    std::string uuid = "scan_regular_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", "", "0", false));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scanopen should NOT include internal column";

    // Should have exactly 3 user-defined columns
    EXPECT_EQ(schema->num_fields(), 3);

    // Fetch remaining batches and verify they also don't have internal columns
    int total_rows = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);

      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 3);
      total_rows += table->num_rows();
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scanopen schema (no internal columns):"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scanopen with "*" SHOULD show internal columns
  {
    std::string uuid = "scan_internal_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scanopen should include internal column: "
        << internal_column_name;

    // Should have 4 columns: id, name, value, __deleted_flag
    EXPECT_EQ(schema->num_fields(), 4);

    // Fetch remaining batches and verify they also have internal columns
    int total_rows = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);

      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 4);
      total_rows += table->num_rows();

      // Verify internal column still present in subsequent batches
      bool batch_has_internal = false;
      for (int i = 0; i < table->schema()->num_fields(); i++) {
        if (table->schema()->field(i)->name() == internal_column_name) {
          batch_has_internal = true;
          break;
        }
      }
      EXPECT_TRUE(batch_has_internal)
          << "All batches should include internal column";
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout
        << "Scenario 2 - Internal scanopen schema (with internal columns):"
        << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Internal scanopen with ONLY internal column (no visible
  // columns)
  {
    std::string uuid = "scan_internal_only_" + table_name;
    std::string projection = internal_column_name;

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have exactly 1 column: __deleted_flag only
    EXPECT_EQ(schema->num_fields(), 1);
    EXPECT_EQ(schema->field(0)->name(), internal_column_name);

    // Verify all values are false (no deleted records) in all batches
    int total_rows = 0;
    while (true) {
      auto deleted_flag_column = table->column(0);
      ASSERT_GT(deleted_flag_column->num_chunks(), 0);
      for (int chunk_idx = 0; chunk_idx < deleted_flag_column->num_chunks();
           ++chunk_idx) {
        auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
            deleted_flag_column->chunk(chunk_idx));
        for (int64_t i = 0; i < chunk->length(); i++) {
          EXPECT_FALSE(chunk->Value(i))
              << "All __deleted_flag values should be false (no deletes yet)";
          total_rows++;
        }
      }

      if (!has_next) break;

      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);
      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 1);
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Internal column ONLY projection:" << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scanopen with specific internal column + visible
  // column
  {
    std::string uuid = "scan_specific_" + table_name;
    std::string projection = "id, " + internal_column_name;

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have exactly 2 columns: id and __deleted_flag
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), internal_column_name);

    // Count total rows across all batches
    int total_rows = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);
      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 2);
      total_rows += table->num_rows();
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 4 - Specific internal column + visible column:"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Regular scanopen with specific columns (no internal column)
  {
    std::string uuid = "scan_regular_specific_" + table_name;
    std::string projection = "id, name";

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", false));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scanopen with specific columns should NOT include internal "
           "column";

    // Should have exactly 2 columns: id, name
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");

    // Verify all batches
    int total_rows = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);

      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 2);
      total_rows += table->num_rows();
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Regular scanopen with specific columns:"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 6: Internal scanopen with specific regular columns only
  // (internal column NOT explicitly requested)
  // include_internal_columns=true but only visible columns requested
  // Result: Should NOT include internal columns
  {
    std::string uuid = "scan_internal_regular_" + table_name;
    std::string projection = "id, name";

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column is NOT present (not explicitly requested)
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Internal scan flag only ALLOWS internal columns, doesn't "
           "auto-add them";

    // Should have exactly 2 columns: id, name (no automatic internal column)
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");

    // Verify all batches
    int total_rows = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);

      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 2);
      total_rows += table->num_rows();

      // Verify internal column still not present in subsequent batches
      bool batch_has_internal = false;
      for (int i = 0; i < table->schema()->num_fields(); i++) {
        if (table->schema()->field(i)->name() == internal_column_name) {
          batch_has_internal = true;
          break;
        }
      }
      EXPECT_FALSE(batch_has_internal)
          << "Internal scan flag only ALLOWS internal columns";
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 6 - Internal scanopen with specific regular "
                 "columns only:"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }

  // Scenario 7: Internal scanopen with mix of regular and internal columns
  {
    std::string uuid = "scan_mixed_" + table_name;
    std::string projection = "id, " + internal_column_name + ", name";

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have 3 columns in the requested order: id, __deleted_flag, name
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), internal_column_name);
    EXPECT_EQ(schema->field(2)->name(), "name");

    // Verify all batches
    int total_rows = table->num_rows();
    while (has_next) {
      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);

      ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
      EXPECT_EQ(table->schema()->num_fields(), 3);
      EXPECT_EQ(table->schema()->field(0)->name(), "id");
      EXPECT_EQ(table->schema()->field(1)->name(), internal_column_name);
      EXPECT_EQ(table->schema()->field(2)->name(), "name");
      total_rows += table->num_rows();
    }

    EXPECT_EQ(total_rows, 10) << "Should have processed all 10 records";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 7 - Internal scanopen with mixed columns:"
              << std::endl;
    std::cout << schema->ToString() << std::endl;
#endif
  }
}

// Test failure cases of scanopen internal column projection
TEST_F(NoIndexTest, ScanOpenInternalColumnProjectionFailure) {
  std::string table_name = "test_scanopen_internal_fail";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert test data
  for (int i = 0; i < 10; i++) {
    std::string data = std::to_string(i) + "\u001eUser" + std::to_string(i) +
                       "\u001e" + std::to_string(i * 100);
    status = vdb::_InsertCommand(table_name, data);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scanopen requesting ONLY internal column should fail
  {
    std::string uuid = "scan_regular_internal_only_" + table_name;
    std::string projection = internal_column_name;

    auto result =
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", false);
    EXPECT_FALSE(result.ok())
        << "Regular scanopen requesting only internal column should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column_name) != std::string::npos)
          << "Error message should mention the internal column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1 - Regular scanopen with only internal column "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 2: Regular scanopen with mixed columns (visible + internal) should
  // fail
  {
    std::string uuid = "scan_mixed_" + table_name;
    std::string projection = "id, " + internal_column_name;
    auto result =
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", false);
    EXPECT_FALSE(result.ok()) << "Regular scanopen with mixed columns (visible "
                                 "+ internal) should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column_name) != std::string::npos)
          << "Error message should mention the internal column name";

#ifdef _DEBUG_GTEST
      std::cout
          << "Scenario 2 - Regular scanopen with mixed columns correctly fails:"
          << std::endl;
      std::cout << "  Requested: id, " << internal_column_name << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 3: Empty projection should fail
  {
    std::string uuid = "scan_empty_" + table_name;
    auto result = vdb::_ScanOpenCommand(uuid, table_name, "", "", "0", false);
    EXPECT_FALSE(result.ok()) << "Empty projection should fail with an error";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Empty projection list") != std::string::npos)
          << "Error message should contain 'Empty projection list', got: "
          << error_msg;

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 3 - Empty projection correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 4: Empty projection with internal scanopen should also fail
  {
    std::string uuid = "scan_empty_internal_" + table_name;
    auto result = vdb::_ScanOpenCommand(uuid, table_name, "", "", "0", true);
    EXPECT_FALSE(result.ok()) << "Empty projection should fail even with "
                                 "include_internal_columns=true";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Empty projection list") != std::string::npos)
          << "Error message should contain 'Empty projection list', got: "
          << error_msg;

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 4 - Empty projection with internal scanopen "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 5: Regular scanopen with non-existent column should fail
  {
    std::string uuid = "scan_non_existent_" + table_name;
    std::string projection = "id, non_existent_column";
    auto result =
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", false);
    EXPECT_FALSE(result.ok())
        << "Regular scanopen with non-existent column should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find("non_existent_column") != std::string::npos)
          << "Error message should mention the non-existent column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 5 - Regular scanopen with non-existent column "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }

  // Scenario 6: Internal scanopen with non-existent column should also fail
  {
    std::string uuid = "scan_non_existent_internal_" + table_name;
    std::string projection = "id, non_existent_column";
    auto result =
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true);
    EXPECT_FALSE(result.ok())
        << "Internal scanopen with non-existent column should fail";

    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Error message should contain 'Could not find column', got: "
          << error_msg;
      EXPECT_TRUE(error_msg.find("non_existent_column") != std::string::npos)
          << "Error message should mention the non-existent column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 6 - Internal scanopen with non-existent column "
                   "correctly fails:"
                << std::endl;
      std::cout << "  Error: " << error_msg << std::endl;
#endif
    }
  }
}

// Test internal column projection with scanopen and delete operations
TEST_F(NoIndexTest, ScanOpenInternalColumnProjectionWithDelete) {
  std::string table_name = "test_scanopen_delete_visibility";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert multiple records to simulate dataset
  for (int i = 0; i < 10; i++) {
    std::string data = std::to_string(i) + "\u001eUser" + std::to_string(i) +
                       "\u001e" + std::to_string(i * 100);
    status = vdb::_InsertCommand(table_name, data);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1-1: Delete a record and verify behavior difference between
  // regular scanopen and internal scanopen
  {
    // Delete the record with id=5
    std::string filter = "id = 5";
    ASSERT_OK_AND_ASSIGN(auto deleted_count,
                         vdb::_DeleteCommand(table_name, filter));
    EXPECT_EQ(deleted_count, 1) << "Should delete exactly 1 record";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1-1 - Deleted record with id=5" << std::endl;
#endif

    // 1-1a: Regular scanopen should NOT see the deleted record
    {
      std::string uuid = "scan_deleted_regular_" + table_name;
      ASSERT_OK_AND_ASSIGN(
          auto open_result,
          vdb::_ScanOpenCommand(uuid, table_name, "id, name", "", "0", false));

      auto serialized_table = std::get<0>(open_result);
      auto has_next = std::get<1>(open_result);

      int total_rows = 0;

      // Verify all batches
      while (true) {
        ASSERT_OK_AND_ASSIGN(auto table,
                             DeserializeToTableFrom(serialized_table));
        total_rows += table->num_rows();

        // Verify that id=5 is NOT present
        auto id_col = table->column(0);
        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks(); ++chunk_idx) {
          auto chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));
          for (int64_t i = 0; i < chunk->length(); i++) {
            auto id_value = chunk->Value(i);
            EXPECT_NE(id_value, 5)
                << "Regular scanopen should not see deleted record (id=5)";
          }
        }

        if (!has_next) break;

        ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
        serialized_table = std::get<0>(fetch_result);
        has_next = std::get<1>(fetch_result);
      }

      // Should have 9 records (10 - 1 deleted)
      EXPECT_EQ(total_rows, 9);

#ifdef _DEBUG_GTEST
      std::cout << "  Regular scanopen: " << total_rows
                << " rows (deleted record hidden)" << std::endl;
#endif
    }

    // 1-1b: Internal scanopen SHOULD see the deleted record with
    // __deleted_flag = true
    {
      std::string uuid = "scan_deleted_internal_" + table_name;
      std::string projection = "id, name, " + internal_column_name;

      ASSERT_OK_AND_ASSIGN(
          auto open_result,
          vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

      auto serialized_table = std::get<0>(open_result);
      auto has_next = std::get<1>(open_result);

      int total_rows = 0;
      bool found_deleted_record = false;

      // Check all batches
      while (true) {
        ASSERT_OK_AND_ASSIGN(auto table,
                             DeserializeToTableFrom(serialized_table));
        total_rows += table->num_rows();

        auto id_col = table->column(0);
        auto deleted_flag_col = table->column(2);

        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks(); ++chunk_idx) {
          auto id_chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));
          auto flag_chunk = std::static_pointer_cast<arrow::BooleanArray>(
              deleted_flag_col->chunk(chunk_idx));

          for (int64_t i = 0; i < id_chunk->length(); i++) {
            auto id_value = id_chunk->Value(i);
            auto deleted_flag = flag_chunk->Value(i);

            if (id_value == 5) {
              found_deleted_record = true;
              EXPECT_TRUE(deleted_flag)
                  << "Deleted record (id=5) should have __deleted_flag = true";
            }
          }
        }

        if (!has_next) break;

        ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
        serialized_table = std::get<0>(fetch_result);
        has_next = std::get<1>(fetch_result);
      }

      // Should have all 10 records (including the deleted one)
      EXPECT_EQ(total_rows, 10);
      EXPECT_TRUE(found_deleted_record)
          << "Internal scanopen should see deleted record (id=5)";

#ifdef _DEBUG_GTEST
      std::cout << "  Internal scanopen: " << total_rows
                << " rows (deleted record visible with flag=true)" << std::endl;
#endif
    }
  }

  // Scenario 1-2: Internal scanopen without filter should show ALL records
  // (active + deleted)
  {
    std::string uuid = "scan_no_filter_" + table_name;
    std::string projection = "id, " + internal_column_name;

    // Internal scanopen with NO filter
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    int total_rows = 0;
    int active_count = 0;
    int deleted_count = 0;

    // Count in all batches
    while (true) {
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));
      total_rows += table->num_rows();

      auto deleted_flag_col = table->column(1);
      for (int chunk_idx = 0; chunk_idx < deleted_flag_col->num_chunks();
           ++chunk_idx) {
        auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
            deleted_flag_col->chunk(chunk_idx));
        for (int64_t i = 0; i < chunk->length(); i++) {
          auto deleted_flag = chunk->Value(i);
          if (deleted_flag) {
            deleted_count++;
          } else {
            active_count++;
          }
        }
      }

      if (!has_next) break;

      ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
      serialized_table = std::get<0>(fetch_result);
      has_next = std::get<1>(fetch_result);
    }

    // Should have all 10 records (9 active + 1 deleted from previous scenario)
    EXPECT_EQ(total_rows, 10);
    EXPECT_EQ(active_count, 9) << "Should have 9 active records";
    EXPECT_EQ(deleted_count, 1) << "Should have 1 deleted record";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1-2 - Internal scanopen without filter:"
              << std::endl;
    std::cout << "  Total records: " << total_rows << std::endl;
    std::cout << "  - Active: " << active_count << std::endl;
    std::cout << "  - Deleted: " << deleted_count << std::endl;
#endif

    // Compare with regular scanopen (should only show active records)
    std::string uuid_regular = "scan_regular_compare_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto regular_result,
        vdb::_ScanOpenCommand(uuid_regular, table_name, "id", "", "0", false));

    auto regular_serialized = std::get<0>(regular_result);
    auto regular_has_next = std::get<1>(regular_result);

    int regular_total = 0;
    ASSERT_OK_AND_ASSIGN(auto regular_table,
                         DeserializeToTableFrom(regular_serialized));
    regular_total += regular_table->num_rows();

    while (regular_has_next) {
      ASSERT_OK_AND_ASSIGN(auto regular_fetch,
                           vdb::_FetchNextCommand(uuid_regular));
      regular_serialized = std::get<0>(regular_fetch);
      regular_has_next = std::get<1>(regular_fetch);

      ASSERT_OK_AND_ASSIGN(regular_table,
                           DeserializeToTableFrom(regular_serialized));
      regular_total += regular_table->num_rows();
    }

    EXPECT_EQ(regular_total, 9)
        << "Regular scanopen should only show active records";

#ifdef _DEBUG_GTEST
    std::cout << "  Regular scanopen (for comparison): " << regular_total
              << " rows (active only)" << std::endl;
#endif
  }

  // Scenario 1-3: Filter by __deleted_flag to query specific records
  {
    std::string projection = "id, name, " + internal_column_name;

    // 1-3a: Query ONLY deleted records (__deleted_flag = true)
    {
      std::string uuid = "scan_deleted_only_" + table_name;
      std::string filter = internal_column_name + " = true";

      ASSERT_OK_AND_ASSIGN(auto open_result,
                           vdb::_ScanOpenCommand(uuid, table_name, projection,
                                                 filter, "0", true));

      auto serialized_table = std::get<0>(open_result);
      auto has_next = std::get<1>(open_result);

      int total_rows = 0;

      // Verify all batches
      while (true) {
        ASSERT_OK_AND_ASSIGN(auto table,
                             DeserializeToTableFrom(serialized_table));
        total_rows += table->num_rows();

        // Verify all have __deleted_flag = true
        auto deleted_flag_col = table->column(2);
        for (int chunk_idx = 0; chunk_idx < deleted_flag_col->num_chunks();
             ++chunk_idx) {
          auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
              deleted_flag_col->chunk(chunk_idx));
          for (int64_t i = 0; i < chunk->length(); i++) {
            auto deleted_flag = chunk->Value(i);
            EXPECT_TRUE(deleted_flag)
                << "Filtered record should have flag=true";
          }
        }

        if (!has_next) break;

        ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
        serialized_table = std::get<0>(fetch_result);
        has_next = std::get<1>(fetch_result);
      }

      // Should have 1 deleted record (id=5)
      EXPECT_EQ(total_rows, 1);

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1-3a - Filter deleted only: " << total_rows
                << " rows (id=5)" << std::endl;
#endif
    }

    // 1-3b: Query ONLY active records (__deleted_flag = false)
    {
      std::string uuid = "scan_active_only_" + table_name;
      std::string filter = internal_column_name + " = false";

      ASSERT_OK_AND_ASSIGN(auto open_result,
                           vdb::_ScanOpenCommand(uuid, table_name, projection,
                                                 filter, "0", true));

      auto serialized_table = std::get<0>(open_result);
      auto has_next = std::get<1>(open_result);

      int total_rows = 0;

      // Verify all batches
      while (true) {
        ASSERT_OK_AND_ASSIGN(auto table,
                             DeserializeToTableFrom(serialized_table));
        total_rows += table->num_rows();

        // Verify all have __deleted_flag = false
        auto deleted_flag_col = table->column(2);
        for (int chunk_idx = 0; chunk_idx < deleted_flag_col->num_chunks();
             ++chunk_idx) {
          auto chunk = std::static_pointer_cast<arrow::BooleanArray>(
              deleted_flag_col->chunk(chunk_idx));
          for (int64_t i = 0; i < chunk->length(); i++) {
            auto deleted_flag = chunk->Value(i);
            EXPECT_FALSE(deleted_flag)
                << "Filtered records should all have flag=false";
          }
        }

        if (!has_next) break;

        ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
        serialized_table = std::get<0>(fetch_result);
        has_next = std::get<1>(fetch_result);
      }

      // Should have 9 active records
      EXPECT_EQ(total_rows, 9);

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1-3b - Filter active only: " << total_rows
                << " rows" << std::endl;
#endif
    }

    // 1-3c: Complex filter with __deleted_flag
    {
      std::string uuid = "scan_complex_filter_" + table_name;
      std::string filter = internal_column_name + " = true AND id = 5";

      ASSERT_OK_AND_ASSIGN(auto open_result,
                           vdb::_ScanOpenCommand(uuid, table_name, projection,
                                                 filter, "0", true));

      auto serialized_table = std::get<0>(open_result);
      auto has_next = std::get<1>(open_result);

      int total_rows = 0;
      ASSERT_OK_AND_ASSIGN(auto table,
                           DeserializeToTableFrom(serialized_table));
      total_rows += table->num_rows();

      // Verify it's the correct deleted record
      if (table->num_rows() > 0) {
        auto id_col = table->column(0);
        // Find the row across all chunks (data might not be in first chunk)
        bool found_row = false;
        for (int chunk_idx = 0; chunk_idx < id_col->num_chunks() && !found_row;
             ++chunk_idx) {
          auto id_chunk = std::static_pointer_cast<arrow::UInt32Array>(
              id_col->chunk(chunk_idx));
          if (id_chunk->length() > 0) {
            auto id_value = id_chunk->Value(0);
            EXPECT_EQ(id_value, 5);
            found_row = true;
          }
        }
        EXPECT_TRUE(found_row) << "Should find exactly 1 row";
      }

      // Verify all batches
      while (has_next) {
        ASSERT_OK_AND_ASSIGN(auto fetch_result, vdb::_FetchNextCommand(uuid));
        serialized_table = std::get<0>(fetch_result);
        has_next = std::get<1>(fetch_result);

        ASSERT_OK_AND_ASSIGN(table, DeserializeToTableFrom(serialized_table));
        total_rows += table->num_rows();
      }

      // Should have exactly 1 record (deleted AND id=5)
      EXPECT_EQ(total_rows, 1);

#ifdef _DEBUG_GTEST
      std::cout << "Scenario 1-3c - Complex filter (deleted AND id=5): "
                << total_rows << " rows" << std::endl;
#endif
    }
  }
}

// Test successful cases of filter with internal columns
TEST_F(NoIndexTest, ScanInternalColumnFilterSuccess) {
  std::string table_name = "test_filter_success";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Insert test data
  status = vdb::_InsertCommand(table_name, "0\u001eAlice\u001e100");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "1\u001eBob\u001e200");
  ASSERT_TRUE(status.ok()) << status.ToString();
  status = vdb::_InsertCommand(table_name, "2\u001eCharlie\u001e300");
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scan with empty filter
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have only user columns (no internal columns)
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");
    EXPECT_EQ(schema->field(2)->name(), "value");

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scan should NOT include internal column";

    // Should return all rows
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scan with empty filter:" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scan with empty filter
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

    // Should return all rows
    EXPECT_EQ(table->num_rows(), 3);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 2 - Internal scan with empty filter:" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Regular scan with normal filter (id > 0)
  {
    std::string filter = "id > 0";
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, "*", filter, "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have only user columns
    EXPECT_EQ(schema->num_fields(), 3);

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scan should NOT include internal column";

    // Should return filtered rows (id > 0: Bob, Charlie)
    EXPECT_EQ(table->num_rows(), 2);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Regular scan with filter 'id > 0':" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scan with normal filter (id > 0)
  {
    std::string filter = "id > 0";
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", filter, "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

    // Should return filtered rows (id > 0: Bob, Charlie)
    EXPECT_EQ(table->num_rows(), 2);

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 4 - Internal scan with filter 'id > 0':"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Internal scan with internal column only filter
  {
    // First, delete one record (id=1, Bob)
    std::string delete_filter = "id = 1";
    auto delete_status = vdb::_DeleteCommand(table_name, delete_filter);
    ASSERT_TRUE(delete_status.ok()) << delete_status.status().ToString();

    // Now scan with internal column only filter
    std::string filter = internal_column_name + " = true";
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", filter, "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

    // Should return only deleted row (id=1, Bob)
    EXPECT_EQ(table->num_rows(), 1);

    // Verify the deleted row is Bob (id=1)
    if (table->num_rows() > 0) {
      auto id_column = table->column(0);
      ASSERT_GT(id_column->num_chunks(), 0)
          << "ID column should have at least one chunk";

      for (int chunk_idx = 0; chunk_idx < id_column->num_chunks();
           ++chunk_idx) {
        auto id_array = std::static_pointer_cast<arrow::UInt32Array>(
            id_column->chunk(chunk_idx));
        if (id_array->length() > 0) {
          EXPECT_EQ(id_array->Value(0), 1)
              << "Deleted record should have id=1 (Bob)";
          break;
        }
      }
    }

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Internal scan with internal column filter '"
              << filter << "':" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << " (deleted records)"
              << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 6: Internal scan with combined filter (internal + user column)
  {
    // Scan with combined filter (internal column + user column)
    std::string filter = internal_column_name + " = true AND id > 0";
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", filter, "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

    // Should return only deleted row with id > 0 (id=1, Bob)
    EXPECT_EQ(table->num_rows(), 1);

    // Verify the deleted row is Bob (id=1)
    if (table->num_rows() > 0) {
      auto id_column = table->column(0);
      ASSERT_GT(id_column->num_chunks(), 0)
          << "ID column should have at least one chunk";

      for (int chunk_idx = 0; chunk_idx < id_column->num_chunks();
           ++chunk_idx) {
        auto id_array = std::static_pointer_cast<arrow::UInt32Array>(
            id_column->chunk(chunk_idx));
        if (id_array->length() > 0) {
          EXPECT_EQ(id_array->Value(0), 1)
              << "Deleted record should have id=1 (Bob)";
          break;
        }
      }
    }

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 6 - Internal scan with combined filter '" << filter
              << "':" << std::endl;
    std::cout << "  Rows: " << table->num_rows()
              << " (deleted records with id > 0)" << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }
}

// Test that internal columns in filter are rejected for Scan, Delete, Update
TEST_F(NoIndexTest, ScanInternalColumnFilterFailure) {
  std::string table_name = "test_internal_filter";

  // Create table with simple schema (no index needed)
  auto id_field = arrow::field("id", arrow::int64());
  auto name_field = arrow::field("name", arrow::utf8());
  auto value_field = arrow::field("value", arrow::int32());
  auto schema = arrow::schema({id_field, name_field, value_field});

  std::unordered_map<std::string, std::string> metadata;
  metadata["table name"] = table_name;

  auto schema_with_metadata =
      schema->WithMetadata(std::make_shared<arrow::KeyValueMetadata>(metadata));

  auto maybe_serialized_schema =
      arrow::ipc::SerializeSchema(*schema_with_metadata, &vdb::arrow_pool);
  ASSERT_TRUE(maybe_serialized_schema.ok());
  auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
  sds schema_sds =
      sdsnewlen(serialized_schema->data(), serialized_schema->size());
  auto create_status = vdb::_CreateTableCommand(schema_sds);
  sdsfree(schema_sds);
  ASSERT_TRUE(create_status.ok()) << create_status.ToString();

  // Insert test data
  {
    arrow::Int64Builder id_builder;
    arrow::StringBuilder name_builder;
    arrow::Int32Builder value_builder;

    ASSERT_OK(id_builder.Append(1));
    ASSERT_OK(name_builder.Append("test1"));
    ASSERT_OK(value_builder.Append(100));

    ASSERT_OK(id_builder.Append(2));
    ASSERT_OK(name_builder.Append("test2"));
    ASSERT_OK(value_builder.Append(200));

    std::shared_ptr<arrow::Array> id_array, name_array, value_array;
    ASSERT_OK(id_builder.Finish(&id_array));
    ASSERT_OK(name_builder.Finish(&name_array));
    ASSERT_OK(value_builder.Finish(&value_array));

    auto batch = arrow::RecordBatch::Make(schema, 2,
                                          {id_array, name_array, value_array});
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch};
    auto maybe_serialized = SerializeRecordBatches(schema, batches);
    ASSERT_TRUE(maybe_serialized.ok());
    auto serialized_rb = maybe_serialized.ValueUnsafe();
    sds batch_sds = sdsnewlen(serialized_rb->data(), serialized_rb->size());
    auto insert_status = vdb::_BatchInsertCommand(table_name, batch_sds);
    sdsfree(batch_sds);
    ASSERT_TRUE(insert_status.ok()) << insert_status.ToString();
  }

  std::string internal_column = vdb::kDeletedFlagColumn;  // "__deleted_flag"

  // Test 1: Scan with internal column in filter should fail
  {
    std::string filter = internal_column + " = false";
    auto result = vdb::_ScanCommand(table_name, "*", filter, "0", false);

    EXPECT_FALSE(result.ok())
        << "Scan with internal column in filter should fail";
    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Expected 'Could not find column' error, got: " << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column) != std::string::npos)
          << "Error message should mention the internal column name";

#ifdef _DEBUG_GTEST
      std::cout << "Scan correctly rejected internal column in filter: "
                << error_msg << std::endl;
#endif
    }
  }

  // Test 2: Scan with internal column in complex filter should fail
  {
    std::string filter = "id > 0 AND " + internal_column + " = false";
    auto result = vdb::_ScanCommand(table_name, "*", filter, "0", false);

    EXPECT_FALSE(result.ok())
        << "Scan with internal column in complex filter should fail";
    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos);
      EXPECT_TRUE(error_msg.find(internal_column) != std::string::npos);

#ifdef _DEBUG_GTEST
      std::cout << "Scan correctly rejected internal column in complex filter: "
                << error_msg << std::endl;
#endif
    }
  }

  // Test 3: Delete with internal column in filter should fail
  {
    std::string filter = internal_column + " = true";
    auto result = vdb::_DeleteCommand(table_name, filter);

    EXPECT_FALSE(result.ok())
        << "Delete with internal column in filter should fail";
    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Expected 'Could not find column' error, got: " << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column) != std::string::npos);

#ifdef _DEBUG_GTEST
      std::cout << "Delete correctly rejected internal column in filter: "
                << error_msg << std::endl;
#endif
    }
  }

  // Test 4: Delete with internal column in complex filter should fail
  {
    std::string filter = "id = 1 AND " + internal_column + " = true";
    auto result = vdb::_DeleteCommand(table_name, filter);

    EXPECT_FALSE(result.ok())
        << "Delete with internal column in complex filter should fail";
    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos);
      EXPECT_TRUE(error_msg.find(internal_column) != std::string::npos);

#ifdef _DEBUG_GTEST
      std::cout
          << "Delete correctly rejected internal column in complex filter: "
          << error_msg << std::endl;
#endif
    }
  }

  // Test 5: Update with internal column in filter should fail
  {
    std::string updates = "name='updated'";
    std::string filter = internal_column + " = false";
    auto result = vdb::_UpdateCommand(table_name, updates, filter);

    EXPECT_FALSE(result.ok())
        << "Update with internal column in filter should fail";
    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos)
          << "Expected 'Could not find column' error, got: " << error_msg;
      EXPECT_TRUE(error_msg.find(internal_column) != std::string::npos);

#ifdef _DEBUG_GTEST
      std::cout << "Update correctly rejected internal column in filter: "
                << error_msg << std::endl;
#endif
    }
  }

  // Test 6: Update with internal column in complex filter should fail
  {
    std::string updates = "value=999";
    std::string filter = "id > 0 AND " + internal_column + " = false";
    auto result = vdb::_UpdateCommand(table_name, updates, filter);

    EXPECT_FALSE(result.ok())
        << "Update with internal column in complex filter should fail";
    if (!result.ok()) {
      std::string error_msg = result.status().ToString();
      EXPECT_TRUE(error_msg.find("Could not find column") != std::string::npos);
      EXPECT_TRUE(error_msg.find(internal_column) != std::string::npos);

#ifdef _DEBUG_GTEST
      std::cout
          << "Update correctly rejected internal column in complex filter: "
          << error_msg << std::endl;
#endif
    }
  }

  // Cleanup
  auto drop_status = vdb::_DropTableCommand(table_name, false);
  ASSERT_TRUE(drop_status.ok()) << drop_status.ToString();
}

// Test internal column projection on empty table (success cases)
TEST_F(NoIndexTest, ScanEmptyTableInternalColumnProjectionSuccess) {
  std::string table_name = "test_empty_table_projection";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // No data inserted - empty table
  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scan with "*" on empty table
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column) << "Regular scan should NOT include "
                                         "internal column even on empty table";

    // Should have exactly 3 user-defined columns
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scan on empty table:" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scan with "*" on empty table SHOULD show internal
  // columns
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column even on empty table";

    // Should have 4 columns: id, name, value, __deleted_flag
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 2 - Internal scan on empty table:" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Internal scan with ONLY internal column on empty table
  {
    std::string projection = internal_column_name;
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have exactly 1 column: __deleted_flag only
    EXPECT_EQ(schema->num_fields(), 1);
    EXPECT_EQ(schema->field(0)->name(), internal_column_name);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Internal column ONLY projection on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scan with specific internal + visible columns on empty
  // table
  {
    std::string projection = "id, " + internal_column_name;
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have exactly 2 columns: id and __deleted_flag
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), internal_column_name);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

#ifdef _DEBUG_GTEST
    std::cout
        << "Scenario 4 - Internal + visible columns projection on empty table:"
        << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Regular scan with specific columns on empty table
  {
    std::string projection = "id, name";
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, projection, "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scan should NOT include internal column";

    // Should have exactly 2 columns: id, name
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Regular scan with specific columns on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }
}

// Test internal column filtering on empty table (success cases)
TEST_F(NoIndexTest, ScanEmptyTableInternalColumnFilterSuccess) {
  std::string table_name = "test_empty_table_filter";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // No data inserted - empty table
  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scan with empty filter on empty table
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have only user columns (no internal columns)
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scan should NOT include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scan with empty filter on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scan with empty filter on empty table
  {
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", "", "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 2 - Internal scan with empty filter on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Internal scan with __deleted_flag = true filter on empty table
  {
    std::string filter = internal_column_name + " = true";
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", filter, "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with filter";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Internal scan with __deleted_flag = true filter "
                 "on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scan with __deleted_flag = false filter on empty table
  {
    std::string filter = internal_column_name + " = false";
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", filter, "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with filter";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 4 - Internal scan with __deleted_flag = false "
                 "filter on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Internal scan with combined filter on empty table
  {
    std::string filter = internal_column_name + " = false AND id > 0";
    ASSERT_OK_AND_ASSIGN(auto serialized_table,
                         vdb::_ScanCommand(table_name, "*", filter, "0", true));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with combined filter";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scan should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Internal scan with combined filter on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 6: Regular scan with normal filter on empty table
  {
    std::string filter = "id > 0";
    ASSERT_OK_AND_ASSIGN(
        auto serialized_table,
        vdb::_ScanCommand(table_name, "*", filter, "0", false));
    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));

    auto schema = table->schema();

    // Should have only user columns
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with filter";

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scan should NOT include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 6 - Regular scan with normal filter on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }
}

// Test internal column projection on empty table with ScanOpen (success cases)
TEST_F(NoIndexTest, ScanOpenEmptyTableInternalColumnProjectionSuccess) {
  std::string table_name = "test_scanopen_empty_projection";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // No data inserted - empty table
  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scanopen with "*" on empty table
  {
    std::string uuid = "scan_regular_empty_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", "", "0", false));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scanopen should NOT include internal column on empty table";

    // Should have exactly 3 user-defined columns
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scanopen on empty table:" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scanopen with "*" on empty table SHOULD show internal
  // columns
  {
    std::string uuid = "scan_internal_empty_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scanopen should include internal column on empty table";

    // Should have 4 columns: id, name, value, __deleted_flag
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 2 - Internal scanopen on empty table:" << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Internal scanopen with ONLY internal column on empty table
  {
    std::string uuid = "scan_internal_only_empty_" + table_name;
    std::string projection = internal_column_name;

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have exactly 1 column: __deleted_flag only
    EXPECT_EQ(schema->num_fields(), 1);
    EXPECT_EQ(schema->field(0)->name(), internal_column_name);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Internal column ONLY projection on empty table "
                 "(scanopen):"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scanopen with specific internal + visible columns on
  // empty table
  {
    std::string uuid = "scan_specific_empty_" + table_name;
    std::string projection = "id, " + internal_column_name;

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have exactly 2 columns: id and __deleted_flag
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), internal_column_name);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 4 - Internal + visible columns projection on empty "
                 "table (scanopen):"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Regular scanopen with specific columns on empty table
  {
    std::string uuid = "scan_regular_cols_empty_" + table_name;
    std::string projection = "id, name";

    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, projection, "", "0", false));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scanopen should NOT include internal column";

    // Should have exactly 2 columns: id, name
    EXPECT_EQ(schema->num_fields(), 2);
    EXPECT_EQ(schema->field(0)->name(), "id");
    EXPECT_EQ(schema->field(1)->name(), "name");
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Regular scanopen with specific columns on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }
}

// Test internal column filtering on empty table with ScanOpen (success cases)
TEST_F(NoIndexTest, ScanOpenEmptyTableInternalColumnFilterSuccess) {
  std::string table_name = "test_scanopen_empty_filter";
  std::string schema_string = "id uint32 not null, name String, value int32";

  auto status = CreateTableForTest(table_name, schema_string);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // No data inserted - empty table
  std::string internal_column_name = vdb::kDeletedFlagColumn;

  // Scenario 1: Regular scanopen with empty filter on empty table
  {
    std::string uuid = "scan_empty_filter_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", "", "0", false));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have only user columns (no internal columns)
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scanopen should NOT include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 1 - Regular scanopen with empty filter on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 2: Internal scanopen with empty filter on empty table
  {
    std::string uuid = "scan_internal_empty_filter_" + table_name;
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", "", "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0) << "Empty table should return 0 rows";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scanopen should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 2 - Internal scanopen with empty filter on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 3: Internal scanopen with __deleted_flag = true filter on empty
  // table
  {
    std::string uuid = "scan_deleted_true_" + table_name;
    std::string filter = internal_column_name + " = true";
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", filter, "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with filter";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scanopen should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 3 - Internal scanopen with __deleted_flag = true "
                 "filter on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 4: Internal scanopen with __deleted_flag = false filter on empty
  // table
  {
    std::string uuid = "scan_deleted_false_" + table_name;
    std::string filter = internal_column_name + " = false";
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", filter, "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with filter";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scanopen should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 4 - Internal scanopen with __deleted_flag = false "
                 "filter on empty table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 5: Internal scanopen with combined filter on empty table
  {
    std::string uuid = "scan_combined_filter_" + table_name;
    std::string filter = internal_column_name + " = false AND id > 0";
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", filter, "0", true));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have both user columns and internal columns
    EXPECT_EQ(schema->num_fields(), 4);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with combined filter";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

    // Verify internal column IS present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_TRUE(has_internal_column)
        << "Internal scanopen should include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 5 - Internal scanopen with combined filter on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }

  // Scenario 6: Regular scanopen with normal filter on empty table
  {
    std::string uuid = "scan_normal_filter_" + table_name;
    std::string filter = "id > 0";
    ASSERT_OK_AND_ASSIGN(
        auto open_result,
        vdb::_ScanOpenCommand(uuid, table_name, "*", filter, "0", false));

    auto serialized_table = std::get<0>(open_result);
    auto has_next = std::get<1>(open_result);

    ASSERT_OK_AND_ASSIGN(auto table, DeserializeToTableFrom(serialized_table));
    auto schema = table->schema();

    // Should have only user columns
    EXPECT_EQ(schema->num_fields(), 3);
    EXPECT_EQ(table->num_rows(), 0)
        << "Empty table should return 0 rows even with filter";
    EXPECT_FALSE(has_next) << "Empty table should not have next batch";

    // Verify internal column is NOT present
    bool has_internal_column = false;
    for (int i = 0; i < schema->num_fields(); i++) {
      if (schema->field(i)->name() == internal_column_name) {
        has_internal_column = true;
        break;
      }
    }
    EXPECT_FALSE(has_internal_column)
        << "Regular scanopen should NOT include internal column";

#ifdef _DEBUG_GTEST
    std::cout << "Scenario 6 - Regular scanopen with normal filter on empty "
                 "table:"
              << std::endl;
    std::cout << "  Rows: " << table->num_rows() << std::endl;
    std::cout << "  Schema: " << schema->ToString() << std::endl;
#endif
  }
}
}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
