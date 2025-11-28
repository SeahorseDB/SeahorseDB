#include <memory>
#include <string>
#include <regex>
#include <sstream>
#include <set>

#include <arrow/api.h>
#include <arrow/testing/gtest_util.h>

#include <gtest/gtest.h>

#include "vdb/common/status.hh"
#include "vdb/tests/util_for_test.hh"
#include "vdb/vdb_api.hh"
#include "vdb/common/defs.hh"
#include "vdb/common/util.hh"
#include "vdb/data/mutable_array.hh"
#include "vdb/data/table.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/data/index_info.hh"
#include "vdb/data/primary_key_index.hh"
#include "vdb/tests/base_environment.hh"

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/DataStructureTestSuite";

class DataStructureTestSuite : public BaseTestSuite {
 protected:
  void SetUp() override {
    BaseTestSuite::SetUp();
    server.vdb_active_set_size_limit = 1000;
  }
};

class TableTest : public DataStructureTestSuite {};
class FilterTest : public DataStructureTestSuite {};
class MutableArrayTest : public DataStructureTestSuite {};
class EmbeddingStoreTest : public DataStructureTestSuite {};
class SchemaTest : public DataStructureTestSuite {};
class SkipListTest : public DataStructureTestSuite {};
class NumberKeyFlatArrayTest : public DataStructureTestSuite {};
class FileKeyMapTest : public DataStructureTestSuite {};
class IndexInfoTest : public DataStructureTestSuite {};
// TODO - The tests should be performed automatically without human eyes..

TEST_F(TableTest, ParseStringTest) {
  std::string table_schema_string =
      "ID INT32 not null, Name String, Attributes List[  String  ], Vector "
      "Fixed_Size_list[  1024,    floAt32 ]";
  arrow::FieldVector fields;
  fields.emplace_back(
      std::make_shared<arrow::Field>("id", arrow::int32(), false));
  fields.emplace_back(std::make_shared<arrow::Field>("name", arrow::utf8()));
  fields.emplace_back(
      std::make_shared<arrow::Field>("attributes", arrow::list(arrow::utf8())));
  fields.emplace_back(std::make_shared<arrow::Field>(
      "vector", arrow::fixed_size_list(arrow::float32(), 1024)));
  auto schema = std::make_shared<arrow::Schema>(fields);
  auto parsed_schema = vdb::ParseSchemaFrom(table_schema_string);

  std::cout << schema->ToString() << std::endl;
  std::cout << parsed_schema->ToString() << std::endl;

  EXPECT_TRUE(schema->Equals(parsed_schema));
}

TEST_F(TableTest, CreateTableTest) {
  std::string table_schema_string =
      "id INT32 not null, Name String, Attributes List[  String  ], Vector "
      "Fixed_Size_list[  1024,    floAt32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());
#ifdef _DEBUG_GTEST
  std::cout << table->ToString() << std::endl;
#endif
  auto segment = table->AddSegment(table, "test_segment");
  EXPECT_TRUE(segment != nullptr);
}

TEST_F(MutableArrayTest, AppendAndInactiveTest) {
  vdb::NumericArray<int16_t, arrow::Int16Array> int16arr;
  EXPECT_TRUE(int16arr.Append(23).ok());
  EXPECT_TRUE(int16arr.Append(24).ok());
  EXPECT_TRUE(int16arr.AppendNull().ok());
  EXPECT_TRUE(int16arr.Append(25).ok());
  EXPECT_TRUE(int16arr.Append(26).ok());
  EXPECT_EQ(int16arr.Size(), 5);
  EXPECT_EQ(int16arr.GetValue(4).value(), 26);
  EXPECT_FALSE(int16arr.GetValue(2).has_value());
  EXPECT_FALSE(int16arr.GetValue(8).has_value());

  vdb::FixedSizeListArray<float, arrow::FloatArray> fslarr(4);
  EXPECT_TRUE(fslarr.Append({128.0, 32.8, 192.6, 48.1}).ok());
  EXPECT_TRUE(fslarr.Append({423.3, 418.1, 191.4, 0.512}).ok());
  EXPECT_TRUE(fslarr.AppendNull().ok());
  EXPECT_TRUE(fslarr.Append({127.0, 0, 0, 1.0}).ok());
  EXPECT_EQ(fslarr.Size(), 4);
  EXPECT_EQ(fslarr.GetValue(0).value().size(), 4);
  EXPECT_EQ(fslarr.GetValue(1).value().size(), 4);
  EXPECT_FALSE(fslarr.GetValue(2).has_value());
  EXPECT_EQ(fslarr.GetValue(3).value().size(), 4);

  vdb::ListArray<uint32_t, arrow::UInt32Array> larr;
  EXPECT_TRUE(larr.Append({128, 64, 32, 16}).ok());
  EXPECT_TRUE(larr.AppendNull().ok());
  EXPECT_TRUE(larr.Append({8, 4}).ok());
  EXPECT_TRUE(larr.AppendNull().ok());
  EXPECT_TRUE(larr.Append({2}).ok());
  EXPECT_TRUE(larr.AppendNull().ok());
  EXPECT_TRUE(larr.Append(std::vector<uint32_t>{}).ok());
  EXPECT_EQ(larr.Size(), 7);
  EXPECT_EQ(larr.GetValue(0).value(), std::vector<uint32_t>({128, 64, 32, 16}));
  EXPECT_FALSE(larr.GetValue(1).has_value());
  EXPECT_EQ(larr.GetValue(2).value(), std::vector<uint32_t>({8, 4}));
  EXPECT_FALSE(larr.GetValue(3).has_value());
  EXPECT_EQ(larr.GetValue(4).value(), std::vector<uint32_t>({2}));
  EXPECT_FALSE(larr.GetValue(5).has_value());

  vdb::StringArray strarr;
  EXPECT_TRUE(strarr.Append(std::string_view{"Hello World!"}).ok());
  EXPECT_TRUE(strarr.Append(std::string_view{"Nice to meet you!"}).ok());
  EXPECT_TRUE(strarr.AppendNull().ok());
  EXPECT_TRUE(strarr.Append(std::string_view{"There is no spoon."}).ok());
  EXPECT_EQ(strarr.Size(), 4);
  EXPECT_EQ(strarr.GetValue(0).value(), "Hello World!");
  EXPECT_EQ(strarr.GetValue(1).value(), "Nice to meet you!");
  EXPECT_FALSE(strarr.GetValue(2).has_value());
  EXPECT_EQ(strarr.GetValue(3).value(), "There is no spoon.");
  EXPECT_FALSE(strarr.GetValue(4).has_value());

  vdb::LargeStringArray lstrarr;
  EXPECT_TRUE(lstrarr.Append(std::string_view{"Hello World!"}).ok());
  EXPECT_TRUE(lstrarr.Append(std::string_view{"Nice to meet you!"}).ok());
  EXPECT_TRUE(lstrarr.AppendNull().ok());
  EXPECT_TRUE(lstrarr.Append(std::string_view{"There is no spoon."}).ok());
  EXPECT_EQ(lstrarr.Size(), 4);
  EXPECT_EQ(lstrarr.GetValue(0).value(), "Hello World!");
  EXPECT_EQ(lstrarr.GetValue(1).value(), "Nice to meet you!");
  EXPECT_FALSE(lstrarr.GetValue(2).has_value());
  EXPECT_EQ(lstrarr.GetValue(3).value(), "There is no spoon.");
  EXPECT_FALSE(lstrarr.GetValue(4).has_value());

  vdb::StringListArray slarr;
  EXPECT_TRUE(slarr.Append({"Hello", "World"}).ok());
  EXPECT_TRUE(slarr.AppendNull().ok());
  EXPECT_TRUE(slarr.Append({"There", "is", "no", "spoon"}).ok());
  EXPECT_TRUE(slarr.Append({"Nothing", "is", "impossible"}).ok());
  EXPECT_TRUE(slarr.Append(std::vector<std::string>{}).ok());
  EXPECT_EQ(slarr.Size(), 5);
  EXPECT_EQ(slarr.GetValue(0).value(),
            std::vector<std::string_view>({"Hello", "World"}));
  EXPECT_FALSE(slarr.GetValue(1).has_value());
  EXPECT_EQ(slarr.GetValue(2).value(),
            std::vector<std::string_view>({"There", "is", "no", "spoon"}));
  EXPECT_EQ(slarr.GetValue(3).value(),
            std::vector<std::string_view>({"Nothing", "is", "impossible"}));

  vdb::StringFixedSizeListArray sfslarr(2);
  EXPECT_TRUE(sfslarr.AppendNull().ok());
  EXPECT_TRUE(sfslarr.Append({"Hello", "World"}).ok());
  EXPECT_TRUE(sfslarr.Append({"Nice to", "Meet you"}).ok());
  EXPECT_EQ(sfslarr.Size(), 3);
  EXPECT_FALSE(sfslarr.GetValue(0).has_value());
  EXPECT_EQ(sfslarr.GetValue(1),
            std::vector<std::string_view>({"Hello", "World"}));
  EXPECT_EQ(sfslarr.GetValue(2),
            std::vector<std::string_view>({"Nice to", "Meet you"}));

  vdb::BooleanArray barr;
  EXPECT_TRUE(barr.Append(true).ok());
  EXPECT_TRUE(barr.AppendNull().ok());
  EXPECT_TRUE(barr.Append(false).ok());
  EXPECT_TRUE(barr.Append(true).ok());
  EXPECT_EQ(barr.Size(), 4);
  EXPECT_TRUE(barr.GetValue(0).value());
  EXPECT_FALSE(barr.GetValue(1).has_value());
  EXPECT_FALSE(barr.GetValue(2).value());
  EXPECT_TRUE(barr.GetValue(3).value());

  vdb::BooleanListArray blarr;
  EXPECT_TRUE(blarr.Append({true, true, false}).ok());
  EXPECT_TRUE(blarr.Append({false, false, true, false}).ok());
  EXPECT_TRUE(blarr.AppendNull().ok());
  EXPECT_TRUE(blarr.Append({true, false}).ok());
  EXPECT_TRUE(blarr.Append(std::vector<bool>{}).ok());
  EXPECT_EQ(blarr.Size(), 5);
  EXPECT_EQ(blarr.GetValue(0), std::vector<bool>({true, true, false}));
  EXPECT_EQ(blarr.GetValue(1), std::vector<bool>({false, false, true, false}));
  EXPECT_FALSE(blarr.GetValue(2).has_value());
  EXPECT_EQ(blarr.GetValue(3), std::vector<bool>({true, false}));

  vdb::BooleanFixedSizeListArray bfslarr(4);
  EXPECT_TRUE(bfslarr.Append({true, false, false, false}).ok());
  EXPECT_TRUE(bfslarr.AppendNull().ok());
  EXPECT_TRUE(bfslarr.Append({false, true, false, false}).ok());
  EXPECT_TRUE(bfslarr.Append({false, false, true, false}).ok());
  EXPECT_TRUE(bfslarr.AppendNull().ok());
  EXPECT_TRUE(bfslarr.Append({false, false, false, true}).ok());
  EXPECT_TRUE(bfslarr.GetValue(0).has_value());
  EXPECT_FALSE(bfslarr.GetValue(1).has_value());
  EXPECT_TRUE(bfslarr.GetValue(2).has_value());
  EXPECT_TRUE(bfslarr.GetValue(3).has_value());
  EXPECT_FALSE(bfslarr.GetValue(4).has_value());
  EXPECT_TRUE(bfslarr.GetValue(5).has_value());
  EXPECT_EQ(bfslarr.GetValue(0).value(), std::vector<bool>({1, 0, 0, 0}));
  EXPECT_EQ(bfslarr.GetValue(2).value(), std::vector<bool>({0, 1, 0, 0}));
  EXPECT_EQ(bfslarr.GetValue(3).value(), std::vector<bool>({0, 0, 1, 0}));
  EXPECT_EQ(bfslarr.GetValue(5).value(), std::vector<bool>({0, 0, 0, 1}));
}

TEST_F(TableTest, AppendRecordTest) {
  std::string table_schema_string =
      "id int32 not null, Name String, Height float32, Gender Bool";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  // data
  std::vector<std::string> records = {"0\u001eTom\u001e180.3\u001e0",
                                      "1\u001eJane\u001e190.1\u001e1",
                                      "2\u001eJohn\u001e168.7\u001e0"};

  for (auto& record : records) {
    auto status = table->AppendRecord(record);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      EXPECT_TRUE(status.ok());
    }
  }

  EXPECT_EQ(table->GetSegmentCount(), 3);

  // Verify that segments are created with correct IDs based on the record
  // values
  for (size_t i = 0; i < records.size(); i++) {
    auto tokens = vdb::Tokenize(records[i], vdb::kRS);
    // Expected segment ID follows the new specification format:
    // {version}::{type}::comp::{column_names}::{json_values}::::{metadata}
    std::string expected_segment_id =
        "v1::value::single::id::[\"" + std::string(tokens[0]) + "\"]::::";

    // Find the segment with this ID
    auto segment = table->GetSegment(expected_segment_id);
    ASSERT_NE(segment, nullptr)
        << "Segment with ID " << expected_segment_id << " not found";
    ASSERT_EQ(tokens.size(), 4) << "Record should have 4 fields";
  }
}

TEST_F(TableTest, AppendRecordTestWithLargeString) {
  std::string table_schema_string =
      "id int32 not null, Name LaRge_String, Height float32, Gender Bool";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  // data
  std::vector<std::string> records = {"0\u001eTom\u001e180.3\u001e0",
                                      "1\u001eJane\u001e190.1\u001e1",
                                      "2\u001eJohn\u001e168.7\u001e0"};

  for (auto& record : records) {
    auto status = table->AppendRecord(record);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  EXPECT_EQ(table->GetSegmentCount(), 3);

  // Verify that segments are created with correct IDs based on the record
  // values
  for (size_t i = 0; i < records.size(); i++) {
    auto tokens = vdb::Tokenize(records[i], vdb::kRS);
    // Expected segment ID follows the new specification format:
    // {version}::{type}::comp::{column_names}::{json_values}::::{metadata}
    std::string expected_segment_id =
        "v1::value::single::id::[\"" + std::string(tokens[0]) + "\"]::::";

    // Find the segment with this ID
    auto segment = table->GetSegment(expected_segment_id);
    ASSERT_NE(segment, nullptr)
        << "Segment with ID " << expected_segment_id << " not found";
    ASSERT_EQ(tokens.size(), 4) << "Record should have 4 fields";
  }
}

TEST_F(TableTest, BatchAppendRecordsTest) {
  std::string table_schema_string =
      "segment_id int32 not null, id int32, Name String, Height float32, "
      "Attributes List[String]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "segment_id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  // Create 50 distinct segment IDs
  const int num_segments = 50;
  const int total_records = 5000;
  const int records_per_segment = total_records / num_segments;

  std::vector<std::string> names = {"Alice",  "Bob",   "Carol", "Dave",
                                    "Eve",    "Frank", "Grace", "Henry",
                                    "Isabel", "Jack"};
  std::vector<std::string> languages = {"C",          "C++",   "Python", "Java",
                                        "JavaScript", "Ruby",  "Go",     "Rust",
                                        "Swift",      "Kotlin"};

  // Instead of creating shuffled batches, create one batch per segment ID
  // This ensures each segment gets its own records
  for (int segment_idx = 0; segment_idx < num_segments; segment_idx++) {
    // Create arrays for each column
    arrow::Int32Builder segment_id_builder;
    arrow::Int32Builder id_builder;
    arrow::StringBuilder name_builder;
    arrow::FloatBuilder height_builder;
    arrow::ListBuilder list_builder(arrow::default_memory_pool(),
                                    std::make_shared<arrow::StringBuilder>());
    auto& value_builder =
        static_cast<arrow::StringBuilder&>(*list_builder.value_builder());

    // Add data for this segment
    for (int i = 0; i < records_per_segment; i++) {
      int record_idx = segment_idx * records_per_segment + i;
      ASSERT_OK(segment_id_builder.Append(segment_idx));
      ASSERT_OK(id_builder.Append(record_idx));
      ASSERT_OK(name_builder.Append(names[record_idx % names.size()]));
      ASSERT_OK(height_builder.Append(160.0f + (rand() % 300) / 10.0f));

      // Add list of languages
      int num_langs = 1 + (rand() % 3);
      ASSERT_OK(list_builder.Append());
      for (int j = 0; j < num_langs; j++) {
        ASSERT_OK(value_builder.Append(
            languages[(record_idx + j) % languages.size()]));
      }
    }

    // Create arrays from builders
    std::shared_ptr<arrow::Array> segment_id_array;
    ASSERT_OK(segment_id_builder.Finish(&segment_id_array));

    std::shared_ptr<arrow::Array> id_array;
    ASSERT_OK(id_builder.Finish(&id_array));

    std::shared_ptr<arrow::Array> name_array;
    ASSERT_OK(name_builder.Finish(&name_array));

    std::shared_ptr<arrow::Array> height_array;
    ASSERT_OK(height_builder.Finish(&height_array));

    std::shared_ptr<arrow::Array> attributes_array;
    ASSERT_OK(list_builder.Finish(&attributes_array));

    // Create record batch for this segment
    auto record_batch =
        arrow::RecordBatch::Make(schema, records_per_segment,
                                 {segment_id_array, id_array, name_array,
                                  height_array, attributes_array});

    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    for (int i = 0; i < 5; i++) {
      // insert multiple batches
      batches.push_back(record_batch);
    }
    // test segmentation by segment_id
    auto status = table->AppendRecords(batches);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  // Verify that segments are created with correct IDs
  EXPECT_EQ(table->GetSegmentCount(), num_segments);

  // Verify total number of records
  size_t total_count = 0;
  for (auto& [segment_id, segment] : table->GetSegments()) {
    total_count += segment->Size();
  }
  EXPECT_EQ(total_count, total_records * 5);

  // Verify that each segment has the expected number of records
  for (int i = 0; i < num_segments; i++) {
    // Expected segment ID follows the new specification format:
    // {version}::{type}::comp::{column_names}::{json_values}::::{metadata}
    std::string expected_segment_id =
        "v1::value::single::segment_id::[\"" + std::to_string(i) + "\"]::::";
    auto segment = table->GetSegment(expected_segment_id);
    ASSERT_NE(segment, nullptr)
        << "Segment with ID " << expected_segment_id << " not found";
    EXPECT_EQ(segment->Size(), records_per_segment * 5)
        << "Segment " << expected_segment_id << " has " << segment->Size()
        << " records instead of " << records_per_segment * 5;
  }

  // Make all segments inactive
  for (auto& [_, segment] : table->GetSegments()) {
    auto status = segment->MakeInactive();
    ASSERT_TRUE(status.ok()) << status.ToString();
  }
}

TEST_F(EmbeddingStoreTest, SingleFileWriteAndReadTest) {
  EmbeddingStore es(EmbeddingStoreDirectoryPath(), 0, 4);
  auto status = es.CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  std::vector<float> raw_embeddings(400);
  for (int i = 0; i < 400; ++i) {
    raw_embeddings[i] = i;
  }
  status = es.Write(raw_embeddings.data(), 0, 100);
  EXPECT_TRUE(status.ok()) << status.ToString();
  /* sequential read */
  {
    uint64_t starting_label = 0;
    uint64_t count = 100;
    ASSERT_OK_AND_ASSIGN(auto read_array,
                         es.ReadToArray(starting_label, count));
    EXPECT_TRUE(memcmp(raw_embeddings.data(),
                       read_array->values()->data()->GetValues<float>(1),
                       100 * 4 * sizeof(float)) == 0);
  }
  {
    uint64_t starting_label = 0;
    uint64_t count = 100;
    ASSERT_OK_AND_ASSIGN(auto read_buffer,
                         es.ReadToBuffer(starting_label, count));
    EXPECT_TRUE(memcmp(raw_embeddings.data(), read_buffer->data(),
                       100 * 4 * sizeof(float)) == 0);
  }

  /* random read */
  std::vector<uint64_t> labels = {
      LabelInfo::Build(0, 0, 4), LabelInfo::Build(0, 0, 9),
      LabelInfo::Build(0, 0, 10), LabelInfo::Build(0, 0, 24)};

  std::vector<float> expected_embeddings = {
      16.0f, 17.0f, 18.0f, 19.0f,  // record 4
      36.0f, 37.0f, 38.0f, 39.0f,  // record 9
      40.0f, 41.0f, 42.0f, 43.0f,  // record 10
      96.0f, 97.0f, 98.0f, 99.0f   // record 24
  };

  {
    ASSERT_OK_AND_ASSIGN(auto label_read_array,
                         es.ReadToArray(labels.data(), labels.size()));
    EXPECT_TRUE(memcmp(expected_embeddings.data(),
                       label_read_array->values()->data()->GetValues<float>(1),
                       4 * 4 * sizeof(float)) == 0)
        << "label_read_array: " << label_read_array->ToString() << std::endl;
  }
  {
    ASSERT_OK_AND_ASSIGN(auto label_read_buffer,
                         es.ReadToBuffer(labels.data(), labels.size()));
    EXPECT_TRUE(memcmp(expected_embeddings.data(), label_read_buffer->data(),
                       4 * 4 * sizeof(float)) == 0)
        << "label_read_buffer: " << label_read_buffer->ToString() << std::endl;
  }
}

TEST_F(EmbeddingStoreTest, MultiFileWriteAndReadTest) {
  EmbeddingStore es(EmbeddingStoreDirectoryPath(), 0, 4);
  auto status = es.CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  std::vector<float> raw_embeddings(400);
  for (int i = 0; i < 400; ++i) {
    raw_embeddings[i] = i;
  }
  uint16_t set_count = 10;
  for (int i = 0; i < set_count; i++) {
    uint64_t label = LabelInfo::Build(0, i, 0);
    for (int j = 0; j < 5; j++) {
      auto buffer = raw_embeddings.data() + 20 * j * es.Dimension();
      auto status = es.Write(buffer, label, 20);
      EXPECT_TRUE(status.ok()) << status.ToString();
      label += 20;
    }
  }

  /* sequential read */
  {
    for (int i = 0; i < set_count; i++) {
      uint64_t label = LabelInfo::Build(0, i, 0);
      ASSERT_OK_AND_ASSIGN(auto read_array, es.ReadToArray(label, 100));
      EXPECT_TRUE(memcmp(raw_embeddings.data(),
                         read_array->values()->data()->GetValues<float>(1),
                         100 * es.Dimension() * sizeof(float)) == 0)
          << "read_array->ToString(): " << read_array->ToString();
    }
  }
  {
    for (int i = 0; i < set_count; i++) {
      uint64_t label = LabelInfo::Build(0, i, 0);
      ASSERT_OK_AND_ASSIGN(auto read_buffer, es.ReadToBuffer(label, 100));
      EXPECT_TRUE(memcmp(raw_embeddings.data(), read_buffer->data(),
                         100 * es.Dimension() * sizeof(float)) == 0)
          << "read_buffer->ToString(): " << read_buffer->ToString();
    }
  }

  /* random read */
  std::vector<uint64_t> labels = {
      LabelInfo::Build(0, 0, 3),   // record 3: [12, 13, 14, 15]
      LabelInfo::Build(0, 1, 15),  // record 15: [60, 61, 62, 63]
      LabelInfo::Build(0, 2, 8),   // record 8: [32, 33, 34, 35]
      LabelInfo::Build(0, 3, 21),  // record 21: [84, 85, 86, 87]
      LabelInfo::Build(0, 4, 5),   // record 5: [20, 21, 22, 23]
      LabelInfo::Build(0, 5, 17),  // record 17: [68, 69, 70, 71]
      LabelInfo::Build(0, 6, 2),   // record 2: [8, 9, 10, 11]
      LabelInfo::Build(0, 7, 12),  // record 12: [48, 49, 50, 51]
      LabelInfo::Build(0, 8, 7),   // record 7: [28, 29, 30, 31]
      LabelInfo::Build(0, 9, 11)   // record 11: [44, 45, 46, 47]
  };
  std::vector<float> expected_embeddings = {
      12.0f, 13.0f, 14.0f, 15.0f,  // set 0, record 3
      60.0f, 61.0f, 62.0f, 63.0f,  // set 1, record 15
      32.0f, 33.0f, 34.0f, 35.0f,  // set 2, record 8
      84.0f, 85.0f, 86.0f, 87.0f,  // set 3, record 21
      20.0f, 21.0f, 22.0f, 23.0f,  // set 4, record 5
      68.0f, 69.0f, 70.0f, 71.0f,  // set 5, record 17
      8.0f,  9.0f,  10.0f, 11.0f,  // set 6, record 2
      48.0f, 49.0f, 50.0f, 51.0f,  // set 7, record 12
      28.0f, 29.0f, 30.0f, 31.0f,  // set 8, record 7
      44.0f, 45.0f, 46.0f, 47.0f   // set 9, record 11
  };
  {
    ASSERT_OK_AND_ASSIGN(auto label_read_array,
                         es.ReadToArray(labels.data(), labels.size()));
    EXPECT_TRUE(memcmp(expected_embeddings.data(),
                       label_read_array->values()->data()->GetValues<float>(1),
                       4 * es.Dimension() * sizeof(float)) == 0)
        << "label_read_array: " << label_read_array->ToString() << std::endl;
  }
  {
    ASSERT_OK_AND_ASSIGN(auto label_read_buffer,
                         es.ReadToBuffer(labels.data(), labels.size()));
    EXPECT_TRUE(memcmp(expected_embeddings.data(), label_read_buffer->data(),
                       4 * es.Dimension() * sizeof(float)) == 0)
        << "label_read_array: " << label_read_buffer->ToString() << std::endl;
  }

  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, 4);
  status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  /* async random read with distance calculation */
  auto index = std::make_shared<vdb::Hnsw>(vdb::DistanceSpace::kL2, 4, 100, 3,
                                           1000, embedding_store);
  auto dist_func = index->GetDistFunc();
  auto dist_func_param = index->GetDistFuncParam();
  std::vector<float> distances(labels.size());
  float query_embedding[4] = {1.1f, 1.2f, 1.3f, 1.4f};
  for (size_t i = 0; i < labels.size(); i++) {
    distances[i] = dist_func(
        query_embedding, expected_embeddings.data() + 4 * i, dist_func_param);
  }
  {
    ASSERT_OK_AND_ASSIGN(auto distance_array,
                         es.ReadAndCalculateDistances(
                             labels.data(), labels.size(), query_embedding,
                             dist_func, dist_func_param));
    EXPECT_TRUE(memcmp(distances.data(), distance_array->values()->data(),
                       distances.size() * sizeof(float)) == 0)
        << "distance_array: " << distance_array->ToString() << std::endl;
  }
}

TEST_F(EmbeddingStoreTest, ComplexTest) {
  // 4 columns, 3 segments
  const int num_columns = 4;
  const int num_segments = 3;
  const int dimension = 4;

  // create embedding store for each column
  std::vector<std::unique_ptr<EmbeddingStore>> stores;
  for (int col = 0; col < num_columns; ++col) {
    auto store = std::make_unique<EmbeddingStore>(EmbeddingStoreDirectoryPath(),
                                                  col, dimension);
    stores.push_back(std::move(store));
  }

  // create directory for each segment and column
  for (int seg = 0; seg < num_segments; ++seg) {
    for (int col = 0; col < num_columns; ++col) {
      auto status = stores[col]->CreateSegmentAndColumnDirectory(seg);
      ASSERT_TRUE(status.ok()) << "Failed to create directory for segment "
                               << seg << " column " << col;
    }
  }

  // create different data for each segment
  std::vector<std::vector<std::vector<float>>> column_segment_embeddings(
      num_segments, std::vector<std::vector<float>>(num_columns));
  for (int seg = 0; seg < num_segments; ++seg) {
    for (int col = 0; col < num_columns; ++col) {
      column_segment_embeddings[seg][col].resize(400);
      for (int i = 0; i < 400; ++i) {
        column_segment_embeddings[seg][col][i] =
            i + seg * 1000 +
            col * 10000;  // different offset for each segment and column
      }
    }
  }

  uint16_t set_count = 10;
  // write data for each segment and column
  for (int seg = 0; seg < num_segments; ++seg) {
    for (int col = 0; col < num_columns; ++col) {
      for (int i = 0; i < set_count; i++) {
        uint64_t label = LabelInfo::Build(seg, i, 0);
        for (int j = 0; j < 5; j++) {
          auto buffer =
              column_segment_embeddings[seg][col].data() + 20 * j * dimension;
          auto status = stores[col]->Write(buffer, label, 20);
          EXPECT_TRUE(status.ok())
              << "Write failed for segment " << seg << " column " << col
              << " set " << i << " batch " << j << ": " << status.ToString();
          label += 20;
        }
      }
    }
  }

  /* sequential read - test for all segments and columns */
  for (int seg = 0; seg < num_segments; ++seg) {
    for (int col = 0; col < num_columns; ++col) {
      for (int i = 0; i < set_count; i++) {
        uint64_t label = LabelInfo::Build(seg, i, 0);
        ASSERT_OK_AND_ASSIGN(auto read_array,
                             stores[col]->ReadToArray(label, 100));
        EXPECT_TRUE(memcmp(column_segment_embeddings[seg][col].data(),
                           read_array->values()->data()->GetValues<float>(1),
                           100 * dimension * sizeof(float)) == 0)
            << "ReadToArray failed for segment " << seg << " column " << col
            << " set " << i << ": " << read_array->ToString();
      }
    }
  }

  for (int seg = 0; seg < num_segments; ++seg) {
    for (int col = 0; col < num_columns; ++col) {
      for (int i = 0; i < set_count; i++) {
        uint64_t label = LabelInfo::Build(seg, i, 0);
        ASSERT_OK_AND_ASSIGN(auto read_buffer,
                             stores[col]->ReadToBuffer(label, 100));
        EXPECT_TRUE(memcmp(column_segment_embeddings[seg][col].data(),
                           read_buffer->data(),
                           100 * dimension * sizeof(float)) == 0)
            << "ReadToBuffer failed for segment " << seg << " column " << col
            << " set " << i << ": " << read_buffer->ToString();
      }
    }
  }

  /* random read - test for random records in each segment */
  for (int seg = 0; seg < num_segments; ++seg) {
    for (int col = 0; col < num_columns; ++col) {
      std::vector<uint64_t> labels = {
          LabelInfo::Build(seg, 0,
                           3),  // record 3: [12, 13, 14, 15] + seg*1000
          LabelInfo::Build(seg, 1,
                           15),  // record 15: [60, 61, 62, 63] + seg*1000
          LabelInfo::Build(seg, 2,
                           8),  // record 8: [32, 33, 34, 35] + seg*1000
          LabelInfo::Build(seg, 3,
                           21),  // record 21: [84, 85, 86, 87] + seg*1000
          LabelInfo::Build(seg, 4,
                           5),  // record 5: [20, 21, 22, 23] + seg*1000
          LabelInfo::Build(seg, 5,
                           17),  // record 17: [68, 69, 70, 71] + seg*1000
          LabelInfo::Build(seg, 6,
                           2),  // record 2: [8, 9, 10, 11] + seg*1000
          LabelInfo::Build(seg, 7,
                           12),  // record 12: [48, 49, 50, 51] + seg*1000
          LabelInfo::Build(seg, 8,
                           7),  // record 7: [28, 29, 30, 31] + seg*1000
          LabelInfo::Build(seg, 9,
                           11)  // record 11: [44, 45, 46, 47] + seg*1000
      };
      // expected embeddings considering the offset of each segment
      std::vector<float> expected_embeddings = {
          12.0f + seg * 1000 + col * 10000, 13.0f + seg * 1000 + col * 10000,
          14.0f + seg * 1000 + col * 10000, 15.0f + seg * 1000 + col * 10000,
          60.0f + seg * 1000 + col * 10000, 61.0f + seg * 1000 + col * 10000,
          62.0f + seg * 1000 + col * 10000, 63.0f + seg * 1000 + col * 10000,
          32.0f + seg * 1000 + col * 10000, 33.0f + seg * 1000 + col * 10000,
          34.0f + seg * 1000 + col * 10000, 35.0f + seg * 1000 + col * 10000,
          84.0f + seg * 1000 + col * 10000, 85.0f + seg * 1000 + col * 10000,
          86.0f + seg * 1000 + col * 10000, 87.0f + seg * 1000 + col * 10000,
          20.0f + seg * 1000 + col * 10000, 21.0f + seg * 1000 + col * 10000,
          22.0f + seg * 1000 + col * 10000, 23.0f + seg * 1000 + col * 10000,
          68.0f + seg * 1000 + col * 10000, 69.0f + seg * 1000 + col * 10000,
          70.0f + seg * 1000 + col * 10000, 71.0f + seg * 1000 + col * 10000,
          8.0f + seg * 1000 + col * 10000,  9.0f + seg * 1000 + col * 10000,
          10.0f + seg * 1000 + col * 10000, 11.0f + seg * 1000 + col * 10000,
          48.0f + seg * 1000 + col * 10000, 49.0f + seg * 1000 + col * 10000,
          50.0f + seg * 1000 + col * 10000, 51.0f + seg * 1000 + col * 10000,
          28.0f + seg * 1000 + col * 10000, 29.0f + seg * 1000 + col * 10000,
          30.0f + seg * 1000 + col * 10000, 31.0f + seg * 1000 + col * 10000,
          44.0f + seg * 1000 + col * 10000, 45.0f + seg * 1000 + col * 10000,
          46.0f + seg * 1000 + col * 10000, 47.0f + seg * 1000 + col * 10000};

      {
        ASSERT_OK_AND_ASSIGN(
            auto label_read_array,
            stores[col]->ReadToArray(labels.data(), labels.size()));
        EXPECT_TRUE(
            memcmp(expected_embeddings.data(),
                   label_read_array->values()->data()->GetValues<float>(1),
                   labels.size() * dimension * sizeof(float)) == 0)
            << "Random ReadToArray failed for segment " << seg << " column "
            << col << ": " << label_read_array->ToString();
      }
      {
        ASSERT_OK_AND_ASSIGN(
            auto label_read_buffer,
            stores[col]->ReadToBuffer(labels.data(), labels.size()));
        EXPECT_TRUE(memcmp(expected_embeddings.data(),
                           label_read_buffer->data(),
                           labels.size() * dimension * sizeof(float)) == 0)
            << "Random ReadToBuffer failed for segment " << seg << " column "
            << col << ": " << label_read_buffer->ToString();
      }
    }
  }

  /* async random read with distance calculation - test for first column */
  auto embedding_store = std::make_shared<EmbeddingStore>(
      EmbeddingStoreDirectoryPath(), 0, dimension);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());

  auto index = std::make_shared<vdb::Hnsw>(vdb::DistanceSpace::kL2, dimension,
                                           100, 3, 1000, embedding_store);
  auto dist_func = index->GetDistFunc();
  auto dist_func_param = index->GetDistFuncParam();

  // test for distance calculation in first segment and first column
  std::vector<uint64_t> labels = {
      LabelInfo::Build(0, 0, 3),   // record 3: [12, 13, 14, 15]
      LabelInfo::Build(0, 1, 15),  // record 15: [60, 61, 62, 63]
      LabelInfo::Build(0, 2, 8),   // record 8: [32, 33, 34, 35]
      LabelInfo::Build(0, 3, 21),  // record 21: [84, 85, 86, 87]
      LabelInfo::Build(0, 4, 5),   // record 5: [20, 21, 22, 23]
      LabelInfo::Build(0, 5, 17),  // record 17: [68, 69, 70, 71]
      LabelInfo::Build(0, 6, 2),   // record 2: [8, 9, 10, 11]
      LabelInfo::Build(0, 7, 12),  // record 12: [48, 49, 50, 51]
      LabelInfo::Build(0, 8, 7),   // record 7: [28, 29, 30, 31]
      LabelInfo::Build(0, 9, 11)   // record 11: [44, 45, 46, 47]
  };

  std::vector<float> expected_embeddings = {
      12.0f, 13.0f, 14.0f, 15.0f,  // set 0, record 3
      60.0f, 61.0f, 62.0f, 63.0f,  // set 1, record 15
      32.0f, 33.0f, 34.0f, 35.0f,  // set 2, record 8
      84.0f, 85.0f, 86.0f, 87.0f,  // set 3, record 21
      20.0f, 21.0f, 22.0f, 23.0f,  // set 4, record 5
      68.0f, 69.0f, 70.0f, 71.0f,  // set 5, record 17
      8.0f,  9.0f,  10.0f, 11.0f,  // set 6, record 2
      48.0f, 49.0f, 50.0f, 51.0f,  // set 7, record 12
      28.0f, 29.0f, 30.0f, 31.0f,  // set 8, record 7
      44.0f, 45.0f, 46.0f, 47.0f   // set 9, record 11
  };

  std::vector<float> distances(labels.size());
  float query_embedding[4] = {1.1f, 1.2f, 1.3f, 1.4f};
  for (size_t i = 0; i < labels.size(); i++) {
    distances[i] =
        dist_func(query_embedding, expected_embeddings.data() + dimension * i,
                  dist_func_param);
  }

  {
    ASSERT_OK_AND_ASSIGN(auto distance_array,
                         stores[0]->ReadAndCalculateDistances(
                             labels.data(), labels.size(), query_embedding,
                             dist_func, dist_func_param));
    EXPECT_TRUE(memcmp(distances.data(), distance_array->values()->data(),
                       distances.size() * sizeof(float)) == 0)
        << "Distance calculation failed: " << distance_array->ToString();
  }
}

TEST_F(SchemaTest, SchemaIsolationTest) {
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
      ann_column_id, "Hnsw", "L2Space", ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}, {"max_threads", "1"}});
  tests::TableWrapper::AddMetadata(table, add_metadata);
  auto external_schema = table->GetSchema();
  auto internal_schema = table->GetInternalSchema();
  auto extended_schema = table->GetExtendedSchema();

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
             "External Schema: %s", external_schema->ToString().c_str());
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
             "Internal Schema: %s", internal_schema->ToString().c_str());

  /* External Schema Field Check */
  EXPECT_EQ(external_schema->field(0)->name(), "id");
  EXPECT_EQ(external_schema->field(0)->type()->id(), arrow::Type::UINT32);
  EXPECT_EQ(external_schema->field(1)->name(), "name");
  EXPECT_EQ(external_schema->field(1)->type()->id(), arrow::Type::STRING);
  EXPECT_EQ(external_schema->field(2)->name(), "feature");
  EXPECT_EQ(external_schema->field(2)->type()->id(),
            arrow::Type::FIXED_SIZE_LIST);

  /* Internal Schema Field Check */
  EXPECT_EQ(internal_schema->field(0)->name(), "id");
  EXPECT_EQ(internal_schema->field(0)->type()->id(), arrow::Type::UINT32);
  EXPECT_EQ(internal_schema->field(1)->name(), "name");
  EXPECT_EQ(internal_schema->field(1)->type()->id(), arrow::Type::STRING);
  EXPECT_EQ(internal_schema->field(2)->name(), "feature");
  EXPECT_EQ(internal_schema->field(2)->type()->id(), arrow::Type::UINT64);
  EXPECT_EQ(internal_schema->field(3)->name(), vdb::kDeletedFlagColumn);
  EXPECT_EQ(internal_schema->field(3)->type()->id(), arrow::Type::BOOL);

  /* Extended Schema Field Check */
  EXPECT_EQ(extended_schema->field(0)->name(), "id");
  EXPECT_EQ(extended_schema->field(0)->type()->id(), arrow::Type::UINT32);
  EXPECT_EQ(extended_schema->field(1)->name(), "name");
  EXPECT_EQ(extended_schema->field(1)->type()->id(), arrow::Type::STRING);
  EXPECT_EQ(extended_schema->field(2)->name(), "feature");
  EXPECT_EQ(extended_schema->field(2)->type()->id(),
            arrow::Type::FIXED_SIZE_LIST);
  EXPECT_EQ(extended_schema->field(3)->name(), vdb::kDeletedFlagColumn);
  EXPECT_EQ(extended_schema->field(3)->type()->id(), arrow::Type::BOOL);
}

TEST_F(NumberKeyFlatArrayTest, UniqueScanTest) {
  auto setup_base_nodes = []() {
    auto flat_array = std::make_shared<vdb::NumberKeyFlatArray>();
    flat_array->Initialize(20, 60);
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    for (uint32_t i = 20; i < 30; i++) pairs.push_back({i, i});
    for (uint32_t i = 50; i < 60; i++) pairs.push_back({i, i});
    flat_array->Insert(pairs);
    return flat_array;
  };

  /* [start, end) start is included, end is excluded */
  struct Range {
    uint32_t start, end;
  };
  std::vector<Range> test_ranges = {
      {20, 30}, {50, 60}, {20, 60}, {20, 52}, {20, 50}, {20, 45},
      {28, 60}, {28, 52}, {28, 50}, {28, 45}, {30, 60}, {30, 52},
      {30, 50}, {30, 45}, {35, 60}, {35, 52}, {35, 50}, {35, 45}};

  for (const auto& r : test_ranges) {
    auto flat_array = setup_base_nodes();
    std::vector<std::pair<uint32_t, uint32_t>> insert_pairs;
    for (uint32_t i = r.start; i < r.end; i++) insert_pairs.push_back({i, i});
    flat_array->Insert(insert_pairs);

    auto result =
        flat_array->UniqueScan(insert_pairs, vdb::ContextType::kNoNeed);
    ASSERT_TRUE(result.ok());

    std::vector<uint32_t> expected_offsets;
    for (uint32_t i = r.start; i < r.end; i++) {
      if ((20 <= i && i < 30) || (50 <= i && i < 60))
        expected_offsets.push_back(i);
      else if (std::find_if(insert_pairs.begin(), insert_pairs.end(),
                            [&](const std::pair<uint32_t, uint32_t>& p) {
                              return p.first == i;
                            }) != insert_pairs.end()) {
        expected_offsets.push_back(i);
      }
    }
    EXPECT_EQ(result->offsets, expected_offsets)
        << "Range [" << r.start << "," << r.end << ")";
  }
}

TEST_F(NumberKeyFlatArrayTest, InsertTest) {
  // Base setup: Fix front/back nodes to [20,30), [50,60)
  auto setup_base_nodes = []() {
    auto flat_array = std::make_shared<vdb::NumberKeyFlatArray>();
    // Insert ranges [20,30) and [50,60)
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    for (uint32_t i = 20; i < 30; i++) {
      pairs.push_back({i, i});  // offset = number
    }
    for (uint32_t i = 50; i < 60; i++) {
      pairs.push_back({i, i});  // offset = number
    }
    flat_array->Insert(pairs);
    return flat_array;
  };

  // 1. Full-Full: [20,60) - Complete merge
  auto flat_array = setup_base_nodes();
  std::vector<std::pair<uint32_t, uint32_t>> insert_pairs;
  for (uint32_t i = 20; i < 60; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 2. Full-Partial: [20,52) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 20; i < 52; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 3. Full-Adjacent: [20,50) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 20; i < 50; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 4. Full-Disjoint: [20,45) - Merge with front only
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 20; i < 45; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 44], [50, 59]");

  // 5. Partial-Full: [28,60) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 28; i < 60; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 6. Partial-Partial: [28,52) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 28; i < 52; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 7. Partial-Adjacent: [28,50) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 28; i < 50; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 8. Partial-Disjoint: [28,45) - Merge with front only
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 28; i < 45; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 44], [50, 59]");

  // 9. Adjacent-Full: [30,60) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 30; i < 60; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 10. Adjacent-Partial: [30,52) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 30; i < 52; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 11. Adjacent-Adjacent: [30,50) - Complete merge
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 30; i < 50; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 59]");

  // 12. Adjacent-Disjoint: [30,45) - Merge with front only
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 30; i < 45; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 44], [50, 59]");

  // 13. Disjoint-Full: [35,60) - Merge with back only
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 35; i < 60; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 29], [35, 59]");

  // 14. Disjoint-Partial: [35,52) - Merge with back only
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 35; i < 52; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 29], [35, 59]");

  // 15. Disjoint-Adjacent: [35,50) - Merge with back only
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 35; i < 50; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 29], [35, 59]");

  // 16. Disjoint-Disjoint: [35,45) - Add as independent range
  flat_array = setup_base_nodes();
  insert_pairs.clear();
  for (uint32_t i = 35; i < 45; i++) {
    insert_pairs.push_back({i, i});
  }
  flat_array->Insert(insert_pairs);
  EXPECT_EQ(flat_array->ToString(), "[20, 29], [35, 44], [50, 59]");
}

TEST_F(NumberKeyFlatArrayTest, DeleteTest) {
  auto setup_base_nodes = []() {
    auto flat_array = std::make_shared<vdb::NumberKeyFlatArray>();
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    for (uint32_t i = 20; i < 30; i++) {
      pairs.push_back({i, i});  // offset = number
    }
    for (uint32_t i = 50; i < 60; i++) {
      pairs.push_back({i, i});  // offset = number
    }
    flat_array->Insert(pairs);
    return flat_array;
  };

  // 1. delete [25,33)
  {
    auto flat_array = setup_base_nodes();
    std::vector<std::pair<uint32_t, uint32_t>> del_pairs;
    for (uint32_t i = 25; i < 33; i++) del_pairs.push_back({i, i});
    flat_array->Delete(del_pairs);
    EXPECT_EQ(flat_array->ToString(), "[20, 24], [50, 59]");
  }

  // 2. delete [21, 23, 25, 27, 29, 51, 53, 55]
  {
    auto flat_array = setup_base_nodes();
    std::vector<uint32_t> nums = {21, 23, 25, 27, 29, 51, 53, 55};
    std::vector<std::pair<uint32_t, uint32_t>> del_pairs;
    for (auto n : nums) del_pairs.push_back({n, n});
    flat_array->Delete(del_pairs);
    EXPECT_EQ(flat_array->ToString(),
              "20, 22, 24, 26, 28, 50, 52, 54, [56, 59]");
  }

  // 3. delete [25, 27]
  {
    auto flat_array = setup_base_nodes();
    std::vector<std::pair<uint32_t, uint32_t>> del_pairs;
    del_pairs.push_back({25, 25});
    del_pairs.push_back({27, 27});
    flat_array->Delete(del_pairs);
    EXPECT_EQ(flat_array->ToString(), "[20, 24], 26, [28, 29], [50, 59]");
  }

  // 4. delete [22, 24, 26]
  {
    auto flat_array = setup_base_nodes();
    std::vector<std::pair<uint32_t, uint32_t>> del_pairs;
    del_pairs.push_back({22, 22});
    del_pairs.push_back({24, 24});
    del_pairs.push_back({26, 26});
    flat_array->Delete(del_pairs);
    EXPECT_EQ(flat_array->ToString(), "[20, 21], 23, 25, [27, 29], [50, 59]");
  }

  // 5. delete all numbers
  {
    auto flat_array = setup_base_nodes();
    std::vector<std::pair<uint32_t, uint32_t>> del_pairs;
    for (uint32_t i = 20; i < 30; i++) del_pairs.push_back({i, i});
    for (uint32_t i = 50; i < 60; i++) del_pairs.push_back({i, i});
    flat_array->Delete(del_pairs);
    EXPECT_EQ(flat_array->ToString(), "");
  }

  // 6. delete random numbers between 20 and 59
  {
    auto flat_array = setup_base_nodes();
    std::vector<uint32_t> all_nums;
    for (uint32_t i = 20; i < 30; i++) all_nums.push_back(i);
    for (uint32_t i = 50; i < 60; i++) all_nums.push_back(i);
    std::vector<uint32_t> nums_to_delete;
    std::sample(all_nums.begin(), all_nums.end(),
                std::back_inserter(nums_to_delete), 15,
                std::mt19937{std::random_device{}()});
    std::sort(nums_to_delete.begin(), nums_to_delete.end());
    std::vector<std::pair<uint32_t, uint32_t>> del_pairs;
    for (auto n : nums_to_delete) del_pairs.push_back({n, n});
    flat_array->Delete(del_pairs);
    std::vector<uint32_t> remaining;
    std::string s = flat_array->ToString();
    std::regex rgx("\\[([0-9]+), ([0-9]+)\\]|([0-9]+)");
    std::smatch match;
    std::string::const_iterator searchStart(s.cbegin());
    while (std::regex_search(searchStart, s.cend(), match, rgx)) {
      if (match[1].matched && match[2].matched) {
        uint32_t a = std::stoi(match[1]);
        uint32_t b = std::stoi(match[2]);
        for (uint32_t i = a; i <= b; i++) remaining.push_back(i);
      } else if (match[3].matched) {
        remaining.push_back(std::stoi(match[3]));
      }
      searchStart = match.suffix().first;
    }
    std::vector<uint32_t> expected;
    std::set<uint32_t> deleted(nums_to_delete.begin(), nums_to_delete.end());
    for (auto n : all_nums) {
      if (deleted.find(n) == deleted.end()) expected.push_back(n);
    }
    EXPECT_EQ(remaining, expected);
  }
}

TEST_F(NumberKeyFlatArrayTest, ComprehensiveOverlapRelationTest) {
  auto flat_array = std::make_shared<vdb::NumberKeyFlatArray>();
  // [20,30), [50,60)
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  for (uint32_t i = 20; i < 30; i++) pairs.push_back({i, i});
  for (uint32_t i = 50; i < 60; i++) pairs.push_back({i, i});
  flat_array->Insert(pairs);

  struct Query {
    uint32_t start, end;
    std::vector<uint32_t> expected;
  };
  std::vector<Query> queries = {
      {20, 60, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59}},
      {20, 55, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54}},
      {20, 49, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
      {20, 45, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
      {25, 60, {25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}},
      {25, 55, {25, 26, 27, 28, 29, 50, 51, 52, 53, 54}},
      {25, 49, {25, 26, 27, 28, 29}},
      {25, 45, {25, 26, 27, 28, 29}},
      {31, 60, {50, 51, 52, 53, 54, 55, 56, 57, 58, 59}},
      {31, 55, {50, 51, 52, 53, 54}},
      {31, 49, {}},
      {31, 45, {}},
      {35, 60, {50, 51, 52, 53, 54, 55, 56, 57, 58, 59}},
      {35, 55, {50, 51, 52, 53, 54}},
      {35, 49, {}},
      {35, 45, {}}};
  for (const auto& q : queries) {
    std::vector<std::pair<uint32_t, uint32_t>> query_pairs;
    for (uint32_t i = q.start; i < q.end; i++) query_pairs.push_back({i, i});
    auto result =
        flat_array->UniqueScan(query_pairs, vdb::ContextType::kNoNeed);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->offsets, q.expected)
        << "Query [" << q.start << "," << q.end << ")";
  }
}

TEST_F(NumberKeyFlatArrayTest, MergeManyDisjointRangesRandomly) {
  auto flat_array = std::make_shared<vdb::NumberKeyFlatArray>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  // Insert 1000 disjoint ranges (each range size 10, gap size 5)
  const int NUM_RANGES = 1000;
  const int RANGE_SIZE = 10;
  const int GAP_SIZE = 5;

  std::vector<std::pair<uint32_t, uint32_t>> all_pairs;
  std::vector<std::pair<int, int>> ranges;
  for (int i = 0; i < NUM_RANGES; i++) {
    int start = i * (RANGE_SIZE + GAP_SIZE);
    int end = start + RANGE_SIZE;
    for (int n = start; n < end; n++)
      all_pairs.push_back({(uint32_t)n, (uint32_t)n});
    ranges.push_back({start, end});
  }
  flat_array->Insert(all_pairs);

  // Connect each gap randomly with either adjacent or partial overlap
  for (size_t i = 0; i < ranges.size() - 1; i++) {
    bool front_adjacent = dis(gen) == 1;
    bool back_adjacent = dis(gen) == 1;
    int gap_start = front_adjacent ? ranges[i].second : ranges[i].second - 2;
    int gap_end = back_adjacent ? ranges[i + 1].first : ranges[i + 1].first + 2;
    std::vector<std::pair<uint32_t, uint32_t>> gap_pairs;
    for (int n = gap_start; n < gap_end; n++)
      gap_pairs.push_back({(uint32_t)n, (uint32_t)n});
    flat_array->Insert(gap_pairs);
  }

  std::string result = flat_array->ToString();
  int expected_start = ranges.front().first;
  int expected_end = ranges.back().second - 1;
  std::stringstream ss;
  ss << "[" << expected_start << ", " << expected_end << "]";
  EXPECT_EQ(result, ss.str());
}

TEST_F(FileKeyMapTest, InsertAndDelete) {
  std::vector<std::string> files = {"a.txt", "b.txt", "c.txt", "d.txt",
                                    "e.txt"};
  vdb::FileKeyMap file_key_map;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (const auto& file : files) {
    std::vector<std::pair<uint32_t, uint32_t>> init_pairs;
    for (uint32_t i = 20; i < 30; i++) init_pairs.push_back({i, i});
    for (uint32_t i = 50; i < 60; i++) init_pairs.push_back({i, i});
    file_key_map.Insert(file, init_pairs);
  }

  for (const auto& file : files) {
    std::vector<uint32_t> all_nums(100);
    std::iota(all_nums.begin(), all_nums.end(), 1);
    std::shuffle(all_nums.begin(), all_nums.end(), gen);
    std::set<uint32_t> insert_set(all_nums.begin(), all_nums.begin() + 40);
    std::vector<std::pair<uint32_t, uint32_t>> insert_pairs;
    for (auto n : insert_set) insert_pairs.push_back({n, n});
    file_key_map.Insert(file, insert_pairs);

    std::vector<uint32_t> delete_candidates(100);
    std::iota(delete_candidates.begin(), delete_candidates.end(), 1);
    std::shuffle(delete_candidates.begin(), delete_candidates.end(), gen);
    std::set<uint32_t> delete_set(delete_candidates.begin(),
                                  delete_candidates.begin() + 60);
    std::vector<std::pair<uint32_t, uint32_t>> delete_pairs;
    for (auto n : delete_set) delete_pairs.push_back({n, n});
    file_key_map.Delete(file, delete_pairs);

    std::set<uint32_t> current;
    for (uint32_t i = 20; i < 30; i++) current.insert(i);
    for (uint32_t i = 50; i < 60; i++) current.insert(i);
    current.insert(insert_set.begin(), insert_set.end());
    std::vector<uint32_t> expected;
    for (auto n : current) {
      if (delete_set.find(n) == delete_set.end()) expected.push_back(n);
    }

    std::string s = file_key_map.ToString();
    std::istringstream iss(s);
    std::string line;
    std::string file_line;
    while (std::getline(iss, line)) {
      if (line.find(file + ":") == 0) {
        file_line = line.substr(file.size() + 2);
        break;
      }
    }
    std::vector<uint32_t> actual;
    std::regex rgx("\\[([0-9]+), ([0-9]+)\\]|([0-9]+)");
    std::smatch match;
    std::string::const_iterator searchStart(file_line.cbegin());
    while (std::regex_search(searchStart, file_line.cend(), match, rgx)) {
      if (match[1].matched && match[2].matched) {
        uint32_t a = std::stoi(match[1]);
        uint32_t b = std::stoi(match[2]);
        for (uint32_t i = a; i <= b; i++) actual.push_back(i);
      } else if (match[3].matched) {
        actual.push_back(std::stoi(match[3]));
      }
      searchStart = match.suffix().first;
    }
    EXPECT_EQ(actual, expected) << file << " remaining values mismatch";
    EXPECT_TRUE(std::is_sorted(actual.begin(), actual.end()));
  }
}

TEST_F(IndexInfoTest, ParseIndexInfo) {
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmentation_info_str =
        R"({"segment_type": "value", "segment_keys": ["id"], "segment_key_composition_type": "single"})";
    std::string index_info_str =
        R"([{"column_id": "3", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmentation_info", segmentation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto maybe_index_infos = index_info_builder.Build();
    EXPECT_TRUE(maybe_index_infos.ok());
    auto index_infos = maybe_index_infos.ValueOrDie();
    EXPECT_EQ(index_infos->size(), 1);
    for (auto& index_info : *index_infos) {
      auto field_type = schema->field(index_info.GetColumnId())->type();
      EXPECT_EQ(field_type->id(), arrow::Type::FIXED_SIZE_LIST);
      auto fixed_size_list_type =
          std::static_pointer_cast<arrow::FixedSizeListType>(field_type);
      auto value_type = fixed_size_list_type->value_type();
      EXPECT_EQ(value_type->id(), arrow::Type::FLOAT);
    }
  }
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column_id": "2", "index_type": "Hnsw", "parameters": {"space": "L1Space", "ef_construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto index_infos = index_info_builder.Build();
    EXPECT_FALSE(index_infos.ok());
  }
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column": "2", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto index_infos = index_info_builder.Build();
    EXPECT_FALSE(index_infos.ok());
  }
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column_id": "2", "type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto index_infos = index_info_builder.Build();
    EXPECT_FALSE(index_infos.ok());
  }
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column_id": "2", "index_type": "Hnsw", "parameters": {"spac": "L2Space", "ef_construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto index_infos = index_info_builder.Build();
    EXPECT_FALSE(index_infos.ok());
  }
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column_id": "2", "index_type": "Hnsw", "parameters": {"space": "L2Space", "construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto index_infos = index_info_builder.Build();
    EXPECT_FALSE(index_infos.ok());
  }
  {
    std::string table_schema_string =
        "id INT32 not null, Name String, Attributes List[  String  ], Vector "
        "Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column_id": "2", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M_": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto index_infos = index_info_builder.Build();
    EXPECT_FALSE(index_infos.ok());
  }
  {
    std::string table_schema_string =
        "Vector0 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector1 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector2 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector3 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector4 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector5 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector6 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector7 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector8 Fixed_Size_list[  1024,    floAt32 ], "
        "Vector9 Fixed_Size_list[  1024,    floAt32 ]";
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    std::string table_name = "test_table";
    std::string segmetation_info_str =
        R"({"segment_type": "value", "segment_by": ["id"]})";
    std::string index_info_str =
        R"([{"column_id": "0", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "1", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "2", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "3", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "4", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "5", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "6", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "7", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "8", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}, )"
        R"({"column_id": "9", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "100", "M": "2"}}])";
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>(
            {{"segmetation_info", segmetation_info_str},
             {"index_info", index_info_str},
             {"table name", table_name},
             {"active_set_size_limit",
              std::to_string(server.vdb_active_set_size_limit)}}));
    schema = schema->WithMetadata(metadata);

    IndexInfoBuilder index_info_builder;
    std::string index_info_json_str =
        schema->metadata()->Get("index_info").ValueOr("");
    EXPECT_NE(index_info_json_str, "");

    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    auto maybe_index_infos = index_info_builder.Build();
    EXPECT_TRUE(maybe_index_infos.ok());
    auto index_infos = maybe_index_infos.ValueOrDie();
    EXPECT_EQ(index_infos->size(), 10);
    for (auto& index_info : *index_infos) {
      auto field_type = schema->field(index_info.GetColumnId())->type();
      EXPECT_EQ(field_type->id(), arrow::Type::FIXED_SIZE_LIST);
      auto fixed_size_list_type =
          std::static_pointer_cast<arrow::FixedSizeListType>(field_type);
      auto value_type = fixed_size_list_type->value_type();
      EXPECT_EQ(value_type->id(), arrow::Type::FLOAT);
    }
  }
}

TEST(ArrowTest, ConversionTest) {
  /*
  std::vector<int32_t> test_vector = {128, 42, 0,  0, 0, 0,  0,
                                      0,   32, 61, 0, 0, 74, 19};
  std::vector<uint8_t> null_vector = {vdb::BitPos[0] | vdb::BitPos[4] |
                                      vdb::BitPos[6]};
  uint8_t *data_ptr = reinterpret_cast<uint8_t *>(test_vector.data());
  int64_t data_size = test_vector.size() * sizeof(int32_t);

  uint8_t *null_ptr = reinterpret_cast<uint8_t *>(null_vector.data());
  int64_t null_size = null_vector.size() * sizeof(uint8_t);

  auto null_buffer = std::make_shared<arrow::Buffer>(null_ptr, null_size);

  auto buffer = std::make_shared<arrow::Buffer>(data_ptr, data_size);
  auto array =
      std::make_shared<arrow::PrimitiveArray>(arrow::int32(), 14, buffer);

  std::cout << array->ToString() << std::endl;

  auto flist_array = std::make_shared<arrow::FixedSizeListArray>(
      arrow::fixed_size_list(arrow::int32(), 2), 7, array, null_buffer);

  std::cout << flist_array->ToString() << std::endl;

  std::unique_ptr<uint8_t> dst_ptr(new uint8_t[128]);
  uint64_t value = -1;
  std::copy((uint8_t *)&value, ((uint8_t *)&value) + 8, dst_ptr.get());
  auto view_ptr = (uint64_t *)dst_ptr.get();
  EXPECT_EQ(value, *view_ptr);

  std::vector<int32_t> offset_vector = {0, 2, 8, 10, 12, 14};
  auto offset_buffer =
      std::make_shared<arrow::Buffer>((uint8_t *)offset_vector.data(), 24);

  auto list_array = std::make_shared<arrow::ListArray>(
      arrow::list(arrow::int32()), 5, offset_buffer, array);

  std::cout << list_array->ToString() << std::endl;

  std::string test_str_vector = "helloworld";
  std::vector<int32_t> offset_vec = {0, 5, 10};
  auto str_buf = std::make_shared<arrow::Buffer>(
      (uint8_t *)test_str_vector.data(), (int64_t)test_str_vector.size());
  auto offset_buf = std::make_shared<arrow::Buffer>(
      (uint8_t *)offset_vec.data(),
      (int64_t)(offset_vec.size() * sizeof(int32_t)));
  auto str_arr = std::make_shared<arrow::StringArray>(2, offset_buf, str_buf);
  std::cout << str_arr->ToString() << std::endl;
  auto vb = std::make_shared<arrow::StringBuilder>();

  auto lb = std::make_shared<arrow::ListBuilder>(arrow::default_memory_pool(),
  vb); std::shared_ptr<arrow::Array> arr; lb->Append(); vb->Append("");
  lb->Append();
  vb->Append("");
  lb->Append();
  vb->Append("");
  vb->Append("");
  vb->Append("");
  lb->Append();
  vb->Append("");
  vb->Append("");
  vb->Append("");
  vb->Append("");
  lb->Append();
  vb->Append("");
  vb->Append("");
  vb->Append("");
  vb->Append("");
  vb->Append("");
  lb->Append(false);
  lb->Finish(&arr);
  std::cout << arr->ToString() << std::endl;
  auto vb = std::make_shared<arrow::StringBuilder>();

  auto lb =
  std::make_shared<arrow::FixedSizeListBuilder>(arrow::default_memory_pool(),
  vb, 3); std::shared_ptr<arrow::Array> arr; vb->Append(""); vb->Append("");
  vb->Append("");
  lb->Append();
  lb->Finish(&arr);
  std::cout << arr->ToString() << std::endl;
  // StringListArray slice
  int64_t offset = 1;
  int64_t length = 2;
  auto sliced_array = arr->Slice(offset, length);

  auto org_array = std::static_pointer_cast<arrow::ListArray>(arr);
  std::cout << org_array->offset() << " " << org_array->length() << std::endl;
  auto org_values =
  std::static_pointer_cast<arrow::StringArray>(org_array->values());

  std::cout << org_values->offset() << " " << org_values->length() << " " <<
  org_values->raw_value_offsets() << std::endl;
  // Print data and length of sliced StringListArray and child data
  std::cout << "sliced" << std::endl;
  auto string_list_array =
  std::static_pointer_cast<arrow::ListArray>(sliced_array); std::cout <<
  string_list_array->offset() << " " << string_list_array->length() <<
  std::endl; auto sl_values =
  std::static_pointer_cast<arrow::StringArray>(string_list_array->values());
  std::cout << org_values->offset() << " " << org_values->length() << " " <<
  org_values->raw_value_offsets() << std::endl;
  */
}

TEST_F(MutableArrayTest, TrimEndComprehensiveTest) {
  // 1. BooleanArray TrimEnd test
  {
    vdb::BooleanArray bool_arr;
    EXPECT_TRUE(bool_arr.Append(true).ok());
    EXPECT_TRUE(bool_arr.Append(false).ok());
    EXPECT_TRUE(bool_arr.AppendNull().ok());
    EXPECT_TRUE(bool_arr.Append(true).ok());
    EXPECT_TRUE(bool_arr.Append(false).ok());
    EXPECT_EQ(bool_arr.Size(), 5);

    // Remove 2 elements from the end
    bool_arr.TrimEnd(2);
    EXPECT_EQ(bool_arr.Size(), 3);
    EXPECT_TRUE(bool_arr.GetValue(0).value());
    EXPECT_FALSE(bool_arr.GetValue(1).value());
    EXPECT_FALSE(bool_arr.GetValue(2).has_value());  // null

    // Remove all elements
    bool_arr.TrimEnd(5);
    EXPECT_EQ(bool_arr.Size(), 0);

    // TrimEnd on empty array
    bool_arr.TrimEnd(1);
    EXPECT_EQ(bool_arr.Size(), 0);
  }

  // 2. NumericArray types TrimEnd test
  {
    // Int8Array
    vdb::Int8Array int8_arr;
    EXPECT_TRUE(int8_arr.Append(10).ok());
    EXPECT_TRUE(int8_arr.Append(20).ok());
    EXPECT_TRUE(int8_arr.AppendNull().ok());
    EXPECT_TRUE(int8_arr.Append(30).ok());
    EXPECT_EQ(int8_arr.Size(), 4);

    int8_arr.TrimEnd(1);
    EXPECT_EQ(int8_arr.Size(), 3);
    EXPECT_EQ(int8_arr.GetValue(0).value(), 10);
    EXPECT_EQ(int8_arr.GetValue(1).value(), 20);
    EXPECT_FALSE(int8_arr.GetValue(2).has_value());

    // Int16Array
    vdb::Int16Array int16_arr;
    EXPECT_TRUE(int16_arr.Append(1000).ok());
    EXPECT_TRUE(int16_arr.Append(2000).ok());
    EXPECT_TRUE(int16_arr.Append(3000).ok());
    EXPECT_EQ(int16_arr.Size(), 3);

    int16_arr.TrimEnd(2);
    EXPECT_EQ(int16_arr.Size(), 1);
    EXPECT_EQ(int16_arr.GetValue(0).value(), 1000);

    // Int32Array
    vdb::Int32Array int32_arr;
    EXPECT_TRUE(int32_arr.Append(100000).ok());
    EXPECT_TRUE(int32_arr.Append(200000).ok());
    EXPECT_TRUE(int32_arr.Append(300000).ok());
    EXPECT_EQ(int32_arr.Size(), 3);

    int32_arr.TrimEnd(1);
    EXPECT_EQ(int32_arr.Size(), 2);
    EXPECT_EQ(int32_arr.GetValue(0).value(), 100000);
    EXPECT_EQ(int32_arr.GetValue(1).value(), 200000);

    // Int64Array
    vdb::Int64Array int64_arr;
    EXPECT_TRUE(int64_arr.Append(10000000000LL).ok());
    EXPECT_TRUE(int64_arr.Append(20000000000LL).ok());
    EXPECT_EQ(int64_arr.Size(), 2);

    int64_arr.TrimEnd(1);
    EXPECT_EQ(int64_arr.Size(), 1);
    EXPECT_EQ(int64_arr.GetValue(0).value(), 10000000000LL);

    // UInt8Array
    vdb::UInt8Array uint8_arr;
    EXPECT_TRUE(uint8_arr.Append(255).ok());
    EXPECT_TRUE(uint8_arr.Append(128).ok());
    EXPECT_TRUE(uint8_arr.Append(64).ok());
    EXPECT_EQ(uint8_arr.Size(), 3);

    uint8_arr.TrimEnd(2);
    EXPECT_EQ(uint8_arr.Size(), 1);
    EXPECT_EQ(uint8_arr.GetValue(0).value(), 255);

    // UInt16Array
    vdb::UInt16Array uint16_arr;
    EXPECT_TRUE(uint16_arr.Append(65535).ok());
    EXPECT_TRUE(uint16_arr.Append(32768).ok());
    EXPECT_EQ(uint16_arr.Size(), 2);

    uint16_arr.TrimEnd(1);
    EXPECT_EQ(uint16_arr.Size(), 1);
    EXPECT_EQ(uint16_arr.GetValue(0).value(), 65535);

    // UInt32Array
    vdb::UInt32Array uint32_arr;
    EXPECT_TRUE(uint32_arr.Append(4294967295U).ok());
    EXPECT_TRUE(uint32_arr.Append(2147483648U).ok());
    EXPECT_EQ(uint32_arr.Size(), 2);

    uint32_arr.TrimEnd(1);
    EXPECT_EQ(uint32_arr.Size(), 1);
    EXPECT_EQ(uint32_arr.GetValue(0).value(), 4294967295U);

    // UInt64Array
    vdb::UInt64Array uint64_arr;
    EXPECT_TRUE(uint64_arr.Append(18446744073709551615ULL).ok());
    EXPECT_TRUE(uint64_arr.Append(9223372036854775808ULL).ok());
    EXPECT_EQ(uint64_arr.Size(), 2);

    uint64_arr.TrimEnd(1);
    EXPECT_EQ(uint64_arr.Size(), 1);
    EXPECT_EQ(uint64_arr.GetValue(0).value(), 18446744073709551615ULL);

    // FloatArray
    vdb::FloatArray float_arr;
    EXPECT_TRUE(float_arr.Append(3.14159f).ok());
    EXPECT_TRUE(float_arr.Append(2.71828f).ok());
    EXPECT_TRUE(float_arr.Append(1.41421f).ok());
    EXPECT_EQ(float_arr.Size(), 3);

    float_arr.TrimEnd(1);
    EXPECT_EQ(float_arr.Size(), 2);
    EXPECT_FLOAT_EQ(float_arr.GetValue(0).value(), 3.14159f);
    EXPECT_FLOAT_EQ(float_arr.GetValue(1).value(), 2.71828f);

    // DoubleArray
    vdb::DoubleArray double_arr;
    EXPECT_TRUE(double_arr.Append(3.141592653589793).ok());
    EXPECT_TRUE(double_arr.Append(2.718281828459045).ok());
    EXPECT_EQ(double_arr.Size(), 2);

    double_arr.TrimEnd(1);
    EXPECT_EQ(double_arr.Size(), 1);
    EXPECT_DOUBLE_EQ(double_arr.GetValue(0).value(), 3.141592653589793);
  }

  // 3. StringArray TrimEnd test
  {
    vdb::StringArray str_arr;
    EXPECT_TRUE(str_arr.Append(std::string_view{"Hello"}).ok());
    EXPECT_TRUE(str_arr.Append(std::string_view{"World"}).ok());
    EXPECT_TRUE(str_arr.AppendNull().ok());
    EXPECT_TRUE(str_arr.Append(std::string_view{"Test"}).ok());
    EXPECT_TRUE(str_arr.Append(std::string_view{"String"}).ok());
    EXPECT_EQ(str_arr.Size(), 5);

    str_arr.TrimEnd(2);
    EXPECT_EQ(str_arr.Size(), 3);
    EXPECT_EQ(str_arr.GetValue(0).value(), "Hello");
    EXPECT_EQ(str_arr.GetValue(1).value(), "World");
    EXPECT_FALSE(str_arr.GetValue(2).has_value());

    str_arr.TrimEnd(1);
    EXPECT_EQ(str_arr.Size(), 2);
    EXPECT_EQ(str_arr.GetValue(0).value(), "Hello");
    EXPECT_EQ(str_arr.GetValue(1).value(), "World");
  }

  // 4. LargeStringArray TrimEnd test
  {
    vdb::LargeStringArray large_str_arr;
    EXPECT_TRUE(large_str_arr.Append(std::string_view{"Large"}).ok());
    EXPECT_TRUE(large_str_arr.Append(std::string_view{"String"}).ok());
    EXPECT_TRUE(large_str_arr.AppendNull().ok());
    EXPECT_TRUE(large_str_arr.Append(std::string_view{"Array"}).ok());
    EXPECT_EQ(large_str_arr.Size(), 4);

    large_str_arr.TrimEnd(1);
    EXPECT_EQ(large_str_arr.Size(), 3);
    EXPECT_EQ(large_str_arr.GetValue(0).value(), "Large");
    EXPECT_EQ(large_str_arr.GetValue(1).value(), "String");
    EXPECT_FALSE(large_str_arr.GetValue(2).has_value());
  }

  // 5. ListArray types TrimEnd test
  {
    // Int32ListArray
    vdb::Int32ListArray int32_list_arr;
    EXPECT_TRUE(int32_list_arr.Append({1, 2, 3}).ok());
    EXPECT_TRUE(int32_list_arr.Append({4, 5}).ok());
    EXPECT_TRUE(int32_list_arr.AppendNull().ok());
    EXPECT_TRUE(int32_list_arr.Append({6, 7, 8, 9}).ok());
    EXPECT_EQ(int32_list_arr.Size(), 4);

    int32_list_arr.TrimEnd(1);
    EXPECT_EQ(int32_list_arr.Size(), 3);
    EXPECT_EQ(int32_list_arr.GetValue(0).value(),
              std::vector<int32_t>({1, 2, 3}));
    EXPECT_EQ(int32_list_arr.GetValue(1).value(), std::vector<int32_t>({4, 5}));
    EXPECT_FALSE(int32_list_arr.GetValue(2).has_value());

    // FloatListArray
    vdb::FloatListArray float_list_arr;
    EXPECT_TRUE(float_list_arr.Append({1.1f, 2.2f}).ok());
    EXPECT_TRUE(float_list_arr.Append({3.3f, 4.4f, 5.5f}).ok());
    EXPECT_EQ(float_list_arr.Size(), 2);

    float_list_arr.TrimEnd(1);
    EXPECT_EQ(float_list_arr.Size(), 1);
    EXPECT_EQ(float_list_arr.GetValue(0).value(),
              std::vector<float>({1.1f, 2.2f}));

    // StringListArray
    vdb::StringListArray str_list_arr;
    EXPECT_TRUE(str_list_arr.Append({"Hello", "World"}).ok());
    EXPECT_TRUE(str_list_arr.Append({"Test", "String", "Array"}).ok());
    EXPECT_TRUE(str_list_arr.AppendNull().ok());
    EXPECT_EQ(str_list_arr.Size(), 3);

    str_list_arr.TrimEnd(2);
    EXPECT_EQ(str_list_arr.Size(), 1);
    EXPECT_EQ(str_list_arr.GetValue(0).value(),
              std::vector<std::string_view>({"Hello", "World"}));

    // BooleanListArray
    vdb::BooleanListArray bool_list_arr;
    EXPECT_TRUE(bool_list_arr.Append({true, false, true}).ok());
    EXPECT_TRUE(bool_list_arr.Append({false, false}).ok());
    EXPECT_EQ(bool_list_arr.Size(), 2);

    bool_list_arr.TrimEnd(1);
    EXPECT_EQ(bool_list_arr.Size(), 1);
    EXPECT_EQ(bool_list_arr.GetValue(0).value(),
              std::vector<bool>({true, false, true}));
  }

  // 6. FixedSizeListArray types TrimEnd test
  {
    // Int32FixedSizeListArray
    vdb::Int32FixedSizeListArray int32_fsl_arr(3);
    EXPECT_TRUE(int32_fsl_arr.Append({1, 2, 3}).ok());
    EXPECT_TRUE(int32_fsl_arr.Append({4, 5, 6}).ok());
    EXPECT_TRUE(int32_fsl_arr.AppendNull().ok());
    EXPECT_TRUE(int32_fsl_arr.Append({7, 8, 9}).ok());
    EXPECT_EQ(int32_fsl_arr.Size(), 4);

    int32_fsl_arr.TrimEnd(1);
    EXPECT_EQ(int32_fsl_arr.Size(), 3);
    EXPECT_EQ(int32_fsl_arr.GetValue(0).value(),
              std::vector<int32_t>({1, 2, 3}));
    EXPECT_EQ(int32_fsl_arr.GetValue(1).value(),
              std::vector<int32_t>({4, 5, 6}));
    EXPECT_FALSE(int32_fsl_arr.GetValue(2).has_value());

    // FloatFixedSizeListArray
    vdb::FloatFixedSizeListArray float_fsl_arr(2);
    EXPECT_TRUE(float_fsl_arr.Append({1.1f, 2.2f}).ok());
    EXPECT_TRUE(float_fsl_arr.Append({3.3f, 4.4f}).ok());
    EXPECT_TRUE(float_fsl_arr.Append({5.5f, 6.6f}).ok());
    EXPECT_EQ(float_fsl_arr.Size(), 3);

    float_fsl_arr.TrimEnd(2);
    EXPECT_EQ(float_fsl_arr.Size(), 1);
    EXPECT_EQ(float_fsl_arr.GetValue(0).value(),
              std::vector<float>({1.1f, 2.2f}));

    // BooleanFixedSizeListArray
    vdb::BooleanFixedSizeListArray bool_fsl_arr(4);
    EXPECT_TRUE(bool_fsl_arr.Append({true, false, true, false}).ok());
    EXPECT_TRUE(bool_fsl_arr.AppendNull().ok());
    EXPECT_TRUE(bool_fsl_arr.Append({false, true, false, true}).ok());
    EXPECT_EQ(bool_fsl_arr.Size(), 3);

    bool_fsl_arr.TrimEnd(1);
    EXPECT_EQ(bool_fsl_arr.Size(), 2);
    EXPECT_EQ(bool_fsl_arr.GetValue(0).value(),
              std::vector<bool>({true, false, true, false}));
    EXPECT_FALSE(bool_fsl_arr.GetValue(1).has_value());

    // StringFixedSizeListArray
    vdb::StringFixedSizeListArray str_fsl_arr(2);
    EXPECT_TRUE(str_fsl_arr.Append({"Hello", "World"}).ok());
    EXPECT_TRUE(str_fsl_arr.Append({"Test", "String"}).ok());
    EXPECT_TRUE(str_fsl_arr.AppendNull().ok());
    EXPECT_EQ(str_fsl_arr.Size(), 3);

    str_fsl_arr.TrimEnd(2);
    EXPECT_EQ(str_fsl_arr.Size(), 1);
    EXPECT_EQ(str_fsl_arr.GetValue(0).value(),
              std::vector<std::string_view>({"Hello", "World"}));
  }

  // 7. Edge cases test
  {
    // TrimEnd on empty array
    vdb::Int32Array empty_arr;
    EXPECT_EQ(empty_arr.Size(), 0);
    empty_arr.TrimEnd(5);
    EXPECT_EQ(empty_arr.Size(), 0);

    // TrimEnd with count larger than array size
    vdb::StringArray str_arr;
    EXPECT_TRUE(str_arr.Append(std::string_view{"Test"}).ok());
    EXPECT_TRUE(str_arr.Append(std::string_view{"String"}).ok());
    EXPECT_EQ(str_arr.Size(), 2);

    str_arr.TrimEnd(10);  // Larger than array size
    EXPECT_EQ(str_arr.Size(), 0);

    // TrimEnd with count = 0
    vdb::Int32Array int_arr;
    EXPECT_TRUE(int_arr.Append(100).ok());
    EXPECT_TRUE(int_arr.Append(200).ok());
    EXPECT_EQ(int_arr.Size(), 2);

    int_arr.TrimEnd(0);
    EXPECT_EQ(int_arr.Size(), 2);  // No change
    EXPECT_EQ(int_arr.GetValue(0).value(), 100);
    EXPECT_EQ(int_arr.GetValue(1).value(), 200);
  }

  // 8. Sequential TrimEnd test
  {
    vdb::Int32Array arr;
    for (int i = 0; i < 10; i++) {
      EXPECT_TRUE(arr.Append(i * 10).ok());
    }
    EXPECT_EQ(arr.Size(), 10);

    // Step-by-step TrimEnd
    arr.TrimEnd(3);  // 10 -> 7
    EXPECT_EQ(arr.Size(), 7);
    EXPECT_EQ(arr.GetValue(6).value(), 60);

    arr.TrimEnd(2);  // 7 -> 5
    EXPECT_EQ(arr.Size(), 5);
    EXPECT_EQ(arr.GetValue(4).value(), 40);

    arr.TrimEnd(5);  // 5 -> 0
    EXPECT_EQ(arr.Size(), 0);
  }
}

TEST_F(MutableArrayTest, TrimEndBooleanBitZeroingTest) {
  // Helper lambda function to check bit zeroing in BooleanArray
  auto CheckBooleanArrayBitZeroing =
      [](std::shared_ptr<arrow::BooleanArray> boolean_array,
         size_t original_length) {
        const uint8_t* data = boolean_array->values()->data();
        size_t current_length = boolean_array->length();
        size_t current_bytes =
            (current_length + 7) /
            8;  // The number of bytes needed to store the current length

        // Check all bytes that are currently in use
        for (size_t byte_idx = 0; byte_idx < current_bytes; byte_idx++) {
          uint8_t byte_value = data[byte_idx];

          // Calculate how many bits are valid in this byte
          size_t bits_start = byte_idx * 8;
          size_t bits_end = (bits_start + 8 < current_length) ? bits_start + 8
                                                              : current_length;
          size_t valid_bits_in_byte = bits_end - bits_start;

          // Check unused bits in this byte (if any)
          if (valid_bits_in_byte < 8) {
            for (int bit = valid_bits_in_byte; bit < 8; bit++) {
              uint8_t bit_mask = 1 << bit;
              uint8_t bit_value = byte_value & bit_mask;
              EXPECT_EQ(bit_value, 0)
                  << "Unused bit " << bit << " in byte " << byte_idx
                  << " should be zero. "
                  << "Byte value: 0x" << std::hex
                  << static_cast<int>(byte_value) << ", Bit mask: 0x"
                  << static_cast<int>(bit_mask)
                  << ", Valid bits in byte: " << std::dec << valid_bits_in_byte
                  << ", Current length: " << current_length
                  << ", Byte index: " << byte_idx;
            }
          }
        }

        // Note: We don't check completely unused bytes because Arrow
        // implementation may not zero them for performance reasons. This is
        // acceptable behavior. The critical requirement is that unused bits
        // within used bytes are zeroed.
      };
  // 1. BooleanArray bit zeroing test
  {
    vdb::BooleanArray bool_arr;

    // Fill 10 elements (2 bytes: 8 bits + 2 bits)
    for (int i = 0; i < 10; i++) {
      EXPECT_TRUE(bool_arr.Append(i % 2 == 0).ok());  // alternating true/false
    }
    EXPECT_EQ(bool_arr.Size(), 10);

    // TrimEnd to 6 elements (should zero upper 2 bits of first byte)
    bool_arr.TrimEnd(4);  // 10 -> 6
    EXPECT_EQ(bool_arr.Size(), 6);

    // Check bit zeroing
    auto arrow_array = bool_arr.ToArrowArray();
    auto boolean_array =
        std::static_pointer_cast<arrow::BooleanArray>(arrow_array);
    CheckBooleanArrayBitZeroing(boolean_array, 10);

    // TrimEnd to 3 elements (should zero upper 5 bits of first byte)
    bool_arr.TrimEnd(3);  // 6 -> 3
    EXPECT_EQ(bool_arr.Size(), 3);

    arrow_array = bool_arr.ToArrowArray();
    boolean_array = std::static_pointer_cast<arrow::BooleanArray>(arrow_array);
    CheckBooleanArrayBitZeroing(boolean_array, 6);
  }

  // 2. BooleanListArray bit zeroing test
  {
    vdb::BooleanListArray bool_list_arr;

    // Add lists with total 19 boolean elements across lists
    EXPECT_TRUE(
        bool_list_arr
            .Append({true, false, true, false, true, false, true, false, true})
            .ok());  // 9 elements
    EXPECT_TRUE(bool_list_arr
                    .Append({false, true, false, true, false, true, false, true,
                             false, true})
                    .ok());  // 10 elements
    EXPECT_EQ(bool_list_arr.Size(), 2);

    // TrimEnd to 1 list (should zero bits for the second list)
    bool_list_arr.TrimEnd(1);  // 2 -> 1
    EXPECT_EQ(bool_list_arr.Size(), 1);

    // Convert to Arrow array and check internal child array
    auto arrow_array = bool_list_arr.ToArrowArray();
    auto list_array = std::static_pointer_cast<arrow::ListArray>(arrow_array);
    auto child_array =
        std::static_pointer_cast<arrow::BooleanArray>(list_array->values());

    // The child array should have been trimmed to 9 elements (first list only)
    EXPECT_EQ(child_array->length(), 9);
    CheckBooleanArrayBitZeroing(child_array, 19);
  }

  // 3. BooleanFixedSizeListArray bit zeroing test
  {
    vdb::BooleanFixedSizeListArray bool_fsl_arr(6);  // 6 booleans per element

    // Add 3 elements (total 18 bits: 2 full bytes + 2 bits)
    EXPECT_TRUE(
        bool_fsl_arr.Append({true, false, true, false, true, false}).ok());
    EXPECT_TRUE(
        bool_fsl_arr.Append({false, true, false, true, false, true}).ok());
    EXPECT_TRUE(
        bool_fsl_arr.Append({true, true, false, false, true, true}).ok());
    EXPECT_EQ(bool_fsl_arr.Size(), 3);

    // TrimEnd to 2 elements (should zero bits for the third element)
    bool_fsl_arr.TrimEnd(1);  // 3 -> 2
    EXPECT_EQ(bool_fsl_arr.Size(), 2);

    // Convert to Arrow array and check internal child array
    auto arrow_array = bool_fsl_arr.ToArrowArray();
    auto fsl_array =
        std::static_pointer_cast<arrow::FixedSizeListArray>(arrow_array);
    auto child_array =
        std::static_pointer_cast<arrow::BooleanArray>(fsl_array->values());

    // The child array should have been trimmed to 12 elements (2 * 6)
    EXPECT_EQ(child_array->length(), 12);
    CheckBooleanArrayBitZeroing(child_array, 18);
  }

  // 4. Complex bit boundary test - Focus on same-byte bit zeroing
  {
    vdb::BooleanArray bool_arr;

    // Test cases where TrimEnd affects bits within the same byte
    std::vector<std::pair<int, int>> test_cases = {
        {10, 6},   // 10 -> 6: 2 bytes -> 1 byte (upper 2 bits of byte 0)
        {15, 9},   // 15 -> 9: 2 bytes -> 2 bytes (upper 7 bits of byte 1)
        {24, 17},  // 24 -> 17: 3 bytes -> 3 bytes (upper 7 bits of byte 2)
        {7, 3},    // 7 -> 3: 1 byte -> 1 byte (upper 4 bits of byte 0)
    };

    for (const auto& [original_size, target_size] : test_cases) {
      bool_arr.Reset();

      // Fill with alternating pattern
      for (int i = 0; i < original_size; i++) {
        EXPECT_TRUE(bool_arr.Append(i % 2 == 0).ok());
      }

      // TrimEnd to target size
      bool_arr.TrimEnd(original_size - target_size);
      EXPECT_EQ(bool_arr.Size(), target_size);

      // Check bit zeroing
      auto arrow_array = bool_arr.ToArrowArray();
      auto boolean_array =
          std::static_pointer_cast<arrow::BooleanArray>(arrow_array);
      CheckBooleanArrayBitZeroing(boolean_array, original_size);
    }
  }
}

}  // namespace vdb

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
