#include <gtest/gtest.h>
#include <arrow/api.h>
#include <arrow/json/from_string.h>
#include <arrow/testing/gtest_util.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "vdb/data/metadata.hh"
#include "vdb/data/segmentation.hh"
#include "vdb/data/statistics_collector.hh"
#include "vdb/common/defs.hh"
#include "vdb/tests/base_environment.hh"

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/SegmentTestSuite";
}  // namespace vdb

using namespace vdb::tests;

namespace vdb {
namespace {

using json = nlohmann::json;

class SegmentationInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a sample schema
    schema_ = arrow::schema({arrow::field("id", arrow::int32()),
                             arrow::field("name", arrow::utf8()),
                             arrow::field("age", arrow::int32()),
                             arrow::field("city", arrow::utf8())});
  }

  std::shared_ptr<arrow::Schema> schema_;
};

// ValueSegmentIdGenerator Tests
TEST_F(SegmentationInfoTest, ValueSegmentIdGeneratorValid) {
  ValueSegmentIdGenerator generator;
  generator.SetSegmentKeys({"region", "date", "category"});
  generator.SetSegmentKeyCompositionType(SegmentKeyCompositionType::kComposite);
  std::vector<std::string> parts = {"seoul", "2024-01-01", "electronics"};
  auto result = generator.GenerateSegmentId(parts);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie(),
            "v1::value::comp::region+date+category::[\"seoul\",\"2024-01-01\","
            "\"electronics\"]::::");
}

TEST_F(SegmentationInfoTest, ValueSegmentIdGeneratorEmpty) {
  ValueSegmentIdGenerator generator;
  std::vector<std::string> parts;
  generator.SetSegmentKeys({"id", "name", "age"});
  generator.SetSegmentKeyCompositionType(SegmentKeyCompositionType::kComposite);
  auto result = generator.GenerateSegmentId(parts);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie(), "v1::value::comp::id+name+age::[]::::");
}

// HashSegmentIdGenerator Tests
TEST_F(SegmentationInfoTest, HashSegmentIdGeneratorValid) {
  HashSegmentIdGenerator generator;
  generator.SetVersion("v1");
  generator.SetNumBuckets(10);
  generator.SetSegmentKeys({"id"});
  generator.SetSegmentKeyCompositionType(SegmentKeyCompositionType::kSingle);
  std::vector<std::string> parts = {"123"};
  auto result = generator.GenerateSegmentId(parts);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie(), "v1::hash::single::id::8::10::");
}

TEST_F(SegmentationInfoTest, HashSegmentIdGeneratorInvalidMultipleColumns) {
  HashSegmentIdGenerator generator;
  generator.SetSegmentKeyCompositionType(SegmentKeyCompositionType::kComposite);
  std::vector<std::string> parts = {"123", "456"};
  auto result = generator.GenerateSegmentId(parts);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().message(),
            "Multiple columns are not supported for hash segment type");
}

// SegmentIdGeneratorFactory Tests
TEST_F(SegmentationInfoTest, SegmentIdGeneratorFactoryValid) {
  auto value_generator =
      SegmentIdGeneratorFactory::CreateGenerator(SegmentType::kValue);
  ASSERT_NE(value_generator, nullptr);

  auto hash_generator =
      SegmentIdGeneratorFactory::CreateGenerator(SegmentType::kHash);
  ASSERT_NE(hash_generator, nullptr);
}

TEST_F(SegmentationInfoTest, SegmentIdGeneratorFactoryInvalid) {
  auto generator =
      SegmentIdGeneratorFactory::CreateGenerator(SegmentType::kUndefined);
  EXPECT_EQ(generator, nullptr);
}

// RecordViewPartsExtractor Tests
TEST_F(SegmentationInfoTest, RecordViewPartsExtractorValid) {
  std::string record = "1" + std::string(1, kRS) + "John" +
                       std::string(1, kRS) + "25" + std::string(1, kRS) +
                       "New York";
  RecordViewPartsExtractor extractor(record);
  std::vector<uint32_t> indices = {0, 1, 2};

  auto result = extractor.ExtractParts(indices);
  ASSERT_TRUE(result.ok());
  auto parts = result.ValueOrDie();
  EXPECT_EQ(parts.size(), 3);
  EXPECT_EQ(parts[0], "1");
  EXPECT_EQ(parts[1], "John");
  EXPECT_EQ(parts[2], "25");
}

TEST_F(SegmentationInfoTest, RecordViewPartsExtractorInvalidIndex) {
  std::string record = "1" + std::string(1, kRS) + "John" +
                       std::string(1, kRS) + "25" + std::string(1, kRS) +
                       "New York";
  RecordViewPartsExtractor extractor(record);
  std::vector<uint32_t> indices = {0, 4};  // 4 is out of range

  auto result = extractor.ExtractParts(indices);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().message(),
            "Index out of range when generating segment ID from record");
}

// RecordBatchPartsExtractor Tests
TEST_F(SegmentationInfoTest, RecordBatchPartsExtractorValid) {
  std::shared_ptr<arrow::Array> id_array, name_array, age_array, city_array;
  ASSERT_OK_AND_ASSIGN(id_array,
                       arrow::json::ArrayFromJSONString(arrow::int32(), "[1]"));
  ASSERT_OK_AND_ASSIGN(name_array, arrow::json::ArrayFromJSONString(
                                       arrow::utf8(), R"(["John"])"));
  ASSERT_OK_AND_ASSIGN(
      age_array, arrow::json::ArrayFromJSONString(arrow::int32(), "[25]"));
  ASSERT_OK_AND_ASSIGN(city_array, arrow::json::ArrayFromJSONString(
                                       arrow::utf8(), R"(["New York"])"));

  auto rb = arrow::RecordBatch::Make(
      schema_, 1, {id_array, name_array, age_array, city_array});
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  RecordBatchPartsExtractor extractor(rbs);

  std::vector<uint32_t> indices = {0, 1, 2};
  auto result = extractor.ExtractParts(indices);
  ASSERT_TRUE(result.ok());
  auto parts = result.ValueOrDie();
  EXPECT_EQ(parts.size(), 3);
  EXPECT_EQ(parts[0], "1");
  EXPECT_EQ(parts[1], "John");
  EXPECT_EQ(parts[2], "25");
}

TEST_F(SegmentationInfoTest, RecordBatchPartsExtractorEmpty) {
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  RecordBatchPartsExtractor extractor(rbs);

  std::vector<uint32_t> indices = {0};
  auto result = extractor.ExtractParts(indices);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().message(),
            "Empty record batch for segment id generation");
}

// SegmentationInfo Tests
TEST_F(SegmentationInfoTest, SegmentationInfoValid) {
  json segmetation_info = {{"segment_type", "value"},
                           {"segment_keys", {"id", "name"}},
                           {"version", "v1"},
                           {"segment_key_composition_type", "composite"}};

  std::vector<uint32_t> indices = {0, 1};
  std::vector<std::string> names = {"id", "name"};

  SegmentationInfo info(std::move(segmetation_info), SegmentType::kValue,
                        schema_, std::move(names), std::move(indices));

  EXPECT_EQ(info.GetSegmentType(), SegmentType::kValue);
  EXPECT_EQ(info.GetSegmentKeysIndices().size(), 2);
  EXPECT_EQ(info.GetSegmentKeys().size(), 2);

  std::string record = "1" + std::string(1, kRS) + "John" +
                       std::string(1, kRS) + "25" + std::string(1, kRS) +
                       "New York";
  auto result = info.GetSegmentId(RecordViewPartsExtractor(record));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie(),
            "v1::value::comp::id+name::[\"1\",\"John\"]::::");
}

TEST_F(SegmentationInfoTest, SegmentationInfoHashValid) {
  json segmetation_info = {{"segment_type", "hash"},
                           {"segment_keys", {"id"}},
                           {"segment_key_composition_type", "single"},
                           {"version", "v1"},
                           {"num_buckets", 10}};

  std::vector<uint32_t> indices = {0};
  std::vector<std::string> names = {"id"};

  SegmentationInfo info(std::move(segmetation_info), SegmentType::kHash,
                        schema_, std::move(names), std::move(indices));

  EXPECT_EQ(info.GetSegmentType(), SegmentType::kHash);
  EXPECT_EQ(info.GetSegmentKeysIndices().size(), 1);
  EXPECT_EQ(info.GetSegmentKeys().size(), 1);

  auto result = info.GetSegmentId(RecordViewPartsExtractor("123"));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.ValueOrDie(), "v1::hash::single::id::8::10::");
}

// SegmentationInfoBuilder Tests
TEST_F(SegmentationInfoTest, SegmentationInfoBuilderValid) {
  std::string info_json = R"({
    "segment_type": "value",
    "segment_keys": ["id", "name"],
    "segment_key_composition_type": "composite"
  })";

  SegmentationInfoBuilder builder;
  builder.SetSegmentationInfo(info_json).SetSchema(schema_);

  auto result = builder.Build();
  ASSERT_TRUE(result.ok());
  auto info = result.ValueOrDie();

  EXPECT_EQ(info.GetSegmentType(), SegmentType::kValue);
  EXPECT_EQ(info.GetSegmentKeysIndices().size(), 2);
  EXPECT_EQ(info.GetSegmentKeys().size(), 2);
}

// SegmentationInfoChecker Tests
TEST_F(SegmentationInfoTest, SegmentationInfoCheckerInvalidJson) {
  std::string invalid_json = "{invalid json}";

  auto result = SegmentationInfoChecker().Check(invalid_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(
      result.message(),
      "Failed to parse segmentation info: [json.exception.parse_error.101] "
      "parse error at line 1, column 2: syntax error while parsing object key "
      "- invalid literal; last read: '{i'; expected string literal");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerMissingSegmentType) {
  std::string info_json = R"({
    "segment_keys": ["id", "name"]
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(
      result.message(),
      "Missing required field: \"segment_type\" for \"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerInvalidSegmentType) {
  std::string info_json = R"({
    "segment_type": "invalid_type",
    "segment_keys": ["id", "name"],
    "segment_key_composition_type": "composite"
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.message(),
            "Invalid value for \"segment_type\": invalid_type for "
            "\"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerMissingSegmentKeys) {
  std::string info_json = R"({
    "segment_type": "value"
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(
      result.message(),
      "Missing required field: \"segment_keys\" for \"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerInvalidColumn) {
  std::string info_json = R"({
    "segment_type": "value",
    "segment_keys": ["invalid_column"],
    "segment_key_composition_type": "composite"
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.message(),
            "Invalid element in \"segment_keys\": \"invalid_column\" for "
            "\"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerHashTypeMissingNumBuckets) {
  std::string info_json = R"({
    "segment_type": "hash",
    "segment_keys": ["id"],
    "version": "v1"
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(
      result.message(),
      "Missing required field: \"num_buckets\" for \"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerHashTypeInvalidNumBuckets) {
  std::string info_json = R"({
    "segment_type": "hash",
    "segment_keys": ["id"],
    "version": "v1",
    "num_buckets": 0
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.message(),
            "Invalid value for \"num_buckets\": 0 for \"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerHashTypeMultipleKeys) {
  std::string info_json = R"({
    "segment_type": "hash",
    "segment_keys": ["id", "name"],
    "segment_key_composition_type": "single",
    "version": "v1",
    "num_buckets": 10
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.message(),
            "\"segment_keys\" array size must be exactly 1, but got 2 for "
            "\"segmentation_info\"");
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerValidValueType) {
  std::string info_json = R"({
    "segment_type": "value",
    "segment_keys": ["id", "name"],
    "segment_key_composition_type": "composite"
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_TRUE(result.ok());
}

TEST_F(SegmentationInfoTest, SegmentationInfoCheckerValidHashType) {
  std::string info_json = R"({
    "segment_type": "hash",
    "segment_keys": ["id"],
    "version": "v1",
    "num_buckets": 10,
    "segment_key_composition_type": "single"
  })";

  auto result = SegmentationInfoChecker().Check(info_json, schema_);
  ASSERT_TRUE(result.ok());
}

class SegmentStatisticsCollectorTest : public BaseTestSuite {
 protected:
  void SetUp() override {
    BaseTestSuite::SetUp();

    // Get table dictionary
    table_dictionary_ = vdb::GetTableDictionary();

    // Create table1 with schema
    std::vector<std::shared_ptr<arrow::Field>> schema_vector1 = {
        arrow::field("id", arrow::int64(), false),
        arrow::field("name", arrow::utf8()),
        arrow::field("age", arrow::int32()),
        arrow::field("feature", arrow::fixed_size_list(arrow::float32(), 3))};
    std::string segment_type = "value";
    std::string segment_keys = "id";
    std::string segment_key_composition_type = "single";
    std::string segmentation_info_str1 = MakeSegmentationInfoString(
        segment_type, segment_keys, segment_key_composition_type);
    size_t ann_column_id = 3;
    size_t ef_construction = 100;
    size_t M = 16;
    std::string index_info_str1 = MakeDenseIndexInfoString(
        ann_column_id, "Hnsw", "L2Space", ef_construction, M);
    auto metadata1 = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"segmentation_info", segmentation_info_str1},
            {"table name", "table1"},
            {"active_set_size_limit", "1000"},
            {"index_info", index_info_str1}});
    auto schema1 = std::make_shared<arrow::Schema>(schema_vector1, metadata1);
    vdb::TableBuilderOptions options1;
    vdb::TableBuilder builder1{
        std::move(options1.SetTableName("table1").SetSchema(schema1))};
    ASSERT_OK_AND_ASSIGN(auto table1, builder1.Build());

    // Create table2 with schema
    std::vector<std::shared_ptr<arrow::Field>> schema_vector2 = {
        arrow::field("id", arrow::int64(), false),
        arrow::field("value", arrow::float64())};
    std::string segment_type2 = "value";
    std::string segment_keys2 = "id";
    std::string segment_key_composition_type2 = "single";
    std::string segmentation_info_str2 = MakeSegmentationInfoString(
        segment_type2, segment_keys2, segment_key_composition_type2);
    auto metadata2 = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"segmentation_info", segmentation_info_str2},
            {"table name", "table2"},
            {"active_set_size_limit", "1000"}});
    auto schema2 = std::make_shared<arrow::Schema>(schema_vector2, metadata2);
    vdb::TableBuilderOptions options2;
    vdb::TableBuilder builder2{
        std::move(options2.SetTableName("table2").SetSchema(schema2))};
    ASSERT_OK_AND_ASSIGN(auto table2, builder2.Build());

    // Add tables to dictionary
    (*table_dictionary_)["table1"] = table1;
    (*table_dictionary_)["table2"] = table2;

    // Create collector
    collector_ =
        std::make_unique<SegmentStatisticsCollector>(table_dictionary_);
  }

  vdb::map<std::string, std::shared_ptr<vdb::Table>>* table_dictionary_;
  std::unique_ptr<SegmentStatisticsCollector> collector_;
};

TEST_F(SegmentStatisticsCollectorTest, CollectAllEmpty) {
  auto result = collector_->CollectAll();
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 0);
}

TEST_F(SegmentStatisticsCollectorTest, CollectByTable) {
  // Add a segment to table1
  auto table1 = (*table_dictionary_)["table1"];
  auto segment = std::make_shared<vdb::Segment>(table1, "segment1", 0);
  table1->AddSegment(table1, "segment1");

  auto result = collector_->CollectByTable("table1");
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 1);

  const auto& info = (*result)[0];
  ASSERT_EQ(info.table_name, "table1");
  ASSERT_EQ(info.segment_id, "segment1");
  ASSERT_EQ(info.row_count, 0);
  ASSERT_EQ(info.deleted_row_count, 0);
  ASSERT_EQ(info.active_set_row_count, 0);
  ASSERT_EQ(info.inactive_set_count, 0);
  ASSERT_FALSE(info.indexed_row_count.empty());
}

TEST_F(SegmentStatisticsCollectorTest, CollectByTables) {
  // Add segments to both tables
  auto table1 = (*table_dictionary_)["table1"];
  auto table2 = (*table_dictionary_)["table2"];

  auto segment1 = std::make_shared<vdb::Segment>(table1, "segment1", 0);
  auto segment2 = std::make_shared<vdb::Segment>(table2, "segment2", 0);

  table1->AddSegment(table1, "segment1");
  table2->AddSegment(table2, "segment2");

  auto result = collector_->CollectByTables({"table1", "table2"});
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 2);

  // Verify table1 segment info
  const auto& info1 = (*result)[0];
  ASSERT_EQ(info1.table_name, "table1");
  ASSERT_EQ(info1.segment_id, "segment1");

  // Verify table2 segment info
  const auto& info2 = (*result)[1];
  ASSERT_EQ(info2.table_name, "table2");
  ASSERT_EQ(info2.segment_id, "segment2");
}

TEST_F(SegmentStatisticsCollectorTest, CollectWithIndexedColumns) {
  auto table1 = (*table_dictionary_)["table1"];
  auto segment = table1->AddSegment(table1, "segment1");

  // Create record batch
  arrow::Int64Builder id_builder;
  arrow::StringBuilder name_builder;
  arrow::Int32Builder age_builder;
  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder feature_builder(arrow::default_memory_pool(),
                                              value_builder, 3);

  // Add values
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(name_builder.Append("John"));
  ARROW_EXPECT_OK(age_builder.Append(25));

  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(1.0f));
  ARROW_EXPECT_OK(value_builder->Append(2.0f));
  ARROW_EXPECT_OK(value_builder->Append(3.0f));

  ARROW_EXPECT_OK(id_builder.Append(2));
  ARROW_EXPECT_OK(name_builder.Append("Jane"));
  ARROW_EXPECT_OK(age_builder.Append(30));

  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(4.0f));
  ARROW_EXPECT_OK(value_builder->Append(5.0f));
  ARROW_EXPECT_OK(value_builder->Append(6.0f));

  // Finish arrays
  std::shared_ptr<arrow::Array> id_array, name_array, age_array, feature_array;
  ARROW_EXPECT_OK(id_builder.Finish(&id_array));
  ARROW_EXPECT_OK(name_builder.Finish(&name_array));
  ARROW_EXPECT_OK(age_builder.Finish(&age_array));
  ARROW_EXPECT_OK(feature_builder.Finish(&feature_array));

  // Create record batch
  auto rb = arrow::RecordBatch::Make(
      table1->GetSchema(), 2, {id_array, name_array, age_array, feature_array});
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                       SerializeRecordBatches(table1->GetSchema(), rbs));

  // Insert record batch
  sds serialized_rb_sds =
      sdsnewlen(reinterpret_cast<const void*>(serialized_rb->data()),
                static_cast<size_t>(serialized_rb->size()));
  auto status = vdb::_BatchInsertCommand("table1", serialized_rb_sds);
  sdsfree(serialized_rb_sds);
  ASSERT_TRUE(status.ok());

  auto result = collector_->CollectByTable("table1");
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 2);

  const auto& info =
      std::find_if(result->begin(), result->end(), [](const auto& entry) {
        return entry.segment_id == "v1::value::single::id::[\"1\"]::::";
      });
  ASSERT_NE(info, result->end());
  ASSERT_EQ(info->indexed_row_count.at("feature"), 2);
  ASSERT_EQ(info->row_count, 2);
}

TEST_F(SegmentStatisticsCollectorTest, JsonSerialization) {
  // Create a segment with some data
  auto table1 = (*table_dictionary_)["table1"];

  // Create record batch
  arrow::Int64Builder id_builder;
  arrow::StringBuilder name_builder;
  arrow::Int32Builder age_builder;
  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder feature_builder(arrow::default_memory_pool(),
                                              value_builder, 3);

  // Add values
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(name_builder.Append("John"));
  ARROW_EXPECT_OK(age_builder.Append(25));

  ARROW_EXPECT_OK(feature_builder.Append());
  ARROW_EXPECT_OK(value_builder->Append(1.0f));
  ARROW_EXPECT_OK(value_builder->Append(2.0f));
  ARROW_EXPECT_OK(value_builder->Append(3.0f));

  // Finish arrays
  std::shared_ptr<arrow::Array> id_array, name_array, age_array, feature_array;
  ARROW_EXPECT_OK(id_builder.Finish(&id_array));
  ARROW_EXPECT_OK(name_builder.Finish(&name_array));
  ARROW_EXPECT_OK(age_builder.Finish(&age_array));
  ARROW_EXPECT_OK(feature_builder.Finish(&feature_array));

  // Create record batch
  auto rb = arrow::RecordBatch::Make(
      table1->GetSchema(), 1, {id_array, name_array, age_array, feature_array});
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  ASSERT_OK_AND_ASSIGN(auto serialized_rb,
                       SerializeRecordBatches(table1->GetSchema(), rbs));

  // Insert record batch
  sds serialized_rb_sds =
      sdsnewlen(reinterpret_cast<const void*>(serialized_rb->data()),
                static_cast<size_t>(serialized_rb->size()));
  auto status = vdb::_BatchInsertCommand("table1", serialized_rb_sds);
  sdsfree(serialized_rb_sds);
  ASSERT_TRUE(status.ok());

  // Collect info
  auto result = collector_->CollectByTable("table1");
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 1);

  // Convert to JSON
  auto json = collector_->ToJson();
  ASSERT_TRUE(json.is_object());
  ASSERT_TRUE(json.contains("segment_statistics"));
  ASSERT_TRUE(json["segment_statistics"].is_array());
  ASSERT_EQ(json["segment_statistics"].size(), 1);

  // Verify JSON content
  const auto& json_obj = json["segment_statistics"][0];
  EXPECT_EQ(json_obj["table_name"], "table1");
  EXPECT_EQ(json_obj["segment_id"], "v1::value::single::id::[\"1\"]::::");
  EXPECT_EQ(json_obj["row_count"], 1);
  EXPECT_EQ(json_obj["deleted_row_count"], 0);
  EXPECT_EQ(json_obj["active_set_row_count"], 1);
  EXPECT_EQ(json_obj["inactive_set_count"], 0);

  // Verify indexed row count
  const auto& indexed_json = json_obj["indexed_row_count"];
  ASSERT_TRUE(indexed_json.is_object());
  EXPECT_EQ(indexed_json.size(), 1);
  EXPECT_EQ(indexed_json["feature"], 1);

  // Test deserialization
  auto deserialized = SegmentStatisticsCollector::FromJson(json);
  ASSERT_EQ(deserialized.size(), 1);
  EXPECT_EQ(deserialized[0].table_name, "table1");
  EXPECT_EQ(deserialized[0].segment_id, "v1::value::single::id::[\"1\"]::::");
  EXPECT_EQ(deserialized[0].indexed_row_count.at("feature"), 1);
}

TEST_F(SegmentStatisticsCollectorTest, InvalidTableName) {
  auto result = collector_->CollectByTable("nonexistent_table");
  ASSERT_FALSE(result.ok());
  ASSERT_EQ(result.status().message(), "Table not found: nonexistent_table");
}

TEST_F(SegmentStatisticsCollectorTest, TableWithIndexInfos) {
  // 1. Define schema (the last column must be a vector/embedding column for
  // index_infos to be meaningful)
  auto schema = arrow::schema(
      {arrow::field("id", arrow::int64(), false),
       arrow::field("name", arrow::utf8(), false),
       arrow::field("feature", arrow::fixed_size_list(arrow::float32(), 3))});

  // 2. Create index_info JSON string (feature column is at index 2)
  size_t ann_column_id = 2;
  size_t ef_construction = 100;
  size_t M = 16;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, "Hnsw", "L2Space", ef_construction, M);

  // 3. Create metadata including segmentation_info
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"segmentation_info", segmentation_info_str},
          {"index_info", index_info_str},
          {"table name", "table_with_index"},
          {"active_set_size_limit",
           std::to_string(server.vdb_active_set_size_limit)}});
  schema = schema->WithMetadata(metadata);

  // 4. Create table using TableBuilder
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("table_with_index").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  // 5. Add table to dictionary
  (*table_dictionary_)["table_with_index"] = table;

  // 6. Verify that index_infos are parsed correctly
  auto index_infos = table->GetIndexInfos();
  ASSERT_TRUE(index_infos != nullptr);
  ASSERT_EQ(index_infos->size(), 1);
  EXPECT_EQ(index_infos->at(0).GetColumnId(), 2);
  EXPECT_EQ(index_infos->at(0).GetIndexType(), "Hnsw");
}

// ============================================================================
// Segment ID Parser Tests
// ============================================================================

TEST_F(SegmentationInfoTest, ValueSegmentIdParserValid) {
  ValueSegmentIdParser parser;
  std::string segment_id =
      "v1::value::comp::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";
  std::vector<std::string> column_names = {"region", "date", "category"};

  auto result = parser.ParseSegmentId(segment_id, column_names);
  ASSERT_TRUE(result.ok());
  auto mapping = result.ValueOrDie();

  EXPECT_EQ(mapping["region"], "seoul");
  EXPECT_EQ(mapping["date"], "2024-01-01");
  EXPECT_EQ(mapping["category"], "electronics");
}

TEST_F(SegmentationInfoTest, ValueSegmentIdParserExtractField) {
  ValueSegmentIdParser parser;
  std::string segment_id =
      "v1::value::comp::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";

  auto version_result = parser.ExtractField(segment_id, "version");
  ASSERT_TRUE(version_result.ok());
  EXPECT_EQ(version_result.ValueOrDie(), "v1");

  auto type_result = parser.ExtractField(segment_id, "type");
  ASSERT_TRUE(type_result.ok());
  EXPECT_EQ(type_result.ValueOrDie(), "value");

  auto values_result = parser.ExtractField(segment_id, "values");
  ASSERT_TRUE(values_result.ok());
  EXPECT_EQ(values_result.ValueOrDie(),
            "[\"seoul\",\"2024-01-01\",\"electronics\"]");
}

TEST_F(SegmentationInfoTest, ValueSegmentIdParserValidate) {
  ValueSegmentIdParser parser;

  // Valid segment ID
  std::string valid_id =
      "v1::value::comp::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";
  auto valid_result = parser.ValidateSegmentId(valid_id);
  ASSERT_TRUE(valid_result.ok());
  EXPECT_TRUE(valid_result.ValueOrDie());

  // Invalid segment ID - wrong type
  std::string invalid_type =
      "v1::hash::comp::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";
  auto invalid_type_result = parser.ValidateSegmentId(invalid_type);
  ASSERT_TRUE(invalid_type_result.ok());
  EXPECT_FALSE(invalid_type_result.ValueOrDie());

  // Invalid segment ID - wrong composition type
  std::string invalid_comp =
      "v1::value::single::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";
  auto invalid_comp_result = parser.ValidateSegmentId(invalid_comp);
  ASSERT_TRUE(invalid_comp_result.ok());
  EXPECT_FALSE(invalid_comp_result.ValueOrDie());

  // Invalid segment ID - invalid JSON
  std::string invalid_json =
      "v1::value::comp::region+date+category::invalid_json::::";
  auto invalid_json_result = parser.ValidateSegmentId(invalid_json);
  ASSERT_TRUE(invalid_json_result.ok());
  EXPECT_FALSE(invalid_json_result.ValueOrDie());
}

TEST_F(SegmentationInfoTest, HashSegmentIdParserValid) {
  HashSegmentIdParser parser;
  std::string segment_id = "v1::hash::single::id::9::10::";
  std::vector<std::string> column_names = {"id"};

  auto result = parser.ParseSegmentId(segment_id, column_names);
  ASSERT_TRUE(result.ok());
  auto mapping = result.ValueOrDie();

  EXPECT_EQ(mapping["id"], "9");
}

TEST_F(SegmentationInfoTest, HashSegmentIdParserExtractField) {
  HashSegmentIdParser parser;
  std::string segment_id = "v1::hash::single::id::9::10::";

  auto version_result = parser.ExtractField(segment_id, "version");
  ASSERT_TRUE(version_result.ok());
  EXPECT_EQ(version_result.ValueOrDie(), "v1");

  auto type_result = parser.ExtractField(segment_id, "type");
  ASSERT_TRUE(type_result.ok());
  EXPECT_EQ(type_result.ValueOrDie(), "hash");

  auto bucket_result = parser.ExtractField(segment_id, "bucket");
  ASSERT_TRUE(bucket_result.ok());
  EXPECT_EQ(bucket_result.ValueOrDie(), "9");

  auto num_buckets_result = parser.ExtractField(segment_id, "num_buckets");
  ASSERT_TRUE(num_buckets_result.ok());
  EXPECT_EQ(num_buckets_result.ValueOrDie(), "10");
}

TEST_F(SegmentationInfoTest, HashSegmentIdParserValidate) {
  HashSegmentIdParser parser;

  // Valid segment ID
  std::string valid_id = "v1::hash::single::id::9::10::";
  auto valid_result = parser.ValidateSegmentId(valid_id);
  ASSERT_TRUE(valid_result.ok());
  EXPECT_TRUE(valid_result.ValueOrDie());

  // Invalid segment ID - wrong type
  std::string invalid_type = "v1::value::single::id::9::10::";
  auto invalid_type_result = parser.ValidateSegmentId(invalid_type);
  ASSERT_TRUE(invalid_type_result.ok());
  EXPECT_FALSE(invalid_type_result.ValueOrDie());

  // Invalid segment ID - invalid composition type
  std::string invalid_comp = "v1::hash::invalid::id::9::10::";
  auto invalid_comp_result = parser.ValidateSegmentId(invalid_comp);
  ASSERT_TRUE(invalid_comp_result.ok());
  EXPECT_FALSE(invalid_comp_result.ValueOrDie());

  // Invalid segment ID - non-numeric bucket
  std::string invalid_bucket = "v1::hash::single::id::abc::10::";
  auto invalid_bucket_result = parser.ValidateSegmentId(invalid_bucket);
  ASSERT_TRUE(invalid_bucket_result.ok());
  EXPECT_FALSE(invalid_bucket_result.ValueOrDie());
}

TEST_F(SegmentationInfoTest, SegmentIdParserFactory) {
  // Test creating parsers by type
  auto value_parser = SegmentIdParserFactory::CreateParser(SegmentType::kValue);
  ASSERT_NE(value_parser, nullptr);

  auto hash_parser = SegmentIdParserFactory::CreateParser(SegmentType::kHash);
  ASSERT_NE(hash_parser, nullptr);

  auto undefined_parser =
      SegmentIdParserFactory::CreateParser(SegmentType::kUndefined);
  EXPECT_EQ(undefined_parser, nullptr);

  // Test creating parsers from segment ID
  std::string value_segment_id =
      "v1::value::comp::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";
  auto value_parser_from_id =
      SegmentIdParserFactory::CreateParserFromSegmentId(value_segment_id);
  ASSERT_NE(value_parser_from_id, nullptr);

  std::string hash_segment_id = "v1::hash::single::id::9::10::";
  auto hash_parser_from_id =
      SegmentIdParserFactory::CreateParserFromSegmentId(hash_segment_id);
  ASSERT_NE(hash_parser_from_id, nullptr);

  std::string invalid_segment_id = "invalid::format";
  auto invalid_parser =
      SegmentIdParserFactory::CreateParserFromSegmentId(invalid_segment_id);
  EXPECT_EQ(invalid_parser, nullptr);
}

TEST_F(SegmentationInfoTest, SegmentIdUtils) {
  // Test ParseSegmentId
  std::string value_segment_id =
      "v1::value::comp::region+date+category::[\"seoul\",\"2024-01-01\","
      "\"electronics\"]::::";
  std::vector<std::string> column_names = {"region", "date", "category"};

  auto parse_result = SegmentIdUtils::ParseSegmentId(
      value_segment_id, column_names, SegmentType::kValue);
  ASSERT_TRUE(parse_result.ok());
  auto mapping = parse_result.ValueOrDie();
  EXPECT_EQ(mapping["region"], "seoul");
  EXPECT_EQ(mapping["date"], "2024-01-01");
  EXPECT_EQ(mapping["category"], "electronics");

  // Test ExtractValues
  auto extract_result = SegmentIdUtils::ExtractValues(
      value_segment_id, column_names, SegmentType::kValue);
  ASSERT_TRUE(extract_result.ok());
  auto values = extract_result.ValueOrDie();
  EXPECT_EQ(values.size(), 3);
  EXPECT_EQ(values[0], "seoul");
  EXPECT_EQ(values[1], "2024-01-01");
  EXPECT_EQ(values[2], "electronics");

  // Test ValidateSegmentId
  auto validate_result =
      SegmentIdUtils::ValidateSegmentId(value_segment_id, SegmentType::kValue);
  ASSERT_TRUE(validate_result.ok());
  EXPECT_TRUE(validate_result.ValueOrDie());

  // Test GetSegmentType
  auto type_result = SegmentIdUtils::GetSegmentType(value_segment_id);
  ASSERT_TRUE(type_result.ok());
  EXPECT_EQ(type_result.ValueOrDie(), SegmentType::kValue);

  std::string hash_segment_id = "v1::hash::single::id::9::10::";
  auto hash_type_result = SegmentIdUtils::GetSegmentType(hash_segment_id);
  ASSERT_TRUE(hash_type_result.ok());
  EXPECT_EQ(hash_type_result.ValueOrDie(), SegmentType::kHash);
}

}  // namespace
}  // namespace vdb

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}