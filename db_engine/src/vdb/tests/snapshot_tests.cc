#include <filesystem>
#include <memory>
#include <string>
#include <strings.h>
#include <filesystem>
#include <arrow/record_batch.h>
#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>

#include "arrow/api.h"

#include "vdb/tests/util_for_test.hh"
#include "vdb/vdb.hh"
#include "vdb/vdb_api.hh"
#include "vdb/common/defs.hh"
#include "vdb/common/util.hh"
#include "vdb/data/table.hh"
#include "vdb/data/mutable_array.hh"
#include "vdb/data/index_handler.hh"

#include "vdb/tests/base_environment.hh"

using namespace vdb::tests;

namespace vdb {

std::string test_suite_directory_path =
    test_root_directory_path + "/SnapshotTestSuite";

struct SnapshotTestParams {
  std::string index_type;
  std::string space_type;

  SnapshotTestParams(const std::string &index_type,
                     const std::string &space_type)
      : index_type(index_type), space_type(space_type) {}

  std::string ToString() const { return index_type + "_" + space_type; }
};

const std::vector<std::string> INDEX_TYPES = {"Hnsw"};
const std::vector<std::string> SPACE_TYPES = {"l2space", "ipspace"};

std::vector<SnapshotTestParams> GenerateSnapshotTestParams() {
  std::vector<SnapshotTestParams> params;
  for (const auto &index_type : INDEX_TYPES) {
    for (const auto &space_type : SPACE_TYPES) {
      if (index_type == "Hnsw") {
        params.emplace_back(index_type, space_type);
      }
    }
  }
  return params;
}

class SnapshotTestSuite : public BaseTestSuiteWithParam<SnapshotTestParams> {
 public:
  void SetUp() override { BaseTestSuiteWithParam<SnapshotTestParams>::SetUp(); }
  vdb::DistanceSpace GetDistanceSpace() const {
    if (GetParam().space_type == "L2Space") {
      return vdb::DistanceSpace::kL2;
    } else if (GetParam().space_type == "IPSpace") {
      return vdb::DistanceSpace::kIP;
    } else if (GetParam().space_type == "CosineSpace") {
      return vdb::DistanceSpace::kCosine;
    }
    return vdb::DistanceSpace::kL2;
  }
};

class IndexTest : public SnapshotTestSuite {
 public:
  std::shared_ptr<DenseVectorIndex> CreateDenseIndex(
      DistanceSpace space, size_t dim, size_t ef_construction, size_t M,
      size_t max_elem, std::shared_ptr<EmbeddingStore> embedding_store);
  std::shared_ptr<DenseVectorIndex> LoadDenseIndex(
      DistanceSpace space, size_t dim, std::string save_path,
      std::shared_ptr<EmbeddingStore> embedding_store);
};
std::shared_ptr<DenseVectorIndex> IndexTest::CreateDenseIndex(
    DistanceSpace space, size_t dim, size_t ef_construction, size_t M,
    size_t max_elem, std::shared_ptr<EmbeddingStore> embedding_store) {
  if (GetParam().index_type == "Hnsw") {
    return std::make_shared<vdb::Hnsw>(space, dim, ef_construction, M, max_elem,
                                       embedding_store);
  }
  return nullptr;
}

std::shared_ptr<DenseVectorIndex> IndexTest::LoadDenseIndex(
    DistanceSpace space, size_t dim, std::string save_path,
    std::shared_ptr<EmbeddingStore> embedding_store) {
  if (GetParam().index_type == "Hnsw") {
    return std::make_shared<vdb::Hnsw>(save_path, space, dim, embedding_store);
  }
  return nullptr;
}

class SegmentTest : public SnapshotTestSuite {};
class TableTest : public SnapshotTestSuite {};
class VdbSnapshotTest : public SnapshotTestSuite {};
class TableSnapshotTest : public SnapshotTestSuite {};

INSTANTIATE_TEST_SUITE_P(
    IndexTypes, IndexTest, testing::ValuesIn(GenerateSnapshotTestParams()),
    [](const testing::TestParamInfo<IndexTest::ParamType> &info) {
      return info.param.ToString();
    });
INSTANTIATE_TEST_SUITE_P(
    IndexTypes, VdbSnapshotTest,
    testing::ValuesIn(GenerateSnapshotTestParams()),
    [](const testing::TestParamInfo<VdbSnapshotTest::ParamType> &info) {
      return info.param.ToString();
    });
INSTANTIATE_TEST_SUITE_P(
    IndexTypes, SegmentTest, ::testing::ValuesIn(GenerateSnapshotTestParams()),
    [](const testing::TestParamInfo<SegmentTest::ParamType> &info) {
      return info.param.ToString();
    });
INSTANTIATE_TEST_SUITE_P(
    IndexTypes, TableTest, ::testing::ValuesIn(GenerateSnapshotTestParams()),
    [](const testing::TestParamInfo<TableTest::ParamType> &info) {
      return info.param.ToString();
    });
INSTANTIATE_TEST_SUITE_P(
    IndexTypes, TableSnapshotTest,
    ::testing::ValuesIn(GenerateSnapshotTestParams()),
    [](const testing::TestParamInfo<TableSnapshotTest::ParamType> &info) {
      return info.param.ToString();
    });

TEST_P(IndexTest, SingleIndex8DSmallDataTest) {
  size_t data_cnt = 10;
  size_t dim = 8;
  auto data = generateSequentialFloatArray(data_cnt, dim, 1);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif
  /* {1,1} ... {10,10} */
  size_t ef_construction = 100;
  size_t M = 3;
  size_t max_elem = 10;
  auto index_space = GetDistanceSpace();
  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, dim);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  auto index = CreateDenseIndex(index_space, dim, ef_construction, M, max_elem,
                                embedding_store);
  for (size_t i = 0; i < data_cnt; i++) {
    float *point = &data[i][0];
    index->AddEmbedding(point, i);
  }

  std::string index_file_path = TestDirectoryPath() + "/index";
  status = index->Save(index_file_path);

#ifdef _DEBUG_GTEST
  std::cout << "save index to : " << index_file_path << std::endl;
#endif

  auto loaded_index =
      LoadDenseIndex(index_space, dim, index_file_path, embedding_store);
  ASSERT_TRUE(DenseIndexEquals(index, loaded_index));
}

TEST_P(IndexTest, SingleIndex8DLargeDataTest) {
  size_t data_cnt = 10000;
  size_t dim = 8;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif
  /* {1,1} ... {10,10} */
  size_t ef_construction = 100;
  size_t M = 3;
  size_t max_elem = 10000;
  auto index_space = GetDistanceSpace();
  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, dim);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  auto index = CreateDenseIndex(index_space, dim, ef_construction, M, max_elem,
                                embedding_store);
  for (size_t i = 0; i < data_cnt; i++) {
    float *point = &data[i][0];
    index->AddEmbedding(point, i);
  }
  std::string index_file_path = TestDirectoryPath() + "/index";
  status = index->Save(index_file_path);

#ifdef _DEBUG_GTEST
  std::cout << "save index to : " << index_file_path << std::endl;
#endif

  auto loaded_index =
      LoadDenseIndex(index_space, dim, index_file_path, embedding_store);
  ASSERT_TRUE(DenseIndexEquals(index, loaded_index));
}

TEST_P(IndexTest, SingleIndex128DLargeDataTest) {
  size_t data_cnt = 10000;
  size_t dim = 128;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif
  /* {1,1} ... {10,10} */
  size_t ef_construction = 100;
  size_t M = 3;
  size_t max_elem = 10000;
  auto index_space = GetDistanceSpace();
  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, dim);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  auto index = CreateDenseIndex(index_space, dim, ef_construction, M, max_elem,
                                embedding_store);
  for (size_t i = 0; i < data_cnt; i++) {
    float *point = &data[i][0];
    index->AddEmbedding(point, i);
  }

  std::string index_file_path = TestDirectoryPath() + "/index";
  status = index->Save(index_file_path);

#ifdef _DEBUG_GTEST
  std::cout << "save index to : " << index_file_path << std::endl;
#endif

  auto loaded_index =
      LoadDenseIndex(index_space, dim, index_file_path, embedding_store);
  ASSERT_TRUE(DenseIndexEquals(index, loaded_index));
}

TEST_P(IndexTest, IndexHandlerTest) {
  auto index_type = GetParam();
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 16;

  std::string test_table_name = "dummy_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  server.vdb_active_set_size_limit = 1000;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  auto table = table_dictionary->at(test_table_name);

  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto org_index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  size_t data_cnt = 1000;
  auto data = generateSequentialFloatArray(data_cnt, dim, 1);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif

  for (size_t i = 0; i < data.size(); i++) {
    float *point = &data[i][0];
    org_index_handler->AddEmbeddingForTest(point, i, ann_column_id);
  }

#ifdef _DEBUG_GTEST
  std::cout << "index count=" << org_index_handler->Size() << std::endl;
#endif

  std::string index_directory_path = TestDirectoryPath() + "/indexes";
  status = org_index_handler->Save(index_directory_path);
  ASSERT_TRUE(status.ok());

  auto loaded_index_handler = std::make_shared<IndexHandlerForTest>(
      table, 0, index_directory_path, org_index_handler->Size());
  auto org_handler =
      static_cast<std::shared_ptr<vdb::IndexHandler>>(org_index_handler);
  auto loaded_handler =
      static_cast<std::shared_ptr<vdb::IndexHandler>>(loaded_index_handler);
  ASSERT_TRUE(IndexHandlerEquals(org_handler, loaded_handler));
}

TEST_P(IndexTest, IndexHandlerMoreDataTest) {
  auto index_type = GetParam();
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 128;
  server.vdb_active_set_size_limit = 10000;

  std::string test_table_name = "dummy_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);

  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto org_index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  size_t data_cnt = 10000;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif

  for (size_t i = 0; i < data.size(); i++) {
    float *point = &data[i][0];
    org_index_handler->AddEmbeddingForTest(point, i, ann_column_id);
  }

  std::string index_directory_path = TestDirectoryPath() + "/indexes";
  status = org_index_handler->Save(index_directory_path);
  ASSERT_TRUE(status.ok());

  auto loaded_index_handler = std::make_shared<IndexHandlerForTest>(
      table, 0, index_directory_path, org_index_handler->Size());
  auto org_handler =
      static_cast<std::shared_ptr<vdb::IndexHandler>>(org_index_handler);
  auto loaded_handler =
      static_cast<std::shared_ptr<vdb::IndexHandler>>(loaded_index_handler);
  ASSERT_TRUE(IndexHandlerEquals(org_handler, loaded_handler));
}

TEST_P(SegmentTest, EmptyTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
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
  std::string dummy_id = "";
  auto org_segment = std::make_shared<vdb::Segment>(table, dummy_id, 0);

  std::string segment_directory_path = TestDirectoryPath() + "/segment";
  auto status = org_segment->Save(segment_directory_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  bool load_error = false;
  try {
    auto loaded_segment = std::make_shared<vdb::Segment>(
        table, dummy_id, 0, segment_directory_path);
    auto load_status = loaded_segment->LoadInactiveSets(segment_directory_path);
    if (!load_status.ok()) {
      loaded_segment.reset();
      std::cerr << load_status.ToString() << std::endl;
      ASSERT_TRUE(load_status.ok());
    }
    ASSERT_TRUE(SegmentEquals(org_segment, loaded_segment));
#ifdef _DEBUG_GTEST
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
    std::cout << "org_segment" << std::endl
              << org_segment->ToString() << std::endl;
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
    std::cout << "loaded_segment" << std::endl
              << loaded_segment->ToString() << std::endl;
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
#endif
  } catch (const std::exception &e) {
    load_error = true;
    std::cerr << "General error: " << e.what() << std::endl;
  }
  ASSERT_FALSE(load_error);
}

TEST_P(SegmentTest, IncompleteTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
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
  std::string dummy_id = "";
  auto org_segment = std::make_shared<vdb::Segment>(table, dummy_id, 0);

  std::string segment_directory_path = TestDirectoryPath() + "/segment";
  auto status = org_segment->Save(segment_directory_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string manifest_file_path = segment_directory_path + "/manifest";
  std::filesystem::remove(manifest_file_path);
  bool std_error = false;
  try {
    auto loaded_segment = std::make_shared<vdb::Segment>(
        table, dummy_id, 0, segment_directory_path);
    auto load_status = loaded_segment->LoadInactiveSets(segment_directory_path);
    if (!load_status.ok()) {
      loaded_segment.reset();
      std::cerr << load_status.ToString() << std::endl;
      ASSERT_TRUE(load_status.ok());
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "Filesystem error: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "General error: " << e.what() << std::endl;
    std_error = true;
  }
  ASSERT_TRUE(std_error);
}

TEST_P(SegmentTest, EmptyWithIndexHandlerTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "10000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());
  std::string dummy_id = "";
  auto org_segment = std::make_shared<vdb::Segment>(table, dummy_id, 0);

  std::string segment_directory_path = TestDirectoryPath() + "/segment";
  auto status = org_segment->Save(segment_directory_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  bool load_error = false;
  try {
    auto loaded_segment = std::make_shared<vdb::Segment>(
        table, dummy_id, 0, segment_directory_path);
    auto load_status = loaded_segment->LoadInactiveSets(segment_directory_path);
    if (!load_status.ok()) {
      loaded_segment.reset();
      std::cerr << load_status.ToString() << std::endl;
      ASSERT_TRUE(load_status.ok());
    }
    ASSERT_TRUE(SegmentEquals(org_segment, loaded_segment));
#ifdef _DEBUG_GTEST
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
    std::cout << "org_segment" << std::endl
              << org_segment->ToString(false) << std::endl;
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
    std::cout << "loaded_segment" << std::endl
              << loaded_segment->ToString(false) << std::endl;
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
#endif
  } catch (const std::exception &e) {
    std::cerr << "General error: " << e.what() << std::endl;
    load_error = true;
  }
  ASSERT_FALSE(load_error);
}

TEST_P(SegmentTest, ManyDataWithIndexHandlerTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

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

  auto maybe_rb = GenerateRecordBatch(schema, 5000, dim);
  ASSERT_TRUE(maybe_rb.ok());
  auto rb = maybe_rb.ValueUnsafe();
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  auto status = org_segment->AppendRecords(rbs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string segment_directory_path = TestDirectoryPath() + "/segment";
  status = org_segment->Save(segment_directory_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  bool load_error = false;
  try {
    auto loaded_segment = std::make_shared<vdb::Segment>(
        table, dummy_id, 0, segment_directory_path);
    auto load_status = loaded_segment->LoadInactiveSets(segment_directory_path);
    if (!load_status.ok()) {
      loaded_segment.reset();
      std::cerr << load_status.ToString() << std::endl;
      ASSERT_TRUE(load_status.ok());
    }
    ASSERT_TRUE(SegmentEquals(org_segment, loaded_segment));
#ifdef _DEBUG_GTEST
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
    std::cout << "org_segment" << std::endl
              << org_segment->ToString(false) << std::endl;
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
    std::cout << "loaded_segment" << std::endl
              << loaded_segment->ToString(false) << std::endl;
    std::cout
        << "---------------------------------------------------------------"
        << std::endl;
#endif
  } catch (const std::exception &e) {
    std::cerr << "General error: " << e.what() << std::endl;
    load_error = true;
  }
  ASSERT_FALSE(load_error);
}

TEST_P(TableTest, EmptyTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
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
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  org_table->AddSegment(org_table, "1");

  std::string snapshot_directory_path = TestDirectoryPath();
  std::string_view snapshot_directory_path_view(snapshot_directory_path);
  auto status = org_table->Save(snapshot_directory_path_view);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string table_directory_path = snapshot_directory_path + "/test_table";
  vdb::TableBuilderOptions options2;
  vdb::TableBuilder builder2(
      std::move(options2.SetTableName(table_name)
                    .SetTableDirectoryPath(table_directory_path)));
  ASSERT_OK_AND_ASSIGN(auto loaded_table, builder2.Build());
  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(TableEquals(org_table, loaded_table));
#ifdef _DEBUG_GTEST
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "org_table" << std::endl
            << org_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "loaded_table" << std::endl
            << loaded_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
#endif
}

TEST_P(TableTest, IncompleteTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
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
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  org_table->AddSegment(org_table, "1");

  std::string snapshot_directory_path = TestDirectoryPath();
  std::string_view snapshot_directory_path_view(snapshot_directory_path);
  auto status = org_table->Save(snapshot_directory_path_view);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string table_directory_path = snapshot_directory_path + "/test_table";
  std::string manifest_file_path = table_directory_path + "/manifest";
  std::filesystem::remove(manifest_file_path);

  vdb::TableBuilderOptions options2;
  vdb::TableBuilder builder2(
      std::move(options2.SetTableName(table_name)
                    .SetTableDirectoryPath(table_directory_path)));
  auto loaded_table = builder2.Build();
  if (!loaded_table.ok()) {
    if (loaded_table.status().ToStringWithoutContextLines().find(
            "saving snapshot of table is not completed") == std::string::npos) {
      ASSERT_OK(loaded_table);
    }
  }
}

TEST_P(TableTest, EmptyWithIndexTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "10000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());

  auto org_segment = org_table->AddSegment(org_table, "1");

  std::string snapshot_directory_path = TestDirectoryPath();
  std::string_view snapshot_directory_path_view(snapshot_directory_path);
  auto status = org_table->Save(snapshot_directory_path_view);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string table_directory_path = snapshot_directory_path + "/test_table";
  vdb::TableBuilderOptions options2;
  vdb::TableBuilder builder2(
      std::move(options2.SetTableName("test_table")
                    .SetTableDirectoryPath(table_directory_path)));
  ASSERT_OK_AND_ASSIGN(auto loaded_table, builder2.Build());
  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(TableEquals(org_table, loaded_table));
#ifdef _DEBUG_GTEST
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "org_table" << std::endl
            << org_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "loaded_table" << std::endl
            << loaded_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
#endif
}

TEST_P(TableTest, ManyDataWithIndexTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "1000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  auto org_segment = org_table->AddSegment(org_table, "1");

  auto maybe_rb = GenerateRecordBatch(schema, 5000, dim);
  ASSERT_TRUE(maybe_rb.ok());
  auto rb = maybe_rb.ValueUnsafe();
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  auto status = org_segment->AppendRecords(rbs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string snapshot_directory_path = TestDirectoryPath();
  std::string_view snapshot_directory_path_view(snapshot_directory_path);
  status = org_table->Save(snapshot_directory_path_view);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string table_directory_path = snapshot_directory_path + "/test_table";
  vdb::TableBuilderOptions options2;
  vdb::TableBuilder builder2(
      std::move(options2.SetTableName("test_table")
                    .SetTableDirectoryPath(table_directory_path)));
  ASSERT_OK_AND_ASSIGN(auto loaded_table, builder2.Build());
  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(TableEquals(org_table, loaded_table));
#ifdef _DEBUG_GTEST
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "org_table" << std::endl
            << org_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "loaded_table" << std::endl
            << loaded_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
#endif
}

TEST_P(TableTest, MultipleSegmentsManyDataWithIndexTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  size_t segment_count = 5;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "1000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  for (size_t i = 0; i < segment_count; i++) {
    auto org_segment = org_table->AddSegment(org_table, std::to_string(i));

    auto maybe_rb = GenerateRecordBatch(schema, 2500, dim);
    ASSERT_TRUE(maybe_rb.ok());
    auto rb = maybe_rb.ValueUnsafe();
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
    auto status = org_segment->AppendRecords(rbs);
    if (!status.ok()) {
      std::cout << status.ToString() << std::endl;
      ASSERT_TRUE(status.ok());
    }
  }
  std::string snapshot_directory_path = TestDirectoryPath();
  std::string_view snapshot_directory_path_view(snapshot_directory_path);
  auto status = org_table->Save(snapshot_directory_path_view);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }
  std::string table_directory_path = snapshot_directory_path + "/test_table";
  vdb::TableBuilderOptions options2;
  vdb::TableBuilder builder2(
      std::move(options2.SetTableName("test_table")
                    .SetTableDirectoryPath(table_directory_path)));
  ASSERT_OK_AND_ASSIGN(auto loaded_table, builder2.Build());
  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(TableEquals(org_table, loaded_table));
#ifdef _DEBUG_GTEST
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "org_table" << std::endl
            << org_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
  std::cout << "loaded_table" << std::endl
            << loaded_table->ToString(false) << std::endl;
  std::cout << "---------------------------------------------------------------"
            << std::endl;
#endif
}

TEST_P(VdbSnapshotTest, EmptyTest) {
  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(table_count, loaded_table_count);
}

TEST_P(VdbSnapshotTest, SingleEmptyTableTest) {
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
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
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  org_table->AddSegment(org_table, "1");
  auto table_dictionary = vdb::GetTableDictionary();
  table_dictionary->insert({"test_table", org_table});

  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  ASSERT_TRUE(PrepareVdbSnapshot()) << "Failed to prepare VDB snapshot";
  ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  PostVdbSnapshot();
  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(table_count, loaded_table_count);
  auto loaded_table_dictionary = vdb::GetTableDictionary();
  for (auto [table_name, table] : *loaded_table_dictionary) {
    ASSERT_TRUE(table_name.compare("test_table") == 0);
  }
}

TEST_P(VdbSnapshotTest, SingleDataTableTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", "test_table"},
          {"active_set_size_limit", "1000"},
          {"index_info", index_info_str}});
  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName("test_table").SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  auto org_segment = org_table->AddSegment(org_table, "1");
  auto table_dictionary = vdb::GetTableDictionary();
  table_dictionary->insert({"test_table", org_table});

  auto maybe_rb = GenerateRecordBatch(schema, 5000, dim);
  ASSERT_TRUE(maybe_rb.ok());
  auto rb = maybe_rb.ValueUnsafe();
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
  auto status = org_segment->AppendRecords(rbs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(table_count, loaded_table_count);
  auto loaded_table_dictionary = vdb::GetTableDictionary();
  for (auto [table_name, table] : *loaded_table_dictionary) {
    ASSERT_TRUE(table_name.compare("test_table") == 0);
    ASSERT_TRUE(TableEquals(org_table, table));
  }
}

TEST_P(VdbSnapshotTest, ComplexTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  size_t segment_count = 5;
  size_t org_table_count = 5;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto table_dictionary = vdb::GetTableDictionary();
  std::map<std::string, std::shared_ptr<vdb::Table>> org_tables;
  for (size_t j = 0; j < org_table_count; j++) {
    std::string table_name = "test_table" + std::to_string(j);
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    size_t ann_column_id = 3;
    size_t ef_construction = 100;
    size_t M = 2;
    std::string index_info_str =
        MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                                 index_type.space_type, ef_construction, M);

    auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"table name", table_name},
            {"active_set_size_limit", "1000"},
            {"index_info", index_info_str}});
    schema = schema->WithMetadata(add_metadata);
    vdb::TableBuilderOptions options;
    vdb::TableBuilder builder{
        std::move(options.SetTableName(table_name).SetSchema(schema))};
    ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
    auto org_segment = org_table->AddSegment(org_table, "1");
    table_dictionary->insert({table_name, org_table});

    for (size_t i = 0; i < segment_count; i++) {
      auto org_segment = org_table->AddSegment(org_table, std::to_string(i));

      auto maybe_rb = GenerateRecordBatch(schema, 970, dim);
      ASSERT_TRUE(maybe_rb.ok());
      auto rb = maybe_rb.ValueUnsafe();
      std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
      auto status = org_segment->AppendRecords(rbs);
      if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        ASSERT_TRUE(status.ok());
      }
    }
    table_dictionary->insert({table_name, org_table});
    org_tables.insert({table_name, org_table});
  }

  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(table_count, loaded_table_count);
  auto loaded_table_dictionary = vdb::GetTableDictionary();
  for (auto [table_name, table] : *loaded_table_dictionary) {
    ASSERT_TRUE(TableEquals(org_tables[table_name], table));
  }
}

TEST_P(VdbSnapshotTest, MultipleSnapshotTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  size_t segment_count = 5;
  size_t org_table_count = 5;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  auto table_dictionary = vdb::GetTableDictionary();
  std::map<std::string, std::shared_ptr<vdb::Table>> org_tables;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  for (size_t j = 0; j < org_table_count; j++) {
    std::string table_name = "test_table" + std::to_string(j);
    auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"table name", table_name},
            {"active_set_size_limit", "1000"},
            {"index_info", index_info_str}});
    schema = schema->WithMetadata(add_metadata);
    vdb::TableBuilderOptions options;
    vdb::TableBuilder builder{
        std::move(options.SetTableName(table_name).SetSchema(schema))};
    ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
    auto org_segment = org_table->AddSegment(org_table, "1");
    table_dictionary->insert({table_name, org_table});

    for (size_t i = 0; i < segment_count; i++) {
      auto org_segment = org_table->AddSegment(org_table, std::to_string(i));

      auto maybe_rb = GenerateRecordBatch(schema, 970, dim);
      ASSERT_TRUE(maybe_rb.ok());
      auto rb = maybe_rb.ValueUnsafe();
      std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
      auto status = org_segment->AppendRecords(rbs);
      if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        ASSERT_TRUE(status.ok());
      }
    }
    table_dictionary->insert({table_name, org_table});
    org_tables.insert({table_name, org_table});
    std::string base_snapshot_name = GetBaseSnapshotDirectoryName(j);
    std::string base_snapshot_path(server.aof_dirname);
    base_snapshot_path.append("/");
    base_snapshot_path.append(base_snapshot_name);
    ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  }

  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();
  std::string base_snapshot_name =
      GetBaseSnapshotDirectoryName(org_table_count - 1);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(table_count, loaded_table_count);
  auto loaded_table_dictionary = vdb::GetTableDictionary();
  for (auto [table_name, table] : *loaded_table_dictionary) {
    ASSERT_TRUE(TableEquals(org_tables[table_name], table));
  }
}

TEST_P(VdbSnapshotTest, ResumeBuildingIncompleteIndexTest) {
  auto index_type = GetParam();
  /* for using IndexingThreadJob() */
  server.allow_bg_index_thread = true;
  uint32_t dim = 128;
  size_t segment_count = 3;
  size_t org_table_count = 2;
  std::string table_schema_string =
      "Id int32, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto table_dictionary = vdb::GetTableDictionary();
  std::map<std::string, std::shared_ptr<vdb::Table>> org_tables;
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  for (size_t j = 0; j < org_table_count; j++) {
    std::string table_name = "test_table" + std::to_string(j);
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"table name", table_name},
            {"active_set_size_limit", "1000"},
            {"index_info", index_info_str}});

    schema = schema->WithMetadata(add_metadata);
    vdb::TableBuilderOptions options;
    vdb::TableBuilder builder{
        std::move(options.SetTableName(table_name).SetSchema(schema))};
    ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
    auto org_segment = org_table->AddSegment(org_table, "1");
    table_dictionary->insert({table_name, org_table});

    for (size_t i = 0; i < segment_count; i++) {
      auto org_segment = org_table->AddSegment(org_table, std::to_string(i));

      auto maybe_rb = GenerateRecordBatch(schema, 3000, dim);
      ASSERT_TRUE(maybe_rb.ok());
      auto rb = maybe_rb.ValueUnsafe();
      std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
      auto status = org_segment->AppendRecords(rbs);
      if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        ASSERT_TRUE(status.ok());
      }
    }
    table_dictionary->insert({table_name, org_table});
    org_tables.insert({table_name, org_table});
    if (j == 0) {
      /* test_table1 will not be able to build index */
      ASSERT_TRUE(PrepareVdbSnapshot()) << "Failed to prepare VDB snapshot";
    }
  }

  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  /* Checking incomplete index exists before snapshot save */
  bool incomplete_index_check_before_snapshot = true;
  for (auto [table_name, table] : *table_dictionary) {
    bool prepare_check = TableIndexFullBuildCheck(table, false);
    if (!prepare_check) {
      incomplete_index_check_before_snapshot = false;
    }
  }
  ASSERT_FALSE(incomplete_index_check_before_snapshot);
  ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(table_count, loaded_table_count);
  auto loaded_table_dictionary = vdb::GetTableDictionary();

  /* Checking all tables are equal before resuming incomplete index building
   */
  for (auto [table_name, table] : *loaded_table_dictionary) {
    ASSERT_TRUE(TableEquals(org_tables[table_name], table));
    if (table_name.compare("test_table1") == 0) {
      ASSERT_FALSE(TableIndexFullBuildCheck(table, false));
    }
  }
  PostVdbSnapshot();
  org_tables.clear();
  /* Resuming the index building and checking all index building job is
   * completed */
  for (auto [table_name, table] : *loaded_table_dictionary) {
    bool building_complete = false;
    while (!building_complete) {
      building_complete = TableIndexFullBuildCheck(table, false);
    }
  }
  /* disable background indexing thread */
  server.allow_bg_index_thread = false;
}

TEST_P(VdbSnapshotTest, ResumeBuildingIncompleteIndexWithDropTableTest) {
  auto index_type = GetParam();
  /* for using IndexingThreadJob() */
  server.allow_bg_index_thread = true;
  uint32_t dim = 128;
  size_t segment_count = 3;
  size_t org_table_count = 2;
  std::string table_schema_string =
      "Id int32, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
  auto table_dictionary = vdb::GetTableDictionary();
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 2;
  std::string index_info_str =
      MakeDenseIndexInfoString(ann_column_id, index_type.index_type,
                               index_type.space_type, ef_construction, M);

  for (size_t j = 0; j < org_table_count; j++) {
    std::string table_name = "test_table" + std::to_string(j);
    auto schema = vdb::ParseSchemaFrom(table_schema_string);
    auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{
            {"table name", table_name},
            {"active_set_size_limit", "1000"},
            {"index_info", index_info_str}});

    schema = schema->WithMetadata(add_metadata);
    vdb::TableBuilderOptions options;
    vdb::TableBuilder builder{
        std::move(options.SetTableName(table_name).SetSchema(schema))};
    ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
    auto org_segment = org_table->AddSegment(org_table, "1");
    table_dictionary->insert({table_name, org_table});

    for (size_t i = 0; i < segment_count; i++) {
      auto org_segment = org_table->AddSegment(org_table, std::to_string(i));

      auto maybe_rb = GenerateRecordBatch(schema, 3000, dim);
      ASSERT_TRUE(maybe_rb.ok());
      auto rb = maybe_rb.ValueUnsafe();
      std::vector<std::shared_ptr<arrow::RecordBatch>> rbs = {rb};
      auto status = org_segment->AppendRecords(rbs);
      if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        ASSERT_TRUE(status.ok());
      }
    }
    table_dictionary->insert({table_name, org_table});
    if (j == 0) {
      /* test_table1 will not be able to build index */
      ASSERT_TRUE(PrepareVdbSnapshot()) << "Failed to prepare VDB snapshot";
    }
  }

  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  /* Checking incomplete index exists before snapshot save */
  bool incomplete_index_check_before_snapshot = true;
  for (auto [table_name, table] : *table_dictionary) {
    bool prepare_check = TableIndexFullBuildCheck(table, false);
    if (!prepare_check) {
      incomplete_index_check_before_snapshot = false;
    }
  }
  ASSERT_FALSE(incomplete_index_check_before_snapshot);
  ASSERT_TRUE(SaveVdbSnapshot(base_snapshot_path.data()));
  /* Remember how many tables were in the snapshot before any drop */
  size_t snapshot_table_count = vdb::GetTableDictionary()->size();

  PostVdbSnapshot();
  /* Drop test_table0 and verify its embedding store directory is removed */
  auto table1_it = table_dictionary->find("test_table1");
  ASSERT_TRUE(table1_it != table_dictionary->end());
  auto table1 = table1_it->second;
  auto embedding_store_directory_path = table1->GetEmbeddingStore(ann_column_id)
                                            ->GetEmbeddingStoreDirectoryPath();
  /* Release local reference to ensure destructor can run on drop */
  table1 = nullptr;
  EXPECT_TRUE(std::filesystem::exists(embedding_store_directory_path));
  auto drop_table_status = vdb::_DropTableCommand("test_table1");
  ASSERT_TRUE(drop_table_status.ok());
  EXPECT_FALSE(std::filesystem::exists(embedding_store_directory_path));
  /* Keep org_tables as-is for equality check after loading snapshot */

  DeallocateTableDictionary();
  AllocateTableDictionary();
  ASSERT_TRUE(LoadVdbSnapshot(base_snapshot_path.data()));
  size_t loaded_table_count = vdb::GetTableDictionary()->size();
  ASSERT_EQ(snapshot_table_count, loaded_table_count);

  /* disable background indexing thread */
  server.allow_bg_index_thread = false;
}

TEST_F(VdbSnapshotTest, InactiveSetTransitionDuringSnapshotTest) {
  // Set active set size limit to 10
  server.vdb_active_set_size_limit = 10;

  // Enable AOF but disable auto rewrite for this test
  server.aof_enabled = 1;
  server.aof_state = AOF_ON;
  server.aof_rewrite_perc = 0;      // Disable auto rewrite
  server.aof_rewrite_min_size = 0;  // Disable auto rewrite

  // Create table without segmentation (all data in single segment)
  std::string table_name = "test_duplicate_data_table";
  std::string schema_string = "ID uint32 not null, Name String";

  auto schema = vdb::ParseSchemaFrom(schema_string);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"table name", table_name}, {"active_set_size_limit", "10"}});

  schema = schema->WithMetadata(add_metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());

  auto table_dictionary = vdb::GetTableDictionary();
  table_dictionary->insert({table_name, org_table});

  // First batch insertion: 1,2,3,4,5,6,7 (stored in active set)
  std::vector<std::string> first_batch = {
      "1\u001eAlice", "2\u001eBob",   "3\u001eCharlie", "4\u001eDavid",
      "5\u001eEve",   "6\u001eFrank", "7\u001eGrace"};

  for (const auto &record : first_batch) {
    auto insert_status =
        vdb::_InsertCommand(table_name, std::string_view(record));
    ASSERT_TRUE(insert_status.ok()) << "Failed to insert record: " << record;
  }

  // Verify state before first VDB snapshot
  auto segments_before_first_snapshot = org_table->GetSegments();
  ASSERT_EQ(segments_before_first_snapshot.size(), 1) << "Expected 1 segment";

  auto segment_before_first_snapshot =
      segments_before_first_snapshot.begin()->second;
  ASSERT_EQ(segment_before_first_snapshot->ActiveSetRecordCount(), 7)
      << "Expected 7 records in active set";
  ASSERT_EQ(segment_before_first_snapshot->InactiveSetCount(), 0)
      << "Expected 0 inactive sets before first snapshot";

  // Create first VDB snapshot using PrepareVdbSnapshot
  std::string first_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string first_snapshot_path(server.aof_dirname);
  first_snapshot_path.append("/");
  first_snapshot_path.append(first_snapshot_name);

  ASSERT_TRUE(PrepareVdbSnapshot()) << "Failed to prepare VDB snapshot";
  ASSERT_TRUE(SaveVdbSnapshot(first_snapshot_path.data()));
  PostVdbSnapshot();

#ifdef _DEBUG_GTEST
  std::cout << "First VDB Snapshot saved: " << first_snapshot_path << std::endl;
#endif
  // Verify state after first VDB snapshot - active set should be converted
  // to inactive set
  auto segments_after_first_snapshot = org_table->GetSegments();
  auto segment_after_first_snapshot =
      segments_after_first_snapshot.begin()->second;
  ASSERT_EQ(segment_after_first_snapshot->InactiveSetCount(), 1)
      << "Expected 1 inactive set after first snapshot (active set "
         "converted)";
  ASSERT_EQ(segment_after_first_snapshot->ActiveSetRecordCount(), 0)
      << "Expected 0 records in active set after first snapshot";

#ifdef _DEBUG_GTEST
  std::cout << "After first VDB snapshot: ActiveSet="
            << segment_after_first_snapshot->ActiveSetRecordCount()
            << ", InactiveSets="
            << segment_after_first_snapshot->InactiveSetCount()
            << ", TotalSize=" << segment_after_first_snapshot->ActualSize()
            << std::endl;
#endif
  // Create second VDB snapshot immediately after first snapshot (no data
  // changes)
  std::string second_snapshot_name = GetBaseSnapshotDirectoryName(1);
  std::string second_snapshot_path(server.aof_dirname);
  second_snapshot_path.append("/");
  second_snapshot_path.append(second_snapshot_name);

  ASSERT_TRUE(PrepareVdbSnapshot()) << "Failed to prepare VDB snapshot";
  ASSERT_TRUE(SaveVdbSnapshot(second_snapshot_path.data()));
  PostVdbSnapshot();

#ifdef _DEBUG_GTEST
  std::cout << "Second VDB Snapshot saved: " << second_snapshot_path
            << std::endl;
#endif
  // Verify state after second VDB snapshot - inactive set count should
  // remain the same
  auto segments_after_second_snapshot_immediate = org_table->GetSegments();
  auto segment_after_second_snapshot_immediate =
      segments_after_second_snapshot_immediate.begin()->second;
  ASSERT_EQ(segment_after_second_snapshot_immediate->InactiveSetCount(), 1)
      << "Expected 1 inactive set after second snapshot (no change)";
  ASSERT_EQ(segment_after_second_snapshot_immediate->ActiveSetRecordCount(), 0)
      << "Expected 0 records in active set after second snapshot";

#ifdef _DEBUG_GTEST
  std::cout << "After second VDB snapshot (immediate): ActiveSet="
            << segment_after_second_snapshot_immediate->ActiveSetRecordCount()
            << ", InactiveSets="
            << segment_after_second_snapshot_immediate->InactiveSetCount()
            << ", TotalSize="
            << segment_after_second_snapshot_immediate->ActualSize()
            << std::endl;
#endif
  // Second batch insertion: 8,9,10,11,12,13 (stored in active set)
  std::vector<std::string> second_batch = {"8\u001eHenry", "9\u001eIvy",
                                           "10\u001eJack", "11\u001eKate",
                                           "12\u001eLiam", "13\u001eMia"};

  for (const auto &record : second_batch) {
    auto insert_status =
        vdb::_InsertCommand(table_name, std::string_view(record));
    ASSERT_TRUE(insert_status.ok()) << "Failed to insert record: " << record;
  }

  // Verify state after second batch insertion - should have 1 inactive set
  auto segments_after_second_batch = org_table->GetSegments();
  auto segment_after_second_batch = segments_after_second_batch.begin()->second;
  ASSERT_EQ(segment_after_second_batch->InactiveSetCount(), 1)
      << "Expected 1 inactive set after second batch insertion";
  ASSERT_EQ(segment_after_second_batch->ActiveSetRecordCount(), 6)
      << "Expected 6 records in active set after second batch insertion";

#ifdef _DEBUG_GTEST
  std::cout << "After second batch insertion: ActiveSet="
            << segment_after_second_batch->ActiveSetRecordCount()
            << ", InactiveSets="
            << segment_after_second_batch->InactiveSetCount()
            << ", TotalSize=" << segment_after_second_batch->ActualSize()
            << std::endl;

  // Log inactive sets details
  for (uint16_t i = 0; i < segment_after_second_batch->InactiveSetCount();
       i++) {
    auto inactive_set = segment_after_second_batch->GetInactiveSet(i);
    if (inactive_set && inactive_set->GetRb()) {
      std::cout << "  Inactive Set " << i << ": "
                << inactive_set->GetRb()->num_rows() << " records" << std::endl;
    }
  }
#endif
  // Verify total records before restart
  size_t total_records_before = segment_after_second_batch->ActualSize();
  ASSERT_EQ(total_records_before, 13)
      << "Expected 13 total records before restart";

  // Simulate process restart
  size_t table_count = vdb::GetTableDictionary()->size();
  DeallocateTableDictionary();
  AllocateTableDictionary();

  // Ensure AOF remains enabled but auto rewrite disabled after restart
  // simulation
  server.aof_enabled = 1;
  server.aof_state = AOF_ON;
  server.aof_rewrite_perc = 0;      // Disable auto rewrite
  server.aof_rewrite_min_size = 0;  // Disable auto rewrite

  // Load second VDB snapshot (load 1,2,3,4,5,6,7 from VDB snapshot)
  ASSERT_TRUE(LoadVdbSnapshot(second_snapshot_path.data()))
      << "Failed to load VDB snapshot";

  // Verify loaded table
  auto loaded_table_dictionary = vdb::GetTableDictionary();
  size_t loaded_table_count = loaded_table_dictionary->size();
  ASSERT_EQ(table_count, loaded_table_count);

  auto it = loaded_table_dictionary->find(table_name);
  ASSERT_TRUE(it != loaded_table_dictionary->end());
  auto loaded_table = it->second;

  auto loaded_segments = loaded_table->GetSegments();
  ASSERT_EQ(loaded_segments.size(), 1) << "Expected 1 segment after load";

  auto loaded_segment = loaded_segments.begin()->second;

  // Verify data loaded from VDB snapshot
  // Should contain 1,2,3,4,5,6,7 from VDB snapshot
  ASSERT_EQ(loaded_segment->InactiveSetCount(), 1)
      << "Expected 1 inactive set after VDB snapshot load";
  ASSERT_EQ(loaded_segment->ActiveSetRecordCount(), 0)
      << "Expected 0 records in active set after VDB snapshot load";

  // Simulate AOF replay: insert 8,9,10,11,12,13 again
  for (const auto &record : second_batch) {
    auto insert_status =
        vdb::_InsertCommand(table_name, std::string_view(record));
    ASSERT_TRUE(insert_status.ok())
        << "Failed to insert record during AOF replay: " << record;
  }

  // Verify final state after AOF replay
  auto final_segments = loaded_table->GetSegments();
  auto final_segment = final_segments.begin()->second;

  // Verify total record count (if duplicates exist: 19, if not: 13)
  auto total_records_after = final_segment->ActualSize();

  // Check if duplicates occurred
  if (total_records_after > 13) {
    // Duplicates detected - test failure
    FAIL() << "Duplicate data detected! Expected 13 records, but got "
           << total_records_after
           << " records. This indicates 8,9,10 were duplicated.";
  }

  // No duplicates - test success
  ASSERT_EQ(total_records_after, 13)
      << "Expected 13 total records after AOF replay";
  // Verify individual records
  std::set<std::string> expected_ids = {"1", "2", "3",  "4",  "5",  "6", "7",
                                        "8", "9", "10", "11", "12", "13"};
  std::set<std::string> actual_ids;
  // Collect IDs from all inactive sets and active set
  for (uint16_t i = 0; i < final_segment->InactiveSetCount(); i++) {
    auto inactive_set = final_segment->GetInactiveSet(i);
    if (inactive_set && inactive_set->GetRb()) {
      auto id_column = std::static_pointer_cast<arrow::UInt32Array>(
          inactive_set->GetRb()->column(0));
      for (int64_t j = 0; j < id_column->length(); j++) {
        actual_ids.insert(std::to_string(id_column->Value(j)));
      }
    }

    // Collect IDs from active set as well
    if (final_segment->ActiveSetRecordCount() > 0) {
      auto active_rb = final_segment->ActiveSetRecordBatch();
      if (active_rb) {
        auto id_column =
            std::static_pointer_cast<arrow::UInt32Array>(active_rb->column(0));
        for (int64_t j = 0; j < id_column->length(); j++) {
          actual_ids.insert(std::to_string(id_column->Value(j)));
        }
      }
    }

    ASSERT_EQ(actual_ids, expected_ids)
        << "Record IDs don't match expected values";
  }
}

TEST_P(TableSnapshotTest, SaveAndLoadTableSnapshotTest) {
  auto index_type = GetParam();
  uint32_t dim = 16;
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";
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
  ASSERT_OK_AND_ASSIGN(auto org_table, builder.Build());
  org_table->AddSegment(org_table, "1");
  auto table_dictionary = vdb::GetTableDictionary();
  table_dictionary->insert({"test_table", org_table});

  std::string base_snapshot_name = GetBaseSnapshotDirectoryName(0);
  std::string base_snapshot_path(server.aof_dirname);
  base_snapshot_path.append("/");
  base_snapshot_path.append(base_snapshot_name);
  PrepareTableSnapshot(table_name.data());
  bool ok = SaveTableSnapshot(table_name.data(), base_snapshot_path.data());
  ASSERT_TRUE(ok);
  PostTableSnapshot();
  DeallocateTableDictionary();
  AllocateTableDictionary();

  ok = LoadTableSnapshot(table_name.data(), base_snapshot_path.data());
  ASSERT_TRUE(ok);
  auto loaded_table_dictionary = vdb::GetTableDictionary();
  for (auto [table_name, table] : *loaded_table_dictionary) {
    ASSERT_TRUE(table_name.compare("test_table") == 0);
  }
}
}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
