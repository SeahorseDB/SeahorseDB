#include <climits>
#include <memory>
#include <queue>
#include <string>
#include <strings.h>
#include <utility>
#include <gtest/gtest.h>

#include <arrow/testing/gtest_util.h>

#include "vdb/tests/base_environment.hh"

using namespace vdb::tests;

namespace vdb {

std::string test_suite_directory_path =
    test_root_directory_path + "/IndexHandlerTestSuite";

struct IndexHandlerTestParams {
  std::string index_type;
  std::string space_type;

  IndexHandlerTestParams(const std::string &index_type,
                         const std::string &space_type)
      : index_type(index_type), space_type(space_type) {}

  std::string ToString() const {
    std::string result = index_type + "_" + space_type;
    return result;
  }
};

class IndexHandlerTestSuite
    : public vdb::BaseTestSuiteWithParam<IndexHandlerTestParams> {
 public:
  vdb::DistanceSpace GetDistanceSpace();
};

class DenseIndexTest : public IndexHandlerTestSuite {
 public:
  void SetUp() override {
    vdb::BaseTestSuiteWithParam<IndexHandlerTestParams>::SetUp();
  }
};
const std::vector<std::string> INDEX_TYPES = {"Hnsw"};
const std::vector<std::string> SPACE_TYPES = {"l2space", "ipspace"};

std::vector<IndexHandlerTestParams> GenerateIndexHandlerTestParams() {
  std::vector<IndexHandlerTestParams> params;
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
    IndexTypes, DenseIndexTest,
    testing::ValuesIn(GenerateIndexHandlerTestParams()),
    [](const testing::TestParamInfo<DenseIndexTest::ParamType> &info) {
      return info.param.ToString();
    });

vdb::DistanceSpace IndexHandlerTestSuite::GetDistanceSpace() {
  const auto &params = GetParam();
  if (params.space_type == "l2space") {
    return DistanceSpace::kL2;
  } else if (params.space_type == "ipspace") {
    return DistanceSpace::kIP;
  } else if (params.space_type == "cosinespace") {
    return DistanceSpace::kCosine;
  }
  return DistanceSpace::kL2;
}

TEST_P(DenseIndexTest, DenseIndexSearchKnnTest) {
  auto space = GetDistanceSpace();
  auto index_type = GetParam().index_type;
  auto space_string = GetParam().space_type;
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 16;

  std::string test_table_name = "dummy_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  server.vdb_active_set_size_limit = 10000;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 24;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_string, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  size_t data_cnt = 1000;
  auto data = generateSequentialFloatArray(data_cnt, dim, 1);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif

  for (size_t i = 0; i < data.size(); i++) {
    float *point = &data[i][0];
    auto status = index_handler->AddEmbeddingForTest(point, i, ann_column_id);
    if (!status.ok()) {
      std::cout << status.ToString() << std::endl;
    }
    ASSERT_TRUE(status.ok());
  }

  auto query_vector = generateSequentialFloatArray(1, dim, 5)[0];
  float *query = &query_vector[0];
  size_t k = 10;
  size_t ef_search = k * 3;
#ifdef _DEBUG_GTEST
  std::cout << "k=" << k << " ef_search=" << ef_search
            << " query=" << VecToStr(query, dim) << std::endl;
#endif
  auto index_result_vec =
      index_handler->SearchKnn(query, ef_search, ann_column_id);
  index_result_vec->resize(k);
  auto index_result = ResultToDistEmbeddingPairs(index_handler, ann_column_id,
                                                 *index_result_vec);

  auto knn_result = SearchExactKnn(query, data, k, space);
  ASSERT_EQ(index_result.size(), knn_result.size());
#ifdef _DEBUG_GTEST
  for (size_t i = 0; i < index_result.size(); i++) {
    auto &index_pair = index_result[i];
    auto &index_dist = index_pair.first;
    auto &index_embedding = index_pair.second;
    auto &knn_pair = knn_result[i];
    auto &knn_dist = knn_pair.first;
    auto &knn_embedding = knn_pair.second;

    std::cout << i << "th index result: dist=" << index_dist
              << ", embedding=" << VecToStr(index_embedding.data(), dim);
    std::cout << " exact knn result: dist=" << knn_dist
              << ", embedding=" << VecToStr(knn_embedding.data(), dim)
              << std::endl;
  }
#endif
  auto accuracy = CalculateKnnAccuracy(index_result, knn_result);
  ASSERT_GE(accuracy, 90.0);
#ifdef _DEBUG_GTEST
  std::cout << "Accuracy=" << accuracy << std::endl;
  std::cout << "index count=" << index_handler->Size() << std::endl;
#endif
}

TEST_P(DenseIndexTest, MultiIndexSearchKnnTest) {
  auto space = GetDistanceSpace();
  auto index_type = GetParam().index_type;
  auto space_string = GetParam().space_type;
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 16;

  std::string test_table_name = "dummy_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  server.vdb_active_set_size_limit = 100;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  size_t ann_column_id = 3;
  size_t ef_construction = 100;
  size_t M = 24;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_string, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  size_t data_cnt = 1000;
  auto data = generateSequentialFloatArray(data_cnt, dim, 1);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif

  uint32_t set_number = 0;
  uint32_t record_number = 0;
  for (size_t i = 0; i < data.size(); i++) {
    if (i > 0 && (i % server.vdb_active_set_size_limit) == 0) {
      set_number++;
      record_number = 0;
    }
    float *point = &data[i][0];
    auto label = LabelInfo::Build(0, set_number, record_number);
    auto status =
        index_handler->AddEmbeddingForTest(point, label, ann_column_id);
    if (!status.ok()) {
      std::cout << status.ToString() << std::endl;
    }
    ASSERT_TRUE(status.ok());
    record_number++;
  }

  auto query_vector = generateSequentialFloatArray(1, dim, 5)[0];
  float *query = &query_vector[0];
  size_t k = 10;
  size_t ef_search = k * 3;
#ifdef _DEBUG_GTEST
  std::cout << "k=" << k << " ef_search=" << ef_search
            << " query=" << VecToStr(query, dim) << std::endl;
#endif
  auto index_result_vec =
      index_handler->SearchKnn(query, ef_search, ann_column_id);
  index_result_vec->resize(k);
  auto index_result = ResultToDistEmbeddingPairs(index_handler, ann_column_id,
                                                 *index_result_vec);

  auto knn_result = SearchExactKnn(query, data, k, space);
  ASSERT_EQ(index_result.size(), knn_result.size());
#ifdef _DEBUG_GTEST
  for (size_t i = 0; i < index_result.size(); i++) {
    auto &index_pair = index_result[i];
    auto &index_dist = index_pair.first;
    auto &index_embedding = index_pair.second;
    auto &knn_pair = knn_result[i];
    auto &knn_dist = knn_pair.first;
    auto &knn_embedding = knn_pair.second;

    std::cout << i << "th index result: dist=" << index_dist
              << ", embedding=" << VecToStr(index_embedding.data(), dim);
    std::cout << " exact knn result: dist=" << knn_dist
              << ", embedding=" << VecToStr(knn_embedding.data(), dim)
              << std::endl;
  }
#endif
  auto accuracy = CalculateKnnAccuracy(index_result, knn_result);
  ASSERT_GE(accuracy, 90.0);
#ifdef _DEBUG_GTEST
  std::cout << "Accuracy=" << accuracy << std::endl;
  std::cout << "index count=" << index_handler->Size() << std::endl;
#endif
}

TEST_P(DenseIndexTest, ManyPointsSearchKnnTest) {
  auto space = GetDistanceSpace();
  auto index_type = GetParam().index_type;
  auto space_string = GetParam().space_type;
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 16;

  std::string test_table_name = "dummy_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  server.vdb_active_set_size_limit = 3000;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  size_t ann_column_id = 3;
  size_t ef_construction = 10;
  size_t M = 24;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_string, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  size_t data_cnt = 3000;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif

  size_t active_set_size_limit = table->GetActiveSetSizeLimit();
  uint32_t set_number = 0;
  uint32_t record_number = 0;
  for (size_t i = 0; i < data.size(); i++) {
    if (i > 0 && (i % active_set_size_limit) == 0) {
      set_number++;
      record_number = 0;
    }
    float *point = &data[i][0];
    auto label = LabelInfo::Build(0, set_number, record_number);
    auto status =
        index_handler->AddEmbeddingForTest(point, label, ann_column_id);
    if (!status.ok()) {
      std::cout << status.ToString() << std::endl;
    }
    ASSERT_TRUE(status.ok());
    record_number++;
  }

  auto query_vector = generateSequentialFloatArray(1, dim, 5)[0];
  float *query = &query_vector[0];
  size_t k = 100;
  size_t ef_search = k * 10;
#ifdef _DEBUG_GTEST
  std::cout << "k=" << k << " ef_search=" << ef_search
            << " query=" << VecToStr(query, dim) << std::endl;
#endif
  auto index_result_vec =
      index_handler->SearchKnn(query, ef_search, ann_column_id);
  index_result_vec->resize(k);
  auto index_result = ResultToDistEmbeddingPairs(index_handler, ann_column_id,
                                                 *index_result_vec);

  auto knn_result = SearchExactKnn(query, data, k, space);
  ASSERT_EQ(index_result.size(), knn_result.size());
#ifdef _DEBUG_GTEST
  for (size_t i = 0; i < index_result.size(); i++) {
    auto &index_pair = index_result[i];
    auto &index_dist = index_pair.first;
    auto &index_embedding = index_pair.second;
    auto &knn_pair = knn_result[i];
    auto &knn_dist = knn_pair.first;
    auto &knn_embedding = knn_pair.second;

    std::cout << i << "th index result: dist=" << index_dist
              << ", embedding=" << VecToStr(index_embedding.data(), dim);
    std::cout << " exact knn result: dist=" << knn_dist
              << ", embedding=" << VecToStr(knn_embedding.data(), dim)
              << std::endl;
  }
#endif
  auto accuracy = CalculateKnnAccuracy(index_result, knn_result);
  ASSERT_GE(accuracy, 90.0);
#ifdef _DEBUG_GTEST
  std::cout << "Accuracy=" << accuracy << std::endl;
  std::cout << "index count=" << index_handler->Size() << std::endl;
#endif
}

TEST_P(DenseIndexTest, GetEmbeddingArrayTest) {
  auto space = GetDistanceSpace();
  auto index_type = GetParam().index_type;
  auto space_string = GetParam().space_type;
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 16;

  std::string test_table_name = "dummy_table";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  server.vdb_active_set_size_limit = 100;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  size_t ann_column_id = 3;
  size_t ef_construction = 10;
  size_t M = 24;
  std::string index_info_str = MakeDenseIndexInfoString(
      ann_column_id, index_type, space_string, ef_construction, M);
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  size_t data_cnt = 1000;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif

  size_t active_set_size_limit = table->GetActiveSetSizeLimit();
  size_t set_number = 0;
  for (size_t i = 0; i < data.size(); i++) {
    if (i > 0 && (i % active_set_size_limit) == 0) {
      set_number++;
    }
    float *point = &data[i][0];
    uint64_t label = vdb::LabelInfo::Build(0, i / active_set_size_limit,
                                           i % active_set_size_limit);
    auto status =
        index_handler->AddEmbeddingForTest(point, label, ann_column_id);
    if (!status.ok()) {
      std::cout << status.ToString() << std::endl;
    }
    ASSERT_TRUE(status.ok());
  }

  std::vector<uint64_t> selected_labels = {
      LabelInfo::Build(0, 3, 15),  // set 3, record 15
      LabelInfo::Build(0, 7, 8),   // set 7, record 8
      LabelInfo::Build(0, 1, 42),  // set 1, record 42
      LabelInfo::Build(0, 8, 23),  // set 8, record 23
      LabelInfo::Build(0, 4, 37),  // set 4, record 37
      LabelInfo::Build(0, 0, 61),  // set 0, record 61
      LabelInfo::Build(0, 9, 19),  // set 9, record 19
      LabelInfo::Build(0, 2, 45),  // set 2, record 45
      LabelInfo::Build(0, 6, 12),  // set 6, record 12
      LabelInfo::Build(0, 5, 33)   // set 5, record 33
  };

  ASSERT_OK_AND_ASSIGN(
      auto embedding_array,
      index_handler->GetEmbeddingArray(ann_column_id, selected_labels.data(),
                                       selected_labels.size()));
  ASSERT_EQ(embedding_array->length(), selected_labels.size());
  for (size_t i = 0; i < selected_labels.size(); i++) {
    auto &label = selected_labels[i];
    auto inserted_order =
        LabelInfo::GetSetNumber(label) * active_set_size_limit +
        LabelInfo::GetRecordNumber(label);
    auto embedding_array_ =
        std::dynamic_pointer_cast<arrow::FixedSizeListArray>(embedding_array);
    std::shared_ptr<arrow::Array> embedding = embedding_array_->value_slice(i);
    auto embedding_value =
        std::dynamic_pointer_cast<arrow::FloatArray>(embedding);
    for (size_t j = 0; j < dim; j++) {
      ASSERT_FLOAT_EQ(embedding_value->Value(j), data[inserted_order][j]);
    }
  }
}

TEST_P(DenseIndexTest, GetEmbeddingArrayInvalidSetTest) {
  auto index_type = GetParam().index_type;
  /* dummy table for set metadata */
  auto table_dictionary = vdb::GetTableDictionary();
  size_t dim = 16;

  std::string test_table_name = "dummy_table_invalid_set";
  std::string test_schema_string =
      "ID uint32 not null, Name String, Attributes List[ String ], Feature "
      "Fixed_Size_List[ " +
      std::to_string(dim) + ",   Float32 ]";

  server.vdb_active_set_size_limit = 100;
  auto status = CreateTableForTest(test_table_name, test_schema_string);

  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    ASSERT_TRUE(status.ok());
  }

  auto table = table_dictionary->at(test_table_name);
  std::string index_info_str;
  if (index_type == "Hnsw") {
    index_info_str =
        R"([{"column_id": "3", "index_type": "Hnsw", "parameters": {"space": "L2Space", "ef_construction": "10", "M": "24"}}])";
  }
  auto add_metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>{
          {"index_info", index_info_str}});
  TableWrapper::AddMetadata(table, add_metadata);
  uint64_t ann_column_id = 3;
  TableWrapper::AddEmbeddingStore(table, ann_column_id);

  auto index_handler = std::make_shared<IndexHandlerForTest>(table, 0);

  // Insert data for only 3 sets (sets 0, 1, 2)
  size_t data_cnt = 300;  // Will create 3 sets with 100 records each
  auto data = generateRandomFloatArray(data_cnt, dim);

  size_t active_set_size_limit = table->GetActiveSetSizeLimit();
  for (size_t i = 0; i < data.size(); i++) {
    float *point = &data[i][0];
    uint64_t label = vdb::LabelInfo::Build(0, i / active_set_size_limit,
                                           i % active_set_size_limit);
    auto status =
        index_handler->AddEmbeddingForTest(point, label, ann_column_id);
    ASSERT_TRUE(status.ok());
  }

#ifdef _DEBUG_GTEST
  std::cout << "Created " << (data_cnt / active_set_size_limit) << " sets with "
            << active_set_size_limit << " records each" << std::endl;
#endif

  // Test Case 1: Request embedding from non-existent set (set 5)
  {
    std::vector<uint64_t> invalid_labels = {
        LabelInfo::Build(0, 5, 10),  // set 5 doesn't exist (only 0,1,2 exist)
    };

    auto result = index_handler->GetEmbeddingArray(
        ann_column_id, invalid_labels.data(), invalid_labels.size());

    ASSERT_FALSE(result.ok());
    ASSERT_TRUE(result.status().IsInvalid());

    std::string error_msg = result.status().ToString();
    ASSERT_TRUE(error_msg.find("Invalid set id: 5") != std::string::npos);
    ASSERT_TRUE(error_msg.find("Available sets: 3") != std::string::npos);

#ifdef _DEBUG_GTEST
    std::cout << "Expected error for set 5: " << error_msg << std::endl;
#endif
  }

  // Test Case 2: Mix of valid and invalid set requests
  {
    std::vector<uint64_t> mixed_labels = {
        LabelInfo::Build(0, 1, 15),  // valid: set 1, record 15
        LabelInfo::Build(0, 8, 23),  // invalid: set 8 doesn't exist
        LabelInfo::Build(0, 2, 45),  // valid: set 2, record 45
    };

    auto result = index_handler->GetEmbeddingArray(
        ann_column_id, mixed_labels.data(), mixed_labels.size());

    ASSERT_FALSE(result.ok());
    ASSERT_TRUE(result.status().IsInvalid());

    std::string error_msg = result.status().ToString();
    ASSERT_TRUE(error_msg.find("Invalid set id: 8") != std::string::npos);
    ASSERT_TRUE(error_msg.find("Available sets: 3") != std::string::npos);

#ifdef _DEBUG_GTEST
    std::cout << "Expected error for mixed case: " << error_msg << std::endl;
#endif
  }

  // Test Case 3: Very large invalid set number
  {
    std::vector<uint64_t> large_invalid_labels = {
        LabelInfo::Build(0, 9999, 0),  // extremely large set number
    };

    auto result = index_handler->GetEmbeddingArray(ann_column_id,
                                                   large_invalid_labels.data(),
                                                   large_invalid_labels.size());

    ASSERT_FALSE(result.ok());
    ASSERT_TRUE(result.status().IsInvalid());

    std::string error_msg = result.status().ToString();
    ASSERT_TRUE(error_msg.find("Invalid set id: 9999") != std::string::npos);
    ASSERT_TRUE(error_msg.find("Available sets: 3") != std::string::npos);

#ifdef _DEBUG_GTEST
    std::cout << "Expected error for large invalid set: " << error_msg
              << std::endl;
#endif
  }

  // Test Case 4: Valid case to ensure normal operation still works
  {
    std::vector<uint64_t> valid_labels = {
        LabelInfo::Build(0, 0, 15),  // set 0, record 15
        LabelInfo::Build(0, 1, 8),   // set 1, record 8
        LabelInfo::Build(0, 2, 42),  // set 2, record 42
    };

    ASSERT_OK_AND_ASSIGN(
        auto embedding_array,
        index_handler->GetEmbeddingArray(ann_column_id, valid_labels.data(),
                                         valid_labels.size()));

    ASSERT_EQ(embedding_array->length(), valid_labels.size());

#ifdef _DEBUG_GTEST
    std::cout << "Valid case worked correctly with "
              << embedding_array->length() << " embeddings" << std::endl;
#endif
  }
}
}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
