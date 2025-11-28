#include <memory>
#include <queue>
#include <string>
#include <strings.h>
#include <utility>
#include <gtest/gtest.h>

#include <arrow/testing/gtest_util.h>

#include "vdb/tests/base_environment.hh"
#include "vdb/tests/util_for_test.hh"
#include "vdb/data/embedding_store.hh"
#include "vdb/data/index_handler.hh"

using namespace vdb::tests;

namespace vdb {

std::string test_suite_directory_path =
    test_root_directory_path + "/IndexTestSuite";

struct IndexTestParams {
  std::string index_type;
  std::string space_type;

  IndexTestParams(const std::string &index_type, const std::string &space_type)
      : index_type(index_type), space_type(space_type) {}

  std::string ToString() const {
    std::string result = index_type + "_" + space_type;
    return result;
  }
};

const std::vector<std::string> INDEX_TYPES = {"Hnsw"};
const std::vector<std::string> SPACE_TYPES = {"l2space", "ipspace"};

std::vector<IndexTestParams> GenerateIndexTestParams() {
  std::vector<IndexTestParams> params;
  for (const auto &index_type : INDEX_TYPES) {
    for (const auto &space_type : SPACE_TYPES) {
      if (index_type == "Hnsw") {
        params.emplace_back(index_type, space_type);
      }
    }
  }
  return params;
}

class IndexTestSuite : public vdb::BaseTestSuiteWithParam<IndexTestParams> {
 protected:
  void SetUp() override {
    vdb::BaseTestSuiteWithParam<IndexTestParams>::SetUp();
  }
};

class IndexTest : public IndexTestSuite {
 public:
  std::shared_ptr<DenseVectorIndex> CreateDenseIndex(
      DistanceSpace space, size_t dim, size_t ef_construction, size_t M,
      size_t max_elem, std::shared_ptr<EmbeddingStore> embedding_store);

  vdb::DistanceSpace GetDistanceSpace();
};

INSTANTIATE_TEST_SUITE_P(
    IndexTypes, IndexTest, testing::ValuesIn(GenerateIndexTestParams()),
    [](const testing::TestParamInfo<IndexTest::ParamType> &info) {
      return info.param.ToString();
    });

std::shared_ptr<DenseVectorIndex> IndexTest::CreateDenseIndex(
    DistanceSpace space, size_t dim, size_t ef_construction, size_t M,
    size_t max_elem, std::shared_ptr<EmbeddingStore> embedding_store) {
  const auto &params = GetParam();

  if (params.index_type == "Hnsw") {
    return std::make_shared<vdb::Hnsw>(space, dim, ef_construction, M, max_elem,
                                       embedding_store);
  }
  return nullptr;
}

vdb::DistanceSpace IndexTest::GetDistanceSpace() {
  const auto &params = GetParam();
  if (params.space_type == "l2space") {
    return DistanceSpace::kL2;
  } else if (params.space_type == "ipspace") {
    return DistanceSpace::kIP;
  }
  return DistanceSpace::kL2;
}

TEST_P(IndexTest, SearchKnnSimple8DTest) {
  DistanceSpace space = GetDistanceSpace();
  size_t ef_construction = 100;
  size_t M = 3;
  size_t max_elem = 1000;
  size_t dim = 8;
  size_t data_cnt = 10;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "Index type: " << GetParam().index_type
            << ", Space type: " << GetParam().space_type
            << ", dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif
  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, dim);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  auto index = CreateDenseIndex(space, dim, ef_construction, M, max_elem,
                                embedding_store);
  /* {1,1} ... {10,10} */
  for (size_t i = 0; i < data_cnt; i++) {
    float *point = &data[i][0];
    index->AddEmbedding(point, i);
  }
  auto query_vector = generateSequentialFloatArray(1, dim, 5)[0];
  float *query = &query_vector[0];
  size_t k = 5;
#ifdef _DEBUG_GTEST
  std::cout << "k=" << k << " ef_search=" << k
            << " query=" << VecToStr(query, dim) << std::endl;
#endif
  auto index_result_pq = index->SearchKnn(query, k);
  size_t pq_size = index_result_pq.size();
  auto index_result = ResultToDistEmbeddingPairs(index, index_result_pq);
  auto knn_result = SearchExactKnn(query, data, k, space);
  ASSERT_EQ(pq_size, index_result.size());
  ASSERT_EQ(index_result.size(), knn_result.size());
#ifdef _DEBUG_GTEST
  for (size_t i = 0; i < index_result.size(); i++) {
    auto &index_pair = index_result[i];
    auto &index_dist = index_pair.first;
    auto &index_embedding = index_pair.second;
    auto &knn_pair = knn_result[i];
    auto &knn_dist = knn_pair.first;
    auto &knn_embedding = knn_pair.second;

    std::cout << i << "th index result: dist=" << index_dist << ", embedding=["
              << index_embedding[0] << "," << index_embedding[1] << "] ";
    std::cout << " exact knn result: dist=" << knn_dist << ", embedding=["
              << knn_embedding[0] << "," << knn_embedding[1] << "]"
              << std::endl;
  }
#endif
  auto accuracy = CalculateKnnAccuracy(index_result, knn_result);
  ASSERT_GE(accuracy, 90.0);
#ifdef _DEBUG_GTEST
  std::cout << "Accuracy=" << accuracy << std::endl;
#endif
}

TEST_P(IndexTest, SearchKnnSimple16DTest) {
  DistanceSpace space = GetDistanceSpace();
  size_t data_cnt = 10;
  size_t dim = 16;
  auto data = generateSequentialFloatArray(data_cnt, dim, 1);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif
  /* {1,1} ... {10,10} */
  size_t ef_construction = 100;
  size_t M = 3;
  size_t max_elem = 1000;
  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, dim);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  auto index = CreateDenseIndex(space, dim, ef_construction, M, max_elem,
                                embedding_store);
  for (size_t i = 0; i < data_cnt; i++) {
    float *point = &data[i][0];
    index->AddEmbedding(point, i);
  }
  auto query_vector = generateSequentialFloatArray(1, dim, 5)[0];
  float *query = &query_vector[0];
  size_t k = 5;
#ifdef _DEBUG_GTEST
  std::cout << "k=" << k << " ef_search=" << k
            << " query=" << VecToStr(query, dim) << std::endl;
#endif
  auto index_result_pq = index->SearchKnn(query, k);
  size_t pq_size = index_result_pq.size();
  auto index_result = ResultToDistEmbeddingPairs(index, index_result_pq);
  auto knn_result = SearchExactKnn(query, data, k, space);
  ASSERT_EQ(pq_size, index_result.size());
  ASSERT_EQ(index_result.size(), knn_result.size());
#ifdef _DEBUG_GTEST
  for (size_t i = 0; i < index_result.size(); i++) {
    auto &index_pair = index_result[i];
    auto &index_dist = index_pair.first;
    auto &index_embedding = index_pair.second;
    // auto &index_embedding = index_pair.second;
    auto &knn_pair = knn_result[i];
    auto &knn_dist = knn_pair.first;
    auto &knn_embedding = knn_pair.second;
    // auto &knn_embedding = knn_pair.second;

    ASSERT_FLOAT_EQ(index_dist, knn_dist);
    std::cout << i << "th index result: dist=" << index_dist << ", embedding=["
              << index_embedding[0] << "," << index_embedding[1] << "] ";
    std::cout << " exact knn result: dist=" << knn_dist << ", embedding=["
              << knn_embedding[0] << "," << knn_embedding[1] << "]"
              << std::endl;
  }
#endif
  auto accuracy = CalculateKnnAccuracy(index_result, knn_result);
  ASSERT_GE(accuracy, 90.0);
#ifdef _DEBUG_GTEST
  std::cout << "Accuracy=" << accuracy << std::endl;
#endif
}

TEST_P(IndexTest, SearchKnnComplex1024DTest) {
  DistanceSpace space = GetDistanceSpace();
  size_t data_cnt = 3000;
  size_t dim = 1024;
  auto data = generateRandomFloatArray(data_cnt, dim);
#ifdef _DEBUG_GTEST
  std::cout << "dimension=" << dim << " data size=" << data_cnt << std::endl;
#endif
  size_t ef_construction = 100;
  size_t M = 24;
  size_t max_elem = 10000;
  auto embedding_store =
      std::make_shared<EmbeddingStore>(EmbeddingStoreDirectoryPath(), 0, dim);
  auto status = embedding_store->CreateSegmentAndColumnDirectory(0);
  ASSERT_TRUE(status.ok());
  auto index = CreateDenseIndex(space, dim, ef_construction, M, max_elem,
                                embedding_store);
  for (size_t i = 0; i < data_cnt; i++) {
    float *point = &data[i][0];
    index->AddEmbedding(point, i);
  }
  auto query_vector = generateSequentialFloatArray(1, dim, 5)[0];
  float *query = &query_vector[0];
  size_t k = 100;
  size_t ef_search = k * 10;
#ifdef _DEBUG_GTEST
  std::cout << "k=" << k << " ef_search=" << ef_search
            << " query=" << VecToStr(query, dim) << std::endl;
#endif
  auto index_result_pq = index->SearchKnn(query, ef_search);
  size_t pq_size = index_result_pq.size();
  auto index_result = ResultToDistEmbeddingPairs(index, index_result_pq);
  ASSERT_EQ(pq_size, index_result.size());
  index_result.resize(k);

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
#endif
}
}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
