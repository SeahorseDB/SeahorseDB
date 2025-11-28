#pragma once
#include <memory>

#include "vdb/data/table.hh"
#include "vdb/vdb.hh"
#include "vdb/vdb_api.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/data/embedding_store.hh"
#include "vdb/vector/distance/space_header.hh"

#ifdef __cplusplus
extern "C" {
#endif
#include "sds.h"
#include "server.h"
#undef __str
#undef min
#undef max
#ifdef __cplusplus
}
#endif
extern std::string empty_string;

#define CAST_OR_HANDLE_ERROR_IMPL(handle_error, status_name, lhs, type, rexpr) \
  auto &&status_name = (rexpr);                                                \
  handle_error(status_name.status());                                          \
  lhs = static_cast<type>(std::move(status_name).ValueOrDie());

#define ASSERT_OK_AND_CAST(lhs, type, rexpr)                               \
  CAST_OR_HANDLE_ERROR_IMPL(                                               \
      ASSERT_OK, ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), \
      lhs, type, rexpr);

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>>
CreateFixedSizeListArray(const float *data, size_t dimension);

/* index handler for test */
class IndexHandlerForTest : public vdb::IndexHandler {
 public:
  explicit IndexHandlerForTest(std::shared_ptr<vdb::Table> table,
                               const uint16_t segment_number)
      : vdb::IndexHandler{table, segment_number} {}
  explicit IndexHandlerForTest(std::shared_ptr<vdb::Table> table,
                               const uint16_t segment_number,
                               std::string &index_directory_path,
                               const uint64_t index_count)
      : vdb::IndexHandler{table, segment_number, index_directory_path,
                          index_count} {}
  vdb::Status AddEmbeddingForTest(const float *data, uint64_t label,
                                  uint64_t column_id) {
    auto indexes = indexes_[column_id];
    auto index = indexes.back();
    if (index->IsFull()) {
      if (auto status = CreateIndex(); !status.ok()) {
        return status;
      }
      indexes = indexes_[column_id];
      index = indexes.back();
    }
    auto status = index->AddEmbedding(data, label);
    if (!status.ok()) {
      return status;
    }
    auto embedding_store = GetEmbeddingStore(column_id);
    status = embedding_store->Write(data, label, 1);
    if (!status.ok()) {
      return status;
    }

    return vdb::Status::Ok();
  }

  vdb::map<uint64_t, vdb::vector<std::shared_ptr<vdb::VectorIndex>>>
  GetIndexMap() {
    return indexes_;
  }
};
/* to string */
std::string VecToStr(const float *vec, size_t dim);
/* comparison */
bool EmbeddingEquals(const float *lhs, const float *rhs, size_t dimension);
bool EmbeddingEquals(const std::vector<float> &lhs,
                     const std::vector<float> &rhs, size_t dimension);
bool DenseIndexEquals(std::shared_ptr<vdb::DenseVectorIndex> &lhs,
                      std::shared_ptr<vdb::DenseVectorIndex> &rhs);
bool IndexHandlerEquals(std::shared_ptr<vdb::IndexHandler> &lhs,
                        std::shared_ptr<vdb::IndexHandler> &rhs);
bool SegmentEquals(std::shared_ptr<vdb::Segment> &lhs,
                   std::shared_ptr<vdb::Segment> &rhs);
bool TableEquals(std::shared_ptr<vdb::Table> &lhs,
                 std::shared_ptr<vdb::Table> &rhs);
/* check index build complete */
bool TableIndexFullBuildCheck(std::shared_ptr<vdb::Table> &table,
                              const bool print_log);
/* generate sample data */
std::vector<std::vector<float>> generateSequentialFloatArray(int rows, int dim,
                                                             int start);
std::vector<std::vector<float>> generateRandomFloatArray(int rows, int dim,
                                                         bool xy_same = false);

arrow::Result<std::shared_ptr<arrow::Array>> GenerateRandomInt32Array(
    int64_t length, bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::Array>> GenerateRandomFloat32Array(
    int64_t length, bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::Array>> GenerateRandomStringArray(
    int64_t length, bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::Array>> GeneratePrimaryKeyStringArray(
    int64_t length, std::string prefix, int64_t start_number,
    bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::Array>> GeneratePrimaryKeyLargeStringArray(
    int64_t length, std::string prefix, int64_t start_number,
    bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::Array>>
GenerateRandomFloatFixedSizeListArray(int64_t length, int32_t list_size,
                                      bool use_vdb_pool);
arrow::Result<std::shared_ptr<arrow::RecordBatch>> GenerateRecordBatch(
    std::shared_ptr<arrow::Schema> schema, int64_t length, int32_t dimension,
    bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::RecordBatch>>
GenerateRecordBatchWithPrimaryKey(std::shared_ptr<arrow::Schema> schema,
                                  std::string prefix, int64_t start_number,
                                  int64_t length, int32_t dimension,
                                  bool use_large_string = false,
                                  bool use_vdb_pool = true);
arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromRecordBatch(
    const std::shared_ptr<arrow::RecordBatch> &batch, int column_id,
    int record_id, int record_count);
arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromRecordBatch(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const std::string &column_name, int record_id, int record_count);
arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromTable(
    const std::shared_ptr<arrow::Table> &table, int column_id, int record_id,
    int record_count);
arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromTable(
    const std::shared_ptr<arrow::Table> &table, const std::string &column_name,
    int record_id, int record_count);

/* exact knn */
float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProduct(const void *pVect1v, const void *pVect2v,
                   const void *qty_ptr);
float InnerProductDistance(const void *pVect1v, const void *pVect2v,
                           const void *qty_ptr);

std::vector<std::pair<float, std::vector<float>>> SearchExactKnn(
    const float *query, const std::vector<std::vector<float>> &points, size_t k,
    vdb::DistanceSpace space);

/* handle IPC to Table/RecordBatches */
arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeRecordBatches(
    std::shared_ptr<arrow::Schema> schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>> &record_batches);
arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeTable(
    std::shared_ptr<arrow::Table> table);
arrow::Result<std::shared_ptr<arrow::Table>> DeserializeToTableFrom(
    const std::shared_ptr<arrow::Buffer> &serialized_rb);
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
DeserializeToRecordBatchesFrom(
    const std::shared_ptr<arrow::Buffer> &serialized_rb);

/* sort a table */
arrow::Result<std::shared_ptr<arrow::Table>> SortTable(
    const std::shared_ptr<arrow::Table> &table);

/* snapshot test names */
std::string GetBaseSnapshotDirectoryName(long long sequence);
std::string GetTempSnapshotDirectoryName(pid_t pid);
/* etc */
std::vector<std::pair<float, std::vector<float>>> ResultToDistEmbeddingPairs(
    const std::shared_ptr<vdb::DenseVectorIndex> &index,
    std::priority_queue<std::pair<float, uint64_t>> &pq);
std::vector<std::pair<float, std::vector<float>>> ResultToDistEmbeddingPairs(
    const std::shared_ptr<IndexHandlerForTest> &index_handler,
    const uint64_t column_id, std::vector<std::pair<float, uint64_t>> &vec);
float CalculateKnnAccuracy(
    std::vector<std::pair<float, std::vector<float>>> &index_result,
    std::vector<std::pair<float, std::vector<float>>> &knn_result);

vdb::Status CreateTableForTest(std::string table_name,
                               std::string schema_string,
                               bool without_segment_id = false);

std::string MakeSegmentationInfoString(
    std::string segment_type, std::string segment_keys,
    std::string segment_key_composition_type);
std::string MakeDenseIndexInfoString(size_t column_id, std::string index_type,
                                     std::string space_type,
                                     size_t ef_construction, size_t M);

arrow::Result<sds> CreateSerializedDfData(
    const std::vector<std::pair<uint64_t, uint64_t>> &term_df_pairs);

namespace vdb::tests {

class TableWrapper {
 public:
  static void AddMetadata(
      std::shared_ptr<vdb::Table> table,
      const std::shared_ptr<arrow::KeyValueMetadata> &metadata) {
    table->AddMetadata(metadata);
  }
  static void AddMetadataToField(
      std::shared_ptr<vdb::Table> table, uint32_t field_idx,
      const std::shared_ptr<arrow::KeyValueMetadata> &metadata) {
    table->AddMetadataToField(field_idx, metadata);
  }
  static void AddMetadataToField(
      std::shared_ptr<vdb::Table> table, const std::string &field_name,
      const std::shared_ptr<arrow::KeyValueMetadata> &metadata) {
    table->AddMetadataToField(field_name, metadata);
  }
  static void AddEmbeddingStore(std::shared_ptr<vdb::Table> table,
                                const uint64_t column_id) {
    auto embedding_store =
        vdb::make_shared<vdb::EmbeddingStore>(table.get(), column_id);
    table->AddEmbeddingStore(column_id, embedding_store);
  }
};

}  // namespace vdb::tests
