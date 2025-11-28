#include <memory>
#include <random>

#include <arrow/api.h>
#include <arrow/acero/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/type_fwd.h>
#include <string>

#include "vdb/common/defs.hh"
#include "vdb/common/memory_allocator.hh"
#include "vdb/data/table.hh"
#include "vdb/tests/util_for_test.hh"

std::string empty_string = "";
/* to string */
std::string VecToStr(const float *vec, size_t dim) {
  std::stringstream ss;

  ss << "[ ";
  for (size_t i = 0; i < dim; i++) {
    ss << vec[i];
    if (i != dim - 1) ss << ", ";
  }

  ss << " ]";
  return ss.str();
}
/* comparison */
bool EmbeddingEquals(const float *lhs, const float *rhs, size_t dimension) {
  for (size_t i = 0; i < dimension; i++) {
    if (lhs[i] != rhs[i]) {
      std::cout << i << "th Value in Embedding is NOT MATCHED!" << std::endl;
      std::cout << lhs[i] << std::endl;
      std::cout << rhs[i] << std::endl;
      return false;
    }
  }
  return true;
}

bool EmbeddingEquals(const std::vector<float> &lhs,
                     const std::vector<float> &rhs, size_t dimension) {
  if (lhs.size() != dimension || rhs.size() != dimension) {
    std::cout << "Embedding size is NOT MATCHED!" << std::endl;
    std::cout << lhs.size() << std::endl;
    std::cout << rhs.size() << std::endl;
    return false;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (lhs[i] != rhs[i]) {
      std::cout << i << "th Value in Embedding is NOT MATCHED!" << std::endl;
      std::cout << lhs[i] << std::endl;
      std::cout << rhs[i] << std::endl;
      return false;
    }
  }
  return true;
}

bool DenseIndexEquals(std::shared_ptr<vdb::DenseVectorIndex> &lhs,
                      std::shared_ptr<vdb::DenseVectorIndex> &rhs) {
  /* compare metadata */
  if (lhs->Size() != rhs->Size()) {
    std::cout << "Index Size Compare is failed" << std::endl;
    std::cout << lhs->Size() << std::endl;
    std::cout << rhs->Size() << std::endl;
    return false;
  }

  if (lhs->Dimension() != rhs->Dimension()) {
    std::cout << "Index Dimension is NOT MATCHED!" << std::endl;
    std::cout << lhs->Dimension() << std::endl;
    std::cout << rhs->Dimension() << std::endl;
    return false;
  }

  if (lhs->ToString().compare(rhs->ToString()) != 0) {
    std::cout << "Index ToString is NOT MATCHED!" << std::endl;
    std::cout << lhs->ToString() << std::endl;
    std::cout << rhs->ToString() << std::endl;
    return false;
  }

  for (size_t i = 0; i < lhs->Size(); i++) {
    auto embedding_1 = lhs->GetEmbeddingByInternalId(i);
    auto embedding_2 = rhs->GetEmbeddingByInternalId(i);
    if (!EmbeddingEquals(embedding_1.data(), embedding_2.data(),
                         lhs->Dimension())) {
      std::cout << i << "th embedding is NOT MATCHED!" << std::endl;
      std::cout << VecToStr(embedding_1.data(), lhs->Dimension()) << std::endl;
      std::cout << VecToStr(embedding_2.data(), rhs->Dimension()) << std::endl;
      return false;
    }
  }
  /* TODO compare edges */
  return true;
}

bool IndexHandlerEquals(std::shared_ptr<vdb::IndexHandler> &lhs,
                        std::shared_ptr<vdb::IndexHandler> &rhs) {
  /* compare index count */
  if (lhs->Size() != rhs->Size()) {
    std::cout << "Index Count is NOT MATCHED!" << std::endl;
    std::cout << lhs->Size() << std::endl;
    std::cout << rhs->Size() << std::endl;
    return false;
  }
  /* compare indexes */
  auto lhs_for_test = std::static_pointer_cast<IndexHandlerForTest>(lhs);
  auto rhs_for_test = std::static_pointer_cast<IndexHandlerForTest>(rhs);
  auto lhs_indexmap = lhs_for_test->GetIndexMap();
  auto rhs_indexmap = rhs_for_test->GetIndexMap();
  for (auto [column_id, indexes] : lhs_indexmap) {
    for (size_t i = 0; i < indexes.size(); i++) {
      auto index_1 = lhs->Index(column_id, i);
      auto index_2 = rhs->Index(column_id, i);

      auto dense_index_1 =
          std::dynamic_pointer_cast<vdb::DenseVectorIndex>(index_1);
      auto dense_index_2 =
          std::dynamic_pointer_cast<vdb::DenseVectorIndex>(index_2);
      if (dense_index_1 && dense_index_2) {
        if (!DenseIndexEquals(dense_index_1, dense_index_2)) {
          std::cout << i << "th Index is NOT MATCHED!" << std::endl;
          return false;
        }
      } else if ((dense_index_1 && !dense_index_2) ||
                 (!dense_index_1 && dense_index_2)) {
        std::cout << i << "th Index type is NOT MATCHED!" << std::endl;
        return false;
      }
    }
  }

  return true;
}

bool SegmentEquals(std::shared_ptr<vdb::Segment> &lhs,
                   std::shared_ptr<vdb::Segment> &rhs) {
  /* compare inactive batches */
  size_t lhs_inactive_set_count = lhs->InactiveSetCount();
  size_t rhs_inactive_set_count = rhs->InactiveSetCount();
  if (lhs_inactive_set_count != rhs_inactive_set_count) {
    std::cout << "Segment Inactive Set Count is NOT MATCHED! "
              << lhs_inactive_set_count << " " << rhs_inactive_set_count
              << std::endl;
    return false;
  }
  auto lhs_segment_id = lhs->GetId();
  auto rhs_segment_id = rhs->GetId();
  if (lhs_segment_id.compare(rhs_segment_id) != 0) {
    std::cout << "Segment Id is NOT MATCHED! " << lhs_segment_id << " "
              << rhs_segment_id << std::endl;
    return false;
  }
  for (size_t i = 0; i < lhs_inactive_set_count; i++) {
    auto lhs_inactive_set = lhs->GetInactiveSet(i);
    auto rhs_inactive_set = rhs->GetInactiveSet(i);
    auto lhs_inactive_rb = lhs_inactive_set->GetRb();
    auto rhs_inactive_rb = rhs_inactive_set->GetRb();

    if (!lhs_inactive_rb || !rhs_inactive_rb) {
      std::cout << "Inactive set " << i << " is NOT MATCHED!" << std::endl;
      return false;
    }

    if (!lhs_inactive_rb->Equals(*rhs_inactive_rb)) {
      std::cout << i << "th Inactive Rb is NOT MATCHED!" << std::endl;
      std::cout << lhs_inactive_rb->ToString() << std::endl;
      std::cout << rhs_inactive_rb->ToString() << std::endl;
      return false;
    }
  }
  /* compare active batches */
  auto lhs_active_rb = lhs->ActiveSetRecordBatch();
  auto rhs_active_rb = rhs->ActiveSetRecordBatch();
  if (lhs_active_rb == nullptr || rhs_active_rb == nullptr) {
    std::cout << "Cannot get active set from segment: "
              << (lhs_active_rb == nullptr ? "lhs" : "rhs") << std::endl;
    return false;
  }

  if (!lhs_active_rb->Equals(*rhs_active_rb)) {
    std::cout << "Active rb is NOT MATCHED!" << std::endl;
    std::cout << lhs_active_rb->ToString() << std::endl;
    std::cout << rhs_active_rb->ToString() << std::endl;
    return false;
  }

  if (lhs->HasIndex() || rhs->HasIndex()) {
    if (lhs->HasIndex() != rhs->HasIndex()) {
      return false;
    }
    auto lhs_handler = lhs->IndexHandler();
    auto rhs_handler = rhs->IndexHandler();
    if (!IndexHandlerEquals(lhs_handler, rhs_handler)) {
      return false;
    }
  }

  return true;
}

bool TableEquals(std::shared_ptr<vdb::Table> &lhs,
                 std::shared_ptr<vdb::Table> &rhs) {
  /* compare schema */
  if (!lhs->GetSchema()->Equals(rhs->GetSchema(), true)) {
    std::cout << "Table Schema is NOT MATCHED!" << std::endl;
    std::cout << lhs->GetSchema()->ToString() << std::endl;
    std::cout << rhs->GetSchema()->ToString() << std::endl;
    return false;
  }
  /* compare segments */
  for (auto [lhs_id, lhs_segment] : lhs->GetSegments()) {
    auto rhs_segment = rhs->GetSegment(lhs_id.data());
    if (rhs_segment == nullptr) {
      std::cout << lhs_id
                << "id: segment in " + rhs->GetTableName() + " is not found"
                << std::endl;
      return false;
    }
    if (!SegmentEquals(lhs_segment, rhs_segment)) {
      std::cout << lhs_id
                << " id: segment in " + rhs->GetTableName() + " is not found"
                << std::endl;
      return false;
    }
  }
  return true;
}

/* check index build complete */
bool TableIndexFullBuildCheck(std::shared_ptr<vdb::Table> &table,
                              const bool print_log) {
  auto segments = table->GetSegments();
  auto index_infos = table->GetIndexInfos();
  for (auto &[segment_id, segment] : segments) {
    std::shared_ptr<vdb::IndexHandler> index_handler = segment->IndexHandler();
    for (auto &index_info : *index_infos) {
      auto column_id = index_info.GetColumnId();
      size_t num_index = index_handler->Size(column_id);

      for (size_t i = 0; i < num_index; i++) {
        auto index = index_handler->Index(column_id, i);
        auto index_element_count = index->Size();

        std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;
        std::shared_ptr<arrow::RecordBatch> rb;

        if (i == segment->ActiveSetId()) {
          auto maybe_rb = segment->GetRecordbatch(i);
          if (!maybe_rb.ok()) {
            std::cout << "Record batch is not found in " << segment_id
                      << ", set number " << i << std::endl;
            return false;
          }
          rb = maybe_rb.ValueUnsafe();
        } else {
          inactive_set = segment->GetInactiveSet(i);
          rb = inactive_set->GetRb();

          if (!rb) {
            std::cout << "Record batch is not found in " << segment_id
                      << ", set number " << i << std::endl;
            return false;
          }
        }

        size_t row_count = (size_t)rb->num_rows();
        if (index_element_count != row_count) {
          if (print_log) {
            std::cout << "Index is not completely built in "
                      << "segment id " << segment_id << ", set number " << i
                      << ". (index element count: " << index_element_count
                      << ", row count: " << row_count << ")" << std::endl;
          }
          return false;
        }
      }
    }
  }
  return true;
}

/* generate sample data */
std::vector<std::vector<float>> generateSequentialFloatArray(int rows, int dim,
                                                             int start) {
  int cols = dim;
  float start_val = static_cast<float>(start);
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[i][j] = start_val;
    }
    start_val++;
  }
  /* normalize */
  for (int i = 0; i < rows; ++i) {
    float norm = 0.0;
    for (int j = 0; j < cols; ++j) {
      norm += result[i][j] * result[i][j];
    }
    norm = std::sqrt(norm);
    for (int j = 0; j < cols; ++j) {
      result[i][j] = result[i][j] / norm;
    }
  }

  return result;
}

std::vector<std::vector<float>> generateRandomFloatArray(int rows, int dim,
                                                         bool xy_same) {
  int cols = dim;
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

  srand(1748584721);
  for (int i = 0; i < rows; ++i) {
    float value = (static_cast<float>(rand()) * 2 / RAND_MAX - 1);  // -1 ~ 1
    for (int j = 0; j < cols; ++j) {
      /* random float number is assigned */
      if (xy_same) {
        result[i][j] = value;
      } else {
        result[i][j] =
            (static_cast<float>(rand()) * 2 / RAND_MAX - 1);  // -1 ~ 1
      }
    }
  }
  /* normalize */
  for (int i = 0; i < rows; ++i) {
    float norm = 0.0;
    for (int j = 0; j < cols; ++j) {
      norm += result[i][j] * result[i][j];
    }
    norm = std::sqrt(norm);
    for (int j = 0; j < cols; ++j) {
      result[i][j] = result[i][j] / norm;
    }
  }

  return result;
}

arrow::Result<std::shared_ptr<arrow::Array>> GenerateRandomInt32Array(
    int64_t length, bool use_vdb_pool) {
  auto pool = (use_vdb_pool) ? &vdb::arrow_pool : arrow::default_memory_pool();
  arrow::Int32Builder builder(pool);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 100);

  for (int64_t i = 0; i < length; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(dis(gen)));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> GenerateRandomFloat32Array(
    int64_t length, bool use_vdb_pool) {
  auto pool = (use_vdb_pool) ? &vdb::arrow_pool : arrow::default_memory_pool();
  arrow::FloatBuilder builder(pool);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);

  for (int64_t i = 0; i < length; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append(dis(gen)));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> GenerateRandomStringArray(
    int64_t length, bool use_vdb_pool) {
  auto pool = (use_vdb_pool) ? &vdb::arrow_pool : arrow::default_memory_pool();
  arrow::StringBuilder builder(pool);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 100);

  for (int64_t i = 0; i < length; ++i) {
    ARROW_RETURN_NOT_OK(builder.Append("str" + std::to_string(dis(gen))));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> GeneratePrimaryKeyLargeStringArray(
    int64_t length, std::string prefix, int64_t start_number,
    bool use_vdb_pool) {
  auto pool = (use_vdb_pool) ? &vdb::arrow_pool : arrow::default_memory_pool();
  arrow::LargeStringBuilder builder(pool);

  for (int64_t i = 0; i < length; ++i) {
    ARROW_RETURN_NOT_OK(
        builder.Append(prefix + vdb::kRS + std::to_string(start_number + i)));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> GeneratePrimaryKeyStringArray(
    int64_t length, std::string prefix, int64_t start_number,
    bool use_vdb_pool) {
  auto pool = (use_vdb_pool) ? &vdb::arrow_pool : arrow::default_memory_pool();
  arrow::StringBuilder builder(pool);

  for (int64_t i = 0; i < length; ++i) {
    ARROW_RETURN_NOT_OK(
        builder.Append(prefix + vdb::kRS + std::to_string(start_number + i)));
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::Array>>
GenerateRandomFloatFixedSizeListArray(int64_t length, int32_t list_size,
                                      bool use_vdb_pool) {
  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  auto pool = (use_vdb_pool) ? &vdb::arrow_pool : arrow::default_memory_pool();
  arrow::FixedSizeListBuilder builder(pool, value_builder, list_size);
  std::mt19937 gen(0);
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  std::vector<float> values(list_size);

  for (int64_t i = 0; i < length; ++i) {
    for (int32_t j = 0; j < list_size; ++j) {
      values[j] = dis(gen);
    }

    float norm = 0.0f;
    for (int32_t j = 0; j < list_size; ++j) {
      norm += values[j] * values[j];
    }
    norm = std::sqrt(norm);

    ARROW_RETURN_NOT_OK(builder.Append());
    for (int32_t j = 0; j < list_size; ++j) {
      ARROW_RETURN_NOT_OK(value_builder->Append(values[j] / norm));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
GenerateRecordBatchWithPrimaryKey(std::shared_ptr<arrow::Schema> schema,
                                  std::string prefix, int64_t start_number,
                                  int64_t length, int32_t dimension,
                                  bool use_large_string, bool use_vdb_pool) {
  auto maybe_int_array = GenerateRandomInt32Array(length, use_vdb_pool);
  if (!maybe_int_array.ok()) {
    return maybe_int_array.status();
  }
  auto int_array = maybe_int_array.ValueUnsafe();

  auto maybe_float_array = GenerateRandomFloat32Array(length, use_vdb_pool);
  if (!maybe_float_array.ok()) {
    return maybe_float_array.status();
  }
  auto float_array = maybe_float_array.ValueUnsafe();
  std::shared_ptr<arrow::Array> string_array;
  if (use_large_string) {
    auto maybe_string_array = GeneratePrimaryKeyLargeStringArray(
        length, prefix, start_number, use_vdb_pool);
    if (!maybe_string_array.ok()) {
      return maybe_string_array.status();
    }
    string_array = maybe_string_array.ValueUnsafe();
  } else {
    auto maybe_string_array = GeneratePrimaryKeyStringArray(
        length, prefix, start_number, use_vdb_pool);
    if (!maybe_string_array.ok()) {
      return maybe_string_array.status();
    }
    string_array = maybe_string_array.ValueUnsafe();
  }
  auto maybe_fixed_size_list_array =
      GenerateRandomFloatFixedSizeListArray(length, dimension, use_vdb_pool);
  if (!maybe_fixed_size_list_array.ok()) {
    return maybe_fixed_size_list_array.status();
  }
  auto fixed_size_list_array = maybe_fixed_size_list_array.ValueUnsafe();
  std::shared_ptr<arrow::RecordBatch> record_batch = arrow::RecordBatch::Make(
      schema, length,
      std::vector{int_array, string_array, float_array, fixed_size_list_array});
  return record_batch;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> GenerateRecordBatch(
    std::shared_ptr<arrow::Schema> schema, int64_t length, int32_t dimension,
    bool use_vdb_pool) {
  auto maybe_int_array = GenerateRandomInt32Array(length, use_vdb_pool);
  if (!maybe_int_array.ok()) {
    return maybe_int_array.status();
  }
  auto int_array = maybe_int_array.ValueUnsafe();

  auto maybe_float_array = GenerateRandomFloat32Array(length, use_vdb_pool);
  if (!maybe_float_array.ok()) {
    return maybe_float_array.status();
  }
  auto float_array = maybe_float_array.ValueUnsafe();
  auto maybe_string_array = GenerateRandomStringArray(length, use_vdb_pool);
  if (!maybe_string_array.ok()) {
    return maybe_string_array.status();
  }
  auto string_array = maybe_string_array.ValueUnsafe();
  auto maybe_fixed_size_list_array =
      GenerateRandomFloatFixedSizeListArray(length, dimension, use_vdb_pool);
  if (!maybe_fixed_size_list_array.ok()) {
    return maybe_fixed_size_list_array.status();
  }
  auto fixed_size_list_array = maybe_fixed_size_list_array.ValueUnsafe();
  std::shared_ptr<arrow::RecordBatch> record_batch = arrow::RecordBatch::Make(
      schema, length,
      std::vector{int_array, string_array, float_array, fixed_size_list_array});
  return record_batch;
}

arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromRecordBatch(
    const std::shared_ptr<arrow::RecordBatch> &batch, int column_id,
    int record_id, int record_count) {
  if (!batch || column_id >= batch->num_columns() ||
      record_id >= batch->num_rows() ||
      record_id + record_count > batch->num_rows()) {
    return arrow::Status::Invalid("Invalid arguments");
  }

  auto column = batch->column(column_id);
  return column->Slice(record_id, record_count);
}

arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromRecordBatch(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const std::string &column_name, int record_id, int record_count) {
  if (!batch) {
    return arrow::Status::Invalid("RecordBatch is null");
  }

  int column_id = batch->schema()->GetFieldIndex(column_name);
  if (column_id == -1) {
    return arrow::Status::Invalid("Column '", column_name, "' not found");
  }

  if (record_id >= batch->num_rows() ||
      record_id + record_count > batch->num_rows()) {
    return arrow::Status::Invalid("Invalid record range: id=", record_id,
                                  ", count=", record_count,
                                  ", total rows=", batch->num_rows());
  }

  auto column = batch->column(column_id);
  return column->Slice(record_id, record_count);
}

arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromTable(
    const std::shared_ptr<arrow::Table> &table, int column_id, int record_id,
    int record_count) {
  if (!table || column_id >= table->num_columns() ||
      record_id >= table->num_rows() ||
      record_id + record_count > table->num_rows()) {
    return arrow::Status::Invalid("Invalid arguments");
  }

  auto column = table->column(column_id);

  int64_t current_pos = 0;
  for (int chunk_idx = 0; chunk_idx < column->num_chunks(); ++chunk_idx) {
    auto chunk = column->chunk(chunk_idx);
    int64_t chunk_length = chunk->length();

    if (record_id < current_pos + chunk_length) {
      int64_t chunk_start = record_id - current_pos;
      int64_t available_length = chunk_length - chunk_start;
      int64_t actual_count = std::min<int64_t>(record_count, available_length);
      return chunk->Slice(chunk_start, actual_count);
    }

    current_pos += chunk_length;
  }

  return arrow::Status::Invalid("Record not found in any chunk");
}

arrow::Result<std::shared_ptr<arrow::Array>> GetArrayFromTable(
    const std::shared_ptr<arrow::Table> &table, const std::string &column_name,
    int record_id, int record_count) {
  if (!table) {
    return arrow::Status::Invalid("Table is null");
  }

  int column_id = table->schema()->GetFieldIndex(column_name);
  if (column_id == -1) {
    return arrow::Status::Invalid("Column '", column_name, "' not found");
  }

  return GetArrayFromTable(table, column_id, record_id, record_count);
}
/* exact knn */
float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float t = *pVect1 - *pVect2;
    pVect1++;
    pVect2++;
    res += t * t;
  }
  return (res);
}

float InnerProduct(const void *pVect1, const void *pVect2,
                   const void *qty_ptr) {
  size_t qty = *((size_t *)qty_ptr);
  float res = 0;
  for (unsigned i = 0; i < qty; i++) {
    res += ((float *)pVect1)[i] * ((float *)pVect2)[i];
  }
  return res;
}

float InnerProductDistance(const void *pVect1, const void *pVect2,
                           const void *qty_ptr) {
  return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

std::vector<std::pair<float, std::vector<float>>> SearchExactKnn(
    const float *query, const std::vector<std::vector<float>> &points, size_t k,
    vdb::DistanceSpace space) {
  size_t dim = points[0].size();
  std::vector<std::pair<float, std::vector<float>>> distances;
  float norm_query = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    norm_query += query[i] * query[i];
  }
  norm_query = std::sqrt(norm_query);

  for (const auto &point_vec : points) {
    float dist = 0;
    if (space == vdb::DistanceSpace::kL2) {
      dist = L2Sqr(query, point_vec.data(), &dim);
    } else {
      dist = InnerProductDistance(query, point_vec.data(), &dim);
    }
    if (space == vdb::DistanceSpace::kCosine) {
      auto inner_product = 1.0f - dist;
      float norm_point = 0.0f;
      for (size_t i = 0; i < dim; i++) {
        norm_point += point_vec[i] * point_vec[i];
      }
      norm_point = std::sqrt(norm_point);
      dist = 1.0f - inner_product / (norm_query * norm_point);
    }
    const float epsilon = 1e-6;
    if (dist < epsilon) {
      dist = 0.0f;
    }
    distances.push_back(std::make_pair(dist, point_vec));
  }

  std::sort(distances.begin(), distances.end());

  distances.resize(k);
  return distances;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeRecordBatches(
    std::shared_ptr<arrow::Schema> schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>> &record_batches) {
  // Create an in-memory output stream
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::BufferOutputStream::Create());

  // Create an IPC stream writer
  ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeStreamWriter(out, schema));

  for (auto &record_batch : record_batches) {
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*record_batch));
  }

  ARROW_RETURN_NOT_OK(writer->Close());
  return out->Finish();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> SerializeTable(
    std::shared_ptr<arrow::Table> table) {
  // Create an in-memory output stream
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::BufferOutputStream::Create());

  // Create an IPC stream writer
  ARROW_ASSIGN_OR_RAISE(auto writer,
                        arrow::ipc::MakeStreamWriter(out, table->schema()));

  ARROW_RETURN_NOT_OK(writer->WriteTable(*table));

  ARROW_RETURN_NOT_OK(writer->Close());
  return out->Finish();
}

arrow::Result<std::shared_ptr<arrow::Table>> DeserializeToTableFrom(
    const std::shared_ptr<arrow::Buffer> &serialized_rb) {
  ARROW_ASSIGN_OR_RAISE(
      auto reader,
      arrow::ipc::RecordBatchStreamReader::Open(
          std::make_shared<arrow::io::BufferReader>(serialized_rb)));
  return reader->ToTable();
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
DeserializeToRecordBatchesFrom(
    const std::shared_ptr<arrow::Buffer> &serialized_rb) {
  ARROW_ASSIGN_OR_RAISE(
      auto reader,
      arrow::ipc::RecordBatchStreamReader::Open(
          std::make_shared<arrow::io::BufferReader>(serialized_rb)));
  return reader->ToRecordBatches();
}

arrow::Result<std::shared_ptr<arrow::Table>> SortTable(
    const std::shared_ptr<arrow::Table> &table) {
  std::vector<arrow::compute::SortKey> sort_keys;
  for (const auto &field : table->schema()->fields()) {
    if (field->type()->id() != arrow::Type::LIST) {
      sort_keys.emplace_back(field->name());
    }
  }

  std::vector<arrow::acero::Declaration> decls;
  std::shared_ptr<arrow::Table> output_table;
  decls.emplace_back(arrow::acero::Declaration(
      "table_source", arrow::acero::TableSourceNodeOptions(table)));
  decls.emplace_back(arrow::acero::Declaration{
      "order_by", arrow::acero::OrderByNodeOptions(sort_keys)});
  decls.emplace_back(arrow::acero::Declaration{
      "table_sink", arrow::acero::TableSinkNodeOptions(&output_table)});
  arrow::acero::Declaration decl = arrow::acero::Declaration::Sequence(decls);

  auto maybe_plan = arrow::acero::ExecPlan::Make();
  ARROW_ASSIGN_OR_RAISE(auto plan, maybe_plan);
  ARROW_RETURN_NOT_OK(decl.AddToPlan(plan.get()).status());
  ARROW_RETURN_NOT_OK(plan->Validate());

  /* The plan involves using one or more threads for pipelined execution
   * processing. */
  plan->StartProducing();
  auto finished = plan->finished();
  ARROW_RETURN_NOT_OK(finished.status());

  return output_table;
}

/* snapshot test names */
std::string GetBaseSnapshotDirectoryName(long long sequence) {
  std::stringstream ss;
  ss << server.aof_filename << "." << sequence;
  ss << ".base.vdb";

  return ss.str();
}

std::string GetTempSnapshotDirectoryName(pid_t pid) {
  /* vdb snapshot is always base of aof file. */
  std::stringstream ss;
  ss << "temp-rewriteaof-vdb-" << pid;
  ss << ".aof";

  return ss.str();
}

/* etc */
std::vector<std::pair<float, std::vector<float>>> ResultToDistEmbeddingPairs(
    const std::shared_ptr<vdb::DenseVectorIndex> &index,
    std::priority_queue<std::pair<float, uint64_t>> &pq) {
  std::vector<std::pair<float, std::vector<float>>> result;

  while (!pq.empty()) {
    auto index_pair = pq.top();
    auto index_dist = index_pair.first;
    auto index_point =
        index->GetEmbeddingByLabel<std::vector<float>>(index_pair.second);
    result.emplace(result.begin(), index_dist, index_point);
    pq.pop();
  }
  return result;
}

std::vector<std::pair<float, std::vector<float>>> ResultToDistEmbeddingPairs(
    const std::shared_ptr<IndexHandlerForTest> &index_handler,
    const uint64_t column_id, std::vector<std::pair<float, uint64_t>> &vec) {
  std::vector<std::pair<float, std::vector<float>>> result;

  for (auto index_pair : vec) {
    auto index_dist = index_pair.first;
    auto index_point =
        index_handler->GetDenseEmbedding(column_id, index_pair.second);
    result.emplace_back(index_dist, index_point);
  }
  return result;
}

float CalculateKnnAccuracy(
    std::vector<std::pair<float, std::vector<float>>> &index_result,
    std::vector<std::pair<float, std::vector<float>>> &knn_result) {
  float accept_count = 0.0;
  float farthest = knn_result.back().first;
  for (size_t i = 0; i < knn_result.size(); i++) {
    auto index_pair = index_result[i];
    auto index_dist = index_pair.first;
    if (index_dist <= farthest) {
      accept_count++;
    }
  }
  return accept_count * 100.0 / static_cast<float>(knn_result.size());
}

vdb::Status CreateTableForTest(std::string table_name,
                               std::string schema_string,
                               bool without_segment_id) {
  auto table_dictionary = vdb::GetTableDictionary();

  if (std::any_of(table_name.begin(), table_name.end(),
                  [](char c) { return std::isspace(c); }))
    return vdb::Status::InvalidArgument(
        "CreateTableForTestCommand: Invalid table name.");

  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    auto schema = vdb::ParseSchemaFrom(schema_string);
    if (schema != nullptr) {
      std::vector<std::string> keys = {"table name", "active_set_size_limit"};
      std::vector<std::string> values = {
          table_name, std::to_string(server.vdb_active_set_size_limit)};

      if (!without_segment_id) {
        std::string segment_type = "value";
        std::string segment_keys = "id";
        std::string segment_key_composition_type = "single";
        std::string segmentation_info_str = MakeSegmentationInfoString(
            segment_type, segment_keys, segment_key_composition_type);
        keys.push_back("segmentation_info");
        values.push_back(segmentation_info_str);
      }

      auto metadata = std::make_shared<arrow::KeyValueMetadata>(keys, values);
      schema = schema->WithMetadata(metadata);
      vdb::TableBuilderOptions options;
      vdb::TableBuilder builder{
          std::move(options.SetTableName(table_name).SetSchema(schema))};
      ARROW_ASSIGN_OR_RAISE(auto table, builder.Build());
      table_dictionary->insert({table_name, table});
    } else {
      return vdb::Status::InvalidArgument(
          "CreateTableForTestCommand: Invalid schema.");
    }
    return vdb::Status::Ok();
  }

  return vdb::Status::AlreadyExists(
      "CreateTableForTestCommand: Table already exists.");
}

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>>
CreateFixedSizeListArray(const float *data, size_t dimension) {
  arrow::FloatBuilder float_builder;
  ARROW_RETURN_NOT_OK(float_builder.AppendValues(data, data + dimension));
  std::shared_ptr<arrow::FloatArray> float_array;
  ARROW_RETURN_NOT_OK(float_builder.Finish(&float_array));

  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder list_builder(arrow::default_memory_pool(),
                                           value_builder, dimension);

  ARROW_RETURN_NOT_OK(list_builder.Append());
  ARROW_RETURN_NOT_OK(value_builder->AppendValues(data, data + dimension));

  std::shared_ptr<arrow::FixedSizeListArray> list_array;
  ARROW_RETURN_NOT_OK(list_builder.Finish(&list_array));

  return list_array;
}

std::string MakeSegmentationInfoString(
    std::string segment_type, std::string segment_keys,
    std::string segment_key_composition_type) {
  std::stringstream ss;
  if (segment_key_composition_type != "") {
    ss << R"({"segment_type": ")" << segment_type << R"(", "segment_keys": [")"
       << segment_keys << R"("], "segment_key_composition_type": ")"
       << segment_key_composition_type << R"("})";
  } else {
    ss << R"({"segment_type": ")" << segment_type << R"(", "segment_keys": [")"
       << segment_keys << R"("]})";
  }
  return ss.str();
}

std::string MakeDenseIndexInfoString(size_t column_id, std::string index_type,
                                     std::string space_type,
                                     size_t ef_construction, size_t M) {
  std::stringstream ss;
  ss << R"([{"column_id": ")" << column_id << R"(", "index_type": ")"
     << index_type << R"(", "parameters": {"space": ")" << space_type
     << R"(", "ef_construction": ")" << ef_construction << R"(", "M": ")" << M
     << R"("}}])";
  return ss.str();
}

// Helper function to create Arrow IPC serialized df data for testing
arrow::Result<sds> CreateSerializedDfData(
    const std::vector<std::pair<uint64_t, uint64_t>> &term_df_pairs) {
  // Create Arrow schema for df data
  auto schema = arrow::schema({arrow::field("term_id", arrow::uint64()),
                               arrow::field("df", arrow::uint64())});

  // Create builders
  arrow::UInt64Builder term_id_builder;
  arrow::UInt64Builder df_builder;

  // Add data
  for (const auto &pair : term_df_pairs) {
    ARROW_RETURN_NOT_OK(term_id_builder.Append(pair.first));
    ARROW_RETURN_NOT_OK(df_builder.Append(pair.second));
  }

  // Finish arrays
  std::shared_ptr<arrow::Array> term_id_array;
  std::shared_ptr<arrow::Array> df_array;
  ARROW_RETURN_NOT_OK(term_id_builder.Finish(&term_id_array));
  ARROW_RETURN_NOT_OK(df_builder.Finish(&df_array));

  // Create record batch
  auto record_batch = arrow::RecordBatch::Make(schema, term_df_pairs.size(),
                                               {term_id_array, df_array});

  // Serialize to Arrow IPC format
  ARROW_ASSIGN_OR_RAISE(auto serialized_buffer,
                        vdb::SerializeRecordBatch(record_batch));

  // Convert to sds
  sds result = sdsnewlen(serialized_buffer->data(), serialized_buffer->size());
  return result;
}