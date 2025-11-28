#include <filesystem>
#include <stdexcept>
#include <strings.h>
#include <future>

#include "vdb/common/status.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/server_configuration.hh"
#include "vdb/data/embedding_store.hh"
#include "vdb/data/label_info.hh"
#include "vdb/data/table.hh"
#include "vdb/metrics/metrics_api.hh"
#include "vdb/common/util.hh"

namespace vdb {
EmbeddingStore::EmbeddingStore(const vdb::Table *table,
                               const uint64_t column_id,
                               const std::string embedding_store_root_dirname)
    : embedding_store_table_directory_path_{embedding_store_root_dirname},
      column_id_{column_id} {
  if (embedding_store_root_dirname.empty()) {
    embedding_store_table_directory_path_ =
        vdb::ServerConfiguration::GetEmbeddingStoreRootDirname();
  }
  /* root directory is created if not exists. */
  auto status = CreateDirectory(embedding_store_table_directory_path_);
  if (!status.ok()) {
    throw std::invalid_argument(status.ToString());
  }

  embedding_store_table_directory_path_.append("/");
  embedding_store_table_directory_path_.append(table->GetTableName());
  /* root/table_name directory is created if not exists. */
  status = CreateDirectory(embedding_store_table_directory_path_);
  if (!status.ok()) {
    throw std::invalid_argument(status.ToString());
  }

  auto maybe_dimension = table->GetAnnDimension(column_id);
  if (!maybe_dimension.ok()) {
    throw std::invalid_argument(maybe_dimension.status().ToString());
  }
  dimension_ = maybe_dimension.ValueUnsafe();
}

EmbeddingStore::EmbeddingStore(
    const std::string embedding_store_table_directory_path,
    const uint64_t column_id, const int64_t dimension)
    : embedding_store_table_directory_path_{embedding_store_table_directory_path},
      dimension_{dimension},
      column_id_{column_id} {
  auto status = CreateDirectory(embedding_store_table_directory_path_);
  if (!status.ok()) {
    throw std::invalid_argument(status.ToString());
  }
}

Status EmbeddingStore::DropDirectory() {
  if (embedding_store_table_directory_path_.empty()) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "EmbeddingStore is not initialized. Skipping removal of "
               "embedding store directory.");
    return vdb::Status::Ok();
  }

  try {
    std::error_code error_code;
    auto removed_count = std::filesystem::remove_all(
        embedding_store_table_directory_path_, error_code);
    if (error_code) {
      SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
                 "Failed to remove embedding store directory: %s, error: %s",
                 embedding_store_table_directory_path_.c_str(),
                 error_code.message().c_str());
      return vdb::Status::InvalidArgument(
          "Failed to remove embedding store directory: " +
          embedding_store_table_directory_path_ +
          ", error: " + error_code.message());
    }

    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Removed embedding store directory: %s (removed %llu entries)",
               embedding_store_table_directory_path_.c_str(),
               static_cast<unsigned long long>(removed_count));
    return vdb::Status::Ok();
  } catch (const std::exception &e) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Failed to remove embedding store directory: %s, exception: %s",
               embedding_store_table_directory_path_.c_str(), e.what());
    return vdb::Status::InvalidArgument(
        std::string("Failed to remove embedding store directory: ") +
        embedding_store_table_directory_path_ + ", exception: " + e.what());
  }
}

Status EmbeddingStore::CreateDirectory(const std::string &directory_path) {
  if (!std::filesystem::exists(directory_path)) {
    if (!std::filesystem::create_directory(directory_path)) {
      return vdb::Status::InvalidArgument(
          directory_path + " cannot be created: " + std::strerror(errno));
    }
  } else if (std::filesystem::is_regular_file(directory_path)) {
    return vdb::Status::InvalidArgument(directory_path +
                                        " is A FILE, NOT A DIRECTORY");
  }
  return vdb::Status::Ok();
}

Status EmbeddingStore::CreateSegmentAndColumnDirectory(
    const uint16_t segment_number) {
  std::string segment_directory_path = embedding_store_table_directory_path_ +
                                       "/segment." +
                                       std::to_string(segment_number);
  auto status = CreateDirectory(segment_directory_path);
  if (!status.ok()) {
    return status;
  }
  std::string column_directory_path =
      segment_directory_path + "/column." + std::to_string(column_id_);
  return CreateDirectory(column_directory_path);
}

std::string EmbeddingStore::MakeFilePath(
    const std::string &embedding_store_table_directory_path,
    const uint64_t column_id, const uint64_t label) {
  uint16_t segment_number = LabelInfo::GetSegmentNumber(label);
  uint16_t set_number = LabelInfo::GetSetNumber(label);
  return MakeFilePath(embedding_store_table_directory_path, column_id,
                      segment_number, set_number);
}

std::string EmbeddingStore::MakeFilePath(
    const std::string &embedding_store_table_directory_path,
    const uint64_t column_id, const uint16_t segment_number,
    const uint16_t set_number) {
  std::stringstream ss;
  ss << embedding_store_table_directory_path << "/segment." << segment_number
     << "/column." << column_id << "/set." << set_number;
  return ss.str();
}

std::string EmbeddingStore::GetFilePath(const uint64_t label) const {
  uint16_t segment_number = LabelInfo::GetSegmentNumber(label);
  uint16_t set_number = LabelInfo::GetSetNumber(label);
  return GetFilePath(segment_number, set_number);
}

std::string EmbeddingStore::GetFilePath(const uint16_t segment_number,
                                        const uint16_t set_number) const {
  return MakeFilePath(embedding_store_table_directory_path_, column_id_,
                      segment_number, set_number);
}

std::string EmbeddingStore::GetEmbeddingStoreDirectoryPath() const {
  return embedding_store_table_directory_path_;
}

bool EmbeddingStore::HasAnySetFiles() const {
  // embedding_store_table_directory_path_/segment.* / column.<id> / set.*
  try {
    for (const auto &seg_entry : std::filesystem::directory_iterator(
             embedding_store_table_directory_path_)) {
      if (!seg_entry.is_directory()) continue;
      const auto &seg_path = seg_entry.path();
      if (seg_path.filename().string().rfind("segment.", 0) != 0) continue;

      std::string column_dir =
          (seg_path / ("column." + std::to_string(column_id_))).string();
      if (!std::filesystem::exists(column_dir)) continue;
      for (const auto &set_entry :
           std::filesystem::directory_iterator(column_dir)) {
        if (!set_entry.is_regular_file()) continue;
        if (set_entry.path().filename().string().rfind("set.", 0) == 0) {
          return true;
        }
      }
    }
  } catch (const std::exception &) {
    return false;
  }
  return false;
}

Status EmbeddingStore::Write(const float *embeddings_raw,
                             const uint64_t starting_label,
                             const uint64_t count) {
  std::string file_path = GetFilePath(starting_label);
  uint32_t record_number = LabelInfo::GetRecordNumber(starting_label);

  size_t size = count * dimension_ * sizeof(float);
  size_t offset = record_number * dimension_ * sizeof(float);
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "Writing embeddings (starting_label=%s, count=%lu) "
             "to %s",
             LabelInfo::ToString(starting_label).c_str(), count,
             file_path.c_str());
  return WriteTo(file_path, reinterpret_cast<const uint8_t *>(embeddings_raw),
                 size, offset);
}

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>>
EmbeddingStore::ReadToArray(const uint64_t *labels, const size_t &count) {
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> buffer,
                        ReadToBuffer(labels, count));
  return BufferToArray(buffer, count);
}

arrow::Result<std::shared_ptr<arrow::Buffer>> EmbeddingStore::ReadToBuffer(
    const uint64_t *labels, const size_t &count) {
  metrics::CollectDurationStart(
      metrics::MetricIndex::RandomReadEmbeddingsLatency);
  // Create buffer for values
  auto total_size = count * dimension_ * sizeof(float);
  auto maybe_buffer = arrow::AllocateBuffer(total_size);
  if (!maybe_buffer.ok()) {
    metrics::CollectDurationEnd(
        metrics::MetricIndex::RandomReadEmbeddingsLatency);
    return arrow::Status::Invalid(maybe_buffer.status().ToString());
  }
  std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer.ValueUnsafe());

  /* TODO: optimize by multi-threading, sorting etc. */
  for (size_t i = 0; i < count; i++) {
    uint64_t label = labels[i];
    uint32_t record_number = LabelInfo::GetRecordNumber(label);
    std::string file_path = GetFilePath(label);
    size_t size = dimension_ * sizeof(float);
    size_t offset = record_number * dimension_ * sizeof(float);
    auto buffer_ptr = static_cast<uint8_t *>(buffer->mutable_data() +
                                             i * dimension_ * sizeof(float));
    auto status = ReadFrom(file_path, buffer_ptr, size, offset);
    if (!status.ok()) {
      metrics::CollectDurationEnd(
          metrics::MetricIndex::RandomReadEmbeddingsLatency);
      return arrow::Status::Invalid(status.ToString());
    }
  }
  metrics::CollectDurationEnd(
      metrics::MetricIndex::RandomReadEmbeddingsLatency);
  metrics::CollectValue(metrics::MetricIndex::RandomReadEmbeddingsSize, count);
  return buffer;
}

arrow::Result<std::shared_ptr<arrow::FloatArray>>
EmbeddingStore::ReadAndCalculateDistances(const uint64_t *labels,
                                          const size_t &count,
                                          const float *query_embedding,
                                          DISTFUNC<float> dist_func,
                                          void *dist_func_param) {
  metrics::CollectDurationStart(
      metrics::MetricIndex::ReadAndCalculateDistancesLatency);
  // Create buffer for values
  auto total_size = count * dimension_ * sizeof(float);
  auto maybe_buffer = arrow::AllocateBuffer(total_size);
  if (!maybe_buffer.ok()) {
    metrics::CollectDurationEnd(
        metrics::MetricIndex::ReadAndCalculateDistancesLatency);
    return arrow::Status::Invalid(maybe_buffer.status().ToString());
  }
  std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer.ValueUnsafe());
  auto distance_buffer_size = count * sizeof(float);
  auto maybe_distance_buffer = arrow::AllocateBuffer(distance_buffer_size);
  if (!maybe_distance_buffer.ok()) {
    metrics::CollectDurationEnd(
        metrics::MetricIndex::ReadAndCalculateDistancesLatency);
    return arrow::Status::Invalid(maybe_distance_buffer.status().ToString());
  }
  std::shared_ptr<arrow::Buffer> distance_buffer =
      std::move(maybe_distance_buffer.ValueUnsafe());

  /* TODO: optimize by multi-threading, sorting etc. */
  for (size_t i = 0; i < count; i++) {
    uint64_t label = labels[i];
    uint32_t record_number = LabelInfo::GetRecordNumber(label);
    std::string file_path = GetFilePath(label);
    size_t size = dimension_ * sizeof(float);
    size_t offset = record_number * dimension_ * sizeof(float);
    auto buffer_ptr = static_cast<uint8_t *>(buffer->mutable_data() +
                                             i * dimension_ * sizeof(float));
    auto status = ReadFrom(file_path, buffer_ptr, size, offset);
    uint8_t *distance_buffer_ptr = static_cast<uint8_t *>(
        distance_buffer->mutable_data() + i * sizeof(float));
    *reinterpret_cast<float *>(distance_buffer_ptr) =
        dist_func(query_embedding, buffer_ptr, dist_func_param);

    if (!status.ok()) {
      metrics::CollectDurationEnd(
          metrics::MetricIndex::ReadAndCalculateDistancesLatency);
      return arrow::Status::Invalid(status.ToString());
    }
  }

  // Create array data
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);          // validity buffer
  buffers.push_back(distance_buffer);  // value buffer

  auto array_data = arrow::ArrayData::Make(
      arrow::float32(), static_cast<int64_t>(count), std::move(buffers),
      /*null_count=*/0,
      /*offset=*/0);

  metrics::CollectDurationEnd(
      metrics::MetricIndex::ReadAndCalculateDistancesLatency);
  metrics::CollectValue(metrics::MetricIndex::ReadAndCalculateDistancesSize,
                        count);
  return std::make_shared<arrow::FloatArray>(array_data);
}

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>>
EmbeddingStore::ReadToArray(const uint64_t starting_label,
                            const uint64_t count) {
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> buffer,
                        ReadToBuffer(starting_label, count));
  return BufferToArray(buffer, count);
}

arrow::Result<std::shared_ptr<arrow::Buffer>> EmbeddingStore::ReadToBuffer(
    const uint64_t starting_label, const uint64_t count) {
  std::string file_path = GetFilePath(starting_label);
  uint32_t record_number = LabelInfo::GetRecordNumber(starting_label);

  size_t size = count * dimension_ * sizeof(float);
  size_t offset = record_number * dimension_ * sizeof(float);
  metrics::CollectDurationStart(
      metrics::MetricIndex::RandomReadEmbeddingsLatency);
  auto maybe_buffer = arrow::AllocateBuffer(size);
  if (!maybe_buffer.ok()) {
    metrics::CollectDurationEnd(
        metrics::MetricIndex::RandomReadEmbeddingsLatency);
    return arrow::Status::Invalid(maybe_buffer.status().ToString());
  }
  std::shared_ptr<arrow::Buffer> buffer = std::move(maybe_buffer.ValueUnsafe());
  auto status =
      ReadFrom(file_path, reinterpret_cast<uint8_t *>(buffer->mutable_data()),
               size, offset);
  if (!status.ok()) {
    metrics::CollectDurationEnd(
        metrics::MetricIndex::RandomReadEmbeddingsLatency);
    return arrow::Status::Invalid(status.ToString());
  }
  metrics::CollectDurationEnd(
      metrics::MetricIndex::RandomReadEmbeddingsLatency);
  metrics::CollectValue(metrics::MetricIndex::RandomReadEmbeddingsSize, count);
  return buffer;
}

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>>
EmbeddingStore::BufferToArray(std::shared_ptr<arrow::Buffer> buffer,
                              const uint64_t count) {
  // Create array data
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);  // validity buffer
  buffers.push_back(buffer);   // value buffer

  auto array_data = arrow::ArrayData::Make(
      arrow::float32(), static_cast<int64_t>(count * dimension_),
      std::move(buffers),
      /*null_count=*/0,
      /*offset=*/0);

  auto values_array = arrow::MakeArray(array_data);

  // Create FixedSizeList type
  auto value_type = arrow::float32();
  auto list_type =
      std::make_shared<arrow::FixedSizeListType>(value_type, dimension_);

  // Create FixedSizeListArray
  auto embeddings = std::make_shared<arrow::FixedSizeListArray>(
      list_type, count, values_array);
  return embeddings;
}

int64_t EmbeddingStore::Dimension() const { return dimension_; }

}  // namespace vdb
