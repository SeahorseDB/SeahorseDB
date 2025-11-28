#pragma once
#include <memory>
#include <stdint.h>
#include <string>

#include "vdb/common/status.hh"
#include "vdb/common/util.hh"

namespace vdb {

class Table;

class EmbeddingStore {
 public:
  EmbeddingStore(const Table *table, const uint64_t column_id,
                 const std::string embedding_store_root_dirname = "");
  EmbeddingStore(const std::string embedding_store_table_directory_path,
                 const uint64_t column_id, const int64_t dimension);

  Status DropDirectory();

  static Status CreateDirectory(const std::string &directory_path);

  Status CreateSegmentAndColumnDirectory(const uint16_t segment_number);

  Status Write(const float *embeddings_raw, const uint64_t starting_label,
               const uint64_t count);
  /* random multi-embeddings read */
  arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>> ReadToArray(
      const uint64_t *labels, const size_t &count);
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadToBuffer(
      const uint64_t *labels, const size_t &count);
  /* random multi-embeddings read and calculate distances */
  arrow::Result<std::shared_ptr<arrow::FloatArray>> ReadAndCalculateDistances(
      const uint64_t *labels, const size_t &count, const float *query_embedding,
      DISTFUNC<float> dist_func, void *dist_func_param);

  /* sequential embeddings read */
  arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>> ReadToArray(
      const uint64_t starting_label, const uint64_t count);
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadToBuffer(
      const uint64_t starting_label, const uint64_t count);

  int64_t Dimension() const;

  static std::string MakeFilePath(
      const std::string &embedding_store_table_directory_path,
      const uint64_t column_id, const uint64_t label);
  static std::string MakeFilePath(
      const std::string &embedding_store_table_directory_path,
      const uint64_t column_id, const uint16_t segment_number,
      const uint16_t set_number);

  std::string GetFilePath(const uint64_t label) const;
  std::string GetFilePath(const uint16_t segment_number,
                          const uint16_t set_number) const;

  std::string GetEmbeddingStoreDirectoryPath() const;

  // Utility for existence checks
  // Returns true if any set.* file exists under segment.* / column.<column_id_>
  bool HasAnySetFiles() const;

 private:
  arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>> BufferToArray(
      std::shared_ptr<arrow::Buffer> buffer, const uint64_t count);

  std::string embedding_store_table_directory_path_;
  int64_t dimension_;
  uint64_t column_id_;
};

}  // namespace vdb
