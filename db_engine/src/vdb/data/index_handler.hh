#pragma once
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>

#include "vdb/index/hnsw/hnsw.hh"
#include "vdb/index/hnsw/bruteforce.hh"

#include "vdb/common/status.hh"
#include "vdb/common/util.hh"
#include "vdb/data/embedding_store.hh"
#include "vdb/vector/distance/space_header.hh"
#include "vdb/compute/execution.hh"

namespace vdb {

/**
 * VectorIndex Class Hierarchy and Relationships
 *
 *                 ┌─────────────────────────────────┐
 *                 │         VectorIndex             │
 *                 │        (Abstract)               │
 *                 └────────────────┬────────────────┘
 *                                  │
 *                                  ▼
 *                       ┌─────────────────────┐
 *                       │  DenseVectorIndex   │
 *                       │    (Abstract)       │
 *                       └──────────┬──────────┘
 *                                  │
 *                                  ▼
 *                       ┌─────────────────────┐
 *                       │      Hnsw           │
 *                       │  (Concrete)         │
 *                       │ + HNSW Index        │
 *                       │ + In-Memory         │
 *                       │ + Dense Only        │
 *                       └─────────────────────┘
 */

class Table;
class EmbeddingStore;

class VectorIndex {
 public:
  enum Type { kHnsw, kIndexTypeMax };
  virtual ~VectorIndex() = default;

  virtual Status Save(const std::string &index_file_path) const = 0;
  virtual bool IsFull() const { return false; }
  virtual size_t Dimension() const { return 0; }
  virtual size_t Size() const = 0;
  virtual size_t CompleteSize() const { return Size(); }
  virtual size_t DeletedSize() const { return 0; }
  virtual size_t MaxSize() const { return 0; }

  virtual Status DeleteEmbedding(const uint64_t label) = 0;

  virtual arrow::Result<std::shared_ptr<arrow::Array>> GetEmbeddingArray(
      const uint64_t *labels, const size_t &count) const = 0;

  template <typename T>
  Status AddEmbedding(const T &embedding, uint64_t label) {
    if constexpr (std::is_same_v<T, float *> ||
                  std::is_same_v<T, const float *>) {
      return AddEmbeddingImpl(static_cast<const void *>(embedding), label);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return AddEmbeddingImpl(static_cast<const void *>(&embedding), label);
    }
    return Status::InvalidArgument("Unsupported embedding type");
  }

  // Template method pattern for SearchKnn - with filter
  template <typename T, typename FilterType>
  std::priority_queue<std::pair<float, uint64_t>> SearchKnn(const T &query,
                                                            const size_t &k,
                                                            FilterType filter) {
    if constexpr (std::is_same_v<T, float *> ||
                  std::is_same_v<T, const float *>) {
      return SearchKnnImpl(static_cast<const void *>(query), k,
                           static_cast<void *>(filter));
    } else if constexpr (std::is_same_v<T, std::string>) {
      return SearchKnnImpl(static_cast<const void *>(&query), k,
                           static_cast<void *>(filter));
    }

    return std::priority_queue<std::pair<float, uint64_t>>();
  }

  // Template method pattern for SearchKnn - without filter
  template <typename T>
  std::priority_queue<std::pair<float, uint64_t>> SearchKnn(const T &query,
                                                            const size_t &k) {
    if constexpr (std::is_same_v<T, float *> ||
                  std::is_same_v<T, const float *>) {
      return SearchKnnImpl(static_cast<const void *>(query), k, nullptr);
    } else if constexpr (std::is_same_v<T, std::string>) {
      return SearchKnnImpl(static_cast<const void *>(&query), k, nullptr);
    }
    return std::priority_queue<std::pair<float, uint64_t>>();
  }

  template <typename ReturnType>
  ReturnType GetEmbeddingByLabel(const uint64_t label) const {
    if constexpr (std::is_same_v<ReturnType, std::vector<float>>) {
      return GetDenseEmbeddingByLabel(label);
    }
  }

 protected:
  virtual Status AddEmbeddingImpl(const void *, uint64_t) {
    return Status::InvalidArgument(
        "Embedding type not supported by this index");
  }

  virtual std::priority_queue<std::pair<float, uint64_t>> SearchKnnImpl(
      const void *, const size_t &, void *) {
    return std::priority_queue<std::pair<float, uint64_t>>();
  }

  virtual std::vector<float> GetDenseEmbeddingByLabel(const uint64_t) const {
    return std::vector<float>();  // Default empty vector for dense
  }

  static std::string EmbeddingToString(const float *embedding,
                                       size_t dimension);

 private:
};

class DenseVectorIndex : public VectorIndex {
 public:
  virtual ~DenseVectorIndex() = default;

  virtual DISTFUNC<float> GetDistFunc() const = 0;
  virtual void *GetDistFuncParam() const = 0;
  virtual std::vector<float> GetEmbeddingByInternalId(
      const uint64_t internal_id) const = 0;
  virtual arrow::Result<std::shared_ptr<arrow::Array>> GetEmbeddingArray(
      const uint64_t *labels, const size_t &count) const = 0;

  virtual std::string ToString(bool show_embeddings = true,
                               bool show_edges = true) const = 0;
};

class Hnsw : public DenseVectorIndex {
 public:
  explicit Hnsw(DistanceSpace space, size_t dim, size_t ef_construction,
                size_t M, size_t max_elem,
                std::shared_ptr<EmbeddingStore> embedding_store);
  explicit Hnsw(const std::string &index_file_path, DistanceSpace space,
                size_t dim, std::shared_ptr<EmbeddingStore> embedding_store);

  Status DeleteEmbedding(const uint64_t label) override;

  const float *GetRawEmbeddingByInternalId(const uint64_t internal_id) const;
  std::vector<float> GetEmbeddingByInternalId(
      const uint64_t internal_id) const override;
  arrow::Result<std::shared_ptr<arrow::Array>> GetEmbeddingArray(
      const uint64_t *labels, const size_t &count) const override;

  DISTFUNC_FLOAT32 GetDistFunc() const override;
  void *GetDistFuncParam() const override;
  Status Save(const std::string &index_file_path) const override;
  bool IsFull() const override;
  size_t Size() const override;
  size_t CompleteSize() const override;
  size_t DeletedSize() const override;
  size_t Dimension() const override;
  size_t MaxSize() const override;
  std::string ToString(bool show_embeddings, bool show_edges) const override;

 protected:
  Status AddEmbeddingImpl(const void *embedding, uint64_t label) override;
  std::priority_queue<std::pair<float, uint64_t>> SearchKnnImpl(
      const void *query, const size_t &k, void *filter) override;
  std::vector<float> GetDenseEmbeddingByLabel(
      const uint64_t label) const override;

 private:
  std::shared_ptr<vdb::hnsw::HierarchicalNSW<float>> index_;
  std::weak_ptr<EmbeddingStore> embedding_store_;
};

/* IndexHandler */
class IndexHandler {
 public:
  explicit IndexHandler(std::shared_ptr<Table> table,
                        const uint16_t segment_number);
  explicit IndexHandler(std::shared_ptr<Table> table,
                        const uint16_t segment_number,
                        std::string &index_directory_path,
                        const uint64_t index_count);

  Status Save(std::string &index_directory_path);

  Status LoadHnswIndexes(std::string &index_directory_path,
                         IndexInfo &index_info, const uint64_t index_count);
  Status LoadHnswIndex(std::string &index_file_path, IndexInfo &index_info);
  Status LoadHnswIndex(std::string &index_file_path, const uint64_t column_id,
                       const DistanceSpace space, const int32_t dimension);

  Status CreateHnswIndex(IndexInfo &index_info);
  Status CreateIndex();

  std::shared_ptr<std::vector<std::pair<float, uint64_t>>> SearchKnn(
      const float *query, const size_t &k, uint64_t column_id,
      vdb::hnsw::BaseFilterFunctor *filter = nullptr);
  Status AddDenseEmbedding(uint64_t column_id, const float *embedding,
                           uint64_t label);
  Status AddDenseEmbeddings(
      uint64_t column_id,
      const std::shared_ptr<arrow::FixedSizeListArray> &embeddings,
      const uint64_t starting_label);
  void DeleteEmbeddings(const std::vector<uint64_t> &labels, uint64_t column_id,
                        uint64_t index_id);
  void DeleteEmbeddings(const std::vector<uint64_t> &labels, uint64_t index_id);

  arrow::Result<vdb::vector<std::shared_ptr<VectorIndex>>> GetIndexes(
      uint64_t column_id) {
    auto it = indexes_.find(column_id);
    if (it != indexes_.end()) {
      return it->second;
    }
    return vdb::vector<std::shared_ptr<VectorIndex>>();
  }
  std::vector<float> GetDenseEmbedding(uint64_t column_id, uint64_t label);
  std::vector<float> GetDenseEmbedding(uint64_t column_id, uint16_t set_id,
                                       uint64_t label);
  arrow::Result<std::shared_ptr<arrow::Array>> GetEmbeddingArray(
      uint64_t column_id, const uint64_t *labels, const size_t &count);
  arrow::Result<std::shared_ptr<arrow::Array>> GetEmbeddingArray(
      uint64_t column_id, uint16_t set_id, const std::vector<uint64_t> &labels);

  std::shared_ptr<EmbeddingStore> GetEmbeddingStore(uint64_t column_id) const {
    auto table = table_.lock();
    return table->GetEmbeddingStore(column_id);
  }

  std::shared_ptr<VectorIndex> Index(uint64_t column_id, uint64_t index_id);
  size_t Size() const;
  size_t Size(uint64_t column_id);
  std::string ToString(bool show_embeddings = true,
                       bool show_edges = true) const;
  size_t CountIndexedElements(uint64_t column_id);

  std::vector<float> GetDistancesFromDenseQuery(
      uint64_t column_id, const float *dense_query,
      const std::vector<uint64_t> &labels);

 protected:
  std::weak_ptr<vdb::Table> table_;
  vdb::map<uint64_t, vdb::vector<std::shared_ptr<VectorIndex>>> indexes_;
};

VectorIndex::Type GetIndexType(const std::string &type_string);
}  // namespace vdb
