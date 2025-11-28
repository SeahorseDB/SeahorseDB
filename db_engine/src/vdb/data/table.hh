#pragma once

#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <sstream>

#include <arrow/api.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include "vdb/common/status.hh"
#include "vdb/common/memory_allocator.hh"
#include "vdb/data/expression.hh"
#include "vdb/data/index_info.hh"
#include "vdb/data/mutable_array.hh"
#include "vdb/data/segmentation.hh"
#include "vdb/data/primary_key_index.hh"

namespace vdb {
namespace tests {
class TableWrapper;
}  // namespace tests

using sds = char*;

class EmbeddingStore;
class Table;
class IndexHandler;
class VectorIndex;

struct EmbeddingJobInfo {
  /**
   * This struct only performs shallow copy/move operations on the 'embeddings'
   * shared_ptr. The caller is responsible for any necessary operations on the
   * actual embeddings data (e.g., performing a deep copy if needed) before
   * passing it to this struct.
   */
  std::string segment_id;
  uint64_t column_id;
  uint64_t index_id;
  std::shared_ptr<arrow::FixedSizeListArray> embeddings;
  uint64_t starting_label;
  std::atomic<uint64_t> inserted_count;

  // Move constructor
  EmbeddingJobInfo(EmbeddingJobInfo&& other) noexcept
      : segment_id(std::move(other.segment_id)),
        column_id(other.column_id),
        index_id(other.index_id),
        embeddings(std::move(other.embeddings)),
        starting_label(other.starting_label),
        inserted_count(other.inserted_count.load()) {}

  // Constructor
  EmbeddingJobInfo(
      const std::string& segment_id_ = "", uint64_t column_id_ = 0,
      uint64_t index_id_ = 0,
      std::shared_ptr<arrow::FixedSizeListArray> embeddings_ = nullptr,
      uint64_t starting_label_ = 0, uint64_t inserted_count_ = 0)
      : segment_id(segment_id_),
        column_id(column_id_),
        index_id(index_id_),
        embeddings(embeddings_),
        starting_label(starting_label_),
        inserted_count(inserted_count_) {}

  std::string ToString() const {
    std::stringstream ss;
    ss << "segment_id: " << segment_id
       << ", embedding_count: " << embeddings->length()
       << ", index_id: " << index_id << ", starting_label: " << starting_label
       << ", inserted_count: " << inserted_count.load();

    return ss.str();
  }
};

struct ScanResult {
  ScanResult() : status{} {}
  Status status;
  std::vector<std::shared_ptr<arrow::RecordBatch>> data;

  bool ok() const { return status.ok(); }
};

class ActiveSetRecordBatchRenewer {
 public:
  ActiveSetRecordBatchRenewer(Segment* segment);
  ~ActiveSetRecordBatchRenewer();

  void SetNeedRenew();

 private:
  Segment* segment_;
  bool need_renew_;
};

class InsertionGuard {
 public:
  friend class Segment;
  enum class Stage {
    kNone = 0,
    kActiveSetModified = 1,
    kInactiveSetCreated = 2
  };

  InsertionGuard(Segment* segment);
  ~InsertionGuard();

  void Success();
  void SetStage(Stage stage) { current_stage_ = stage; }

  void BackupCurrentActiveSetState();

  void CommitCurrentBatch();

  void UpdateInactiveSetBackup();

  void RollbackActiveSet();
  void RollbackInactiveSet();

 private:
  Segment* segment_;
  bool success_;
  Stage current_stage_;

  vdb::vector<std::shared_ptr<vdb::MutableArray>>& active_set_;
  size_t& segment_size_;
  size_t batch_column_size_backup_;
  size_t batch_segment_size_backup_;

  uint32_t backup_inactive_sets_size_;
  vdb::vector<std::shared_ptr<vdb::MutableArray>> backup_active_set_;
  std::shared_ptr<arrow::RecordBatch> backup_active_rb_;
  bool inactive_set_modified_;

  ActiveSetRecordBatchRenewer active_rb_renewer_;
};

/**
 * InactiveSet: Immutable record set with zero-copy mmap-based on-demand
 * loading.
 *
 * Storage format: Arrow IPC (uncompressed)
 * Access method: Memory-mapped file (mmap) for zero-copy access
 *
 * Key features:
 * - Zero-copy: Data accessed directly from mmap (no copying)
 * - Selective loading: Only requested columns loaded to physical memory
 * - OS-level lazy loading: Unused pages never trigger page faults
 *
 * Memory efficiency example (5GB data):
 * - Full scan (all columns): ~10GB peak (5GB data + 5GB serialization)
 * - Selective scan (3 columns): ~100MB peak (only selected columns loaded)
 */
class InactiveSet {
 public:
  InactiveSet(std::shared_ptr<arrow::RecordBatch> rb,
              std::shared_ptr<arrow::Buffer> buffer);
  InactiveSet(std::shared_ptr<arrow::Schema>& schema,
              vdb::vector<std::shared_ptr<vdb::MutableArray>>& arrays);
  InactiveSet(std::shared_ptr<arrow::RecordBatch> rb, InactiveSet& source_set,
              bool copy_members);

  // Get full RecordBatch (loads all columns if needed)
  std::shared_ptr<arrow::RecordBatch> GetRb();

  // Get RecordBatch with specific columns only (selective loading)
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetColumns(
      const std::vector<int>& column_indices);

  inline int64_t NumRows() const { return num_rows_; }
  inline std::shared_ptr<arrow::Schema> GetSchema() const { return schema_; }

  std::string ToString() const;

 private:
  // In-memory RecordBatch (for in-memory sets or cached loads)
  std::shared_ptr<arrow::RecordBatch> rb_;
  vdb::vector<std::shared_ptr<vdb::MutableArray>>
      columns_;                            // From mutable arrays
  std::shared_ptr<arrow::Buffer> buffer_;  // From snapshot loading

  // Common metadata
  std::shared_ptr<arrow::Schema> schema_;
  int64_t num_rows_;
  mutable std::mutex load_mutex_;
};

arrow::Result<int64_t> EstimateInitialSetAndIndexSize(
    std::shared_ptr<arrow::Schema> schema, size_t max_count);

arrow::Result<int64_t> EstimateExpandedSize(
    std::shared_ptr<arrow::Schema> schema,
    const std::string_view& record_string,
    int64_t current_activeset_record_count, int64_t active_set_size_limit);
arrow::Result<int64_t> EstimateExpandedSize(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::RecordBatch>& rb);
arrow::Result<int64_t> EstimateExpandedSize(
    std::shared_ptr<arrow::Schema> schema,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches,
    int64_t current_activeset_record_count, int64_t active_set_size_limit);

class Segment : public std::enable_shared_from_this<Segment> {
  friend class InsertionGuard;

 public:
  /* this function may throw an exception */
  explicit Segment(std::shared_ptr<Table> table, const std::string& segment_id,
                   const uint16_t segment_number);

  /* load from snapshot */
  /* this function may throw an exception */
  explicit Segment(std::shared_ptr<Table> table, const std::string& segment_id,
                   const uint16_t segment_number,
                   std::string& segment_directory_path);

  /* save as snapshot */
  Status Save(std::string& segment_directory_path);

  Status AppendRecord(const std::string_view& record_string);

  Status _AppendRecordsInRange(std::shared_ptr<arrow::RecordBatch>& rb,
                               uint64_t start, uint64_t end,
                               uint64_t& append_record_count,
                               InsertionGuard& guard);

  Status AppendRecords(
      std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches);

  Status AddToIndex(std::shared_ptr<arrow::RecordBatch>& rb,
                    uint32_t start_record_number);
  Status AddRecordBatch(std::shared_ptr<arrow::RecordBatch>& rb);
  Status AddToHiddenColumn(std::shared_ptr<arrow::RecordBatch>& rb);

  // this function is used only for AppendRecord function.
  Status AddDenseEmbedding(uint64_t column_id, const float* embedding,
                           const uint16_t set_number,
                           const uint32_t record_number);
  Status AddDenseEmbeddingsJob(
      uint64_t column_id,
      const std::shared_ptr<arrow::FixedSizeListArray>& embeddings,
      const uint64_t starting_label);

  Status MakeInactive(InsertionGuard& guard);

  Status MakeInactive() {
    InsertionGuard guard(this);
    auto status = MakeInactive(guard);
    if (status.ok()) {
      guard.Success();
    }
    return status;
  }

  arrow::Result<std::shared_ptr<arrow::Array>> FilterRecordIds(
      const std::shared_ptr<arrow::RecordBatch>& rb,
      const std::shared_ptr<expression::Predicate>& predicate, bool is_update);

  void UpdateActiveSetDeletedFlag(std::shared_ptr<arrow::Array> column);
  arrow::Result<uint32_t> DeleteRecords(
      const std::shared_ptr<expression::Predicate>& predicate,
      bool is_update = false,
      std::vector<std::shared_ptr<arrow::RecordBatch>>* deleted_records =
          nullptr);

  ScanResult SearchKnn(const float* query, const size_t& k) const;

  inline std::string_view GetId() const {
    return std::string_view(segment_id_);
  }
  inline uint16_t GetSegmentNumber() const { return segment_number_; }

  inline size_t Size() const { return size_; }

  inline size_t ActualSize() const { return size_ - deleted_size_; }

  inline size_t DeletedSize() const { return deleted_size_; }

  inline uint32_t InactiveSetCount() const {
    return static_cast<uint32_t>(inactive_sets_.size());
  }

  inline uint32_t ActiveSetId() const {
    return static_cast<uint32_t>(inactive_sets_.size());
  }

  inline size_t ActiveSetRecordCount() const { return active_set_[0]->Size(); }

  inline bool HasPrimaryKey() const { return primary_key_index_ != nullptr; }

  inline bool HasIndex() const { return index_handler_ != nullptr; }

  void RenewActiveSetRecordBatch();

  std::shared_ptr<arrow::RecordBatch> ActiveSetRecordBatch();
  vdb::vector<std::shared_ptr<vdb::InactiveSet>>& InactiveSets();

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetRecordbatch(
      uint32_t set_id);
  arrow::Result<std::shared_ptr<arrow::RecordBatch>> GetRecordbatch(
      uint32_t set_id, const std::vector<int>& indices);

  std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
            std::vector<std::shared_ptr<InactiveSet>>>
  GetRecordbatches();

  std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
            std::vector<std::shared_ptr<InactiveSet>>>
  GetRecordbatches(const std::vector<int>& column_indices);

  arrow::Result<std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
                          std::vector<std::shared_ptr<InactiveSet>>>>
  GetFilteredRecordbatches(
      const std::shared_ptr<expression::Predicate>& predicate);

  arrow::Result<std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
                          std::vector<std::shared_ptr<InactiveSet>>>>
  GetFilteredRecordbatches(
      const std::shared_ptr<expression::Predicate>& predicate,
      const std::vector<int>& column_indices);

  std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<InactiveSet>>
  GetRecordbatchWithSet(uint16_t set_id);

  arrow::Result<std::pair<std::shared_ptr<arrow::RecordBatch>,
                          std::shared_ptr<InactiveSet>>>
  GetRecordbatchWithSet(uint16_t set_id,
                        const std::vector<int>& column_indices);

  inline std::shared_ptr<vdb::IndexHandler> IndexHandler() const {
    return index_handler_;
  }

  std::string ToString(bool show_records = true) const;

  std::shared_ptr<arrow::Schema> GetSchema() const;
  std::shared_ptr<arrow::Schema> GetExtendedSchema() const;
  std::shared_ptr<arrow::Schema> GetInternalSchema() const;

  std::shared_ptr<PrimaryKeyIndex> GetPrimaryKeyIndex() const;

  Status _CheckMemoryAvailability(const std::shared_ptr<Table>& table,
                                  const std::string_view& record_string);
  Status _CheckMemoryAvailability(
      const std::shared_ptr<Table>& table,
      std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches);

  std::shared_ptr<vdb::InactiveSet> GetInactiveSet(uint16_t set_number);
  void AppendInactiveSet(std::shared_ptr<InactiveSet>& new_inactive_set);
  bool SwapInactiveSet(uint16_t set_number,
                       std::shared_ptr<InactiveSet>& new_inactive_set,
                       void* compare_address = nullptr);

  Status LoadInactiveSets(const std::string& directory_path);

 private:
  /* utilities for saving segment */
  Status SaveManifest(std::string& file_path) const;
  Status SavePrimaryKeyIndex(std::string& file_path) const;

  std::weak_ptr<Table> table_;
  /* segment number is unique number for each segment. */
  uint16_t segment_number_;
  /* segment id is unique string for each segment.
   * segment id is key to finding segment. */
  std::string segment_id_;
  vdb::vector<std::shared_ptr<vdb::InactiveSet>> inactive_sets_;
  vdb::vector<std::shared_ptr<vdb::MutableArray>> active_set_;
  std::mutex set_mutex_;
  size_t size_;
  size_t deleted_size_;

  std::shared_ptr<PrimaryKeyIndex> primary_key_index_;
  std::shared_ptr<vdb::IndexHandler> index_handler_;
  std::shared_ptr<arrow::RecordBatch> active_rb_;

  /* for load inactive set from segment constructor */
  uint16_t inactive_set_count_for_load;
};

// Table class represents table entity of the database. It should be equivalent
// to the SQL table.
// `table_version_` will be monotonically increased if `schema_` is changed.
class Table : public std::enable_shared_from_this<Table> {
 public:
  explicit Table();

  explicit Table(const std::string& table_name,
                 std::shared_ptr<arrow::Schema>& schema,
                 std::shared_ptr<vdb::vector<IndexInfo>> index_infos);

  ~Table();

  Status LoadSegments(std::shared_ptr<Table> table,
                      std::string& table_directory_path);

  /* save as snapshot */
  Status Save(std::string_view& snapshot_directory_path);

  /* segment */
  std::shared_ptr<Segment> AddSegment(const std::shared_ptr<Table>& table,
                                      const std::string& segment_id);

  std::shared_ptr<Segment> GetSegment(const std::string& segment_id) const;

  arrow::Result<std::shared_ptr<Segment>> GetOrCreateSegment(
      const std::string& segment_id);

  const vdb::map<std::string_view, std::shared_ptr<Segment>>& GetSegments()
      const {
    return segments_;
  }

  size_t GetSegmentCount() const { return segments_.size(); }

  arrow::Result<std::vector<std::shared_ptr<Segment>>> GetFilteredSegments(
      std::shared_ptr<expression::Predicate> predicate,
      const std::shared_ptr<arrow::Schema>& schema) const;

  /* schema */
  Status SetSchema(const std::shared_ptr<arrow::Schema>& schema);
  Status SetMetadata(const char* key, std::string value);

  /* index info */
  Status SetIndexInfos(
      const std::shared_ptr<vdb::vector<IndexInfo>>& index_infos);
  std::shared_ptr<vdb::vector<IndexInfo>> GetIndexInfos() const {
    return index_infos_;
  }

  arrow::Result<std::shared_ptr<arrow::Array>> GetEmbeddingArray(
      uint64_t column_id, const uint64_t* labels, const size_t& count) const;

  /* Get External/Internal Schema
   * (External) Schema: Schema for actual data loading, identical to the schema
   * of data being inserted
   * Internal Schema: Schema for internal storage where
   * Ann Column is converted to uint64 array and Hidden Columns are added
   * Extended Schema: Schema for scan operation (Output schema of ScanImpl)
   */
  std::shared_ptr<arrow::Schema> GetSchema() const { return schema_; }
  std::shared_ptr<arrow::Schema> GetInternalSchema() const;
  std::shared_ptr<arrow::Schema> GetExtendedSchema() const;

  arrow::Result<int32_t> GetAnnDimension(uint64_t column_id) const;

  arrow::Result<int> GetPrimaryKeyColumnId() const;
  Status AppendRecord(const std::string_view& record_string);

  Status AppendRecords(
      std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches);

  /* etc. */
  vdb::string GetTableName() const { return table_name_; }
  int64_t GetDimension(uint32_t column_id) const;

  bool HasIndex() const;
  bool HasDenseIndex() const;
  bool HasPrimaryKey() const;

  bool IsIndexColumn(uint64_t column_id) const;
  bool IsIndexColumn(const std::string& column_name) const;
  bool IsAnnColumn(uint64_t column_id) const;
  bool IsAnnColumn(const std::string& column_name) const;
  bool IsHiddenColumn(uint32_t column_id) const;
  bool IsHiddenColumn(const std::string& column_name) const;
  bool IsPrimaryKeyColumn(uint32_t column_id) const;
  bool IsPrimaryKeyColumn(const std::string& column_name) const;

  Status SetActiveSetSizeLimit();
  Status SetActiveSetSizeLimit(size_t active_set_size_limit);
  size_t GetActiveSetSizeLimit() const;

  std::atomic<uint64_t>& GetActiveReadCommands() {
    return active_read_commands_;
  }
  bool CheckRunningActiveReadCommands() const;

  arrow::Result<SegmentationInfo> GetSegmentationInfo() const;
  std::vector<std::string_view> GetSegmentIdColumnNames() const;
  std::vector<uint32_t> GetSegmentIdColumnIndexes() const;

  std::string ToString(bool show_records = false,
                       bool show_metadata = true) const;

  void StartIndexingThread();
  bool IsIndexing() const;
  void StopIndexingThread();

  Status DropEmbeddingStores();

  vdb::map<uint64_t, std::shared_ptr<EmbeddingStore>>& GetEmbeddingStores();

  std::shared_ptr<EmbeddingStore> GetEmbeddingStore(uint64_t column_id) const;

  std::string GetEmbeddingStoreDirectoryPath() const;

  bool IsDropping() const;

  void SetDropping();
  void ResetDropping();

 private:
  /* utilities for load table */
  Status LoadJobInfoQueue(std::string& file_path);
  Status LoadManifest(std::string& file_path);

  /* utilities for save table */
  Status SaveJobInfoQueue(std::string& file_path);
  Status SaveSegmentIds(std::string& file_path);
  Status SaveManifest(std::string& file_path);

  void DeserializeJobInfo(EmbeddingJobInfo& info, uint64_t& record_count,
                          uint8_t* buffer, uint64_t& buffer_offset);
  void SerializeJobInfo(EmbeddingJobInfo& info, vdb::vector<uint8_t>& buffer,
                        uint64_t& buffer_offset);

  Status AddMetadata(const std::shared_ptr<arrow::KeyValueMetadata>& metadata);

  Status AddMetadataToField(
      uint32_t field_idx,
      const std::shared_ptr<arrow::KeyValueMetadata>& metadata);
  Status AddMetadataToField(
      const std::string& field_name,
      const std::shared_ptr<arrow::KeyValueMetadata>& metadata);

  Status AddEmbeddingStore(
      const uint64_t column_id,
      const std::shared_ptr<EmbeddingStore>& embedding_store);

  vdb::string table_name_;
  std::size_t table_version_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<vdb::vector<IndexInfo>> index_infos_;
  std::shared_ptr<arrow::Schema> internal_schema_ = nullptr;
  vdb::map<std::string_view, std::shared_ptr<Segment>> segments_;
  size_t active_set_size_limit_;
  std::atomic<uint64_t> active_read_commands_;

  vdb::map<uint64_t, std::shared_ptr<EmbeddingStore>> embedding_stores_;
  std::chrono::steady_clock::time_point created_time_;

 protected:
  /* if table has ann feature column and
     server.allow_bg_indexing_thread is on, then indexing thread will
     be spawned. */
  // TODO manage total number of threads for this process. (config)
  // TODO add config for default max_threads?
  void IndexingThreadJob();

 private:
  std::condition_variable cond_;
  std::mutex job_queue_mtx_;
  std::deque<EmbeddingJobInfo> job_queue_;
  std::shared_ptr<std::thread> indexing_thread_ = nullptr;
  std::atomic_bool indexing_ = false;
  std::atomic_bool terminate_thread_ = false;

  bool dropping_ = false;

  friend class Segment;
  friend class TableBuilder;
  friend class tests::TableWrapper;
};

class TableBuilderOptions {
 public:
  explicit TableBuilderOptions() = default;

  explicit TableBuilderOptions(const TableBuilderOptions& rhs) = default;

  explicit TableBuilderOptions(TableBuilderOptions&& rhs) = default;

  TableBuilderOptions& operator=(const TableBuilderOptions& rhs) = default;

  TableBuilderOptions& operator=(TableBuilderOptions&& rhs) = default;

  TableBuilderOptions& SetTableName(std::string_view table_name) {
    table_name_ = table_name;
    return *this;
  }

  TableBuilderOptions& SetSchema(std::shared_ptr<arrow::Schema> schema) {
    schema_ = schema;
    return *this;
  }

  TableBuilderOptions& SetTableDirectoryPath(
      std::string_view table_directory_path) {
    table_directory_path_ = table_directory_path;
    return *this;
  }

 private:
  std::string table_name_;
  std::shared_ptr<arrow::Schema> schema_;
  std::string table_directory_path_;

  friend class TableBuilder;
};

class TableBuilder {
 public:
  explicit TableBuilder(const TableBuilderOptions& options)
      : options_{options} {}

  explicit TableBuilder(TableBuilderOptions&& options)
      : options_{std::move(options)} {}

  arrow::Result<std::shared_ptr<vdb::Table>> Build();

 protected:
  arrow::Result<std::shared_ptr<vdb::Table>> BuildTableFromSchema();

  arrow::Result<std::shared_ptr<vdb::Table>> BuildTableFromSavedFile();

 private:
  TableBuilderOptions options_;
};

/* prototype declaration for unit tests */
arrow::Result<std::pair<std::shared_ptr<arrow::RecordBatch>,
                        std::shared_ptr<arrow::Buffer>>>
_LoadRecordBatchFrom(std::string& file_path,
                     std::shared_ptr<arrow::Schema>& schema);
Status _SaveRecordBatchTo(std::string& file_path,
                          std::shared_ptr<arrow::RecordBatch> rb);
}  // namespace vdb
