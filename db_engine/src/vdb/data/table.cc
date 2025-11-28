#include <arrow/result.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>

#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/array/array_nested.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/feather.h>
#include <arrow/io/api.h>
#include <arrow/compute/api.h>

#include "vdb/data/checker.hh"
#include "vdb/metrics/metrics_api.hh"
#include "vdb/common/fd_manager.hh"
#include "vdb/vdb_api.hh"
#include "vdb/common/defs.hh"
#include "vdb/common/memory_allocator.hh"
#include "vdb/common/status.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/server_configuration.hh"
#include "vdb/common/util.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/data/label_info.hh"
#include "vdb/data/metadata.hh"
#include "vdb/data/mutable_array.hh"
#include "vdb/data/segmentation.hh"
#include "vdb/data/table.hh"

namespace vdb {

InsertionGuard::InsertionGuard(Segment* segment)
    : segment_(segment),
      success_(false),
      current_stage_(Stage::kNone),
      active_set_(segment->active_set_),
      segment_size_(segment->size_),
      inactive_set_modified_(false),
      active_rb_renewer_(segment) {
  // Backup Current Active Set (for first batch)
  BackupCurrentActiveSetState();

  // Backup Inactiveset (for first batch)
  backup_inactive_sets_size_ = segment_->inactive_sets_.size();
  backup_active_set_ = segment_->active_set_;
  backup_active_rb_ = segment_->active_rb_;
}

InsertionGuard::~InsertionGuard() {
  if (!success_) {
    try {
      switch (current_stage_) {
        case Stage::kActiveSetModified:
          RollbackActiveSet();
          break;

        case Stage::kInactiveSetCreated:
          RollbackInactiveSet();
          break;

        case Stage::kNone:
        default:
          break;
      }
    } catch (const std::exception& e) {
      vdb::ServerConfiguration::SetTemporaryOomBlocker(true);
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Exception during rollback in InsertionGuard destructor: %s",
                 e.what());
    } catch (...) {
      vdb::ServerConfiguration::SetTemporaryOomBlocker(true);
      SYSTEM_LOG(
          vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
          "Unknown exception during rollback in InsertionGuard destructor.");
    }
  }
}

void InsertionGuard::BackupCurrentActiveSetState() {
  if (!active_set_.empty()) {
    batch_column_size_backup_ = active_set_[0]->Size();
  } else {
    batch_column_size_backup_ = 0;
  }
  batch_segment_size_backup_ = segment_size_;
  /* once active set is modified, renew active set record batch */
  active_rb_renewer_.SetNeedRenew();
}

void InsertionGuard::CommitCurrentBatch() { BackupCurrentActiveSetState(); }

void InsertionGuard::UpdateInactiveSetBackup() {
  backup_inactive_sets_size_ = segment_->inactive_sets_.size();
  backup_active_set_ = segment_->active_set_;
  backup_active_rb_ = segment_->active_rb_;
}

void InsertionGuard::RollbackActiveSet() {
  for (size_t i = 0; i < active_set_.size(); ++i) {
    size_t cur = active_set_[i]->Size();
    if (cur > batch_column_size_backup_) {
      active_set_[i]->TrimEnd(cur - batch_column_size_backup_);
    }
  }
  segment_size_ = batch_segment_size_backup_;
}

void InsertionGuard::RollbackInactiveSet() {
  if (segment_ && inactive_set_modified_) {
    segment_->inactive_sets_.resize(backup_inactive_sets_size_);
    std::swap(segment_->active_set_, backup_active_set_);
    std::swap(segment_->active_rb_, backup_active_rb_);
  }

  inactive_set_modified_ = false;
}

void InsertionGuard::Success() { success_ = true; }

ActiveSetRecordBatchRenewer::ActiveSetRecordBatchRenewer(Segment* segment)
    : segment_(segment), need_renew_(false) {}

ActiveSetRecordBatchRenewer::~ActiveSetRecordBatchRenewer() {
  if (need_renew_) {
    segment_->RenewActiveSetRecordBatch();
  }
}

void ActiveSetRecordBatchRenewer::SetNeedRenew() { need_renew_ = true; }

Table::Table(const std::string& table_name,
             std::shared_ptr<arrow::Schema>& schema,
             std::shared_ptr<vdb::vector<IndexInfo>> index_infos)
    : table_name_{table_name},
      table_version_{0},
      schema_{schema},
      index_infos_{index_infos},
      segments_{},
      active_set_size_limit_{0},
      active_read_commands_{0},
      created_time_{std::chrono::steady_clock::now()} {
  for (auto& index_info : *index_infos) {
    auto column_id = index_info.GetColumnId();
    auto embedding_store = vdb::make_shared<EmbeddingStore>(this, column_id);
    embedding_stores_[column_id] = embedding_store;
  }
}

Table::Table()
    : index_infos_{},
      segments_{},
      active_set_size_limit_{0},
      active_read_commands_{0},
      created_time_{std::chrono::steady_clock::now()} {}

Table::~Table() {
  if (indexing_thread_) StopIndexingThread();
  if (IsDropping()) {
    std::string table_name(table_name_);
    FdManager::GetInstance().CleanupFdsInDirectory(table_name);
    if (HasIndex()) {
      auto status = DropEmbeddingStores();
      if (!status.ok()) {
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "Failed to drop embedding stores for table (%s): %s",
                   table_name_.data(), status.ToString().c_str());
        auto embedding_store_directory_path = GetEmbeddingStoreDirectoryPath();
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "Remove this directory manually: %s",
                   embedding_store_directory_path.data());
      }
    }
  }
}

Status Table::LoadJobInfoQueue(std::string& file_path) {
  uint8_t* buffer;
  size_t file_size;
  file_size = std::filesystem::file_size(file_path);

  if (file_size == 1) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Table (%s) Loading Job Info Queue from (%s) is Done.",
               table_name_.data(), file_path.data());
    return Status::Ok();
  }

  ARROW_CAST_OR_RAISE(buffer, uint8_t*, AllocateAligned(64, file_size));
  /* read job info queue */
  auto status = ReadFrom(file_path, buffer, file_size, 0);
  if (!status.ok()) {
    return status;
  }

  uint64_t buffer_offset = 0;
  uint64_t total_number_of_jobs;
  buffer_offset += GetLengthFrom(buffer, total_number_of_jobs);
  for (uint64_t i = 0; i < total_number_of_jobs; i++) {
    EmbeddingJobInfo info;
    uint64_t record_count;
    DeserializeJobInfo(info, record_count, buffer, buffer_offset);
    /* load embedding store */
    auto embedding_store = GetSegment(info.segment_id)
                               ->IndexHandler()
                               ->GetEmbeddingStore(info.column_id);
    ARROW_ASSIGN_OR_RAISE(
        info.embeddings,
        embedding_store->ReadToArray(info.starting_label, record_count));
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Job Info: %s", info.ToString().data());
    {
      std::lock_guard<std::mutex> lock(job_queue_mtx_);
      job_queue_.push_back(std::move(info));
      indexing_ = true;
    }
    cond_.notify_all();
  }
  DeallocateAligned(buffer, file_size);
  return Status::Ok();
}

Status Table::LoadSegments(std::shared_ptr<Table> table,
                           std::string& table_directory_path) {
  /* load segments */
  std::string segment_ids_file_path(table_directory_path);
  segment_ids_file_path.append("/ids.segment");

  uint8_t* segment_ids_buffer;
  size_t segment_ids_size;
  vdb::Status status;
  /* read segment count and segment ids */
  segment_ids_size = std::filesystem::file_size(segment_ids_file_path);
  ARROW_CAST_OR_RAISE(segment_ids_buffer, uint8_t*,
                      AllocateAligned(64, segment_ids_size));
  status =
      ReadFrom(segment_ids_file_path, segment_ids_buffer, segment_ids_size, 0);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Table (%s) Loading Segment Ids from (%s) is Failed: %s",
               table_name_.data(), segment_ids_file_path.data(),
               status.ToString().data());
    return status;
  }

  uint64_t offset = 0;
  uint64_t segment_count;
  /* get segment count */
  offset += GetLengthFrom(segment_ids_buffer, segment_count);
  for (uint16_t i = 0; i < segment_count; i++) {
    uint64_t segment_id_size;
    /* get segment id */
    std::string segment_id;
    offset += GetLengthFrom(segment_ids_buffer + offset, segment_id_size);
    if (segment_id_size > 0) {
      segment_id.resize(segment_id_size);
      memcpy(segment_id.data(), segment_ids_buffer + offset, segment_id_size);
      offset += segment_id_size;
    }
    std::string segment_directory_path(table_directory_path);
    segment_directory_path.append("/segment.");
    segment_directory_path.append(std::to_string(i));
    /* load segment */
    try {
      auto segment = vdb::make_shared<vdb::Segment>(table, segment_id, i,
                                                    segment_directory_path);
      auto load_status = segment->LoadInactiveSets(segment_directory_path);
      if (!load_status.ok()) {
        segment.reset();
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "Table (%s) Loading InactiveSets for Segment (%s) from (%s) "
                   "is Failed: %s",
                   table_name_.data(), segment_id.data(),
                   segment_directory_path.data(),
                   load_status.ToString().data());
        DeallocateAligned(segment_ids_buffer, segment_ids_size);
        return load_status;
      }
      segments_.emplace(segment->GetId(), segment);
    } catch (const std::exception& e) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Table (%s) Loading Segment (%s) from (%s) is Failed: %s",
                 table_name_.data(), segment_id.data(),
                 segment_directory_path.data(), e.what());
      DeallocateAligned(segment_ids_buffer, segment_ids_size);
      return Status::InvalidArgument(e.what());
    }
  }
  DeallocateAligned(segment_ids_buffer, segment_ids_size);
  if (HasIndex()) {
    bool has_embedding_store_files = true;
    for (auto& kv : embedding_stores_) {
      auto& store = kv.second;
      if (store && !store->HasAnySetFiles()) {
        has_embedding_store_files = false;
        break;
      }
    }

    if (!has_embedding_store_files) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Table (%s) Loading Segments from (%s/segment#) is Done. "
                 "But embedding store files are missing. Maybe it is dropped "
                 "after snapshot",
                 table_name_.data(), table_directory_path.data());
      return Status::Ok();
    }

    std::string job_info_queue_file_path(table_directory_path);
    job_info_queue_file_path.append("/job_info_queue.bin");
    auto status = LoadJobInfoQueue(job_info_queue_file_path);
    if (!status.ok()) {
      return status;
    }
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Loading Segments from (%s/segment#) is Done.",
             table_name_.data(), table_directory_path.data());
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Loading from (%s) is Done.", table_name_.data(),
             table_directory_path.data());
  return Status::Ok();
}

Status Table::LoadManifest(std::string& file_path) {
  uint8_t* buffered_manifest = nullptr;
  size_t manifest_size = 0;
  Status status;
  manifest_size = std::filesystem::file_size(file_path);
  ARROW_CAST_OR_RAISE(buffered_manifest, uint8_t*,
                      AllocateAligned(64, manifest_size));
  /* read table version and schema */
  status = ReadFrom(file_path, buffered_manifest, manifest_size, 0);
  if (!status.ok()) {
    return status;
  }

  uint64_t buffer_offset = 0;
  /* get table version */
  uint64_t table_version;
  buffer_offset += GetLengthFrom(buffered_manifest, table_version);
  table_version_ = static_cast<size_t>(table_version);
  /* get schema */
  uint64_t serialized_schema_size;
  buffer_offset +=
      GetLengthFrom(buffered_manifest + buffer_offset, serialized_schema_size);
  uint8_t* serialized_schema = (buffered_manifest + buffer_offset);
  buffer_offset += serialized_schema_size;
  auto serialized_schema_buffer = vdb::make_shared<arrow::Buffer>(
      serialized_schema, serialized_schema_size);
  auto buf_reader =
      vdb::make_shared<arrow::io::BufferReader>(serialized_schema_buffer);
  arrow::ipc::DictionaryMemo dictionary_memo;
  auto maybe_schema =
      arrow::ipc::ReadSchema(buf_reader.get(), &dictionary_memo);
  DeallocateAligned(buffered_manifest, manifest_size);
  if (!maybe_schema.ok()) {
    return Status::InvalidArgument("Schema is incorrect. - " +
                                   maybe_schema.status().ToString());
  }
  auto schema = maybe_schema.ValueUnsafe();
  status = SetSchema(schema);
  if (!status.ok()) {
    return status;
  }
  auto metadata = schema->metadata();
  auto maybe_table_name = metadata->Get("table name");
  if (!maybe_table_name.ok()) {
    return Status::InvalidArgument("Table name is not provided. - " +
                                   maybe_table_name.status().ToString());
  }
  /* get table name */
  auto table_name = maybe_table_name.ValueUnsafe();
  table_name_ = table_name;

  /* get index infos */
  IndexInfoBuilder index_info_builder;
  index_info_builder.SetIndexInfo(metadata->Get(kIndexInfoKey).ValueOr(""));
  index_info_builder.SetSchema(schema);
  ARROW_ASSIGN_OR_RAISE(auto index_infos, index_info_builder.Build());
  index_infos_ = index_infos;
  try {
    for (auto& index_info : *index_infos) {
      auto column_id = index_info.GetColumnId();
      auto embedding_store = vdb::make_shared<EmbeddingStore>(this, column_id);
      embedding_stores_[column_id] = embedding_store;
    }
  } catch (const std::invalid_argument& e) {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "Creating Table (%s) is Failed: %s", table_name_.data(),
               e.what());
    return Status::InvalidArgument(e.what());
  }

  /* get created time */
  uint64_t created_time_count;
  buffer_offset +=
      GetLengthFrom(buffered_manifest + buffer_offset, created_time_count);
  created_time_ = std::chrono::steady_clock::time_point(
      std::chrono::nanoseconds(created_time_count));

  return Status::Ok();
}

Status Table::Save(std::string_view& snapshot_directory_path) {
  std::error_code ec;
  std::string table_directory_path(snapshot_directory_path);
  table_directory_path.append("/");
  table_directory_path.append(table_name_);

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Saving to (%s) is Started.", table_name_.data(),
             table_directory_path.data());

  if (!std::filesystem::create_directory(table_directory_path, ec)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Table (%s) Saving to (%s) is Failed: %s", table_name_.data(),
               table_directory_path.data(), ec.message().data());
    throw std::invalid_argument(ec.message());
  }

  /* save job info queue */
  if (HasIndex()) {
    std::string job_info_queue_file_path(table_directory_path);
    job_info_queue_file_path.append("/job_info_queue.bin");
    auto status = SaveJobInfoQueue(job_info_queue_file_path);
    if (!status.ok()) {
      return status;
    }
  }

  /* save all segment id in one file */
  std::string segment_id_file_path(table_directory_path);
  segment_id_file_path.append("/ids.segment");
  auto status = SaveSegmentIds(segment_id_file_path);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Table (%s) Saving to (%s) is Failed: %s", table_name_.data(),
               table_directory_path.data(), status.ToString().data());
    return status;
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Saving Segment Ids to (%s) is Done.",
             table_name_.data(), segment_id_file_path.data());
  /* save segments */
  for (auto [key, segment] : GetSegments()) {
    std::string segment_directory_path(table_directory_path);
    segment_directory_path.append("/segment.");
    segment_directory_path.append(std::to_string(segment->GetSegmentNumber()));

    auto status = segment->Save(segment_directory_path);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Table (%s) Saving to (%s) is Failed: %s", table_name_.data(),
                 table_directory_path.data(), status.ToString().data());
      return status;
    }
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Saving Segments to (%s) is Done.", table_name_.data(),
             table_directory_path.data());
  /* save manifest (version, schema, created time) */
  std::string manifest_file_path(table_directory_path);
  manifest_file_path.append("/manifest");
  status = SaveManifest(manifest_file_path);

  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Table (%s) Saving to (%s) is Failed: %s", table_name_.data(),
               table_directory_path.data(), status.ToString().data());
    return status;
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Saving Manifest to (%s) is Done.", table_name_.data(),
             manifest_file_path.data());
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Table (%s) Saving to (%s) is Done.", table_name_.data(),
             table_directory_path.data());

  return Status::Ok();
}

void Table::DeserializeJobInfo(EmbeddingJobInfo& info, uint64_t& record_count,
                               uint8_t* buffer, uint64_t& buffer_offset) {
  /* save format:
   * total length of job info | length of segment id | segment id | index id |
   * starting label | record count */
  uint64_t total_length;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, total_length);
  uint64_t segment_id_length;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, segment_id_length);
  info.segment_id.resize(segment_id_length);
  memcpy(info.segment_id.data(), buffer + buffer_offset, segment_id_length);
  buffer_offset += segment_id_length;
  uint64_t index_id;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, index_id);
  info.index_id = index_id;
  uint64_t column_id;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, column_id);
  info.column_id = column_id;
  uint64_t starting_label;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, starting_label);
  info.starting_label = starting_label;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, record_count);
  info.inserted_count = 0;
}

void Table::SerializeJobInfo(EmbeddingJobInfo& info,
                             vdb::vector<uint8_t>& buffer,
                             uint64_t& buffer_offset) {
  /* save format:
   * total length of job info | length of segment id | segment id | index id |
   * starting label | record count */
  uint64_t saved_label = info.starting_label + info.inserted_count;
  uint64_t record_count = info.embeddings->length() - info.inserted_count;
  uint64_t total_length =
      ComputeBytesFor(info.segment_id.length()) + info.segment_id.length() +
      ComputeBytesFor(info.index_id) + ComputeBytesFor(info.column_id) +
      ComputeBytesFor(saved_label) + ComputeBytesFor(record_count);
  int32_t needed_bytes = ComputeBytesFor(total_length) + total_length;
  buffer.resize(buffer.size() + needed_bytes);
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, total_length);
  buffer_offset +=
      PutLengthTo(buffer.data() + buffer_offset, info.segment_id.length());
  /* save segment id */
  memcpy(buffer.data() + buffer_offset, info.segment_id.data(),
         info.segment_id.length());
  buffer_offset += info.segment_id.length();
  /* save index id */
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, info.index_id);
  /* save column id */
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, info.column_id);
  /* save starting label */
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, saved_label);
  /* save inserted row count */
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, record_count);
}

Status Table::SaveJobInfoQueue(std::string& file_path) {
  /* save job info queue */
  vdb::vector<uint8_t> buffer;
  uint64_t buffer_offset = 0;
  /* save format:
   * total number of jobs | length of first job | first job | length of second
   * job | second job | ... */
  {
    std::lock_guard<std::mutex> lock(job_queue_mtx_);
    /* save total number of jobs */
    auto total_number_of_jobs = 0;
    for (auto& info : job_queue_) {
      if (info.embeddings->length() <=
          static_cast<int64_t>(info.inserted_count))
        continue;
      total_number_of_jobs++;
    }
    int32_t needed_bytes = ComputeBytesFor(total_number_of_jobs);
    buffer.resize(buffer.size() + needed_bytes);
    buffer_offset +=
        PutLengthTo(buffer.data() + buffer_offset, total_number_of_jobs);
    for (auto& info : job_queue_) {
      if (info.embeddings->length() <=
          static_cast<int64_t>(info.inserted_count))
        continue;

      SerializeJobInfo(info, buffer, buffer_offset);

      auto segment = GetSegment(info.segment_id);
      auto embedding_store =
          segment->IndexHandler()->GetEmbeddingStore(info.column_id);

      auto raw_embeddings =
          info.embeddings->values()->data()->GetValues<float>(1);
      raw_embeddings +=
          embedding_store->Dimension() * info.embeddings->offset();
      auto status = embedding_store->Write(raw_embeddings, info.starting_label,
                                           info.embeddings->length());
      if (!status.ok()) {
        return status;
      }
    }

    vdb::Status status = WriteTo(file_path, buffer.data(), buffer_offset, 0);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status Table::SaveSegmentIds(std::string& file_path) {
  size_t segment_count = segments_.size();
  vdb::vector<uint8_t> buffer;
  int buffer_offset = 0;
  /* save format:
   * Total number of segments | length of first segment id | first segment id |
   * length of second segment id | second segment id | ... */
  /* save segment count */
  int32_t needed_bytes = ComputeBytesFor(segment_count);
  buffer.resize(needed_bytes);
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, segment_count);
  vdb::vector<std::shared_ptr<Segment>> segments;
  segments.reserve(segment_count);
  for (auto [segment_id, segment] : segments_) {
    segments.push_back(segment);
  }
  /* sort segments by segment number */
  std::sort(
      segments.begin(), segments.end(),
      [](const std::shared_ptr<Segment>& a, const std::shared_ptr<Segment>& b) {
        return a->GetSegmentNumber() < b->GetSegmentNumber();
      });
  for (auto segment : segments) {
    /* save length of segment_id */
    needed_bytes = ComputeBytesFor(segment->GetId().length());
    buffer.resize(buffer.size() + needed_bytes + segment->GetId().length());
    buffer_offset +=
        PutLengthTo(buffer.data() + buffer_offset, segment->GetId().length());
    /* save segment id */
    memcpy(buffer.data() + buffer_offset, segment->GetId().data(),
           segment->GetId().length());
    buffer_offset += segment->GetId().length();
  }

  /* write byte_vector to file */
  vdb::Status status = WriteTo(file_path, buffer.data(), buffer_offset, 0);
  if (!status.ok()) {
    return status;
  }
  return status;
}

Status Table::SaveManifest(std::string& file_path) {
  vdb::vector<uint8_t> buffer;
  int buffer_offset = 0;
  /* save table version */
  int32_t needed_bytes = ComputeBytesFor(table_version_);
  buffer.resize(needed_bytes);
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, table_version_);
  /* save schema */
  auto maybe_serialized_schema_buffer =
      arrow::ipc::SerializeSchema(*GetSchema(), &arrow_pool);
  if (!maybe_serialized_schema_buffer.ok()) {
    return Status::InvalidArgument(
        maybe_serialized_schema_buffer.status().ToString());
  }
  auto serialized_schema_buffer = maybe_serialized_schema_buffer.ValueUnsafe();
  uint64_t serialized_schema_size =
      static_cast<uint64_t>(serialized_schema_buffer->size());
  needed_bytes = ComputeBytesFor(serialized_schema_size);
  buffer.resize(buffer.size() + needed_bytes + serialized_schema_size);
  buffer_offset +=
      PutLengthTo(buffer.data() + buffer_offset, serialized_schema_size);

  memcpy(buffer.data() + buffer_offset, serialized_schema_buffer->data(),
         serialized_schema_size);
  buffer_offset += serialized_schema_size;

  /* save created time */
  uint64_t created_time_count = created_time_.time_since_epoch().count();
  needed_bytes = ComputeBytesFor(created_time_count);
  buffer.resize(buffer.size() + needed_bytes);
  buffer_offset +=
      PutLengthTo(buffer.data() + buffer_offset, created_time_count);

  vdb::Status status = WriteTo(file_path, buffer.data(), buffer_offset, 0);
  if (!status.ok()) {
    return status;
  }
  return status;
}

std::shared_ptr<Segment> Table::AddSegment(const std::shared_ptr<Table>& table,
                                           const std::string& segment_id) {
  auto segment = segments_.find(segment_id);
  if (segment != segments_.end()) return segment->second;
  try {
    auto new_segment =
        vdb::make_shared<Segment>(table, segment_id, segments_.size());
    segment = segments_.insert({new_segment->GetId(), new_segment}).first;
    return segment->second;
  } catch (std::runtime_error& e) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Creating Segment (%s) is Failed by OOM.", table_name_.data());
    return nullptr;
  }
}

int64_t Table::GetDimension(uint32_t column_id) const {
  auto schema = GetSchema();
  auto type = schema->field(column_id)->type();
  if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
    auto dimension =
        std::static_pointer_cast<arrow::FixedSizeListType>(type)->list_size();
    return dimension;
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
             "Column (%s) type is not fixed sized list.",
             schema->field(column_id)->name().data());
  return 0;
}

std::shared_ptr<Segment> Table::GetSegment(
    const std::string& segment_id) const {
  auto iter = segments_.find(segment_id);
  if (iter != segments_.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

arrow::Result<std::shared_ptr<Segment>> Table::GetOrCreateSegment(
    const std::string& segment_id) {
  auto segment = GetSegment(segment_id);
  if (segment) {
    return segment;
  }
  segment = AddSegment(shared_from_this(), segment_id);
  if (!segment) {
    return arrow::Status::OutOfMemory("Failed to create segment: " +
                                      segment_id);
  }
  return segment;
}

arrow::Result<SegmentationInfo> Table::GetSegmentationInfo() const {
  auto segmentation_info_str =
      GetSchema()->metadata()->Get(kSegmentationInfoKey).ValueOr("");
  SegmentationInfoBuilder segmentation_info_builder;
  segmentation_info_builder.SetSegmentationInfo(segmentation_info_str);
  segmentation_info_builder.SetSchema(GetSchema());
  ARROW_ASSIGN_OR_RAISE(auto segmentation_info,
                        segmentation_info_builder.Build());
  return segmentation_info;
}

std::vector<std::string_view> Table::GetSegmentIdColumnNames() const {
  auto maybe_segmentation_info = GetSegmentationInfo();
  if (!maybe_segmentation_info.ok()) {
    return std::vector<std::string_view>{};
  }
  auto segmentation_info = maybe_segmentation_info.ValueUnsafe();
  if (segmentation_info.GetSegmentType() == SegmentType::kUndefined) {
    return std::vector<std::string_view>{};
  }

  const auto& segment_keys_column_names = segmentation_info.GetSegmentKeys();
  std::vector<std::string_view> ret;
  for (auto column_name : segment_keys_column_names) {
    ret.push_back(column_name);
  }
  return ret;
}

std::vector<uint32_t> Table::GetSegmentIdColumnIndexes() const {
  auto maybe_segmentation_info = GetSegmentationInfo();
  if (!maybe_segmentation_info.ok()) {
    return std::vector<uint32_t>{};
  }
  auto segmentation_info = maybe_segmentation_info.ValueUnsafe();
  if (segmentation_info.GetSegmentType() == SegmentType::kUndefined) {
    return std::vector<uint32_t>{};
  }
  const auto& segment_keys_indices = segmentation_info.GetSegmentKeysIndices();
  return segment_keys_indices;
}

bool Table::IsIndexColumn(uint64_t column_id) const {
  auto index_infos = GetIndexInfos();
  for (auto& index_info : *index_infos) {
    if (index_info.GetColumnId() == column_id) {
      return true;
    }
  }
  return false;
}

bool Table::IsIndexColumn(const std::string& column_name) const {
  auto index_infos = GetIndexInfos();
  auto schema = GetSchema();
  for (auto& index_info : *index_infos) {
    if (schema->field(index_info.GetColumnId())->name() == column_name) {
      return true;
    }
  }
  return false;
}

bool Table::IsAnnColumn(uint64_t column_id) const {
  auto index_infos = GetIndexInfos();
  for (auto& index_info : *index_infos) {
    if (index_info.GetColumnId() == column_id &&
        index_info.IsDenseVectorIndex()) {
      return true;
    }
  }
  return false;
}

bool Table::IsAnnColumn(const std::string& column_name) const {
  auto index_infos = GetIndexInfos();
  auto schema = GetSchema();
  for (auto& index_info : *index_infos) {
    if (schema->field(index_info.GetColumnId())->name() == column_name &&
        index_info.IsDenseVectorIndex()) {
      return true;
    }
  }
  return false;
}

bool Table::IsHiddenColumn(uint32_t column_id) const {
  /* hidden deleted flag column is always at the last */
  return column_id == GetSchema()->fields().size() - 1;
}

bool Table::IsHiddenColumn(const std::string& column_name) const {
  /* hidden deleted flag column is always at the last */
  return column_name ==
         GetSchema()->field(GetSchema()->fields().size() - 1)->name();
}

bool Table::IsPrimaryKeyColumn(uint32_t column_id) const {
  auto schema = GetSchema();
  auto field = schema->field(column_id);
  if (field->metadata() == nullptr) {
    return false;
  }
  auto found = field->metadata()->Get("primary_key").ok();
  return found;
}

bool Table::IsPrimaryKeyColumn(const std::string& column_name) const {
  auto schema = GetSchema();
  auto field_idx = schema->GetFieldIndex(column_name);
  if (field_idx == -1) {
    return false;
  }
  return IsPrimaryKeyColumn(field_idx);
}

size_t Table::GetActiveSetSizeLimit() const { return active_set_size_limit_; }

Status Table::SetActiveSetSizeLimit() {
  ARROW_ASSIGN_OR_RAISE(
      auto active_set_size_limit,
      vdb::stoui64(
          GetSchema()->metadata()->Get(kActiveSetSizeLimitKey).ValueOr("0")));
  active_set_size_limit =
      active_set_size_limit == 0
          ? vdb::ServerConfiguration::GetActiveSetSizeLimit()
          : active_set_size_limit;

  return SetActiveSetSizeLimit(active_set_size_limit);
}

Status Table::SetActiveSetSizeLimit(size_t active_set_size_limit) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Set Active Set Size Limit of Table (%s) as %lu",
             table_name_.data(), active_set_size_limit);

  auto status = SetMetadata(kActiveSetSizeLimitKey,
                            std::to_string(active_set_size_limit));
  if (!status.ok()) {
    return status;
  }
  active_set_size_limit_ = active_set_size_limit;
  return Status::Ok();
}

bool Table::CheckRunningActiveReadCommands() const {
  return active_read_commands_.load() > 0;
}

Status Table::SetSchema(const std::shared_ptr<arrow::Schema>& schema) {
  if (schema == nullptr)
    return Status::InvalidArgument("null schema is not available. ");
  schema_ = schema;

  return Status::Ok();
}

Status Table::SetMetadata(const char* key, std::string value) {
  auto maybe_active_set_size_limit_in_metadata = schema_->metadata()->Get(key);
  if (!maybe_active_set_size_limit_in_metadata.ok()) {
    auto metadata = std::make_shared<arrow::KeyValueMetadata>(
        std::unordered_map<std::string, std::string>{{key, value}});
    return AddMetadata(metadata);
  }

  if (IsImmutable(key)) {
    // check whether try to change existing value in metadata
    auto value_in_metadata =
        maybe_active_set_size_limit_in_metadata.ValueUnsafe();
    if (value != value_in_metadata) {
      std::ostringstream oss;
      oss << key
          << " is immutable metadata, value is different from the value in "
             "metadata. "
          << "value=" << value << ", value in metadata=" << value_in_metadata;
      return Status::InvalidArgument(oss.str());
    }
  }
  return Status::Ok();
}

Status Table::SetIndexInfos(
    const std::shared_ptr<vdb::vector<IndexInfo>>& index_infos) {
  /* Set even when empty to prepare for future separation of create index
   * functionality */
  index_infos_ = index_infos;
  return Status::Ok();
}

Status Table::AddMetadata(
    const std::shared_ptr<arrow::KeyValueMetadata>& metadata) {
  auto schema = GetSchema();

  // check ann column: should be fixed size list of float
  auto index_info_json_str = metadata->Get(kIndexInfoKey).ValueOr("");
  if (!index_info_json_str.empty()) {
    IndexInfoBuilder index_info_builder;
    index_info_builder.SetIndexInfo(index_info_json_str);
    index_info_builder.SetSchema(schema);
    ARROW_ASSIGN_OR_RAISE(auto index_infos, index_info_builder.Build());
    if (index_infos->size() > 0) {
      for (auto& index_info : *index_infos) {
        auto field_type = schema_->field(index_info.GetColumnId())->type();
        if (index_info.IsDenseVectorIndex()) {
          if (field_type->id() != arrow::Type::FIXED_SIZE_LIST) {
            return arrow::Status::Invalid(
                "invalid schema: ann column is not fixed size list type.");
          }
          auto fixed_size_list_type =
              std::static_pointer_cast<arrow::FixedSizeListType>(field_type);
          auto value_type = fixed_size_list_type->value_type();
          if (value_type->id() != arrow::Type::FLOAT) {
            return arrow::Status::Invalid(
                "invalid schema: ann column does not consist of float values.");
          }
        }
      }
    }
    SetIndexInfos(index_infos);
  }

  if (schema->metadata() != nullptr) {
    return SetSchema(
        schema->WithMetadata(schema->metadata()->Merge(*metadata)));
  } else {
    return SetSchema(schema->WithMetadata(metadata));
  }
}

Status Table::AddMetadataToField(
    uint32_t field_idx,
    const std::shared_ptr<arrow::KeyValueMetadata>& metadata) {
  auto schema = GetSchema();
  auto field = schema->field(field_idx);
  auto fields = schema->fields();
  if (field->metadata() != nullptr) {
    auto merged_metadata = field->metadata()->Merge(*metadata);
    fields[field_idx] = field->WithMetadata(merged_metadata);
  } else {
    fields[field_idx] = field->WithMetadata(metadata);
  }
  return SetSchema(std::make_shared<arrow::Schema>(fields, schema->metadata()));
}

Status Table::AddMetadataToField(
    const std::string& field_name,
    const std::shared_ptr<arrow::KeyValueMetadata>& metadata) {
  auto schema = GetSchema();
  auto idx = schema->GetFieldIndex(field_name);
  if (idx == -1) {
    return Status::InvalidArgument("Field not found: " + field_name);
  }
  return AddMetadataToField(idx, metadata);
}

Status Table::AddEmbeddingStore(
    const uint64_t column_id,
    const std::shared_ptr<EmbeddingStore>& embedding_store) {
  embedding_stores_[column_id] = embedding_store;
  return Status::Ok();
}

std::shared_ptr<arrow::Schema> Table::GetExtendedSchema() const {
  auto schema = GetSchema();
  auto extended_fields = schema->fields();
  /* Add Internal Columns (automatically invisible due to __ prefix)
   * Currently: __deleted_flag
   * TODO: For future generalization, consider iterating over a list of
   * internal column definitions instead of manually adding each column */
  extended_fields.push_back(
      arrow::field(vdb::kDeletedFlagColumn, arrow::boolean(), true));
  return std::make_shared<arrow::Schema>(extended_fields, schema->metadata());
}

std::shared_ptr<arrow::Schema> Table::GetInternalSchema() const {
  auto schema = GetSchema();
  auto extended_fields = schema->fields();
  /* Exchange Ann Column from fixed sized list array (Embedding) to uint64 array
   * (Rowid) */
  auto index_infos = GetIndexInfos();
  for (auto index_info : *index_infos) {
    auto ann_column_id = index_info.GetColumnId();
    auto ann_column_name = schema->field(ann_column_id)->name();
    std::shared_ptr<arrow::KeyValueMetadata> field_metadata;
    if (index_info.IsDenseVectorIndex()) {
      auto dimension = GetDimension(ann_column_id);
      field_metadata =
          arrow::key_value_metadata({{"dimension", std::to_string(dimension)}});
    } else {
      field_metadata =
          arrow::key_value_metadata({{"dimension", std::to_string(0)}});
    }
    auto row_id_field =
        arrow::field(ann_column_name, arrow::uint64(), false, field_metadata);
    extended_fields[ann_column_id] = row_id_field;
  }

  /* Add Internal Columns (automatically invisible due to __ prefix)
   * Currently: __deleted_flag
   * TODO: For future generalization, consider iterating over a list of
   * internal column definitions instead of manually adding each column */
  extended_fields.push_back(
      arrow::field(vdb::kDeletedFlagColumn, arrow::boolean(), true));

  return std::make_shared<arrow::Schema>(extended_fields, schema->metadata());
}

arrow::Result<int32_t> Table::GetAnnDimension(uint64_t column_id) const {
  auto field = GetSchema()->field(column_id);
  if (field->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    // assert (field->type()->id() == arrow::Type::FIXED_SIZE_LIST)
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Dense embedding column");
    return std::static_pointer_cast<arrow::FixedSizeListType>(
               GetSchema()->field(column_id)->type())
        ->list_size();
  }
  return arrow::Status::Invalid("Invalid column type: " +
                                field->type()->ToString());
}

arrow::Result<int> Table::GetPrimaryKeyColumnId() const {
  auto schema = GetSchema();

  for (int i = 0; i < schema->num_fields(); i++) {
    auto field = schema->field(i);
    if (field->metadata() != nullptr) {
      auto maybe_primary_key = field->metadata()->Get("primary_key");
      if (maybe_primary_key.ok()) {
        return i;
      }
    }
  }
  return arrow::Status::Invalid(
      "Primary key column not found in any field metadata.");
}

bool Table::HasPrimaryKey() const {
  auto schema = GetSchema();
  auto primary_key_column_id = GetPrimaryKeyColumnId();
  return primary_key_column_id.ok();
}

Status Table::AppendRecord(const std::string_view& record_string) {
  ARROW_ASSIGN_OR_RAISE(auto segmentation_info, GetSegmentationInfo());
  ARROW_ASSIGN_OR_RAISE(
      auto segment_id,
      segmentation_info.GetSegmentId(RecordViewPartsExtractor(record_string)));
  ARROW_ASSIGN_OR_RAISE(auto segment, GetOrCreateSegment(segment_id));
  return segment->AppendRecord(record_string);
}

Status Table::AppendRecords(
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches) {
  ARROW_ASSIGN_OR_RAISE(auto segmentation_info, GetSegmentationInfo());
  ARROW_ASSIGN_OR_RAISE(auto segment_id,
                        segmentation_info.GetSegmentId(
                            RecordBatchPartsExtractor(record_batches)));
  ARROW_ASSIGN_OR_RAISE(auto segment, GetOrCreateSegment(segment_id));
  return segment->AppendRecords(record_batches);
}

bool Table::HasIndex() const {
  auto index_infos = GetIndexInfos();
  if (index_infos->size() > 0) {
    return true;
  }
  return false;
}

bool Table::HasDenseIndex() const {
  auto index_infos = GetIndexInfos();
  for (auto& index_info : *index_infos) {
    if (TransformToLower(index_info.GetIndexType()) == "hnsw") {
      return true;
    }
  }
  return false;
}

void Table::StartIndexingThread() {
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Start Indexing Thread");
  indexing_thread_ =
      std::make_shared<std::thread>(&Table::IndexingThreadJob, this);
}

bool Table::IsIndexing() const { return indexing_; }

void Table::StopIndexingThread() {
  terminate_thread_ = true;
  cond_.notify_all();
  if (indexing_thread_->joinable()) indexing_thread_->join();
}

void Table::IndexingThreadJob() {
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Indexing Thread Started.");
  auto schema = GetSchema();
  auto metadata = schema->metadata();
  auto index_infos = GetIndexInfos();

  auto maybe_max_threads = vdb::stoui64(
      metadata->Get("max_threads").ValueOr("16"));  // TODO: make config
  uint64_t kMaxThreads = 16;
  if (maybe_max_threads.ok()) {
    kMaxThreads = maybe_max_threads.ValueUnsafe();
  }

  if (index_infos->size() == 0) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Table metadata does not have information about the feature "
               "embedding column.");
    return;
  }

  while (true) {
    {  // job queue lock scope
      std::unique_lock<std::mutex> lock(job_queue_mtx_);
      if (job_queue_.empty()) {
        indexing_ = false;
        cond_.wait(lock,
                   [&] { return !job_queue_.empty() || terminate_thread_; });
      }
      if (terminate_thread_) return;
    }  // end of job queue lock scope
    EmbeddingJobInfo& info = job_queue_.front();

    std::vector<std::thread> threads;
    auto index_handler = GetSegment(info.segment_id)->IndexHandler();
    auto index = index_handler->Index(info.column_id, info.index_id);
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Found indexing job. %s", info.ToString().data());
    auto dim = index->Dimension();
    auto raw_embeddings =
        info.embeddings->values()->data()->GetValues<float>(1);
    raw_embeddings += dim * info.embeddings->offset();
    uint64_t embedding_count = info.embeddings->length();

    uint64_t thread_count = std::min(embedding_count, kMaxThreads);
    threads.reserve(thread_count);

    auto ThreadAddEmbedding = [&] {
      while (1) {
        /* If the snapshot operation is in progress, wait until it is
         * completed */
        CheckVdbSnapshot();
        uint64_t cur_id = info.inserted_count.fetch_add(1);
        if (cur_id >= embedding_count) break;
        index->AddEmbedding(raw_embeddings + dim * cur_id,
                            info.starting_label + cur_id);
      }
    };

    for (uint64_t thread_id = 0; thread_id < thread_count - 1; thread_id++) {
      threads.emplace_back(ThreadAddEmbedding);
    }

    ThreadAddEmbedding();

    // TODO: range loop makes pointing error
    for (uint64_t i = 0; i < threads.size(); i++) {
      threads[i].join();
    }

    {  // job queue lock scope
      std::unique_lock<std::mutex> lock(job_queue_mtx_);
      /* job info must be removed after indexing is done for ensuring data
       * persistency. when creating snapshot, indexing job info is saved
       * together in table directory. */
      job_queue_.pop_front();
    }
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Indexing job done. (segment_id=%s, "
               "starting_label=%s, embedding_count=%lu, "
               "index size=%lu)",
               info.segment_id.c_str(),
               LabelInfo::ToString(info.starting_label).c_str(),
               embedding_count, index->Size());
  }
}

std::string Table::ToString(bool show_records, bool show_metadata) const {
  std::stringstream ss;
  ss << "Table Name: " << table_name_ << std::endl;
  ss << "Table Schema: " << GetSchema()->ToString(show_metadata) << std::endl;
  ss << "Active Set Size Limit: " << active_set_size_limit_ << std::endl;
  for (auto& kv : segments_) {
    ss << "----- " << kv.first << " -----" << std::endl;
    ss << kv.second->ToString(show_records) << std::endl;
  }
  return ss.str();
}

arrow::Result<std::vector<std::shared_ptr<Segment>>> Table::GetFilteredSegments(
    std::shared_ptr<expression::Predicate> predicate,
    const std::shared_ptr<arrow::Schema>& schema) const {
  std::vector<std::shared_ptr<Segment>> filtered_segments;
  auto segments = GetSegments();
  if (predicate) {
    ARROW_ASSIGN_OR_RAISE(filtered_segments,
                          predicate->PruneSegments(segments, schema));
  } else {
    filtered_segments.reserve(segments.size());
    for (const auto& [_, segment] : segments) {
      filtered_segments.push_back(segment);
    }
  }
  return filtered_segments;
}

Status Table::DropEmbeddingStores() {
  // Stop indexing thread first to ensure no embedding store access
  if (indexing_thread_) {
    StopIndexingThread();
  }

  // Drop embedding store
  if (!embedding_stores_.empty()) {
    auto first_store_iter = embedding_stores_.begin();
    auto status = first_store_iter->second->DropDirectory();
    if (!status.ok()) {
      return status;
    }
  }

  // Clear the embedding stores map
  embedding_stores_.clear();

  return Status::Ok();
}

vdb::map<uint64_t, std::shared_ptr<EmbeddingStore>>&
Table::GetEmbeddingStores() {
  return embedding_stores_;
}

std::shared_ptr<EmbeddingStore> Table::GetEmbeddingStore(
    uint64_t column_id) const {
  auto it = embedding_stores_.find(column_id);
  if (it != embedding_stores_.end()) {
    return it->second;
  }
  return nullptr;
}

std::string Table::GetEmbeddingStoreDirectoryPath() const {
  if (embedding_stores_.empty()) {
    return "";
  }
  return embedding_stores_.begin()->second->GetEmbeddingStoreDirectoryPath();
}

bool Table::IsDropping() const { return dropping_; }

void Table::SetDropping() { dropping_ = true; }
void Table::ResetDropping() { dropping_ = false; }

arrow::Result<int64_t> EstimateInitialSetAndIndexSize(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<vdb::vector<IndexInfo>> index_infos, size_t max_count) {
  int64_t initial_size = 0;
  for (auto& field : schema->fields()) {
    switch (field->type()->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                             \
  {                                                              \
    case ARROW_TYPE: {                                           \
      initial_size += ARR_TYPE::EstimateInitialSpace(max_count); \
    } break;                                                     \
  }
      __CASE(arrow::Type::BOOL, BooleanArray);
      __CASE(arrow::Type::INT8, Int8Array);
      __CASE(arrow::Type::INT16, Int16Array);
      __CASE(arrow::Type::INT32, Int32Array);
      __CASE(arrow::Type::INT64, Int64Array);
      __CASE(arrow::Type::UINT8, UInt8Array);
      __CASE(arrow::Type::UINT16, UInt16Array);
      __CASE(arrow::Type::UINT32, UInt32Array);
      __CASE(arrow::Type::UINT64, UInt64Array);
      __CASE(arrow::Type::FLOAT, FloatArray);
      __CASE(arrow::Type::DOUBLE, DoubleArray);
      __CASE(arrow::Type::STRING, StringArray);
      __CASE(arrow::Type::LARGE_STRING, LargeStringArray);
#undef __CASE
      case arrow::Type::LIST: {
        auto type = std::static_pointer_cast<arrow::ListType>(field->type())
                        ->value_type();
        switch (type->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                             \
  {                                                              \
    case ARROW_TYPE: {                                           \
      initial_size += ARR_TYPE::EstimateInitialSpace(max_count); \
    } break;                                                     \
  }
          __CASE(arrow::Type::BOOL, BooleanListArray);
          __CASE(arrow::Type::INT8, Int8ListArray);
          __CASE(arrow::Type::INT16, Int16ListArray);
          __CASE(arrow::Type::INT32, Int32ListArray);
          __CASE(arrow::Type::INT64, Int64ListArray);
          __CASE(arrow::Type::UINT8, UInt8ListArray);
          __CASE(arrow::Type::UINT16, UInt16ListArray);
          __CASE(arrow::Type::UINT32, UInt32ListArray);
          __CASE(arrow::Type::UINT64, UInt64ListArray);
          __CASE(arrow::Type::FLOAT, FloatListArray);
          __CASE(arrow::Type::DOUBLE, DoubleListArray);
          __CASE(arrow::Type::STRING, StringListArray);
#undef __CASE
          default:
            return arrow::Status::Invalid("Wrong list schema.");
        }
      } break;
      case arrow::Type::FIXED_SIZE_LIST: {
        auto fixed_size_list_type =
            std::static_pointer_cast<arrow::FixedSizeListType>(field->type());
        auto type = fixed_size_list_type->value_type();
        auto list_size = fixed_size_list_type->list_size();
        switch (type->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                                        \
  {                                                                         \
    case ARROW_TYPE: {                                                      \
      initial_size += ARR_TYPE::EstimateInitialSpace(list_size, max_count); \
    } break;                                                                \
  }
          __CASE(arrow::Type::BOOL, BooleanFixedSizeListArray);
          __CASE(arrow::Type::INT8, Int8FixedSizeListArray);
          __CASE(arrow::Type::INT16, Int16FixedSizeListArray);
          __CASE(arrow::Type::INT32, Int32FixedSizeListArray);
          __CASE(arrow::Type::INT64, Int64FixedSizeListArray);
          __CASE(arrow::Type::UINT8, UInt8FixedSizeListArray);
          __CASE(arrow::Type::UINT16, UInt16FixedSizeListArray);
          __CASE(arrow::Type::UINT32, UInt32FixedSizeListArray);
          __CASE(arrow::Type::UINT64, UInt64FixedSizeListArray);
          __CASE(arrow::Type::FLOAT, FloatFixedSizeListArray);
          __CASE(arrow::Type::DOUBLE, DoubleFixedSizeListArray);
          __CASE(arrow::Type::STRING, StringFixedSizeListArray);
#undef __CASE
          default:
            return arrow::Status::Invalid("Wrong fixed_size_list schema.");
        }
      } break;
      default:
        return arrow::Status::Invalid("Wrong Schema.");
    }
  }

  if (index_infos->size() > 0) {
    for (auto& index_info : *index_infos) {
      auto ann_column_id = index_info.GetColumnId();
      auto dimension = vdb::stoi32(schema->field(ann_column_id)
                                       ->metadata()
                                       ->Get("dimension")
                                       .ValueOr("-1"));
      if (dimension.ok() && dimension.ValueUnsafe() == -1) {
        return arrow::Status::Invalid("Dimension is not set.");
      }

      initial_size +=
          max_count * (/* embedding */ dimension.ValueUnsafe() * sizeof(float) +
                       /* label */ sizeof(uint64_t) +
                       /* level 0 link */ sizeof(void*));
    }
  }
  return initial_size;
}

arrow::Result<int64_t> EstimateExpandedSize(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<vdb::vector<IndexInfo>> index_infos,
    const std::string_view& record_string,
    int64_t current_activeset_record_count, int64_t active_set_size_limit) {
  int64_t expanded_size = 0;
  auto fields = schema->fields();
  size_t column_id = 0;

  auto prev = 0;
  auto pos = record_string.find(kRS, 0);

  while (true) {
    auto token = record_string.substr(prev, pos - prev);

    switch (fields[column_id]->type()->id()) {
      /* Non-list type cases */
#define __CASE(ARROW_TYPE, ARR_TYPE, PARSED_VALUE)                    \
  {                                                                   \
    case ARROW_TYPE: {                                                \
      expanded_size += ARR_TYPE::EstimateExpandedSpace(PARSED_VALUE); \
    } break;                                                          \
  }
      __CASE(arrow::Type::STRING, StringArray, std::string(token));
      __CASE(arrow::Type::LARGE_STRING, LargeStringArray, std::string(token));
#undef __CASE

      /* list type cases */
      case arrow::Type::LIST: {
        auto type =
            std::static_pointer_cast<arrow::ListType>(fields[column_id]->type())
                ->value_field()
                ->type()
                ->id();
        auto prev = 0;
        auto pos = token.find(kGS, prev);
        switch (type) {  // Handling Subtype
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE, PARSED_VALUE)           \
  {                                                                 \
    case ARROW_TYPE: {                                              \
      std::vector<CTYPE> list_value;                                \
      while (true) {                                                \
        auto value_token = token.substr(prev, pos - prev);          \
        PARSED_VALUE;                                               \
        list_value.emplace_back((parsed_value));                    \
        if (pos == std::string::npos) break;                        \
        prev = pos + 1;                                             \
        pos = token.find(kGS, prev);                                \
      }                                                             \
      expanded_size += ARR_TYPE::EstimateExpandedSpace(list_value); \
    } break;                                                        \
  }
          __CASE(arrow::Type::BOOL, bool, BoolListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stobool(std::string(value_token))));
          __CASE(arrow::Type::INT8, int8_t, Int8ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoi32(std::string(value_token))));
          __CASE(arrow::Type::INT16, int16_t, Int16ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoi32(std::string(value_token))));
          __CASE(arrow::Type::INT32, int32_t, Int32ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoi32(std::string(value_token))));
          __CASE(arrow::Type::INT64, int64_t, Int64ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoi64(std::string(value_token))));
          __CASE(arrow::Type::UINT8, uint8_t, UInt8ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoui32(std::string(value_token))));
          __CASE(arrow::Type::UINT16, uint16_t, UInt16ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoui32(std::string(value_token))));
          __CASE(arrow::Type::UINT32, uint32_t, UInt32ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoui32(std::string(value_token))));
          __CASE(arrow::Type::UINT64, uint64_t, UInt64ListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stoui64(std::string(value_token))));
          __CASE(arrow::Type::FLOAT, float, FloatListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stof(std::string(value_token))));
          __CASE(arrow::Type::DOUBLE, double, DoubleListArray,
                 ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                       vdb::stod(std::string(value_token))));
          __CASE(arrow::Type::STRING, std::string, StringListArray,
                 auto parsed_value = std::string(value_token));
          default:
            return arrow::Status::Invalid("Err: Not a valid subtype for list.");
        }
#undef __CASE
      } break;

      /* fixed size list type cases */
      case arrow::Type::FIXED_SIZE_LIST: {
        auto type = std::static_pointer_cast<arrow::FixedSizeListType>(
                        fields[column_id]->type())
                        ->value_field()
                        ->type()
                        ->id();
        auto prev = 0;
        auto pos = token.find(kGS, prev);
        switch (type) {
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE, PARSED_VALUE)           \
  {                                                                 \
    case ARROW_TYPE: {                                              \
      std::vector<CTYPE> list_value;                                \
      while (true) {                                                \
        auto value_token = token.substr(prev, pos - prev);          \
        list_value.emplace_back((PARSED_VALUE));                    \
        if (pos == std::string::npos) break;                        \
        prev = pos + 1;                                             \
        pos = token.find(kGS, prev);                                \
      }                                                             \
      expanded_size += ARR_TYPE::EstimateExpandedSpace(list_value); \
    } break;                                                        \
  }
          __CASE(arrow::Type::STRING, std::string, StringFixedSizeListArray,
                 std::string(value_token));
          default:
            break;
#undef __CASE
        }
      } break;
      default:
        break;
    }

    if (pos == std::string::npos) break;

    prev = pos + 1;
    pos = record_string.find(kRS, prev);
    column_id += 1;
  }

  if (current_activeset_record_count == active_set_size_limit) {
    ARROW_ASSIGN_OR_RAISE(auto initial_set_size,
                          EstimateInitialSetAndIndexSize(
                              schema, index_infos, active_set_size_limit));
    expanded_size += initial_set_size;
  }
  return expanded_size;
}

arrow::Result<int64_t> EstimateExpandedSize(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::RecordBatch>& rb) {
  if (rb->num_rows() == 0) {
    return arrow::Status::Invalid("Don't pass empty recordbatch! ");
  }
  int64_t expanded_size = 0;
  auto fields = schema->fields();

  for (int64_t i = 0; i < rb->num_columns(); i++) {
    auto input_arr = rb->column(i);
    auto field = schema->field(i);

    switch (field->type()->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                                          \
  {                                                                           \
    case ARROW_TYPE: {                                                        \
      expanded_size += vdb::ARR_TYPE::EstimateExpandedSpace(input_arr.get()); \
    } break;                                                                  \
  }
      __CASE(arrow::Type::STRING, StringArray);
      __CASE(arrow::Type::LARGE_STRING, LargeStringArray);
#undef __CASE

      case arrow::Type::LIST: {
        auto type = std::static_pointer_cast<arrow::ListType>(fields[i]->type())
                        ->value_field()
                        ->type()
                        ->id();
        switch (type) {
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE)                                   \
  {                                                                           \
    case ARROW_TYPE: {                                                        \
      expanded_size += vdb::ARR_TYPE::EstimateExpandedSpace(input_arr.get()); \
    } break;                                                                  \
  }
          __CASE(arrow::Type::BOOL, bool, BooleanListArray);
          __CASE(arrow::Type::INT8, int8_t, Int8ListArray);
          __CASE(arrow::Type::INT16, int16_t, Int16ListArray);
          __CASE(arrow::Type::INT32, int32_t, Int32ListArray);
          __CASE(arrow::Type::INT64, int64_t, Int64ListArray);
          __CASE(arrow::Type::UINT8, uint8_t, UInt8ListArray);
          __CASE(arrow::Type::UINT16, uint16_t, UInt16ListArray);
          __CASE(arrow::Type::UINT32, uint32_t, UInt32ListArray);
          __CASE(arrow::Type::UINT64, uint64_t, UInt64ListArray);
          __CASE(arrow::Type::FLOAT, float, FloatListArray);
          __CASE(arrow::Type::DOUBLE, double, DoubleListArray);
          __CASE(arrow::Type::STRING, std::string, StringListArray);
#undef __CASE
          default:
            return arrow::Status::Invalid("Err: Not a valid subtype for list.");
        }
      } break;
      case arrow::Type::FIXED_SIZE_LIST: {
        auto type = std::static_pointer_cast<arrow::FixedSizeListType>(
                        fields[i]->type())
                        ->value_field()
                        ->type()
                        ->id();
        switch (type) {
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE, SUBARR_TYPE)                      \
  {                                                                           \
    case ARROW_TYPE: {                                                        \
      expanded_size += vdb::ARR_TYPE::EstimateExpandedSpace(input_arr.get()); \
    } break;                                                                  \
  }
          __CASE(arrow::Type::STRING, std::string, StringFixedSizeListArray,
                 arrow::StringArray);
#undef __CASE
          default:
            break;
        }
      } break;
      default:
        break;
    }
  }

  return expanded_size;
}

arrow::Result<int64_t> EstimateExpandedSize(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<vdb::vector<IndexInfo>> index_infos,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches,
    int64_t current_activeset_record_count, int64_t active_set_size_limit) {
  int64_t expanded_size = 0;
  size_t append_record_count = 0;
  for (auto& record_batch : record_batches) {
    append_record_count += record_batch->num_rows();
    ARROW_ASSIGN_OR_RAISE(auto rb_expanded_size,
                          EstimateExpandedSize(schema, record_batch));
    expanded_size += rb_expanded_size;
    /* add embedding size */
    auto ann_column_id =
        vdb::stoi32(schema->metadata()->Get("ann_column_id").ValueOr("-1"));
    if (ann_column_id.ok() && ann_column_id.ValueUnsafe() != -1) {
      auto dimension = vdb::stoi32(schema->field(ann_column_id.ValueUnsafe())
                                       ->metadata()
                                       ->Get("dimension")
                                       .ValueOr("-1"));
      if (dimension.ok() && dimension.ValueUnsafe() != -1) {
        expanded_size +=
            sizeof(float) * dimension.ValueUnsafe() * record_batch->num_rows();
      }
    }
  }
  int64_t current_set_size = current_activeset_record_count;
  int64_t new_set_count =
      ((current_set_size + append_record_count) / active_set_size_limit) - 1;
  if (new_set_count > 0) {
    ARROW_ASSIGN_OR_RAISE(auto initial_set_size,
                          EstimateInitialSetAndIndexSize(
                              schema, index_infos, active_set_size_limit));
    expanded_size += initial_set_size;
    /* [DAXE-939] add buffer size */
    expanded_size += 656 * active_set_size_limit;
  }
  return expanded_size;
}

Status Segment::_CheckMemoryAvailability(
    const std::shared_ptr<Table>& table,
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches) {
  for (int64_t retry_cnt = 0;
       retry_cnt <= vdb::ServerConfiguration::GetMaxMemoryRetryCount();
       ++retry_cnt) {
    ARROW_ASSIGN_OR_RAISE(
        auto expanded_size,
        EstimateExpandedSize(table->GetInternalSchema(), table->GetIndexInfos(),
                             record_batches, ActiveSetRecordCount(),
                             table->GetActiveSetSizeLimit()));

    if ((GetRedisAllocatedSize() + expanded_size) <
        vdb::ServerConfiguration::GetMaxMemory()) {
      return Status::Ok();
    }

    if (retry_cnt < vdb::ServerConfiguration::GetMaxMemoryRetryCount()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Waiting for sufficient memory for %ld "
                 "milliseconds. "
                 "mem_used: %ld, retry_cnt: %ld, max_retry_cnt: %ld",
                 table->GetTableName().data(), segment_id_.data(),
                 vdb::ServerConfiguration::GetMaxMemoryRetryInterval(),
                 GetRedisAllocatedSize(), retry_cnt,
                 vdb::ServerConfiguration::GetMaxMemoryRetryCount());

      std::this_thread::sleep_for(std::chrono::milliseconds(
          vdb::ServerConfiguration::GetMaxMemoryRetryInterval()));
    }
  }
  return Status::OutOfMemory("out of memory.");
}

Status Segment::_CheckMemoryAvailability(
    const std::shared_ptr<Table>& table,
    const std::string_view& record_string) {
  for (int64_t retry_cnt = 0;
       retry_cnt <= vdb::ServerConfiguration::GetMaxMemoryRetryCount();
       ++retry_cnt) {
    ARROW_ASSIGN_OR_RAISE(
        auto expanded_size,
        EstimateExpandedSize(table->GetInternalSchema(), table->GetIndexInfos(),
                             record_string, ActiveSetRecordCount(),
                             table->GetActiveSetSizeLimit()));

    if ((GetRedisAllocatedSize() + expanded_size) <
        vdb::ServerConfiguration::GetMaxMemory()) {
      break;
    }

    if (retry_cnt < vdb::ServerConfiguration::GetMaxMemoryRetryCount()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Waiting for sufficient memory for %ld "
                 "milliseconds. "
                 "mem_used: %ld, retry_cnt: %ld, max_retry_cnt: %ld",
                 table->GetTableName().data(), segment_id_.data(),
                 vdb::ServerConfiguration::GetMaxMemoryRetryInterval(),
                 GetRedisAllocatedSize(), retry_cnt,
                 vdb::ServerConfiguration::GetMaxMemoryRetryCount());

      std::this_thread::sleep_for(std::chrono::milliseconds(
          vdb::ServerConfiguration::GetMaxMemoryRetryInterval()));
    }
  }
  return Status::OutOfMemory("out of memory.");
}

Status BuildColumns(
    std::shared_ptr<arrow::Schema>& internal_schema,
    vdb::vector<std::shared_ptr<vdb::MutableArray>>& column_vector,
    size_t max_count) {
  vdb::metrics::ScopedDurationMetric make_active_set_latency(
      vdb::metrics::MakeActiveSetLatency);
  for (auto& field : internal_schema->fields()) {
    switch (field->type()->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                                     \
  {                                                                      \
    case ARROW_TYPE: {                                                   \
      column_vector.emplace_back(vdb::make_shared<ARR_TYPE>(max_count)); \
    } break;                                                             \
  }
      __CASE(arrow::Type::BOOL, BooleanArray);
      __CASE(arrow::Type::INT8, Int8Array);
      __CASE(arrow::Type::INT16, Int16Array);
      __CASE(arrow::Type::INT32, Int32Array);
      __CASE(arrow::Type::INT64, Int64Array);
      __CASE(arrow::Type::UINT8, UInt8Array);
      __CASE(arrow::Type::UINT16, UInt16Array);
      __CASE(arrow::Type::UINT32, UInt32Array);
      __CASE(arrow::Type::UINT64, UInt64Array);
      __CASE(arrow::Type::FLOAT, FloatArray);
      __CASE(arrow::Type::DOUBLE, DoubleArray);
      __CASE(arrow::Type::STRING, StringArray);
      __CASE(arrow::Type::LARGE_STRING, LargeStringArray);
#undef __CASE
      case arrow::Type::LIST: {
        auto type = std::static_pointer_cast<arrow::ListType>(field->type())
                        ->value_type();
        switch (type->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                                     \
  {                                                                      \
    case ARROW_TYPE: {                                                   \
      column_vector.emplace_back(vdb::make_shared<ARR_TYPE>(max_count)); \
    } break;                                                             \
  }
          __CASE(arrow::Type::BOOL, BooleanListArray);
          __CASE(arrow::Type::INT8, Int8ListArray);
          __CASE(arrow::Type::INT16, Int16ListArray);
          __CASE(arrow::Type::INT32, Int32ListArray);
          __CASE(arrow::Type::INT64, Int64ListArray);
          __CASE(arrow::Type::UINT8, UInt8ListArray);
          __CASE(arrow::Type::UINT16, UInt16ListArray);
          __CASE(arrow::Type::UINT32, UInt32ListArray);
          __CASE(arrow::Type::UINT64, UInt64ListArray);
          __CASE(arrow::Type::FLOAT, FloatListArray);
          __CASE(arrow::Type::DOUBLE, DoubleListArray);
          __CASE(arrow::Type::STRING, StringListArray);
#undef __CASE
          default:
            return Status::InvalidArgument("Wrong list schema.");
        }
      } break;
      case arrow::Type::FIXED_SIZE_LIST: {
        auto fixed_size_list_type =
            std::static_pointer_cast<arrow::FixedSizeListType>(field->type());
        auto type = fixed_size_list_type->value_type();
        auto list_size = fixed_size_list_type->list_size();
        switch (type->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                         \
  {                                                          \
    case ARROW_TYPE: {                                       \
      column_vector.emplace_back(                            \
          vdb::make_shared<ARR_TYPE>(list_size, max_count)); \
    } break;                                                 \
  }
          __CASE(arrow::Type::BOOL, BooleanFixedSizeListArray);
          __CASE(arrow::Type::INT8, Int8FixedSizeListArray);
          __CASE(arrow::Type::INT16, Int16FixedSizeListArray);
          __CASE(arrow::Type::INT32, Int32FixedSizeListArray);
          __CASE(arrow::Type::INT64, Int64FixedSizeListArray);
          __CASE(arrow::Type::UINT8, UInt8FixedSizeListArray);
          __CASE(arrow::Type::UINT16, UInt16FixedSizeListArray);
          __CASE(arrow::Type::UINT32, UInt32FixedSizeListArray);
          __CASE(arrow::Type::UINT64, UInt64FixedSizeListArray);
          __CASE(arrow::Type::FLOAT, FloatFixedSizeListArray);
          __CASE(arrow::Type::DOUBLE, DoubleFixedSizeListArray);
          __CASE(arrow::Type::STRING, StringFixedSizeListArray);
#undef __CASE
          default:
            return Status::InvalidArgument("Wrong fixed_size_list schema.");
        }
      } break;
      default:
        return Status::InvalidArgument("Wrong Schema.");
    }
  }

  return Status::Ok();
}

arrow::Result<std::pair<std::shared_ptr<arrow::RecordBatch>,
                        std::shared_ptr<arrow::Buffer>>>
_LoadRecordBatchFrom(std::string& file_path,
                     std::shared_ptr<arrow::Schema>& schema) {
  std::shared_ptr<arrow::Buffer> serialized_buffer;
  std::shared_ptr<arrow::io::BufferReader> buf_reader;

  ARROW_ASSIGN_OR_RAISE(auto infile,
                        arrow::io::ReadableFile::Open(file_path, &arrow_pool));

  ARROW_ASSIGN_OR_RAISE(int64_t file_size, infile->GetSize());

  ARROW_ASSIGN_OR_RAISE(serialized_buffer, infile->Read(file_size));

  buf_reader = std::make_shared<arrow::io::BufferReader>(serialized_buffer);

  arrow::ipc::DictionaryMemo dictionary_memo;

  auto options = arrow::ipc::IpcReadOptions::Defaults();
  options.memory_pool = &arrow_pool;
  ARROW_ASSIGN_OR_RAISE(auto record_batch,
                        arrow::ipc::ReadRecordBatch(schema, &dictionary_memo,
                                                    options, buf_reader.get()));

  return std::make_pair(record_batch, serialized_buffer);
}

Status _SaveRecordBatchTo(std::string& file_path,
                          std::shared_ptr<arrow::RecordBatch> rb) {
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  options.allow_64bit = true;
  options.memory_pool = &arrow_pool;
  ARROW_ASSIGN_OR_RAISE(auto serialized_buffer,
                        arrow::ipc::SerializeRecordBatch(*rb, options));

  ARROW_ASSIGN_OR_RAISE(auto outfile,
                        arrow::io::FileOutputStream::Open(file_path));
  ARROW_RETURN_NOT_OK(
      outfile->Write(serialized_buffer->data(), serialized_buffer->size()));
  ARROW_RETURN_NOT_OK(outfile->Close());
  return arrow::Status::OK();
}

Status Segment::LoadInactiveSets(const std::string& directory_path) {
  uint16_t inactive_set_count = inactive_set_count_for_load;
  auto table = table_.lock();
  auto internal_schema = table->GetInternalSchema();
  auto segment = shared_from_this();
  inactive_sets_.resize(inactive_set_count);

  /* load inactive sets */
  for (uint64_t i = 0; i < inactive_set_count; i++) {
    auto temp_rs_inactive_set =
        std::make_shared<vdb::InactiveSet>(nullptr, nullptr);
    SwapInactiveSet(i, temp_rs_inactive_set, nullptr);
    std::string inactive_set_file_path(directory_path);
    inactive_set_file_path.append("/sets.inactive/set.");
    inactive_set_file_path.append(std::to_string(i));
    auto maybe_rb_buffer_pair =
        _LoadRecordBatchFrom(inactive_set_file_path, internal_schema);
    if (!maybe_rb_buffer_pair.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Loading Inactive Set (%lu) from file (%s) "
                 "- fallback - also Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 static_cast<long unsigned int>(i),
                 inactive_set_file_path.data(),
                 maybe_rb_buffer_pair.status().ToString().data());
      return Status::InvalidArgument(maybe_rb_buffer_pair.status().ToString());
    }
    auto [inactive_rb, buffer] = maybe_rb_buffer_pair.ValueUnsafe();
    if (inactive_rb->num_rows() == 0) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Loading Inactive Set (%lu) from file (%s) "
                 "- fallback - resulted in empty set.",
                 table->GetTableName().data(), segment_id_.data(),
                 static_cast<long unsigned int>(i),
                 inactive_set_file_path.data());
      return Status::InvalidArgument(
          "Inactive set is empty (loaded from file via fallback).");
    }

    auto inactive_set_to_add =
        vdb::make_shared<InactiveSet>(inactive_rb, buffer);
    SwapInactiveSet(i, inactive_set_to_add, nullptr);

    auto inactive_set = GetInactiveSet(i);
    if (inactive_set) {
      size_ += inactive_set->NumRows();
    } else {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Inactive Set (%lu) is null after loading "
                 "attempts. This should not happen if exceptions are properly "
                 "thrown.",
                 table->GetTableName().data(), segment_id_.data(),
                 static_cast<long unsigned int>(i));

      return Status::InvalidArgument(
          "Inactive set is null or has no RecordBatch.");
    }
  }

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Loading Inactive Sets from "
             "(%s/inactive_sets/set#) is Done.",
             table->GetTableName().data(), segment_id_.data(),
             directory_path.data());
  return Status::Ok();
}

Segment::Segment(std::shared_ptr<Table> table, const std::string& segment_id,
                 const uint16_t segment_number)
    : table_{table},
      segment_number_{segment_number},
      segment_id_{segment_id},
      size_{0},
      deleted_size_{0} {
  auto internal_schema = table->GetInternalSchema();
  auto status = BuildColumns(internal_schema, active_set_,
                             table->GetActiveSetSizeLimit());
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Creating is Failed: %s",
               table->GetTableName().data(), segment_id.data(),
               status.ToString().data());
    throw std::invalid_argument(status.ToString());
  }
  RenewActiveSetRecordBatch();

  if (table->HasIndex()) {
    try {
      index_handler_ =
          vdb::make_shared<vdb::IndexHandler>(table, segment_number_);
    } catch (const std::invalid_argument& e) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Creating index handler is Failed: %s",
                 table->GetTableName().data(), segment_id.data(), e.what());
      /* TODO: Handle this error.
       *       - rebuild index handler */
    }
  } else {
    index_handler_ = nullptr;
  }
  auto has_primary_key_column = table->GetPrimaryKeyColumnId().ok();
  if (has_primary_key_column) {
    primary_key_index_ = vdb::make_shared<PrimaryKeyIndex>();
  } else {
    primary_key_index_ = nullptr;
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Creating is Done: %s",
             table->GetTableName().data(), segment_id.data(),
             index_handler_ != nullptr ? "has index." : "has no index.");
}

Segment::Segment(std::shared_ptr<Table> table, const std::string& segment_id,
                 const uint16_t segment_number, std::string& directory_path)
    : table_{table},
      segment_number_{segment_number},
      segment_id_{segment_id},
      size_{0},
      deleted_size_{0} {
  std::error_code ec;
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Loading from (%s) is Started.",
             table->GetTableName().data(), segment_id.data(),
             directory_path.data());

  /* load manifest */
  std::string manifest_file_path(directory_path);
  manifest_file_path.append("/manifest");
  uint8_t* buffered_manifest = nullptr;
  size_t manifest_size = 0;
  Status status;
  /* existence of manifest shows completeness of saving snapshot */
  if (!std::filesystem::exists(manifest_file_path, ec)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Loading Manifest from (%s) is Failed: %s",
               table->GetTableName().data(), segment_id.data(),
               manifest_file_path.data(), ec.message().data());
    throw std::invalid_argument("saving snapshot of segment is not completed.");
  }
  manifest_size = std::filesystem::file_size(manifest_file_path);
  ARROW_CAST_OR_NULL(buffered_manifest, uint8_t*,
                     AllocateAlignedUseThrow(64, manifest_size));
  status = ReadFrom(manifest_file_path, buffered_manifest, manifest_size, 0);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Loading Manifest from (%s) is Failed: %s",
               table->GetTableName().data(), segment_id.data(),
               manifest_file_path.data(), status.ToString().data());
    throw std::invalid_argument(status.ToString());
  }

  uint64_t buffer_offset = 0;
  /* get inactive_set_count */
  uint64_t inactive_set_count;
  buffer_offset += GetLengthFrom(buffered_manifest, inactive_set_count);
  inactive_set_count_for_load =
      static_cast<uint16_t>(inactive_set_count);  // For load inactive set
  /* get index count */
  uint64_t index_count = 0;
  if (table->HasIndex()) {
    buffer_offset +=
        GetLengthFrom(buffered_manifest + buffer_offset, index_count);
  }
  /* get deleted size */
  uint64_t deleted_size = 0;
  buffer_offset +=
      GetLengthFrom(buffered_manifest + buffer_offset, deleted_size);
  deleted_size_ = deleted_size;
  DeallocateAligned(buffered_manifest, manifest_size);
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Loading Manifest from (%s) is Done.",
             table->GetTableName().data(), segment_id.data(),
             manifest_file_path.data());

  auto schema = table->GetSchema();
  /* add schema for hidden column */
  auto internal_schema = table->GetInternalSchema();
  {
    /* load active rb */
    std::string active_set_file_path(directory_path);
    active_set_file_path.append("/set.active");
    auto maybe_rb_buffer_pair =
        _LoadRecordBatchFrom(active_set_file_path, internal_schema);
    if (!maybe_rb_buffer_pair.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Loading Active Set from (%s) is Failed: %s",
                 table->GetTableName().data(), segment_id.data(),
                 active_set_file_path.data(),
                 maybe_rb_buffer_pair.status().ToString().data());
      throw std::invalid_argument(maybe_rb_buffer_pair.status().ToString());
    }
    auto [active_rb, buffer] = maybe_rb_buffer_pair.ValueUnsafe();
    /* append active rb into active set */
    status = BuildColumns(internal_schema, active_set_,
                          table->GetActiveSetSizeLimit());
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Loading Active Set from (%s) is Failed: %s",
                 table->GetTableName().data(), segment_id.data(),
                 active_set_file_path.data(), status.ToString().data());
      throw std::invalid_argument(status.ToString());
    }
    if (active_rb->num_rows() > 0) {
      /* active_rb already has deleted flag column */
      status = AddRecordBatch(active_rb);
      if (!status.ok()) {
        SYSTEM_LOG(
            vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
            "Segment (%s, %s) Loading Active Set from (%s) is Failed: %s",
            table->GetTableName().data(), segment_id.data(),
            active_set_file_path.data(), status.ToString().data());
        throw std::invalid_argument(status.ToString());
      }
    }
    RenewActiveSetRecordBatch();
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Segment (%s, %s) Loading Active Set from (%s) is Done.",
               table->GetTableName().data(), segment_id.data(),
               active_set_file_path.data());
    /* active recordbatch and buffer of serialized recordbatch are freed */
  }

  /* load vector indexes */
  if (index_count > 0) {
    std::string index_directory_path(directory_path);
    index_directory_path.append("/index");
    try {
      index_handler_ = vdb::make_shared<vdb::IndexHandler>(
          table, segment_number_, index_directory_path, index_count);
    } catch (const std::invalid_argument& e) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Loading Vector Indexes from "
                 "(%s/index/index#) is Failed: %s",
                 table->GetTableName().data(), segment_id.data(),
                 directory_path.data(), e.what());
      /* TODO: Handle this error.
       *       - rebuild index using record data */
    }
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Segment (%s, %s) Loading Vector Indexes from "
               "(%s/index/index#) is Done.",
               table->GetTableName().data(), segment_id.data(),
               directory_path.data());
  } else {
    index_handler_ = nullptr;
  }
  if (table->HasPrimaryKey()) {
    auto primary_key_index_file_path(directory_path);
    primary_key_index_file_path.append("/primary_key_index.bin");
    primary_key_index_ =
        vdb::make_shared<PrimaryKeyIndex>(primary_key_index_file_path);
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Loading Primary Key Index from "
               "(%s/primary_key_index.bin) is Done.",
               table->GetTableName().data(), segment_id.data(),
               directory_path.data());
  } else {
    primary_key_index_ = nullptr;
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Loading from (%s) is Done.",
             table->GetTableName().data(), segment_id.data(),
             directory_path.data());
}

Status Segment::Save(std::string& directory_path) {
  auto table = table_.lock();
  std::error_code ec;
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Saving to (%s) is Started.",
             table->GetTableName().data(), segment_id_.data(),
             directory_path.data());
  if (!std::filesystem::create_directory(directory_path, ec)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Saving to (%s) is Failed: %s",
               table->GetTableName().data(), directory_path.data(),
               segment_id_.data(), ec.message().data());
    throw std::invalid_argument(ec.message());
  }
  /* save primary key index */
  if (table->HasPrimaryKey()) {
    auto primary_key_index_file_path(directory_path);
    primary_key_index_file_path.append("/primary_key_index.bin");
    auto status = SavePrimaryKeyIndex(primary_key_index_file_path);
    if (!status.ok()) {
      SYSTEM_LOG(
          vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
          "Segment (%s, %s) Saving Primary Key Index to (%s) is Failed: %s",
          table->GetTableName().data(), segment_id_.data(),
          directory_path.data(), status.ToString().data());
      return status;
    }
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Segment (%s, %s) Saving Primary Key Index to (%s) is Done.",
               table->GetTableName().data(), segment_id_.data(),
               directory_path.data());
  } else {
    primary_key_index_ = nullptr;
  }
  /* save active set */
  std::string active_set_file_path(directory_path);
  active_set_file_path.append("/set.active");
  auto active_rb = ActiveSetRecordBatch();
  if (active_rb == nullptr) {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Saving Active Set to (%s) is Failed: Active "
               "set is null.",
               table->GetTableName().data(), segment_id_.data(),
               active_set_file_path.data());
    return Status::InvalidArgument("Active set is null. It's not possible.");
  }
  auto status = _SaveRecordBatchTo(active_set_file_path, active_rb);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Saving Active Set to (%s) is Failed: %s",
               table->GetTableName().data(), segment_id_.data(),
               active_set_file_path.data(), status.ToString().data());
    return status;
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Saving Active Set to (%s) is Done.",
             table->GetTableName().data(), segment_id_.data(),
             active_set_file_path.data());
  /* save inactive set recordbatches */
  std::string inactive_sets_directory_path(directory_path);
  inactive_sets_directory_path.append("/sets.inactive");

  std::filesystem::create_directory(inactive_sets_directory_path);

  for (size_t i = 0; i < InactiveSets().size(); i++) {
    std::string inactive_set_file_path(inactive_sets_directory_path);
    inactive_set_file_path.append("/set.");
    inactive_set_file_path.append(std::to_string(i));
    auto inactive_set = GetInactiveSet(i);
    auto inactive_rb = inactive_set->GetRb();
    if (!inactive_rb) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Saving Inactive Set(%lu) to (%s) is Failed: "
                 "Inactive set RecordBatch is null.",
                 table->GetTableName().data(), segment_id_.data(), i,
                 inactive_set_file_path.data());
      return Status::InvalidArgument("Inactive set RecordBatch is null.");
    }
    status = _SaveRecordBatchTo(inactive_set_file_path, inactive_rb);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Saving Inactive Set to (%s) is Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 inactive_set_file_path.data(), status.ToString().data());
      return status;
    }
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Saving Inactive Sets to (%s/set#) is Done.",
             table->GetTableName().data(), segment_id_.data(),
             inactive_sets_directory_path.data());
  /* save vector indexes */
  if (HasIndex()) {
    auto index_directory_path(directory_path);
    index_directory_path.append("/index");
    auto status = IndexHandler()->Save(index_directory_path);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Saving Vector Indexes to (%s) is Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 index_directory_path.data(), status.ToString().data());
      return status;
    }
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Segment (%s, %s) Saving Vector Indexes to (%s/index#) is Done.",
               table->GetTableName().data(), segment_id_.data(),
               index_directory_path.data());
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Segment (%s, %s) Saving Vector Indexes is Skipped: No Index.",
               table->GetTableName().data(), segment_id_.data());
  }

  /* save manifest (# of inactive sets and # of indexes) */
  std::string manifest_file_path(directory_path);
  manifest_file_path.append("/manifest");
  status = SaveManifest(manifest_file_path);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Saving Manifest to (%s) is Failed: %s",
               table->GetTableName().data(), segment_id_.data(),
               manifest_file_path.data(), status.ToString().data());
    return status;
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Saving Manifest to (%s) is Done.",
             table->GetTableName().data(), segment_id_.data(),
             manifest_file_path.data());

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Saving to (%s) is Done.",
             table->GetTableName().data(), segment_id_.data(),
             directory_path.data());
  return Status::Ok();
}

Status Segment::SaveManifest(std::string& file_path) const {
  vdb::vector<uint8_t> buffer;
  int buffer_offset = 0;
  /* save # of inactive sets */
  int32_t needed_bytes = ComputeBytesFor(InactiveSetCount());
  buffer.resize(needed_bytes);
  buffer_offset +=
      PutLengthTo(buffer.data() + buffer_offset, InactiveSetCount());
  if (HasIndex()) {
    /* save # of indexes */
    needed_bytes = ComputeBytesFor(IndexHandler()->Size());
    buffer.resize(buffer.size() + needed_bytes);
    buffer_offset +=
        PutLengthTo(buffer.data() + buffer_offset, IndexHandler()->Size());
  }
  /* save # of deleted records */
  needed_bytes = ComputeBytesFor(DeletedSize());
  buffer.resize(buffer.size() + needed_bytes);
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, DeletedSize());

  vdb::Status status = WriteTo(file_path, buffer.data(), buffer_offset, 0);
  if (!status.ok()) {
    return status;
  }
  return status;
}

Status Segment::SavePrimaryKeyIndex(std::string& file_path) const {
  return primary_key_index_->Save(file_path);
}

std::string Segment::ToString(bool show_records) const {
  auto table = table_.lock();
  auto fields = table->GetSchema()->fields();
  std::stringstream ss;
  ss << "SegmentId:  " << segment_id_ << std::endl;
  ss << "SegmentSize: " << size_ << std::endl;
  ss << "SegmentData: " << std::endl;
  ss << "---- Inactive sets (set count=" << InactiveSetCount() << ")----"
     << std::endl;
  if (show_records) {
    for (auto& set : inactive_sets_) {
      ss << set->ToString() << std::endl;
    }
  } else {
    for (auto& set : inactive_sets_) {
      ss << set->NumRows() << " records are not shown." << std::endl;
    }
  }
  ss << "---- Active set (record count=" << ActiveSetRecordCount() << ")----"
     << std::endl;
  if (show_records) {
    for (size_t i = 0; i < fields.size(); i++) {
      ss << fields[i]->name() << ": ";
      ss << active_set_[i]->ToString() << std::endl;
    }
  } else {
    ss << ActiveSetRecordCount() << " records are not shown." << std::endl;
  }
  if (HasIndex()) {
    ss << index_handler_->ToString(false) << std::endl;
  }
  return ss.str();
}

Status Segment::AddDenseEmbedding(uint64_t column_id, const float* embedding,
                                  const uint16_t set_number,
                                  const uint32_t record_number) {
  /* there is no need to distinguish set by label. i th record is mapped to i th
   * embedding. */
  size_t label =
      vdb::LabelInfo::Build(segment_number_, set_number, record_number);
  auto embedding_store = IndexHandler()->GetEmbeddingStore(column_id);
  auto status = embedding_store->Write(embedding, label, 1);
  if (!status.ok()) {
    return status;
  }
  return IndexHandler()->AddDenseEmbedding(column_id, embedding, label);
}

Status Segment::AddDenseEmbeddingsJob(
    uint64_t column_id,
    const std::shared_ptr<arrow::FixedSizeListArray>& embeddings,
    const uint64_t starting_label) {
  auto table = table_.lock();
  /* there is no need to distinguish set by label. i th record is mapped to i th
   * embedding. */
  auto set_number = vdb::LabelInfo::GetSetNumber(starting_label);
  ARROW_ASSIGN_OR_RAISE(auto embeddings_copy, MakeDeepCopy(embeddings));
  {  // job queue lock scope
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Adding new job. (%s, %lu)",
               LabelInfo::ToString(starting_label).data(),
               embeddings_copy->length());
    std::lock_guard<std::mutex> guard(table->job_queue_mtx_);
    table->job_queue_.emplace_back(segment_id_, column_id, set_number,
                                   embeddings_copy, starting_label, 0);
    table->indexing_ = true;
  }  // end of job queue lock scope
  table->cond_.notify_all();
  return Status::Ok();
}

/* AppendRecord is deprecated. Use AppendRecords instead. */
Status Segment::AppendRecord(const std::string_view& record_string) {
  std::shared_ptr<Table> table = table_.lock();
  auto schema = table->GetInternalSchema();
  auto fields = schema->fields();
  size_t column_count = schema->fields().size();
  size_t column_id = 0;

  if (vdb::ServerConfiguration::GetMaxMemory() != 0 &&
      vdb::ServerConfiguration::GetHiddenCheckMemoryAvailability()) {
    auto status = _CheckMemoryAvailability(table, record_string);
    if (!status.ok()) {
      return Status::OutOfMemory("InsertCommand is Failed: out of memory.");
    }
  }

  /* active_rb_renewer_ is set in InsertionGuard */
  InsertionGuard guard(this);

  if (ActiveSetRecordCount() == table->GetActiveSetSizeLimit()) {
    guard.UpdateInactiveSetBackup();
    auto status = MakeInactive(guard);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Appending Single Record is Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 status.ToString().data());
      return status;
    }
    guard.BackupCurrentActiveSetState();
  }

  guard.SetStage(InsertionGuard::Stage::kActiveSetModified);

  size_t prev_record_count = ActiveSetRecordCount();

  struct IndexOperation {
    uint64_t column_id;
    std::string token;
    std::vector<float> dense_data;
  };

  std::vector<IndexOperation> index_operations;

  auto prev = 0;
  auto pos = record_string.find(kRS, 0);

  while (true) {
    auto token = record_string.substr(prev, pos - prev);

    if (table->IsAnnColumn(column_id)) {
      auto dimension = table->GetDimension(column_id);
      std::vector<float> dense_data;
      dense_data.reserve(dimension);

      auto _prev_ = 0;
      auto _pos_ = token.find(kGS, _prev_);
      while (true) {
        auto value_token = token.substr(_prev_, _pos_ - _prev_);
        ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stof(value_token));
        dense_data.emplace_back(parsed_value);
        if (_pos_ == std::string::npos) break;
        _prev_ = _pos_ + 1;
        _pos_ = token.find(kGS, _prev_);
      }
      index_operations.emplace_back(
          IndexOperation{column_id, "", std::move(dense_data)});

      auto rowid =
          LabelInfo::Build(segment_number_, ActiveSetId(), prev_record_count);
      std::static_pointer_cast<UInt64Array>(active_set_[column_id])
          ->Append(rowid);

    } else {
      switch (fields[column_id]->type()->id()) {
        /* Non-list type cases */
#define __CASE(ARROW_TYPE, ARR_TYPE, PARSED_VALUE)               \
  {                                                              \
    case ARROW_TYPE: {                                           \
      PARSED_VALUE;                                              \
      std::static_pointer_cast<ARR_TYPE>(active_set_[column_id]) \
          ->Append(parsed_value);                                \
    } break;                                                     \
  }
        __CASE(arrow::Type::BOOL, BooleanArray,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stobool(token)));
        __CASE(arrow::Type::INT8, Int8Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoi32(token)));
        __CASE(arrow::Type::INT16, Int16Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoi32(token)));
        __CASE(arrow::Type::INT32, Int32Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoi32(token)));
        __CASE(arrow::Type::INT64, Int64Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoi64(token)));
        __CASE(arrow::Type::UINT8, UInt8Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoui32(token)));
        __CASE(arrow::Type::UINT16, UInt16Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoui32(token)));
        __CASE(arrow::Type::UINT32, UInt32Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoui32(token)));
        __CASE(arrow::Type::UINT64, UInt64Array,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stoui64(token)));
        __CASE(arrow::Type::FLOAT, FloatArray,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stof(token)));
        __CASE(arrow::Type::DOUBLE, DoubleArray,
               ARROW_ASSIGN_OR_RAISE(auto parsed_value, vdb::stod(token)));
        __CASE(arrow::Type::STRING, StringArray, auto parsed_value = token);
        __CASE(arrow::Type::LARGE_STRING, LargeStringArray,
               auto parsed_value = token);
#undef __CASE
        /* list type cases */
        case arrow::Type::LIST: {
          auto type = std::static_pointer_cast<arrow::ListType>(
                          fields[column_id]->type())
                          ->value_field()
                          ->type()
                          ->id();
          auto prev = 0;
          auto pos = token.find(kGS, prev);
          switch (type) {  // Handling Subtype
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE, PARSED_VALUE)                      \
  {                                                                            \
    case ARROW_TYPE: {                                                         \
      std::vector<CTYPE> list_value;                                           \
      while (true) {                                                           \
        auto value_token = token.substr(prev, pos - prev);                     \
        PARSED_VALUE;                                                          \
        list_value.emplace_back((parsed_value));                               \
        if (pos == std::string::npos) break;                                   \
        prev = pos + 1;                                                        \
        pos = token.find(kGS, prev);                                           \
      }                                                                        \
      auto status = std::static_pointer_cast<ARR_TYPE>(active_set_[column_id]) \
                        ->Append(list_value);                                  \
      if (!status.ok()) return status;                                         \
    } break;                                                                   \
  }
            __CASE(arrow::Type::BOOL, bool, BoolListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stobool(value_token)));
            __CASE(arrow::Type::INT8, int8_t, Int8ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi32(value_token)));
            __CASE(arrow::Type::INT16, int16_t, Int16ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi32(value_token)));
            __CASE(arrow::Type::INT32, int32_t, Int32ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi32(value_token)));
            __CASE(arrow::Type::INT64, int64_t, Int64ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi64(value_token)));
            __CASE(arrow::Type::UINT8, uint8_t, UInt8ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui32(value_token)));
            __CASE(arrow::Type::UINT16, uint16_t, UInt16ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui32(value_token)));
            __CASE(arrow::Type::UINT32, uint32_t, UInt32ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui32(value_token)));
            __CASE(arrow::Type::UINT64, uint64_t, UInt64ListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui64(value_token)));
            __CASE(arrow::Type::FLOAT, float, FloatListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stof(value_token)));
            __CASE(arrow::Type::DOUBLE, double, DoubleListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stod(value_token)));
            __CASE(arrow::Type::STRING, std::string, StringListArray,
                   auto parsed_value = std::string(value_token));
            default:
              return Status::InvalidArgument("Not a valid subtype for list.");
          }
#undef __CASE
        } break;

        /* fixed size list type cases */
        case arrow::Type::FIXED_SIZE_LIST: {
          auto type = std::static_pointer_cast<arrow::FixedSizeListType>(
                          fields[column_id]->type())
                          ->value_field()
                          ->type()
                          ->id();
          auto _prev_ = 0;
          auto _pos_ = token.find(kGS, _prev_);
          switch (type) {
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE, PARSED_VALUE)                      \
  {                                                                            \
    case ARROW_TYPE: {                                                         \
      std::vector<CTYPE> list_value;                                           \
      while (true) {                                                           \
        auto value_token = token.substr(_prev_, _pos_ - _prev_);               \
        PARSED_VALUE;                                                          \
        list_value.emplace_back((parsed_value));                               \
        if (_pos_ == std::string::npos) break;                                 \
        _prev_ = _pos_ + 1;                                                    \
        _pos_ = token.find(kGS, _prev_);                                       \
      }                                                                        \
      auto status = std::static_pointer_cast<ARR_TYPE>(active_set_[column_id]) \
                        ->Append(list_value);                                  \
      if (!status.ok()) return status;                                         \
    } break;                                                                   \
  }
            __CASE(arrow::Type::BOOL, bool, BoolFixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stobool(value_token)));
            __CASE(arrow::Type::INT8, int8_t, Int8FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi32(value_token)));
            __CASE(arrow::Type::INT16, int16_t, Int16FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi32(value_token)));
            __CASE(arrow::Type::INT32, int32_t, Int32FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi32(value_token)));
            __CASE(arrow::Type::INT64, int64_t, Int64FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoi64(value_token)));
            __CASE(arrow::Type::UINT8, uint8_t, UInt8FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui32(value_token)));
            __CASE(arrow::Type::UINT16, uint16_t, UInt16FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui32(value_token)));
            __CASE(arrow::Type::UINT32, uint32_t, UInt32FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui32(value_token)));
            __CASE(arrow::Type::UINT64, uint64_t, UInt64FixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stoui64(value_token)));
            __CASE(arrow::Type::FLOAT, float, FloatFixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stof(value_token)));
            __CASE(arrow::Type::DOUBLE, double, DoubleFixedSizeListArray,
                   ARROW_ASSIGN_OR_RAISE(auto parsed_value,
                                         vdb::stod(value_token)));
            __CASE(arrow::Type::STRING, std::string, StringFixedSizeListArray,
                   auto parsed_value = std::string(value_token));
            default:
              return Status::InvalidArgument(
                  "Not a valid subtype for fixed_size_list.");
#undef __CASE
          }
        } break;
        default:
          return Status::InvalidArgument(
              "Arrow builder fails to append the value.");
      }
    }

    if (pos == std::string::npos) break;

    prev = pos + 1;
    pos = record_string.find(kRS, prev);
    column_id += 1;
  }

  /* hidden column for deleted flag */
  std::static_pointer_cast<BooleanArray>(active_set_[++column_id])
      ->Append(false);

  if (column_id != column_count - 1) {
    return Status::InvalidArgument("Not enough column is provided.");
  }

  if (!index_operations.empty()) {
    for (const auto& op : index_operations) {
      Status status;
      status = AddDenseEmbedding(op.column_id, op.dense_data.data(),
                                 ActiveSetId(), prev_record_count);

      if (!status.ok()) {
        return status;
      }
    }
  }

  guard.CommitCurrentBatch();
  guard.Success();

  size_++;
  __sync_synchronize();
  return Status::Ok();
}

Status Segment::AddRecordBatch(std::shared_ptr<arrow::RecordBatch>& rb) {
  if (rb->num_rows() == 0) {
    return Status::InvalidArgument("Don't pass empty recordbatch! ");
  }
  std::shared_ptr<Table> table = table_.lock();
  auto table_schema = GetSchema();
  auto rb_schema = rb->schema();
  auto rb_fields = rb_schema->fields();
  size_t prev_active_set_record_count = ActiveSetRecordCount();

  for (int64_t i = 0; i < rb->num_columns(); i++) {
    if (table->IsIndexColumn(i)) {
      if (rb_schema->field(i)->type()->id() == arrow::Type::FIXED_SIZE_LIST ||
          rb_schema->field(i)->type()->id() == arrow::Type::LARGE_STRING) {
        auto starting_rowid = LabelInfo::Build(segment_number_, ActiveSetId(),
                                               prev_active_set_record_count);
        auto rowid_column =
            std::static_pointer_cast<UInt64Array>(active_set_[i]);
        for (int64_t j = 0; j < rb->num_rows(); j++) {
          rowid_column->Append(starting_rowid + j);
        }
        continue;
      }

      if (rb_schema->field(i)->type()->id() != arrow::Type::UINT64) {
        SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                   "ann column (%d, %s) type is not uint64: %s", i,
                   rb_schema->field(i)->name().c_str(),
                   rb_schema->ToString().c_str());
        return Status::InvalidArgument(
            "ann column type is not uint64: CHECK SYSTEM LOG.");
      }
    }

    auto input_arr = rb->column(i);

    if (i < table_schema->num_fields()) {
      auto table_field = table_schema->field(i);
      if (table_field->nullable() == false) {
        if (input_arr->null_count() > 0) {
          return Status::InvalidArgument(
              "Could not insert recordbatch into segment. Field '" +
              table_field->name() +
              "' is not nullable, but the recordbatch contains null values.");
        }
      }
    }

    auto rb_field = rb_schema->field(i);

    switch (rb_field->type()->id()) {
#define __CASE(ARROW_TYPE, ARR_TYPE)                               \
  {                                                                \
    case ARROW_TYPE: {                                             \
      auto casted_mutable_arr =                                    \
          std::static_pointer_cast<vdb::ARR_TYPE>(active_set_[i]); \
      casted_mutable_arr->Append(input_arr.get());                 \
    } break;                                                       \
  }
      __CASE(arrow::Type::BOOL, BooleanArray);
      __CASE(arrow::Type::INT8, Int8Array);
      __CASE(arrow::Type::INT16, Int16Array);
      __CASE(arrow::Type::INT32, Int32Array);
      __CASE(arrow::Type::INT64, Int64Array);
      __CASE(arrow::Type::UINT8, UInt8Array);
      __CASE(arrow::Type::UINT16, UInt16Array);
      __CASE(arrow::Type::UINT32, UInt32Array);
      __CASE(arrow::Type::UINT64, UInt64Array);
      __CASE(arrow::Type::FLOAT, FloatArray);
      __CASE(arrow::Type::DOUBLE, DoubleArray);
      __CASE(arrow::Type::STRING, StringArray);
      __CASE(arrow::Type::LARGE_STRING, LargeStringArray);
#undef __CASE

      case arrow::Type::LIST: {
        auto type = std::static_pointer_cast<arrow::ListType>(rb_field->type())
                        ->value_field()
                        ->type()
                        ->id();
        switch (type) {
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE)                      \
  {                                                              \
    case ARROW_TYPE: {                                           \
      auto casted_mutable_arr =                                  \
          std::static_pointer_cast<ARR_TYPE>(active_set_[i]);    \
      auto status = casted_mutable_arr->Append(input_arr.get()); \
      if (!status.ok()) return status;                           \
    } break;                                                     \
  }
          __CASE(arrow::Type::BOOL, bool, BooleanListArray);
          __CASE(arrow::Type::INT8, int8_t, Int8ListArray);
          __CASE(arrow::Type::INT16, int16_t, Int16ListArray);
          __CASE(arrow::Type::INT32, int32_t, Int32ListArray);
          __CASE(arrow::Type::INT64, int64_t, Int64ListArray);
          __CASE(arrow::Type::UINT8, uint8_t, UInt8ListArray);
          __CASE(arrow::Type::UINT16, uint16_t, UInt16ListArray);
          __CASE(arrow::Type::UINT32, uint32_t, UInt32ListArray);
          __CASE(arrow::Type::UINT64, uint64_t, UInt64ListArray);
          __CASE(arrow::Type::FLOAT, float, FloatListArray);
          __CASE(arrow::Type::DOUBLE, double, DoubleListArray);
          __CASE(arrow::Type::STRING, std::string, StringListArray);
#undef __CASE
          default:
            return Status::InvalidArgument("Not a valid subtype for list.");
        }
      } break;
      case arrow::Type::FIXED_SIZE_LIST: {
        auto type =
            std::static_pointer_cast<arrow::FixedSizeListType>(rb_field->type())
                ->value_field()
                ->type()
                ->id();
        switch (type) {
#define __CASE(ARROW_TYPE, CTYPE, ARR_TYPE, SUBARR_TYPE)         \
  {                                                              \
    case ARROW_TYPE: {                                           \
      auto casted_mutable_arr =                                  \
          std::static_pointer_cast<ARR_TYPE>(active_set_[i]);    \
      auto status = casted_mutable_arr->Append(input_arr.get()); \
      if (!status.ok()) return status;                           \
    } break;                                                     \
  }
          __CASE(arrow::Type::BOOL, bool, BooleanFixedSizeListArray,
                 arrow::BooleanArray);
          __CASE(arrow::Type::INT8, int8_t, Int8FixedSizeListArray,
                 arrow::Int8Array);
          __CASE(arrow::Type::INT16, int16_t, Int16FixedSizeListArray,
                 arrow::Int16Array);
          __CASE(arrow::Type::INT32, int32_t, Int32FixedSizeListArray,
                 arrow::Int32Array);
          __CASE(arrow::Type::INT64, int64_t, Int64FixedSizeListArray,
                 arrow::Int64Array);
          __CASE(arrow::Type::UINT8, uint8_t, UInt8FixedSizeListArray,
                 arrow::UInt8Array);
          __CASE(arrow::Type::UINT16, uint16_t, UInt16FixedSizeListArray,
                 arrow::UInt16Array);
          __CASE(arrow::Type::UINT32, uint32_t, UInt32FixedSizeListArray,
                 arrow::UInt32Array);
          __CASE(arrow::Type::UINT64, uint64_t, UInt64FixedSizeListArray,
                 arrow::UInt64Array);
          __CASE(arrow::Type::FLOAT, float, FloatFixedSizeListArray,
                 arrow::FloatArray);
          __CASE(arrow::Type::DOUBLE, double, DoubleFixedSizeListArray,
                 arrow::DoubleArray);
          __CASE(arrow::Type::STRING, std::string, StringFixedSizeListArray,
                 arrow::StringArray);
#undef __CASE
          default:
            return Status::InvalidArgument(
                "Not a valid subtype for fixed_size_list.");
        }
      } break;
      default:
        return Status::InvalidArgument("");
    }
  }

  size_ += rb->num_rows();
  __sync_synchronize();

  return Status::Ok();
}

Status Segment::AddToIndex(std::shared_ptr<arrow::RecordBatch>& rb,
                           uint32_t start_record_number) {
  if (rb->num_rows() == 0) {
    return Status::InvalidArgument("Don't pass empty recordbatch! ");
  }
  std::shared_ptr<Table> table = table_.lock();
  auto schema = table->GetSchema();

  if (HasIndex()) {
    auto index_infos = table->GetIndexInfos();
    for (auto& index_info : *index_infos) {
      auto index_column_id = index_info.GetColumnId();
      auto index_column = rb->column(index_column_id);

      if (index_info.IsDenseVectorIndex()) {
        auto embeddings =
            std::static_pointer_cast<arrow::FixedSizeListArray>(index_column);
        auto dimension = std::static_pointer_cast<arrow::FixedSizeListType>(
                             embeddings->type())
                             ->list_size();
        auto raw_embeddings = embeddings->values()->data()->GetValues<float>(1);
        raw_embeddings += embeddings->offset() * dimension;
        auto embedding_store =
            IndexHandler()->GetEmbeddingStore(index_column_id);
        auto embedding_count = embeddings->length();
        auto starting_label = LabelInfo::Build(segment_number_, ActiveSetId(),
                                               start_record_number);
        embedding_store->Write(raw_embeddings, starting_label, embedding_count);

        auto status = Status::Ok();
        if (vdb::ServerConfiguration::GetAllowBgIndexThread()) {
          status = AddDenseEmbeddingsJob(index_column_id, embeddings,
                                         starting_label);
        } else {
          status = IndexHandler()->AddDenseEmbeddings(
              index_column_id, embeddings, starting_label);
        }
        if (!status.ok()) return Status::InvalidArgument(status.ToString());
      }
    }
  }
  return Status::Ok();
}

Status Segment::AddToHiddenColumn(std::shared_ptr<arrow::RecordBatch>& rb) {
  /* add rows to __deleted_flag (internal column) column */
  auto schema = GetSchema();
  auto deleted_flag_column =
      std::static_pointer_cast<BooleanArray>(active_set_[schema->num_fields()]);
  for (int64_t i = 0; i < rb->num_rows(); i++) {
    auto status = deleted_flag_column->Append(false);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status Segment::_AppendRecordsInRange(std::shared_ptr<arrow::RecordBatch>& rb,
                                      uint64_t start, uint64_t end,
                                      uint64_t& append_record_count,
                                      InsertionGuard& guard) {
  auto table = table_.lock();
  auto active_set_size_limit = table->GetActiveSetSizeLimit();
  auto length = end - start;
  auto record_batch = rb->Slice(start, length);
  int64_t start_rowno = 0;
  while (start_rowno < record_batch->num_rows()) {
    if (ActiveSetRecordCount() == active_set_size_limit) {
      guard.UpdateInactiveSetBackup();
      auto status = MakeInactive(guard);
      if (!status.ok()) {
        SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                   "Segment (%s, %s) Appending Records is Failed: %s",
                   table->GetTableName().data(), segment_id_.data(),
                   status.ToString().data());
        return status;
      }

      guard.BackupCurrentActiveSetState();

      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
                 "Segment (%s, %s) Making Active Set (size=%ld) into "
                 "Inactive is Done.",
                 table->GetTableName().data(), segment_id_.data(),
                 ActiveSetRecordCount());
    }

    guard.SetStage(InsertionGuard::Stage::kActiveSetModified);

    auto sliced_rb = record_batch->Slice(
        start_rowno, active_set_size_limit - ActiveSetRecordCount());
    uint32_t start_record_number = ActiveSetRecordCount();
    auto status = AddRecordBatch(sliced_rb);

    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Appending Records is Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 status.ToString().data());
      return status;
    }

    status = AddToHiddenColumn(sliced_rb);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Appending deleted flags is Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 status.ToString().data());
      return status;
    }

    status = AddToIndex(sliced_rb, start_record_number);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Appending Embeddings is Failed: %s",
                 table->GetTableName().data(), segment_id_.data(),
                 status.ToString().data());
      return status;
    }

    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
               "Segment (%s, %s) %ld Records are Appended.",
               table->GetTableName().data(), segment_id_.data(),
               (long int)sliced_rb->num_rows());
    start_rowno += sliced_rb->num_rows();

    append_record_count += sliced_rb->num_rows();

    guard.CommitCurrentBatch();
  }
  return Status::Ok();
}

Status Segment::AppendRecords(
    std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches) {
  auto table = table_.lock();

  // Check if record batches are insertable (schema validation including
  // nullable constraints)
  Status status = vdb::CheckRecordBatchIsInsertable(record_batches, table);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) CheckRecordBatchIsInsertable is Failed: %s",
               table->GetTableName().data(), segment_id_.data(),
               status.ToString().data());
    return status;
  }

  if (vdb::ServerConfiguration::GetMaxMemory() != 0 &&
      vdb::ServerConfiguration::GetHiddenCheckMemoryAvailability()) {
    status = _CheckMemoryAvailability(table, record_batches);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) CheckMemoryAvailability is Failed: %s. "
                 "segment_size: %ld, active_set_size_limit: %ld",
                 table->GetTableName().data(), segment_id_.data(),
                 status.ToString().data(), Size(),
                 table->GetActiveSetSizeLimit());
      return Status::OutOfMemory(
          "BatchInsertCommand is Failed: out of memory.");
    }
  }
  /* active_rb_renewer_ is set in InsertionGuard */
  InsertionGuard guard(this);

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Appending Records is Started.",
             table->GetTableName().data(), segment_id_.data());
  uint64_t append_record_count = 0;
  uint64_t input_record_count = 0;
  for (auto& org_record_batch : record_batches) {
    input_record_count += org_record_batch->num_rows();
    auto maybe_primary_key_column_id = table->GetPrimaryKeyColumnId();
    bool has_primary_key_column = maybe_primary_key_column_id.ok();
    PrimaryKeyHandleForInsert pk_handle(GetPrimaryKeyIndex());
    if (has_primary_key_column) {
      auto primary_key_column_id = maybe_primary_key_column_id.ValueUnsafe();
      auto status =
          pk_handle.Initialize(org_record_batch->column(primary_key_column_id));
      if (!status.ok()) {
        return status;
      }
      status = pk_handle.CheckUniqueViolation();
      if (!status.ok()) {
        return status;
      }
      do {
        auto range = pk_handle.GetNextValidOffsetRange();
        if (PrimaryKeyHandleForInsert::IsCompleted(range)) {
          SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                     "Segment (%s, %s) Appending records in all valid offset "
                     "ranges is done.",
                     table->GetTableName().data(), segment_id_.data());
          break;
        }
        auto status =
            _AppendRecordsInRange(org_record_batch, range.first, range.second,
                                  append_record_count, guard);
        if (!status.ok()) {
          SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                     "Segment (%s, %s) Appending records in range (%d, %d) is "
                     "failed: %s",
                     table->GetTableName().data(), segment_id_.data(),
                     range.first, range.second, status.ToString().data());
          return status;
        }
      } while (1);
      status = pk_handle.ApplyToIndex();
      if (!status.ok()) {
        return status;
      }
    } else {
      auto status = _AppendRecordsInRange(org_record_batch, 0,
                                          org_record_batch->num_rows(),
                                          append_record_count, guard);
      if (!status.ok()) {
        return status;
      }
    }
  }
  auto skipped_record_count = input_record_count - append_record_count;
  vdb::metrics::CollectValue(vdb::metrics::BatchInsertRecordCount,
                             append_record_count);
  vdb::metrics::CollectValue(vdb::metrics::BatchInsertRecordSkippedCount,
                             skipped_record_count);

  guard.Success();

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Total %ld Records are Appended.",
             table->GetTableName().data(), segment_id_.data(),
             append_record_count);
  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
             "Segment (%s, %s) Appending Records is Done.",
             table->GetTableName().data(), segment_id_.data());
  return Status::Ok();
}

Status Segment::MakeInactive(InsertionGuard& guard) {
  auto table = table_.lock();
  auto internal_schema = table->GetInternalSchema();
  auto external_schema = table->GetSchema();
  auto active_set_size = ActiveSetRecordCount();

  /* active_data_ is swapped to empty vector.
   * contents of active_data_ is moved into InactiveBatch */
  try {
    guard.SetStage(InsertionGuard::Stage::kInactiveSetCreated);
    guard.inactive_set_modified_ = true;

    auto new_inactive_set =
        vdb::make_shared<vdb::InactiveSet>(internal_schema, active_set_);
    AppendInactiveSet(new_inactive_set);
  } catch (std::exception& e) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) Making Active Set (size=%ld) into Inactive is "
               "Failed: %s",
               table->GetTableName().data(), segment_id_.data(),
               active_set_size, e.what());
    return Status::Unknown("MakeInactive(): Failed to make inactive.");
  }

  /* reset mutable array */
  auto status = BuildColumns(internal_schema, active_set_,
                             table->GetActiveSetSizeLimit());
  if (!status.ok()) {
    return status;
  }

  /* create new vector index */
  if (HasIndex()) {
    status = index_handler_->CreateIndex();
    if (!status.ok()) {
      return status;
    }
  }

  SYSTEM_LOG(
      vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
      "Segment (%s, %s) Making Active Set (size=%ld) into Inactive is Done.",
      table->GetTableName().data(), segment_id_.data(), active_set_size);

  guard.inactive_set_modified_ = false;
  guard.SetStage(InsertionGuard::Stage::kNone);

  SYSTEM_LOG(
      vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
      "Segment (%s, %s) Making Active Set (size=%ld) into Inactive is Done.",
      table->GetTableName().data(), segment_id_.data(), active_set_size);

  return Status::Ok();
}

arrow::Result<std::shared_ptr<arrow::Array>> Segment::FilterRecordIds(
    const std::shared_ptr<arrow::RecordBatch>& rb,
    const std::shared_ptr<expression::Predicate>& predicate,
    [[maybe_unused]] bool is_update) {
  if (!predicate) {
    int64_t num_rows = rb->num_rows();
    arrow::Int64Builder builder;
    auto status = builder.Resize(num_rows);
    if (!status.ok()) {
      return status;
    }
    for (int64_t i = 0; i < num_rows; ++i) {
      builder.UnsafeAppend(i);
    }
    std::shared_ptr<arrow::Array> all_indices;
    status = builder.Finish(&all_indices);
    if (!status.ok()) {
      return status;
    }
    return all_indices;
  }

  ARROW_ASSIGN_OR_RAISE(auto arrow_expr, predicate->BuildArrowExpression());
  ARROW_ASSIGN_OR_RAISE(auto bound_expr, arrow_expr.Bind(*(rb->schema())));
  ARROW_ASSIGN_OR_RAISE(
      auto mask_expr, arrow::compute::ExecuteScalarExpression(
                          bound_expr, arrow::compute::ExecBatch(*(rb.get()))));

  if (!mask_expr.is_array()) {
    return arrow::Status::Invalid("Filter expression is not an array.");
  }
  std::shared_ptr<arrow::Array> mask_array = mask_expr.make_array();

  arrow::Int64Builder builder;
  auto mask_data = std::static_pointer_cast<arrow::BooleanArray>(mask_array);
  for (int64_t i = 0; i < mask_data->length(); ++i) {
    if (mask_data->Value(i)) {
      auto status = builder.Append(i);
      if (!status.ok()) {
        return status;
      }
    }
  }

  std::shared_ptr<arrow::Array> result;
  auto status = builder.Finish(&result);
  if (!status.ok()) {
    return status;
  }
  return result;
}

void Segment::UpdateActiveSetDeletedFlag(std::shared_ptr<arrow::Array> column) {
  auto table = table_.lock();
  std::shared_ptr<BooleanArray> replace_mutable_arr =
      vdb::make_shared<BooleanArray>(table->GetActiveSetSizeLimit());
  replace_mutable_arr->Append(column.get());
  active_set_.back() = replace_mutable_arr;
}

arrow::Result<uint32_t> Segment::DeleteRecords(
    const std::shared_ptr<expression::Predicate>& predicate, bool is_update,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* deleted_records) {
  uint32_t deleted_count = 0;
  ActiveSetRecordBatchRenewer active_rb_renewer(this);
  for (size_t set_index = 0; set_index <= inactive_sets_.size(); ++set_index) {
    std::shared_ptr<arrow::RecordBatch> set = nullptr;
    std::shared_ptr<vdb::InactiveSet> inactive_set = nullptr;

    if (set_index == ActiveSetId()) {
      set = ActiveSetRecordBatch();
      if (set == nullptr) {
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "Segment (%s, %s) Deleting Records is Failed: Active set is "
                   "null.",
                   table_.lock()->GetTableName().data(), segment_id_.data());
        return arrow::Status::Invalid("Active set is null. It's not possible.");
      }
    } else {
      inactive_set = GetInactiveSet(set_index);
      set = inactive_set->GetRb();
    }

    if (set->num_rows() == 0) continue;

    /* Filter records */
    auto filter_result = FilterRecordIds(set, predicate, is_update);
    if (!filter_result.ok()) {
      return filter_result.status();
    }
    auto filtered_indices = filter_result.ValueUnsafe();
    if (filtered_indices->length() == 0) continue;

    /* If update is true, extract deleted records */
    if (is_update && deleted_records) {
      std::shared_ptr<arrow::RecordBatch> deleted_batch;
      if (!predicate) {
        deleted_batch = set;
      } else {
        ARROW_ASSIGN_OR_RAISE(auto datum,
                              arrow::compute::Take(set, filtered_indices));
        deleted_batch = datum.record_batch();
      }

      /* Filter out invisible columns (internal columns)
       * UPDATE command will reinsert these deleted_records after
       * applying modifications. AppendRecords expects user-defined columns
       * only (without internal columns(ex. __deleted_flag)), as internal
       * columns are automatically added during insertion. Including them here
       * would cause schema mismatch or duplication. */
      deleted_batch =
          vdb::RebuildRecordBatchWithoutInternalColumns(deleted_batch);
      deleted_records->push_back(deleted_batch);

      if (vdb::ServerConfiguration::GetMaxMemory() != 0) {
        auto table = table_.lock();
        ARROW_ASSIGN_OR_RAISE(
            auto update_expand_size,
            EstimateExpandedSize(table->GetInternalSchema(),
                                 table->GetIndexInfos(), *deleted_records,
                                 ActiveSetRecordCount(),
                                 table->GetActiveSetSizeLimit()));

        if ((GetRedisAllocatedSize() + update_expand_size) >=
            vdb::ServerConfiguration::GetMaxMemory()) {
          return arrow::Status::OutOfMemory(
              "UpdateCommand is Failed: out of memory.");
        }
      }
    }

    /* Create a new boolean array where only filtered records are set to true */
    arrow::BooleanBuilder boolean_builder;
    auto status = boolean_builder.Resize(set->num_rows());
    if (!status.ok()) {
      return status;
    }

    std::shared_ptr<arrow::Array> existing_flags =
        set->column(set->num_columns() - 1);
    auto existing_flag_data =
        std::static_pointer_cast<arrow::BooleanArray>(existing_flags);

    std::vector<bool> updated_values(set->num_rows(), false);
    for (int64_t i = 0; i < set->num_rows(); ++i) {
      updated_values[i] = existing_flag_data->Value(i);
    }

    for (int64_t i = 0; i < filtered_indices->length(); ++i) {
      uint64_t index =
          std::static_pointer_cast<arrow::Int64Array>(filtered_indices)
              ->Value(i);
      updated_values[index] = true;
    }

    for (bool value : updated_values) {
      boolean_builder.UnsafeAppend(value);
    }

    std::shared_ptr<arrow::Array> true_array;
    status = boolean_builder.Finish(&true_array);
    if (!status.ok()) {
      return status;
    }

    /* Replace the last column of the original RecordBatch with the new boolean
     * array */
    if (set_index == ActiveSetId()) {
      UpdateActiveSetDeletedFlag(true_array);
      if (!is_update) {
        active_rb_renewer.SetNeedRenew();
      }
    } else {
      std::vector<std::shared_ptr<arrow::Array>> new_columns = set->columns();
      new_columns.back() = true_array;

      auto new_batch =
          arrow::RecordBatch::Make(set->schema(), set->num_rows(), new_columns);
      auto new_inactive_set =
          vdb::make_shared<vdb::InactiveSet>(new_batch, *inactive_set, true);

      SwapInactiveSet(set_index, new_inactive_set, nullptr);
    }

    /* If index is enabled, delete node from the index */
    if (HasIndex()) {
      std::vector<uint64_t> labels_vector;
      labels_vector.reserve(filtered_indices->length());
      for (int i = 0; i < filtered_indices->length(); ++i) {
        size_t label = vdb::LabelInfo::Build(
            segment_number_, set_index,
            std::static_pointer_cast<arrow::Int64Array>(filtered_indices)
                ->Value(i));
        labels_vector.push_back(label);
      }
      index_handler_->DeleteEmbeddings(labels_vector, set_index);
    }

    /* If primary key is enabled, delete key from the index */
    auto table = table_.lock();
    auto maybe_primary_key_column_id = table->GetPrimaryKeyColumnId();
    bool has_primary_key_column = maybe_primary_key_column_id.ok();
    if (has_primary_key_column) {
      auto primary_key_column_id = maybe_primary_key_column_id.ValueUnsafe();
      auto schema = set->schema();
      auto composite_key_type =
          schema->field(primary_key_column_id)->type()->id();
      PrimaryKeyHandleForDelete pk_handle(GetPrimaryKeyIndex());
      pk_handle.Initialize(set->column(primary_key_column_id),
                           composite_key_type, filtered_indices);
      auto status = pk_handle.ApplyToIndex();
      if (!status.ok()) {
        return arrow::Status::Invalid(status.ToString());
      }
    }
    deleted_count += filtered_indices->length();
  }

  deleted_size_ += deleted_count;

  return deleted_count;
}

std::shared_ptr<arrow::RecordBatch> Segment::ActiveSetRecordBatch() {
  return active_rb_;
}

void Segment::RenewActiveSetRecordBatch() {
  auto active_set_size = ActiveSetRecordCount();

  auto table = table_.lock();
  auto schema = table->GetInternalSchema();

  std::vector<std::shared_ptr<arrow::Array>> columns;
  for (auto& md : active_set_) {
    columns.push_back(md->ToArrowArray());
  }
  if (active_rb_ == nullptr) {
    SYSTEM_LOG(
        vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
        "Segment (%s, %s) recordbatch of active set is null when renewing "
        "it. It can be possible when creating or loading segment and "
        "when active set has converted to inactive set.",
        table->GetTableName().data(), segment_id_.data());
  }
  active_rb_ = arrow::RecordBatch::Make(schema, active_set_size, columns);
}

vdb::vector<std::shared_ptr<vdb::InactiveSet>>& Segment::InactiveSets() {
  return inactive_sets_;
}

std::shared_ptr<vdb::InactiveSet> Segment::GetInactiveSet(uint16_t set_number) {
  std::lock_guard<std::mutex> lock(set_mutex_);
  return inactive_sets_[set_number];
}

void Segment::AppendInactiveSet(
    std::shared_ptr<InactiveSet>& new_inactive_set) {
  std::lock_guard<std::mutex> lock(set_mutex_);
  inactive_sets_.emplace_back(new_inactive_set);
}

std::shared_ptr<arrow::Schema> Segment::GetSchema() const {
  auto table = table_.lock();
  return table->GetSchema();
}

std::shared_ptr<arrow::Schema> Segment::GetInternalSchema() const {
  auto table = table_.lock();
  return table->GetInternalSchema();
}

std::shared_ptr<arrow::Schema> Segment::GetExtendedSchema() const {
  auto table = table_.lock();
  return table->GetExtendedSchema();
}

bool Segment::SwapInactiveSet(uint16_t set_number,
                              std::shared_ptr<InactiveSet>& new_inactive_set,
                              void* compare_address) {
  std::lock_guard<std::mutex> lock(set_mutex_);
  if (compare_address == nullptr) {
    std::atomic_store(&inactive_sets_[set_number], new_inactive_set);
    return true;
  } else {
    bool success = false;
    std::shared_ptr<InactiveSet>& target = inactive_sets_[set_number];

    /* Since C++20, std::atomic<std::shared_ptr> is supported.
     * If migrating to C++20, the atomic_compare_exchange_strong
     * function will also need to be updated accordingly. */
    if (target.get() == compare_address) {
      auto expected = target;
      success = std::atomic_compare_exchange_strong(&target, &expected,
                                                    new_inactive_set);
    }
    if (!success) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
                 "SwapInactiveSet skipped.(segment %p, set id: %d, "
                 "compare_address %p, inactive set %p)",
                 this, set_number, compare_address, target.get());
    }
    return success;
  }
}

std::shared_ptr<PrimaryKeyIndex> Segment::GetPrimaryKeyIndex() const {
  return primary_key_index_;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Segment::GetRecordbatch(
    uint32_t set_id) {
  if (set_id == ActiveSetId()) {
    auto active_rb = ActiveSetRecordBatch();
    if (active_rb == nullptr) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) GetRecordbatch is Failed: Active set is "
                 "null.",
                 table_.lock()->GetTableName().data(), segment_id_.data());
      return arrow::Status::Invalid("Active set is null. It's not possible.");
    }
    return active_rb;
  } else {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
               "Segment (%s, %s) GetRecordbatch is Failed: Invalid set ID "
               "requested. requested set ID: %d, current active set ID: %d",
               table_.lock()->GetTableName().data(), segment_id_.data(), set_id,
               ActiveSetId());
    return arrow::Status::Invalid(
        "Invalid set ID requested. Requested set ID: " +
        std::to_string(set_id) +
        ", current active set ID is: " + std::to_string(ActiveSetId()));
  }
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Segment::GetRecordbatch(
    uint32_t set_id, const std::vector<int>& indices) {
  ARROW_ASSIGN_OR_RAISE(auto rb, GetRecordbatch(set_id));
  ARROW_ASSIGN_OR_RAISE(auto selected_rb, rb->SelectColumns(indices));
  return selected_rb;
}

std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
          std::vector<std::shared_ptr<InactiveSet>>>
Segment::GetRecordbatches() {
  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  std::vector<std::shared_ptr<InactiveSet>> inactive_sets;
  uint16_t set_index = 0;
  for (auto set : InactiveSets()) {
    if (set->NumRows() != 0) {
      auto rb = set->GetRb();
      if (!rb) {
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "GetRecordbatches: Failed to load InactiveSet (set_index: "
                   "%d, num_rows: %d) - data will be missing!",
                   set_index, set->NumRows());
        set_index++;
        continue;
      }
      rbs.push_back(rb);
      inactive_sets.push_back(set);
    } else {
      SYSTEM_LOG(
          vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
          "GetRecordbatches, Skipping adding inactive set to recordbatches: "
          "Inactive set is empty. (set_index: %d)",
          set_index);
    }
    set_index++;
  }
  auto active_rb = ActiveSetRecordBatch();
  if (active_rb != nullptr) {
    if (active_rb->num_rows() != 0) {
      rbs.emplace_back(active_rb);
    }
  } else {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
               "GetRecordbatches, Skipping adding active set to recordbatches: "
               "Active set is null. It's not possible.");
  }
  return {rbs, inactive_sets};
}

std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
          std::vector<std::shared_ptr<InactiveSet>>>
Segment::GetRecordbatches(const std::vector<int>& column_indices) {
  if (column_indices.empty()) {
    return GetRecordbatches();
  }

  std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
  std::vector<std::shared_ptr<InactiveSet>> inactive_sets;

  for (auto set : InactiveSets()) {
    if (!set || set->NumRows() == 0) continue;

    auto result = set->GetColumns(column_indices);
    if (result.ok()) {
      auto rb = result.ValueUnsafe();
      rbs.push_back(rb);
      inactive_sets.push_back(set);
    }
  }

  auto active_rb = ActiveSetRecordBatch();
  if (active_rb && active_rb->num_rows() != 0) {
    auto select_result = active_rb->SelectColumns(column_indices);
    if (select_result.ok()) {
      auto rb = select_result.ValueUnsafe();
      rbs.emplace_back(rb);
    }
  }

  return {rbs, inactive_sets};
}

arrow::Result<std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
                        std::vector<std::shared_ptr<InactiveSet>>>>
Segment::GetFilteredRecordbatches(
    const std::shared_ptr<expression::Predicate>& predicate) {
  auto [rbs, inactive_sets] = GetRecordbatches();
  if (!predicate) {
    return std::make_pair(std::move(rbs), std::move(inactive_sets));
  }

  ARROW_ASSIGN_OR_RAISE(auto expr, predicate->BuildArrowExpression());
  ARROW_ASSIGN_OR_RAISE(auto bind_expr, expr.Bind(*GetSchema()));

  std::vector<std::shared_ptr<arrow::RecordBatch>> filtered_rbs;
  for (auto& rb : rbs) {
    auto eb = arrow::compute::ExecBatch(*rb);
    ARROW_ASSIGN_OR_RAISE(
        auto filter_expr,
        arrow::compute::ExecuteScalarExpression(bind_expr, eb));
    ARROW_ASSIGN_OR_RAISE(auto filtered_result,
                          arrow::compute::Filter(rb, filter_expr));
    auto filtered_rb = filtered_result.record_batch();
    if (filtered_rb->num_rows() > 0) {
      filtered_rbs.push_back(filtered_rb);
    }
  }

  return std::make_pair(std::move(filtered_rbs),
                        std::vector<std::shared_ptr<InactiveSet>>{});
}

arrow::Result<std::pair<std::vector<std::shared_ptr<arrow::RecordBatch>>,
                        std::vector<std::shared_ptr<InactiveSet>>>>
Segment::GetFilteredRecordbatches(
    const std::shared_ptr<expression::Predicate>& predicate,
    const std::vector<int>& column_indices) {
  if (!predicate) {
    return GetRecordbatches(column_indices);
  }

  std::vector<int> filter_columns;
  auto filter_col_names = predicate->GetReferencedColumns();
  auto schema = GetSchema();
  for (const auto& col_name : filter_col_names) {
    int col_idx = schema->GetFieldIndex(col_name);
    if (col_idx != -1 && std::find(filter_columns.begin(), filter_columns.end(),
                                   col_idx) == filter_columns.end()) {
      filter_columns.push_back(col_idx);
    }
  }

  std::vector<int> combined_columns = column_indices;
  for (int filter_col : filter_columns) {
    if (std::find(combined_columns.begin(), combined_columns.end(),
                  filter_col) == combined_columns.end()) {
      combined_columns.push_back(filter_col);
    }
  }

  auto [rbs, inactive_sets] = GetRecordbatches(combined_columns);

  ARROW_ASSIGN_OR_RAISE(auto expr, predicate->BuildArrowExpression());

  std::vector<std::shared_ptr<arrow::RecordBatch>> filtered_rbs;
  bool need_projection = combined_columns.size() > column_indices.size();

  for (auto& rb : rbs) {
    ARROW_ASSIGN_OR_RAISE(auto bind_expr, expr.Bind(*(rb->schema())));

    auto eb = arrow::compute::ExecBatch(*rb);
    ARROW_ASSIGN_OR_RAISE(
        auto filter_expr,
        arrow::compute::ExecuteScalarExpression(bind_expr, eb));
    ARROW_ASSIGN_OR_RAISE(auto filtered_result,
                          arrow::compute::Filter(rb, filter_expr));

    auto filtered_rb = filtered_result.record_batch();
    if (filtered_rb->num_rows() > 0) {
      if (need_projection) {
        ARROW_ASSIGN_OR_RAISE(auto projected_rb,
                              filtered_rb->SelectColumns(column_indices));
        filtered_rbs.push_back(projected_rb);
      } else {
        filtered_rbs.push_back(filtered_rb);
      }
    }
  }

  return std::make_pair(std::move(filtered_rbs), std::move(inactive_sets));
}

std::pair<std::shared_ptr<arrow::RecordBatch>, std::shared_ptr<InactiveSet>>
Segment::GetRecordbatchWithSet(uint16_t set_id) {
  if (set_id == ActiveSetId()) {
    auto rb = GetRecordbatch(set_id);
    if (!rb.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) Failed to get RecordBatch for set_id %d: %s",
                 table_.lock()->GetTableName().data(), segment_id_.data(),
                 set_id, rb.status().ToString().c_str());
      return {nullptr, nullptr};
    }
    return {rb.ValueUnsafe(), nullptr};
  } else {
    auto inactive_set = GetInactiveSet(set_id);
    if (!inactive_set) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Segment (%s, %s) InactiveSet is null for set_id %d",
                 table_.lock()->GetTableName().data(), segment_id_.data(),
                 set_id);
      return {nullptr, nullptr};
    }

    // GetRb() already supports zero-copy via GetColumns() internally:
    // - If mmap_ipc_region_ exists: uses GetColumns() for zero-copy
    // - If mmap_ipc_file_ exists: uses GetColumns() for zero-copy
    // - If record_batch_file_path_ exists: reopens mmap via GetColumns()
    // - Otherwise: returns cached rb_
    auto rb = inactive_set->GetRb();
    return {rb, inactive_set};
  }
}

arrow::Result<std::pair<std::shared_ptr<arrow::RecordBatch>,
                        std::shared_ptr<InactiveSet>>>
Segment::GetRecordbatchWithSet(uint16_t set_id,
                               const std::vector<int>& column_indices) {
  if (set_id == ActiveSetId()) {
    auto rb = GetRecordbatch(set_id);
    if (!rb.ok()) {
      return rb.status();
    }
    ARROW_ASSIGN_OR_RAISE(auto selected_rb,
                          rb.ValueUnsafe()->SelectColumns(column_indices));
    return std::make_pair(std::move(selected_rb), nullptr);
  } else {
    auto inactive_set = GetInactiveSet(set_id);
    if (!inactive_set) {
      return arrow::Status::Invalid("InactiveSet is null for set_id " +
                                    std::to_string(set_id));
    }

    ARROW_ASSIGN_OR_RAISE(auto rb, inactive_set->GetColumns(column_indices));
    return std::make_pair(std::move(rb), std::move(inactive_set));
  }
}

InactiveSet::InactiveSet(std::shared_ptr<arrow::RecordBatch> rb,
                         std::shared_ptr<arrow::Buffer> buffer)
    : rb_{rb}, buffer_{buffer}, num_rows_{rb ? rb->num_rows() : 0} {}

InactiveSet::InactiveSet(
    std::shared_ptr<arrow::Schema>& schema,
    vdb::vector<std::shared_ptr<vdb::MutableArray>>& arrays)
    : buffer_{nullptr} {
  std::swap(columns_, arrays);
  std::vector<std::shared_ptr<arrow::Array>> arrow_columns;
  size_t size = columns_[0]->Size();
  for (auto& arr : columns_) {
    arrow_columns.emplace_back(arr->ToArrowArray());
    if (size != arr->Size()) {
      throw std::invalid_argument("Unknown error: Column size is different.");
    }
  }

  rb_ = arrow::RecordBatch::Make(schema, columns_[0]->Size(), arrow_columns);
  num_rows_ = rb_ ? rb_->num_rows() : 0;
}

InactiveSet::InactiveSet(std::shared_ptr<arrow::RecordBatch> rb,
                         InactiveSet& source_set, bool copy_members)
    : rb_{rb}, num_rows_{rb ? rb->num_rows() : 0} {
  if (copy_members) {
    columns_ = source_set.columns_;
    buffer_ = source_set.buffer_;
  }
  schema_ = source_set.schema_;
}

std::shared_ptr<arrow::RecordBatch> InactiveSet::GetRb() {
  if (rb_) {
    return rb_;
  }
  return nullptr;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> InactiveSet::GetColumns(
    const std::vector<int>& column_indices) {
  if (column_indices.empty()) {
    return arrow::Status::Invalid("GetColumns: empty column_indices");
  }

  std::lock_guard<std::mutex> lock(load_mutex_);

  // Fast path: data already in memory
  if (rb_) {
    SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogDebug,
               "GetColumns: Fast path - data already in memory");
    // Selective column loading from in-memory RecordBatch
    return rb_->SelectColumns(column_indices);
  } else {
    return arrow::Status::Invalid("GetColumns: data not in memory");
  }
}

std::string InactiveSet::ToString() const {
  if (rb_) {
    return rb_->ToString();
  }
  return "InactiveSet (null)";
}

arrow::Result<std::shared_ptr<vdb::Table>> TableBuilder::Build() {
  std::shared_ptr<vdb::Table> table;
  if (!options_.table_directory_path_.empty()) {
    ARROW_ASSIGN_OR_RAISE(table, BuildTableFromSavedFile());
  } else {
    ARROW_ASSIGN_OR_RAISE(table, BuildTableFromSchema());
  }

  if (table->HasIndex() && vdb::ServerConfiguration::GetAllowBgIndexThread()) {
    table->StartIndexingThread();
  }

  auto status = table->SetActiveSetSizeLimit();
  if (!status.ok()) {
    return arrow::Status::Invalid(status.ToString());
  }

  if (!options_.table_directory_path_.empty()) {
    try {
      auto status = table->LoadSegments(table, options_.table_directory_path_);
      if (!status.ok()) {
        return arrow::Status::Invalid(status.ToString());
      }
    } catch (const std::runtime_error& e) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
                 "Loading Table (%s) is Failed by OOM.",
                 options_.table_name_.data());
      return arrow::Status::OutOfMemory(
          "out of memory: failed to create the table due to insufficient "
          "memory.");
    } catch (const std::invalid_argument& e) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogVerbose,
                 "Loading Table (%s) is Failed: %s",
                 options_.table_name_.data(), e.what());
      return arrow::Status::Invalid(e.what());
    }
  }

  return table;
}

std::map<std::string, std::shared_ptr<vdb::MetadataChecker>>*
GetMetadataCheckers() {
  return static_cast<
      std::map<std::string, std::shared_ptr<vdb::MetadataChecker>>*>(
      vdb::ServerResources::GetMetadataCheckers());
}

arrow::Result<std::shared_ptr<vdb::Table>>
TableBuilder::BuildTableFromSchema() {
  const auto& metadata_checkers = *GetMetadataCheckers();
  for (auto& key : options_.schema_->metadata()->keys()) {
    auto checker = metadata_checkers.find(key);
    if (checker != metadata_checkers.end()) {
      auto status = checker->second->Check(
          options_.schema_->metadata()->Get(key).ValueUnsafe(),
          options_.schema_);
      if (!status.ok()) {
        return status;
      }
    } else {
      return arrow::Status::Invalid("invalid schema: unknown metadata key: " +
                                    key);
    }
  }

  auto segmentation_info_str =
      options_.schema_->metadata()->Get(kSegmentationInfoKey).ValueOr("");
  SegmentationInfoBuilder segmentation_info_builder;
  segmentation_info_builder.SetSegmentationInfo(segmentation_info_str);
  segmentation_info_builder.SetSchema(options_.schema_);
  ARROW_ASSIGN_OR_RAISE(auto segmentation_info,
                        segmentation_info_builder.Build());

  if (segmentation_info.GetSegmentType() == SegmentType::kValue) {
    auto indices = segmentation_info.GetSegmentKeysIndices();
    for (auto index : indices) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
                 "Segment by idx of table column: %u/%lu", index,
                 options_.schema_->fields().size());
      if (index >= options_.schema_->fields().size()) {
        return arrow::Status::Invalid(
            "invalid schema: \"segment_keys\" has invalid column index: " +
            std::to_string(index));
      }
      auto field = options_.schema_->field(index);
      if (field->nullable()) {
        return arrow::Status::Invalid(
            "invalid schema: \"segment_keys\" has nullable column: " +
            field->name());
      }
      if (is_list(field->type()->id())) {
        return arrow::Status::Invalid(
            "invalid schema: \"segment_keys\" has list column: " +
            field->name());
      }
    }
  }

  // Check field information such as types of columns
  for (int i = 0; i < options_.schema_->num_fields(); ++i) {
    auto field = options_.schema_->field(i);
    auto field_name = field->name();
    auto field_type = field->type();

    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Field %d: %s, Type: %s", i, field_name.c_str(),
               field_type->ToString().c_str());

    // Validate: user cannot create columns with internal prefix
    if (vdb::IsInternalColumn(field_name)) {
      return arrow::Status::Invalid(
          "invalid schema: column name '" + field_name +
          "' cannot start with reserved prefix '" +
          std::string(vdb::kInternalColumnPrefix) +
          "'. This prefix is reserved for system internal columns.");
    }

    // check allowed field types
    if (!vdb::IsSupportedType(field_type->id())) {
      return arrow::Status::Invalid(
          "invalid schema: not supported type for column '" + field_name +
          "' with type " + field_type->ToString() + ".");
    }

    // check allowed value types for list columns
    if (is_list(field_type->id())) {
      auto list_type = std::static_pointer_cast<arrow::ListType>(field_type);
      auto value_type = list_type->value_type();
      if (!is_primitive(value_type->id()) && !is_string(value_type->id()) &&
          !is_decimal(value_type->id())) {
        return arrow::Status::Invalid(
            "invalid schema: not supported value type for list column '" +
            field_name + "' with value type " + value_type->ToString() + ".");
      }
    }

    // Add more type-specific checks as needed
    // ...
  }

  // check ann column: should be fixed size list of float
  IndexInfoBuilder index_info_builder;
  index_info_builder.SetIndexInfo(
      options_.schema_->metadata()->Get(kIndexInfoKey).ValueOr(""));
  index_info_builder.SetSchema(options_.schema_);
  ARROW_ASSIGN_OR_RAISE(auto index_infos, index_info_builder.Build());
  if (index_infos->size() > 0) {
    for (auto& index_info : *index_infos) {
      auto field_type =
          options_.schema_->field(index_info.GetColumnId())->type();
      if (index_info.IsDenseVectorIndex()) {
        if (field_type->id() != arrow::Type::FIXED_SIZE_LIST) {
          return arrow::Status::Invalid(
              "invalid schema: ann column is not fixed size list type.");
        }
        auto fixed_size_list_type =
            std::static_pointer_cast<arrow::FixedSizeListType>(field_type);
        auto value_type = fixed_size_list_type->value_type();
        if (value_type->id() != arrow::Type::FLOAT) {
          return arrow::Status::Invalid(
              "invalid schema: ann column does not consist of float values.");
        }
      }
    }
  }

  std::shared_ptr<vdb::Table> table;
  try {
    table = std::make_shared<vdb::Table>(options_.table_name_, options_.schema_,
                                         index_infos);
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Creating Table (%s) is Done.", options_.table_name_.data());
  } catch (const std::runtime_error& e) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Creating Table (%s) is Failed by OOM.",
               options_.table_name_.data());
    return arrow::Status::OutOfMemory(
        "out of memory: failed to create the table due to insufficient "
        "memory.");
  } catch (const std::invalid_argument& e) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Creating Table (%s) is Failed: %s", options_.table_name_.data(),
               e.what());
    return arrow::Status::Invalid(e.what());
  }

  return table;
}

arrow::Result<std::shared_ptr<vdb::Table>>
TableBuilder::BuildTableFromSavedFile() {
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
             "Loading Table (%s) from (%s) is Started.",
             options_.table_name_.data(),
             options_.table_directory_path_.data());

  /* check directory path */
  std::error_code ec;
  if (!std::filesystem::exists(options_.table_directory_path_, ec)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Loading Table (%s) from (%s) is Failed: %s",
               options_.table_name_.data(),
               options_.table_directory_path_.data(), ec.message().data());
    return arrow::Status::Invalid(ec.message());
  }

  /* load manifest */
  std::string manifest_file_path = options_.table_directory_path_ + "/manifest";
  /* existence of manifest shows completeness of saving snapshot */
  if (!std::filesystem::exists(manifest_file_path, ec)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Loading Table (%s) Manifest from (%s) is Failed: %s",
               options_.table_name_.data(), manifest_file_path.data(),
               ec.message().data());
    return arrow::Status::Invalid("saving snapshot of table is not completed");
  }

  std::shared_ptr<vdb::Table> table;
  try {
    table = vdb::make_shared<vdb::Table>();
    auto status = table->LoadManifest(manifest_file_path);
    if (!status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Loading Table (%s) Manifest from (%s) is Failed: %s",
                 options_.table_name_.data(), manifest_file_path.data(),
                 status.ToString().data());
      return arrow::Status::Invalid(status.ToString());
    }

    SYSTEM_LOG(
        vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
        "Loading Table (%s) from (%s) is Done: ", options_.table_name_.data(),
        options_.table_directory_path_.data());
  } catch (const std::runtime_error& e) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Creating Table (%s) is Failed by OOM.",
               options_.table_name_.data());
    return arrow::Status::OutOfMemory(
        "out of memory: failed to create the table due to insufficient "
        "memory.");
  } catch (const std::invalid_argument& e) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogVerbose,
               "Creating Table (%s) is Failed: %s", options_.table_name_.data(),
               e.what());
    return arrow::Status::Invalid(e.what());
  }

  return table;
}

}  // namespace vdb
