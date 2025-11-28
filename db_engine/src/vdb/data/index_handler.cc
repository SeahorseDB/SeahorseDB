#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <thread>
#include <atomic>
#include <cstdint>
#include <string>
#include <memory>

#include <arrow/result.h>

#include "vdb/common/status.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/server_configuration.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/data/embedding_store.hh"
#include "vdb/data/label_info.hh"
#include "vdb/data/table.hh"
#include "vdb/metrics/metrics_api.hh"

namespace vdb {

std::string VectorIndex::EmbeddingToString(const float *embedding,
                                           size_t dimension) {
  std::stringstream ss;
  ss << "[ ";
  if (embedding != nullptr) {
    for (size_t i = 0; i < dimension; i++) {
      ss << embedding[i];
      if (i != dimension - 1) {
        ss << ", ";
      }
    }
  } else {
    ss << "null";
  }
  ss << " ]";
  return ss.str();
}

Hnsw::Hnsw(DistanceSpace space, size_t dim, size_t ef_construction, size_t M,
           size_t max_elem, std::shared_ptr<EmbeddingStore> embedding_store)
    : embedding_store_{embedding_store} {
  index_ = vdb::make_shared<vdb::hnsw::HierarchicalNSW<float>>(
      space, dim, max_elem, M, ef_construction);
}

Hnsw::Hnsw(const std::string &file_path, DistanceSpace space, size_t dim,
           std::shared_ptr<EmbeddingStore> embedding_store)
    : embedding_store_{embedding_store} {
  index_ = vdb::make_shared<vdb::hnsw::HierarchicalNSW<float>>(space, dim,
                                                               file_path);
}

Status Hnsw::AddEmbeddingImpl(const void *embedding, uint64_t label) {
  const float *float_embedding = static_cast<const float *>(embedding);
  try {
    index_->addPoint(float_embedding, label);
    return Status::Ok();
  } catch (std::exception &e) {
    return Status::InvalidArgument(e.what());
  }
}

Status Hnsw::DeleteEmbedding(const uint64_t label) {
  try {
    index_->markDelete(label);
    return Status::Ok();
  } catch (std::exception &e) {
    return Status::InvalidArgument(e.what());
  }
}

std::priority_queue<std::pair<float, uint64_t>> Hnsw::SearchKnnImpl(
    const void *query, const size_t &k, void *filter) {
  const float *float_query = static_cast<const float *>(query);
  vdb::hnsw::BaseFilterFunctor *hnsw_filter =
      static_cast<vdb::hnsw::BaseFilterFunctor *>(filter);

  try {
    metrics::CollectDurationStart(metrics::MetricIndex::HnswSearchKnnLatency);
    std::priority_queue<std::pair<float, vdb::hnsw::labeltype>> result;
    if (vdb::ServerConfiguration::GetEnableHnswMtSingleSearch()) {
      constexpr size_t kEfficientNumThreads = 4;
      result = index_->searchKnn_MT(float_query, k, kEfficientNumThreads,
                                    hnsw_filter);
    } else {
      result = index_->searchKnn(float_query, k, hnsw_filter);
    }
    metrics::CollectDurationEnd(metrics::MetricIndex::HnswSearchKnnLatency);
    metrics::CollectValue(metrics::MetricIndex::HnswSearchKnnSize,
                          result.size());
    return result;
  } catch (std::exception &e) {
    metrics::CollectDurationEnd(metrics::MetricIndex::HnswSearchKnnLatency);
    std::cerr << e.what() << std::endl;
    return {};
  }
}

arrow::Result<std::shared_ptr<arrow::Array>> Hnsw::GetEmbeddingArray(
    const uint64_t *labels, const size_t &count) const {
  auto dim = Dimension();
  auto value_builder = std::make_shared<arrow::FloatBuilder>();
  arrow::FixedSizeListBuilder list_builder(arrow::default_memory_pool(),
                                           value_builder, dim);

  for (size_t i = 0; i < count; i++) {
    auto append_status = list_builder.Append();
    if (!append_status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
                 "GetDenseEmbeddings: Failed to append to list_builder: %s",
                 append_status.ToString().c_str());
      return nullptr;
    }
    auto append_values_status = value_builder->AppendValues(
        GetDenseEmbeddingByLabel(labels[i]).data(), dim);
    if (!append_values_status.ok()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
                 "GetDenseEmbeddings: Failed to append values: %s",
                 append_values_status.ToString().c_str());
      return nullptr;
    }
  }

  std::shared_ptr<arrow::Array> result;
  auto status = list_builder.Finish(&result);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "Failed to make arrow array: %s", status.ToString().c_str());
    return status;
  }

  return result;
}

DISTFUNC_FLOAT32 Hnsw::GetDistFunc() const { return index_->getDistFunc(); }

void *Hnsw::GetDistFuncParam() const { return index_->getDistFuncParam(); }

Status Hnsw::Save(const std::string &file_path) const {
  index_->saveIndex(file_path);
  return Status::Ok();
}

bool Hnsw::IsFull() const {
  return index_->getCurrentElementCount() == index_->getMaxElements();
}

size_t Hnsw::Dimension() const {
  return *(reinterpret_cast<size_t *>(index_->getDistFuncParam()));
}

size_t Hnsw::Size() const { return index_->getCurrentElementCount(); }

size_t Hnsw::CompleteSize() const { return index_->getCompleteElementCount(); }

size_t Hnsw::DeletedSize() const { return index_->getDeletedCount(); }

size_t Hnsw::MaxSize() const { return index_->getMaxElements(); }

std::string Hnsw::ToString(bool show_embeddings, bool show_edges) const {
  std::stringstream ss;
  ss << "offsetLevel0 =" << index_->offsetLevel0_ << std::endl;
  ss << "max_elements =" << index_->max_elements_ << std::endl;
  ss << "cur_element_count =" << index_->cur_element_count << std::endl;
  ss << "size_data_per_element =" << index_->size_data_per_element_
     << std::endl;
  ss << "label_offset =" << index_->label_offset_ << std::endl;
  ss << "offsetData =" << index_->offsetData_ << std::endl;
  ss << "maxlevel =" << index_->maxlevel_ << std::endl;
  ss << "enterpoint_node =" << index_->enterpoint_node_ << std::endl;
  ss << "maxM =" << index_->maxM_ << std::endl;

  ss << "maxM0 =" << index_->maxM0_ << std::endl;
  ss << "M =" << index_->M_ << std::endl;
  ss << "mult =" << index_->mult_ << std::endl;
  ss << "ef_construction =" << index_->ef_construction_ << std::endl;

  if (show_embeddings) {
    ss << "show embeddings (count=" << Size() << ")" << std::endl;
    for (size_t i = 0; i < Size(); i++) {
      if (show_edges) {
        ss << i << "th element level =" << index_->element_levels_[i]
           << std::endl;
      }
      auto embedding = GetRawEmbeddingByInternalId(i);
      if (embedding != nullptr) {
        ss << VectorIndex::EmbeddingToString(embedding, Dimension());
      } else {
        ss << "null";
      }
      if (i != Size() - 1) {
        ss << ", ";
      }
      ss << std::endl;
    }
  } else {
    ss << Size() << " embeddings are not shown. " << std::endl;
  }
  return ss.str();
}

const float *Hnsw::GetRawEmbeddingByInternalId(
    const uint64_t internal_id) const {
  float *point =
      reinterpret_cast<float *>(index_->getDataByInternalId(internal_id));
  return point;
}

std::vector<float> Hnsw::GetEmbeddingByInternalId(
    const uint64_t internal_id) const {
  auto raw_embedding = GetRawEmbeddingByInternalId(internal_id);
  if (raw_embedding == nullptr) {
    auto embedding_store = embedding_store_.lock();
    if (!embedding_store) {
      SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
                 "Failed to get dense embedding by internal id: %ld. Embedding "
                 "store is not valid.",
                 internal_id);

      return std::vector<float>();
    }
    uint64_t label = index_->getExternalLabel(internal_id);
    auto embedding_buffer = embedding_store->ReadToBuffer(&label, 1);
    if (!embedding_buffer.ok()) {
      SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
                 "Failed to get dense embedding by internal id: %ld. Failed to "
                 "read embedding from embedding store.",
                 internal_id);
      return std::vector<float>();
    }

    raw_embedding =
        reinterpret_cast<const float *>(embedding_buffer.ValueUnsafe()->data());
  }

  if (raw_embedding == nullptr) {
    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
               "Failed to get dense embedding by internal id: %ld. Raw "
               "embedding is null.",
               internal_id);
    return std::vector<float>();
  }

  return std::vector<float>(raw_embedding, raw_embedding + Dimension());
}

std::vector<float> Hnsw::GetDenseEmbeddingByLabel(const uint64_t label) const {
  try {
    return index_->getDataByLabel<float>(label);
  } catch (std::exception &e) {
    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
               "Failed to get dense embedding by label: %s. Try to read from "
               "embedding store.",
               e.what());
    auto embedding_store = embedding_store_.lock();
    if (!embedding_store) {
      return std::vector<float>();
    }

    constexpr size_t count = 1;
    auto maybe_embedding_buffer = embedding_store->ReadToBuffer(&label, count);
    if (!maybe_embedding_buffer.ok()) {
      SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
                 "Failed to read embedding from embedding store: %s",
                 maybe_embedding_buffer.status().ToString().c_str());
      return std::vector<float>();
    }

    auto embedding_buffer = maybe_embedding_buffer.ValueUnsafe();
    if (static_cast<size_t>(embedding_buffer->size()) <
        static_cast<size_t>(Dimension() * sizeof(float))) {
      SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
                 "Failed to read embedding from embedding store: buffer size "
                 "(%ld) is less than required size(%ld)",
                 embedding_buffer->size(), Dimension() * sizeof(float));

      return std::vector<float>();
    }
    auto raw_embedding =
        reinterpret_cast<const float *>(embedding_buffer->data());

    if (raw_embedding == nullptr) {
      SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice,
                 "Failed to read embedding from embedding store: raw embedding "
                 "is null");
      return std::vector<float>();
    }

    return std::vector<float>(raw_embedding, raw_embedding + Dimension());
  }
}

VectorIndex::Type GetIndexType(const std::string &type_string) {
  if (TransformToLower(type_string) == "hnsw") {
    return VectorIndex::Type::kHnsw;
  } else {
    return VectorIndex::Type::kIndexTypeMax;
  }
}

IndexHandler::IndexHandler(std::shared_ptr<vdb::Table> table,
                           const uint16_t segment_number)
    : table_{table}, indexes_{} {
  auto embedding_stores = table->GetEmbeddingStores();
  for (auto &[column_id, embedding_store] : embedding_stores) {
    embedding_store->CreateSegmentAndColumnDirectory(segment_number);
  }
  auto status = CreateIndex();
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
  }
}

IndexHandler::IndexHandler(std::shared_ptr<vdb::Table> table,
                           const uint16_t segment_number,
                           std::string &directory_path,
                           const uint64_t index_count)
    : table_{table} {
  auto index_infos = table->GetIndexInfos();
  auto embedding_stores = table->GetEmbeddingStores();
  for (auto &[column_id, embedding_store] : embedding_stores) {
    auto status =
        embedding_store->CreateSegmentAndColumnDirectory(segment_number);
    if (!status.ok()) {
      throw std::invalid_argument(status.ToString());
    }
  }

  for (auto &index_info : *index_infos) {
    /* load vector indexes */
    auto index_type = GetIndexType(index_info.GetIndexType());
    switch (index_type) {
      case VectorIndex::kHnsw: {
        auto status = LoadHnswIndexes(directory_path, index_info, index_count);
        if (!status.ok()) {
          throw std::invalid_argument(status.ToString());
        }
        break;
      }
      case VectorIndex::kIndexTypeMax:
      /* fall through */
      default:
        throw std::invalid_argument("Unknown Index Type");
        break;
    }
  }
}

Status IndexHandler::Save(std::string &directory_path) {
  /* save vector indexes */
  if (!std::filesystem::exists(directory_path)) {
    std::filesystem::create_directory((directory_path));
  }

  for (auto &[column_id, indexes] : indexes_) {
    std::string column_file_path(directory_path);
    column_file_path.append("/column.");
    column_file_path.append(std::to_string(column_id));
    if (!std::filesystem::exists(column_file_path)) {
      std::filesystem::create_directory(column_file_path);
    }

    for (size_t i = 0; i < indexes.size(); i++) {
      std::string index_file_path(column_file_path);
      index_file_path.append("/index.");
      index_file_path.append(std::to_string(i));
      auto index = indexes[i];
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
                 "[INDEX_SAVE] set number: %u, curernt element count: %lu, max "
                 "element count: %lu",
                 i, index->Size(), index->MaxSize());
      auto status = index->Save(index_file_path);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

DistanceSpace GetDistanceSpace(const std::string &metric_string) {
  if (TransformToLower(metric_string) == "ipspace") {
    return DistanceSpace::kIP;
  } else if (TransformToLower(metric_string) == "l2space") {
    return DistanceSpace::kL2;
  } else if (TransformToLower(metric_string) == "cosinespace") {
    return DistanceSpace::kCosine;
  } else if (TransformToLower(metric_string) == "l2spacewithnorm") {
    return DistanceSpace::kL2WithNorm;
  } else {
    return DistanceSpace::kIP;
  }
}

Status IndexHandler::LoadHnswIndex(std::string &index_file_path,
                                   const uint64_t column_id,
                                   const DistanceSpace space,
                                   const int32_t dimension) {
  auto &indexes = indexes_[column_id];
  indexes.emplace_back(vdb::make_shared<Hnsw>(index_file_path, space, dimension,
                                              GetEmbeddingStore(column_id)));
  return Status::Ok();
}

Status IndexHandler::LoadHnswIndex(std::string &file_path,
                                   IndexInfo &index_info) {
  auto table = table_.lock();
  uint64_t column_id = index_info.GetColumnId();
  auto ann_column = table->GetSchema()->field(column_id);
  if (ann_column->type()->id() != arrow::Type::FIXED_SIZE_LIST) {
    return Status::InvalidArgument("ANN column must be a fixed size list");
  }
  ARROW_ASSIGN_OR_RAISE(auto dimension, table->GetAnnDimension(column_id));

  DistanceSpace space = GetDistanceSpace(index_info.GetIndexParam("space"));
  return LoadHnswIndex(file_path, column_id, space, dimension);
}

Status IndexHandler::LoadHnswIndexes(std::string &directory_path,
                                     IndexInfo &index_info,
                                     const uint64_t index_count) {
  auto table = table_.lock();

  uint64_t column_id = index_info.GetColumnId();
  auto ann_column = table->GetSchema()->field(column_id);
  if (ann_column->type()->id() != arrow::Type::FIXED_SIZE_LIST) {
    return Status::InvalidArgument("ANN column must be a fixed size list");
  }
  ARROW_ASSIGN_OR_RAISE(auto dimension, table->GetAnnDimension(column_id));

  DistanceSpace space = GetDistanceSpace(index_info.GetIndexParam("space"));

  for (uint64_t i = 0; i < index_count; i++) {
    std::string index_file_path(directory_path);
    index_file_path.append("/column.");
    index_file_path.append(std::to_string(column_id));
    index_file_path.append("/index.");
    index_file_path.append(std::to_string(i));
    auto status = LoadHnswIndex(index_file_path, column_id, space, dimension);
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "[INDEX_LOAD] set number: %lu, current element count: %lu, max "
               "element count: %lu",
               i, indexes_[column_id][i]->Size(),
               indexes_[column_id][i]->MaxSize());
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status IndexHandler::CreateHnswIndex(IndexInfo &index_info) {
  auto table = table_.lock();
  auto column_id = index_info.GetColumnId();
  auto ann_column = table->GetSchema()->field(column_id);
  if (ann_column->type()->id() != arrow::Type::FIXED_SIZE_LIST) {
    return Status::InvalidArgument("ANN column must be a fixed size list");
  }
  size_t dim =
      std::static_pointer_cast<arrow::FixedSizeListType>(ann_column->type())
          ->list_size();

  DistanceSpace space = GetDistanceSpace(index_info.GetIndexParam("space"));

  ARROW_ASSIGN_OR_RAISE(auto M, vdb::stoui64(index_info.GetIndexParam("M")));
  ARROW_ASSIGN_OR_RAISE(
      auto ef_construction,
      vdb::stoui64(index_info.GetIndexParam("ef_construction")));

  /* index size is same as set size. */
  auto max_elements = table->GetActiveSetSizeLimit();

  auto index =
      vdb::make_shared<Hnsw>(space, dim, ef_construction, M, max_elements,
                             GetEmbeddingStore(column_id));
  auto &indexes = indexes_[column_id];
  indexes.emplace_back(index);
  return Status::Ok();
}

Status IndexHandler::CreateIndex() {
  auto table = table_.lock();
  auto index_infos = table->GetIndexInfos();

  Status status;
  for (auto &index_info : *index_infos) {
    auto index_type = GetIndexType(index_info.GetIndexType());
    switch (index_type) {
      case VectorIndex::kHnsw:
        status = CreateHnswIndex(index_info);
        break;
      case VectorIndex::kIndexTypeMax:
      /* fall through */
      default:
        return Status::InvalidArgument("Unknown Index Type");
    }
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status IndexHandler::AddDenseEmbedding(uint64_t column_id,
                                       const float *embedding, uint64_t label) {
  auto &indexes = indexes_[column_id];
  auto index = indexes.back();
  if (index->IsFull()) {
    return Status::InvalidArgument(
        "index must not be full when inserting embedding. ");
  }

  return index->AddEmbedding(embedding, label);
}

Status IndexHandler::AddDenseEmbeddings(
    uint64_t column_id,
    const std::shared_ptr<arrow::FixedSizeListArray> &embeddings,
    const uint64_t starting_label) {
  auto &indexes = indexes_[column_id];
  auto index = indexes.back();
  auto table = table_.lock();
  auto segment_number = LabelInfo::GetSegmentNumber(starting_label);
  auto set_number = LabelInfo::GetSetNumber(starting_label);
  auto record_number = LabelInfo::GetRecordNumber(starting_label);
  auto dimension = index->Dimension();
  auto raw_embeddings = embeddings->values()->data()->GetValues<float>(1);
  raw_embeddings += embeddings->offset() * dimension;
  for (int64_t i = 0; i < embeddings->length(); i++) {
    auto label =
        LabelInfo::Build(segment_number, set_number, record_number + i);
    auto status =
        AddDenseEmbedding(column_id, raw_embeddings + dimension * i, label);
    if (!status.ok()) return Status::InvalidArgument(status.ToString());
  }

  return Status::Ok();
}

void IndexHandler::DeleteEmbeddings(const std::vector<uint64_t> &labels,
                                    uint64_t column_id, uint64_t index_id) {
  auto &indexes = indexes_[column_id];
  if (index_id >= indexes.size()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogNotice,
               "Invalid index id: %lu", index_id);
    return;
  }
  auto index = indexes[index_id];
  for (auto label : labels) {
    index->DeleteEmbedding(label);
  }
}

void IndexHandler::DeleteEmbeddings(const std::vector<uint64_t> &labels,
                                    uint64_t index_id) {
  auto table = table_.lock();
  auto index_infos = table->GetIndexInfos();

  for (auto &index_info : *index_infos) {
    auto column_id = index_info.GetColumnId();
    auto &indexes = indexes_[column_id];
    if (index_id >= indexes.size()) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogVerbose,
                 "Invalid index id: %lu", index_id);
      continue;
    }
    auto index = indexes[index_id];
    for (auto label : labels) {
      index->DeleteEmbedding(label);
    }
  }
}

std::vector<float> IndexHandler::GetDenseEmbedding(uint64_t column_id,
                                                   uint64_t label) {
  auto set_id = LabelInfo::GetSetNumber(label);
  return GetDenseEmbedding(column_id, set_id, label);
}

std::vector<float> IndexHandler::GetDenseEmbedding(uint64_t column_id,
                                                   uint16_t set_id,
                                                   uint64_t label) {
  auto &indexes = indexes_[column_id];
  if (set_id >= indexes.size()) {
    return std::vector<float>();
  }
  auto index = indexes[set_id];
  return index->GetEmbeddingByLabel<std::vector<float>>(label);
}

arrow::Result<std::shared_ptr<arrow::Array>> IndexHandler::GetEmbeddingArray(
    uint64_t column_id, const uint64_t *labels, const size_t &count) {
  std::unordered_map<uint64_t, std::vector<uint64_t>> label_list_per_set;
  for (size_t i = 0; i < count; i++) {
    auto set_number = LabelInfo::GetSetNumber(labels[i]);
    label_list_per_set[set_number].push_back(labels[i]);
  }

  // TODO: parallelize this when thread pool is ready
  std::vector<std::pair<size_t, std::shared_ptr<arrow::Array>>>
      embeddinglist_per_set;
  auto &indexes = indexes_[column_id];
  for (auto &entry : label_list_per_set) {
    if (entry.first >= indexes.size()) {
      SYSTEM_LOG(
          vdb::LogTopic::Index, vdb::LogLevel::kLogNotice,
          "CRITICAL: Invalid set id detected in GetEmbeddingArray. "
          "set_id=%u, available_sets=%lu, column_id=%lu, "
          "total_labels=%zu, labels_in_invalid_set=%zu, "
          "first_invalid_label=0x%lx, set_from_label=%lu",
          entry.first, indexes.size(), column_id, count, entry.second.size(),
          entry.second.empty() ? 0UL : entry.second[0],
          entry.second.empty() ? 0UL
                               : LabelInfo::GetSetNumber(entry.second[0]));
      return arrow::Status::Invalid(
          "Invalid set id: " + std::to_string(entry.first) +
          ". Available sets: " + std::to_string(indexes.size()));
    }

    auto index = indexes[entry.first];
    ARROW_ASSIGN_OR_RAISE(
        auto embedding_array,
        index->GetEmbeddingArray(entry.second.data(), entry.second.size()));

    embeddinglist_per_set.push_back(
        std::make_pair(entry.first, embedding_array));
  }

  // sort embeddings according to labels
  std::vector<std::shared_ptr<arrow::Array>> embedding_array;
  for (size_t i = 0; i < count; i++) {
    auto set_number = LabelInfo::GetSetNumber(labels[i]);
    auto it = std::find_if(
        embeddinglist_per_set.begin(), embeddinglist_per_set.end(),
        [set_number](const auto &pair) { return pair.first == set_number; });
    auto &embedding = it->second;
    auto first_embedding = embedding->Slice(0, 1);
    it->second = embedding->Slice(1);
    embedding_array.push_back(first_embedding);
  }

  // TODO: avoid COPY, arrow::Concatenate uses copying data internally
  ARROW_ASSIGN_OR_RAISE(auto merged_array, arrow::Concatenate(embedding_array));
  return merged_array;
}

arrow::Result<std::shared_ptr<arrow::Array>> IndexHandler::GetEmbeddingArray(
    uint64_t column_id, uint16_t set_id, const std::vector<uint64_t> &labels) {
  auto &indexes = indexes_[column_id];
  if (set_id >= indexes.size()) {
    return arrow::Status::Invalid("Invalid set id: " + std::to_string(set_id));
  }
  auto index = indexes[set_id];
  ARROW_ASSIGN_OR_RAISE(auto embedding_array,
                        index->GetEmbeddingArray(labels.data(), labels.size()));
  return embedding_array;
}

std::pair<float, uint64_t> PopFarthestFrom(
    std::list<std::priority_queue<std::pair<float, uint64_t>>> &sub_results) {
  float global_farthest = sub_results.begin()->top().first;
  auto farthest_iterator = sub_results.begin();
  for (auto iter = sub_results.begin(); iter != sub_results.end();) {
    if (iter->empty()) {
      iter = sub_results.erase(iter);
      continue;
    }
    auto local_farthest = iter->top().first;
    if (local_farthest > global_farthest) {
      global_farthest = local_farthest;
      farthest_iterator = iter;
    }
    ++iter;
  }

  auto farthest_element = farthest_iterator->top();
  farthest_iterator->pop();
  if (farthest_iterator->empty()) {
    sub_results.erase(farthest_iterator);
  }

  return farthest_element;
}

std::shared_ptr<std::vector<std::pair<float, uint64_t>>>
IndexHandler::SearchKnn(const float *query, const size_t &k,
                        const uint64_t column_id,
                        vdb::hnsw::BaseFilterFunctor *filter) {
  std::list<std::priority_queue<std::pair<float, uint64_t>>> sub_results;
  size_t embedding_count = 0;

  metrics::CollectDurationStart(
      metrics::MetricIndex::IndexHandlerSearchKnnLatency);
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogVerbose,
             "SearchKnn: Start finding %ld nearest neighbors", k);

  /* Collect sub-results from all indexes in parallel using work stealing
   * pattern */
  auto &indexes = indexes_[column_id];
  sub_results.resize(indexes.size());

  // Atomic variable to track the next index to process
  std::atomic<size_t> next_index(0);
  const size_t total_indexes = indexes.size();

  // Define the work function for processing indexes
  auto process_index = [&](void) {
    size_t idx;
    while ((idx = next_index.fetch_add(1)) < total_indexes) {
      auto index = indexes[idx];
      auto iter = sub_results.begin();
      // Move to the list position corresponding to the index
      std::advance(iter, idx);

      metrics::CollectDurationStart(
          metrics::MetricIndex::VectorIndexSearchKnnLatency);

      auto sub_result = index->SearchKnn(query, k, filter);
      *iter = sub_result;

      metrics::CollectDurationEnd(
          metrics::MetricIndex::VectorIndexSearchKnnLatency);

      SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogVerbose,
                 "SearchKnn: %ld embeddings are collected from %ld th index",
                 iter->size(), idx);
    }
  };

  // Determine the number of threads to use (including main thread)
  const int thread_count = vdb::ServerConfiguration::GetIndexThreadNum();
  const int worker_threads = (thread_count > 1) ? thread_count - 1 : 0;

  // Create and start worker threads
  std::vector<std::thread> threads;
  threads.reserve(worker_threads);

  for (int t = 0; t < worker_threads; t++) {
    threads.emplace_back(process_index);
  }

  // Main thread also participates in the work
  process_index();

  // Wait for all worker threads to finish
  for (auto &thread : threads) {
    thread.join();
  }

  // Process results (calculate embedding_count)
  for (auto iter = sub_results.begin(); iter != sub_results.end();) {
    if (iter->empty()) {
      iter = sub_results.erase(iter);
      continue;
    }
    embedding_count += iter->size();
    ++iter;
  }

  /* Retain only the top k elements, and remove the rest */
  if (embedding_count > k) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogVerbose,
               "SearchKnn: retain only top %ld neighbors (%ld -> %ld)", k,
               embedding_count, k);
    while (embedding_count > k) {
      PopFarthestFrom(sub_results);
      embedding_count--;
    }
  }

  auto result = std::make_shared<std::vector<std::pair<float, uint64_t>>>(
      embedding_count);

  /* Merge the sub-results into the main results */
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "SearchKnn: Start print nearest neighbors (%ld embeddings)",
             embedding_count);
  for (int i = embedding_count - 1; i >= 0; i--) {
    (*result)[i] = PopFarthestFrom(sub_results);
    SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
               "SearchKnn: %d th embedding information (dist=%f, label=%s)", i,
               (*result)[i].first,
               LabelInfo::ToString((*result)[i].second).data());
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug,
             "SearchKnn: End print nearest neighbors (%ld embeddings)",
             embedding_count);

  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogVerbose,
             "SearchKnn: Done finding %ld nearest neighbors. %ld embeddings "
             "are found.",
             k, result->size());
  metrics::CollectDurationEnd(
      metrics::MetricIndex::IndexHandlerSearchKnnLatency);
  metrics::CollectValue(metrics::MetricIndex::IndexHandlerSearchKnnSize,
                        result->size());
  return result;
}

std::shared_ptr<VectorIndex> IndexHandler::Index(uint64_t column_id,
                                                 uint64_t index_id) {
  auto &indexes = indexes_[column_id];
  if (index_id >= indexes.size()) {
    return nullptr;
  }
  return indexes[index_id];
}

size_t IndexHandler::Size() const {
  if (indexes_.empty()) return 0;
  return indexes_.begin()->second.size();
}

size_t IndexHandler::Size(uint64_t column_id) {
  auto &indexes = indexes_[column_id];
  return indexes.size();
}

std::vector<float> IndexHandler::GetDistancesFromDenseQuery(
    uint64_t column_id, const float *dense_query,
    const std::vector<uint64_t> &labels) {
  auto &indexes = indexes_[column_id];
  auto index = indexes.back();
  auto dense_index = std::dynamic_pointer_cast<DenseVectorIndex>(index);

  auto distfunc = dense_index->GetDistFunc();
  auto distfunc_param = dense_index->GetDistFuncParam();

  // Use EmbeddingStore::ReadAndCalculateDistances for better efficiency
  auto embedding_store = GetEmbeddingStore(column_id);
  if (!embedding_store) {
    SYSTEM_LOG(vdb::LogTopic::Index, vdb::LogLevel::kLogNotice,
               "Failed to get embedding store");
    return std::vector<float>();
  }

  auto distance_array = embedding_store->ReadAndCalculateDistances(
      labels.data(), labels.size(), dense_query, distfunc, distfunc_param);

  if (!distance_array.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Index, vdb::LogLevel::kLogNotice,
               "Failed to calculate distances: %s",
               distance_array.status().ToString().c_str());
    return std::vector<float>();
  }

  auto array = distance_array.ValueUnsafe();
  std::vector<float> distances;
  distances.reserve(labels.size());

  for (int64_t i = 0; i < array->length(); ++i) {
    distances.push_back(array->Value(i));
  }

  return distances;
}

std::string IndexHandler::ToString(bool show_embeddings,
                                   bool show_edges) const {
  std::stringstream ss;
  ss << "Index Handler " << std::endl;
  ss << "Index Count=" << Size() << std::endl;
  for (auto &[column_id, indexes] : indexes_) {
    ss << "Column ID=" << column_id << std::endl;
    for (size_t i = 0; i < indexes.size(); i++) {
      auto index = indexes[i];
      ss << i << "th Index" << std::endl;
      // Try dense vector index first
      if (auto dense_index =
              std::dynamic_pointer_cast<DenseVectorIndex>(index)) {
        ss << dense_index->ToString(show_embeddings, show_edges) << std::endl;
      } else {
        ss << "Unknown Index Type" << std::endl;
      }
    }
  }
  return ss.str();
}

size_t IndexHandler::CountIndexedElements(uint64_t column_id) {
  size_t total_count = 0;
  auto &indexes = indexes_[column_id];
  for (size_t i = 0; i < indexes.size(); i++) {
    total_count += indexes[i]->Size();
    total_count -= indexes[i]->DeletedSize();
  }
  return total_count;
}

}  // namespace vdb
