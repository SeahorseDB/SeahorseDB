#pragma once

#include <nlohmann/json.hpp>
#include <arrow/api.h>

#include "vdb/common/memory_allocator.hh"
#include "vdb/common/util.hh"

namespace vdb {

constexpr const char* kIndexColumnIdKey = "column_id";
constexpr const char* kIndexTypeKey = "index_type";
constexpr const char* kIndexParamsKey = "parameters";
constexpr const char* kIndexSpaceKey = "space";
constexpr const char* kEfConstructionKey = "ef_construction";
constexpr const char* kMKey = "M";

using json = nlohmann::json;

class IndexInfo {
 public:
  IndexInfo();

  IndexInfo(json&& index_info, std::shared_ptr<arrow::Schema> schema,
            uint64_t column_id, std::string index_type,
            vdb::map<std::string, std::string> index_params);

  uint64_t GetColumnId() const { return column_id_; }
  std::string GetIndexType() const { return index_type_; }
  std::string GetIndexParam(std::string param_name) const;
  bool IsDenseVectorIndex() const {
    return TransformToLower(index_type_) == "hnsw";
  }

 private:
  json index_info_;
  std::shared_ptr<arrow::Schema> schema_;
  uint64_t column_id_;
  std::string index_type_;
  vdb::map<std::string, std::string> index_params_;
};

class IndexInfoBuilder {
 public:
  IndexInfoBuilder() = default;

  IndexInfoBuilder& SetIndexInfo(const std::string& index_info_json_str) {
    index_info_json_str_ = index_info_json_str;
    return *this;
  }

  IndexInfoBuilder& SetSchema(std::shared_ptr<arrow::Schema> schema) {
    schema_ = schema;
    return *this;
  }

  arrow::Result<std::shared_ptr<vdb::vector<IndexInfo>>> Build();

 private:
  std::string index_info_json_str_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace vdb