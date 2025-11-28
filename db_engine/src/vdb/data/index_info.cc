#include "vdb/common/util.hh"
#include "vdb/data/metadata.hh"
#include "vdb/data/index_info.hh"
#include "vdb/data/index_handler.hh"

namespace vdb {

static std::string FindKeyIgnoreCase(const json& obj, const std::string& key) {
  for (const auto& [k, v] : obj.items()) {
    std::string lower_k = TransformToLower(k);
    if (lower_k.compare(key) == 0) {
      return k;
    }
  }
  return "";
}

IndexInfo::IndexInfo()
    : index_info_(), schema_(), column_id_(), index_type_(), index_params_() {}

IndexInfo::IndexInfo(json&& index_info, std::shared_ptr<arrow::Schema> schema,
                     uint64_t column_id, std::string index_type,
                     vdb::map<std::string, std::string> index_params)
    : index_info_(std::move(index_info)),
      schema_(schema),
      column_id_(column_id),
      index_type_(std::move(index_type)),
      index_params_(std::move(index_params)) {}

std::string IndexInfo::GetIndexParam(std::string param_name) const {
  auto it = index_params_.find(param_name);
  if (it == index_params_.end()) {
    return "";
  }
  return it->second;
}

arrow::Result<std::shared_ptr<vdb::vector<IndexInfo>>>
IndexInfoBuilder::Build() {
  if (!schema_) {
    return arrow::Status::Invalid("Schema is not provided for IndexInfo");
  }

  if (index_info_json_str_.empty()) {
    return std::make_shared<vdb::vector<IndexInfo>>();
  }

  try {
    auto index_info_array = json::parse(index_info_json_str_);

    if (!index_info_array.is_array()) {
      return arrow::Status::Invalid("Index info must be an array");
    }

    if (index_info_array.empty()) {
      return arrow::Status::Invalid("Index info has empty array");
    }

    auto result = std::make_shared<vdb::vector<IndexInfo>>();
    result->reserve(index_info_array.size());

    for (const auto& index_info_obj : index_info_array) {
      if (!index_info_obj.is_object()) {
        return arrow::Status::Invalid("Each index info must be an object");
      }

      // Check required fields
      if (!index_info_obj.contains(kIndexColumnIdKey)) {
        return arrow::Status::Invalid("Missing column_id in index info");
      }

      if (!index_info_obj.contains(kIndexTypeKey)) {
        return arrow::Status::Invalid("Missing index_type in index info");
      }

      arrow::Status status;
      // Parse column_id
      const auto& column_id_str = index_info_obj[kIndexColumnIdKey];
      if (!column_id_str.is_string()) {
        return arrow::Status::Invalid("column_id must be a string");
      }
      ARROW_ASSIGN_OR_RAISE(auto column_id,
                            vdb::stoui64(column_id_str.get<std::string>()));
      if (column_id >= static_cast<uint64_t>(schema_->num_fields())) {
        return arrow::Status::Invalid(
            "column_id must be less than the number of fields in the schema: ",
            column_id, " >= ", schema_->num_fields());
      }

      // Parse index_type
      const auto& index_type_str = index_info_obj[kIndexTypeKey];
      if (!index_type_str.is_string()) {
        return arrow::Status::Invalid("index_type must be a string");
      }
      auto index_type_enum = GetIndexType(index_type_str.get<std::string>());
      if (index_type_enum == VectorIndex::Type::kIndexTypeMax) {
        return arrow::Status::Invalid("Invalid index type: ",
                                      index_type_str.get<std::string>());
      }
      std::string index_type = index_type_str.get<std::string>();

      // Parse parameters
      vdb::map<std::string, std::string> index_params;
      switch (index_type_enum) {
        case VectorIndex::Type::kHnsw: {
          auto params_str = FindKeyIgnoreCase(index_info_obj, "parameters");
          if (!params_str.empty() && index_info_obj[params_str].is_object()) {
            const auto& params = index_info_obj[params_str];
            auto index_type_enum = GetIndexType(index_type);
            for (const auto& [key, value] : params.items()) {
              if (value.is_string()) {
                // check index parameters
                switch (index_type_enum) {
                  case VectorIndex::Type::kHnsw:
                    if (key.compare(kIndexSpaceKey) == 0) {
                      auto index_space =
                          TransformToLower(value.get<std::string>());
                      if (index_space != "ipspace" &&
                          index_space != "l2space") {
                        return arrow::Status::Invalid("Invalid index space: ",
                                                      index_space);
                      }
                    } else if (key.compare(kMKey) == 0) {
                      ARROW_ASSIGN_OR_RAISE(
                          [[maybe_unused]] auto m,
                          vdb::stoui64(value.get<std::string>()));
                    } else if (key.compare(kEfConstructionKey) == 0) {
                      ARROW_ASSIGN_OR_RAISE(
                          [[maybe_unused]] auto ef_construction,
                          vdb::stoui64(value.get<std::string>()));
                    } else {
                      return arrow::Status::Invalid("Invalid index parameter: ",
                                                    key);
                    }
                    break;
                  case VectorIndex::Type::kIndexTypeMax:
                  /* fall through */
                  default:
                    break;
                }
                index_params[key] = value.get<std::string>();
              }
            }
          } else {
            return arrow::Status::Invalid("Missing parameters in index info");
          }
          break;
        }
        case VectorIndex::Type::kIndexTypeMax:
          return arrow::Status::Invalid("Invalid index type: kIndexTypeMax");
      }

      // Create IndexInfo object
      json index_info_copy = index_info_obj;
      IndexInfo index_info(std::move(index_info_copy), schema_, column_id,
                           index_type, std::move(index_params));

      // Validate that the parsed values match
      if (index_info.GetColumnId() != column_id) {
        return arrow::Status::Invalid("Column ID mismatch in IndexInfo");
      }

      if (index_info.GetIndexType() != index_type) {
        return arrow::Status::Invalid("Index type mismatch in IndexInfo");
      }

      result->push_back(std::move(index_info));
    }

    /* duplicate column_id check */
    for (size_t i = 0; i < result->size(); i++) {
      for (size_t j = i + 1; j < result->size(); j++) {
        if (result->at(i).GetColumnId() == result->at(j).GetColumnId()) {
          return arrow::Status::Invalid("Duplicate column_id in index info");
        }
      }
    }

    return result;
  } catch (const json::parse_error& e) {
    return arrow::Status::Invalid("Failed to parse index info: ", e.what());
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Error processing index info: ", e.what());
  }
}

}  // namespace vdb
