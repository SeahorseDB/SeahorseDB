#include "segmentation.hh"

#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>
#include "xxhash.hpp"

#include "vdb/common/defs.hh"
#include "vdb/common/util.hh"

namespace vdb {

// Helper function to convert string to SegmentKeyCompositionType
SegmentKeyCompositionType StringToSegmentKeyCompositionType(
    const std::string& str) {
  if (str == "single") {
    return SegmentKeyCompositionType::kSingle;
  } else if (str == "hierarchical") {
    return SegmentKeyCompositionType::kHierarchical;
  } else if (str == "composite") {
    return SegmentKeyCompositionType::kComposite;
  } else {
    return SegmentKeyCompositionType::kSingle;  // default
  }
}

// Helper function to set common properties for segment generators
void SetCommonProperties(SegmentIdGenerator* generator,
                         const json& segmentation_info) {
  if (segmentation_info.contains("version") &&
      segmentation_info["version"].is_string()) {
    generator->SetVersion(segmentation_info["version"].get<std::string>());
  }

  if (segmentation_info.contains("segment_keys") &&
      segmentation_info["segment_keys"].is_array()) {
    std::vector<std::string> segment_by_column_names;
    for (const auto& item : segmentation_info["segment_keys"]) {
      segment_by_column_names.push_back(item.get<std::string>());
    }
    generator->SetSegmentKeys(segment_by_column_names);
  }

  if (segmentation_info.contains("segment_key_composition_type") &&
      segmentation_info["segment_key_composition_type"].is_string()) {
    std::string comp_type_str =
        segmentation_info["segment_key_composition_type"].get<std::string>();
    SegmentKeyCompositionType comp_type =
        StringToSegmentKeyCompositionType(comp_type_str);
    generator->SetSegmentKeyCompositionType(comp_type);
  }
}

SegmentationInfo::SegmentationInfo()
    : segmentation_info_(),
      segment_type_(SegmentType::kUndefined),
      schema_(),
      segment_keys_(),
      segment_keys_indices_() {}

SegmentationInfo::SegmentationInfo(json&& segmentation_info,
                                   SegmentType segment_type,
                                   std::shared_ptr<arrow::Schema> schema,
                                   std::vector<std::string>&& segment_keys,
                                   std::vector<uint32_t>&& segment_keys_indices)
    : segmentation_info_(std::move(segmentation_info)),
      segment_type_(segment_type),
      schema_(schema),
      segment_keys_(segment_keys),
      segment_keys_indices_(segment_keys_indices),
      segment_id_generator_(
          SegmentIdGeneratorFactory::CreateGenerator(segment_type)) {
  if (segment_type_ == SegmentType::kHash) {
    SetCommonProperties(segment_id_generator_.get(), segmentation_info_);
    if (segmentation_info_.contains("num_buckets") &&
        segmentation_info_["num_buckets"].is_number()) {
      dynamic_cast<HashSegmentIdGenerator*>(segment_id_generator_.get())
          ->SetNumBuckets(segmentation_info_["num_buckets"].get<size_t>());
    }
  } else if (segment_type_ == SegmentType::kValue) {
    SetCommonProperties(segment_id_generator_.get(), segmentation_info_);
  }
}

arrow::Result<std::string> SegmentationInfo::GetSegmentId(
    const SegmentPartsExtractor& extractor) const {
  if (segment_keys_indices_.empty()) {
    return std::string{kDefaultSegmentId};
  }

  ARROW_ASSIGN_OR_RAISE(auto segment_parts,
                        extractor.ExtractParts(segment_keys_indices_));

  if (!segment_id_generator_) {
    return std::string{kDefaultSegmentId};
  }

  return segment_id_generator_->GenerateSegmentId(segment_parts);
}

arrow::Result<SegmentationInfo> SegmentationInfoBuilder::Build() {
  if (!schema_) {
    return arrow::Status::Invalid(
        "Schema is not provided for SegmentationInfo");
  }

  if (segmentation_info_json_str_.empty()) {
    return SegmentationInfo();
  }

  try {
    auto segmentation_info = json::parse(segmentation_info_json_str_);

    const std::string type_str =
        segmentation_info["segment_type"].get<std::string>();
    SegmentType segment_type = [&type_str]() {
      if (type_str == "value") {
        return SegmentType::kValue;
      } else if (type_str == "hash") {
        return SegmentType::kHash;
      } else {
        return SegmentType::kUndefined;
      }
    }();

    if (segment_type == SegmentType::kUndefined) {
      return arrow::Status::Invalid("Invalid segment type: ", type_str);
    }

    const auto segment_keys = segmentation_info["segment_keys"];
    std::vector<std::string> segment_keys_column_names;
    std::vector<uint32_t> segment_keys_indices;
    for (const auto& item : segment_keys) {
      auto column_name = item.get<std::string>();
      auto column_index = schema_->GetFieldIndex(column_name);
      segment_keys_indices.push_back(static_cast<uint32_t>(column_index));
      segment_keys_column_names.push_back(column_name);
    }

    return SegmentationInfo(std::move(segmentation_info), segment_type, schema_,
                            std::move(segment_keys_column_names),
                            std::move(segment_keys_indices));
  } catch (const json::parse_error& e) {
    return arrow::Status::Invalid("Failed to parse segmentation info: ",
                                  e.what());
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Error processing segmentation info: ",
                                  e.what());
  }
}

/**
 * ============================================================================
 * Segment ID Specification
 * ============================================================================
 *
 * OVERVIEW
 * --------
 * Format:
 * {version}::{type}::{columns}::{bucket_or_values}::{num_buckets_or_empty}::{metadata}
 *
 * ============================================================================
 * Common Fields
 * ============================================================================
 * {version}: Specification version (e.g., v3)
 * {type}: Segment type
 *   - Hash Segment: hash
 *   - Value Segment: value
 * {columns}: Column definition
 * {metadata}: Additional metadata
 *
 * ============================================================================
 * Column Definition Format
 * ============================================================================
 * {columns} := {col1}>{col2}>{col3}...  // hierarchy
 *            | {col1}+{col2}+{col3}...  // composite
 *            | {col1}                   // single
 *
 * ============================================================================
 * Type-specific Specifications
 * ============================================================================
 *
 * Single Column:
 * --------------
 * {version}::{type}::single::{column}::{bucket}::{num_buckets}::{metadata}
 *
 * Hierarchy:
 * ----------
 * {version}::{type}::hier::{col1}>{col2}>....::{bucket1}.{bucket2}.....::{num_buckets1}.{num_buckets2}.....::{metadata}
 *
 * Composite:
 * ----------
 * {version}::{type}::comp::{col1}+{col2}+.....::{bucket}::{num_buckets}::{metadata}
 *
 * ============================================================================
 * Hash Segment Examples
 * ============================================================================
 * // Single column
 * v3::hash::single::region::5::12::{metadata}
 *
 * // Hierarchy (3-level bucket space)
 * v3::hash::hier::region>date>category::3.10.25::8.16.32::{metadata}
 *
 * // Composite (1-level bucket space)
 * v3::hash::comp::region+date+category::5::12::{metadata}
 *
 * ============================================================================
 * Value Segment
 * ============================================================================
 * Specification Format:
 * ---------------------
 * {version}::{type}::comp::{col1}+{col2}+.....::{json_values}::::{metadata}
 *
 * Characteristics:
 * ----------------
 * - {type}: Always "value"
 * - Column combination: Only "comp" (composite) is supported
 * - {json_values}: Values encoded as JSON array
 * - {num_buckets}: Empty (represented as ::)
 *
 * Value Segment Examples:
 * -----------------------
 * // Basic example
 * v3::value::comp::region+date+category::["seoul","2024-01-01","electronics"]::{metadata}
 *
 * // Values with special characters
 * v3::value::comp::region+date+category::["seoul:kr","2024-01-01","item[db]:log"]::{metadata}
 *
 * // Various data types
 * v3::value::comp::user_id+amount+status::["user123",1500.50,"active:pending"]::{metadata}
 */
arrow::Result<std::string> ValueSegmentIdGenerator::GenerateSegmentId(
    const std::vector<std::string>& segment_parts) const {
  // Value segment format:
  // {version}::{type}::comp::{col1}+{col2}+.....::{json_values}::::{metadata}

  // For value segments, we need to encode the segment_parts as JSON array
  json json_values = json::array();
  for (const auto& part : segment_parts) {
    json_values.push_back(part);
  }

  std::ostringstream ss;
  ss << version_ << "::"                           // version
     << "value" << "::"                            // type
     << GetSegmentKeyCompositionType() << "::"     // column composition type
     << GenerateColumnNamesForSegmentId() << "::"  // column names
     << json_values.dump() << "::"                 // json_values
     << "::"                                       // empty num_buckets
     << metadata_;                                 // metadata

  return ss.str();
}

std::string ValueSegmentIdGenerator::GenerateColumnNamesForSegmentId() const {
  // For value segments, always use composite format with + separator
  return Join(segment_keys_, "+");
}

arrow::Result<std::string> HashSegmentIdGenerator::GenerateSegmentId(
    const std::vector<std::string>& segment_parts) const {
  // Currently only support single partition
  if (segment_parts.size() != 1) {
    return arrow::Status::Invalid(
        "Multiple columns are not supported for hash segment type");
  }

  std::ostringstream ss;
  ss << version_ << "::"                           // version
     << "hash" << "::"                             // type
     << GetSegmentKeyCompositionType() << "::"     // column composition type
     << GenerateColumnNamesForSegmentId() << "::"  // column names
     << GetBucket(segment_parts) << "::"           // bucket
     << GetNumBuckets() << "::"                    // num_buckets
     << GetMetadata();                             // metadata

  return ss.str();
}

std::string HashSegmentIdGenerator::GenerateColumnNamesForSegmentId() const {
  switch (segment_key_composition_type_) {
    case SegmentKeyCompositionType::kSingle:
      return segment_keys_[0];
    case SegmentKeyCompositionType::kHierarchical:
      return Join(segment_keys_, ">");
    case SegmentKeyCompositionType::kComposite:
      return Join(segment_keys_, "+");
    default:
      return Join(segment_keys_, "^");
  }
}

int32_t HashSegmentIdGenerator::GetBucket(
    const std::vector<std::string>& segment_parts) const {
  auto hash_value =
      xxh::xxhash3<64>(segment_parts[0].c_str(), segment_parts[0].size(), 0);
  return static_cast<int32_t>(hash_value % num_buckets_);
}

size_t HashSegmentIdGenerator::GetNumBuckets() const { return num_buckets_; }

std::ostream& operator<<(std::ostream& os,
                         const SegmentKeyCompositionType& partition_type) {
  switch (partition_type) {
    case SegmentKeyCompositionType::kSingle:
      os << "single";
      break;
    case SegmentKeyCompositionType::kHierarchical:
      os << "hier";
      break;
    case SegmentKeyCompositionType::kComposite:
      os << "comp";
      break;
    default:
      os << "unknown";
      break;
  }
  return os;
}

std::shared_ptr<SegmentIdGenerator> SegmentIdGeneratorFactory::CreateGenerator(
    SegmentType type) {
  switch (type) {
    case SegmentType::kValue:
      return std::make_shared<ValueSegmentIdGenerator>();
    case SegmentType::kHash:
      return std::make_shared<HashSegmentIdGenerator>();
    default:
      return nullptr;
  }
}

arrow::Result<std::vector<std::string>> RecordViewPartsExtractor::ExtractParts(
    const std::vector<uint32_t>& indices) const {
  auto tokens = Split(record_view_, kRS);
  std::vector<std::string> segment_parts;
  segment_parts.reserve(indices.size());
  for (auto index : indices) {
    if (index < tokens.size()) {
      segment_parts.push_back(std::string(tokens[index]));
    } else {
      return arrow::Status::Invalid(
          "Index out of range when generating segment ID from record");
    }
  }
  return segment_parts;
}

arrow::Result<std::vector<std::string>> RecordBatchPartsExtractor::ExtractParts(
    const std::vector<uint32_t>& indices) const {
  if (rbs_.empty()) {
    return arrow::Status::Invalid(
        "Empty record batch for segment id generation");
  }
  auto rb = rbs_[0];

  std::vector<std::string> segment_parts;
  segment_parts.reserve(indices.size());
  for (auto index : indices) {
    ARROW_ASSIGN_OR_RAISE(auto scalar, rb->column(index)->GetScalar(0));
    segment_parts.push_back(scalar->ToString());
  }
  return segment_parts;
}

}  // namespace vdb

// ============================================================================
// Segment ID Parser Implementations
// ============================================================================

namespace vdb {

// ValueSegmentIdParser Implementation
arrow::Result<std::unordered_map<std::string, std::string>>
ValueSegmentIdParser::ParseSegmentId(
    const std::string& segment_id,
    const std::vector<std::string>& column_names) const {
  ARROW_ASSIGN_OR_RAISE(auto json_values, ParseJsonValues(segment_id));

  std::unordered_map<std::string, std::string> result;
  for (size_t i = 0; i < column_names.size() && i < json_values.size(); ++i) {
    result[column_names[i]] = json_values[i];
  }

  return result;
}

arrow::Result<std::string> ValueSegmentIdParser::ExtractField(
    const std::string& segment_id, const std::string& field_name) const {
  if (field_name == "version") {
    auto tokens = Split(segment_id, "::");
    if (tokens.size() >= 1) {
      return std::string(tokens[0]);
    }
  } else if (field_name == "type") {
    auto tokens = Split(segment_id, "::");
    if (tokens.size() >= 2) {
      return std::string(tokens[1]);
    }
  } else if (field_name == "composition_type") {
    auto tokens = Split(segment_id, "::");
    if (tokens.size() >= 3) {
      return std::string(tokens[2]);
    }
  } else if (field_name == "column_names") {
    auto tokens = Split(segment_id, "::");
    if (tokens.size() >= 4) {
      return std::string(tokens[3]);
    }
  } else if (field_name == "values") {
    auto tokens = Split(segment_id, "::");
    if (tokens.size() >= 5) {
      return std::string(tokens[4]);
    }
  } else if (field_name == "metadata") {
    auto tokens = Split(segment_id, "::");
    if (tokens.size() >= 7) {
      return std::string(tokens[6]);
    }
  }

  return arrow::Status::Invalid("Field not found: ", field_name);
}

arrow::Result<bool> ValueSegmentIdParser::ValidateSegmentId(
    const std::string& segment_id) const {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 7) {
    return false;
  }

  // Check type is "value"
  if (tokens.size() >= 2 && std::string(tokens[1]) != "value") {
    return false;
  }

  // Check composition type is "comp"
  if (tokens.size() >= 3 && std::string(tokens[2]) != "comp") {
    return false;
  }

  // Validate JSON values format
  if (tokens.size() >= 5) {
    try {
      auto _ = json::parse(std::string(tokens[4]));
    } catch (const json::parse_error&) {
      return false;
    }
  }

  return true;
}

arrow::Result<std::vector<std::string>> ValueSegmentIdParser::ParseJsonValues(
    const std::string& segment_id) const {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 5) {
    return arrow::Status::Invalid("Invalid value segment ID format");
  }

  try {
    auto json_array = json::parse(std::string(tokens[4]));
    if (!json_array.is_array()) {
      return arrow::Status::Invalid("Values field is not a JSON array");
    }

    std::vector<std::string> values;
    for (const auto& item : json_array) {
      values.push_back(item.get<std::string>());
    }
    return values;
  } catch (const json::parse_error& e) {
    return arrow::Status::Invalid("Failed to parse JSON values: ", e.what());
  }
}

arrow::Result<std::vector<std::string>>
ValueSegmentIdParser::ExtractColumnNames(const std::string& segment_id) const {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 4) {
    return arrow::Status::Invalid("Invalid value segment ID format");
  }

  auto column_tokens = Split(std::string(tokens[3]), "+");
  std::vector<std::string> result;
  for (const auto& token : column_tokens) {
    result.push_back(std::string(token));
  }
  return result;
}

// HashSegmentIdParser Implementation
arrow::Result<std::unordered_map<std::string, std::string>>
HashSegmentIdParser::ParseSegmentId(
    const std::string& segment_id,
    const std::vector<std::string>& column_names) const {
  ARROW_ASSIGN_OR_RAISE(auto bucket, ExtractBucket(segment_id));

  std::unordered_map<std::string, std::string> result;
  // For hash segments, we map the bucket value to the first column
  if (!column_names.empty()) {
    result[column_names[0]] = bucket;
  }

  return result;
}

arrow::Result<std::string> HashSegmentIdParser::ExtractField(
    const std::string& segment_id, const std::string& field_name) const {
  auto tokens = Split(segment_id, "::");

  if (field_name == "version") {
    if (tokens.size() >= 1) {
      return std::string(tokens[0]);
    }
  } else if (field_name == "type") {
    if (tokens.size() >= 2) {
      return std::string(tokens[1]);
    }
  } else if (field_name == "composition_type") {
    if (tokens.size() >= 3) {
      return std::string(tokens[2]);
    }
  } else if (field_name == "column_names") {
    if (tokens.size() >= 4) {
      return std::string(tokens[3]);
    }
  } else if (field_name == "bucket") {
    if (tokens.size() >= 5) {
      return std::string(tokens[4]);
    }
  } else if (field_name == "num_buckets") {
    if (tokens.size() >= 6) {
      return std::string(tokens[5]);
    }
  } else if (field_name == "metadata") {
    if (tokens.size() >= 7) {
      return std::string(tokens[6]);
    }
  }

  return arrow::Status::Invalid("Field not found: ", field_name);
}

arrow::Result<bool> HashSegmentIdParser::ValidateSegmentId(
    const std::string& segment_id) const {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 7) {
    return false;
  }

  // Check type is "hash"
  if (tokens.size() >= 2 && std::string(tokens[1]) != "hash") {
    return false;
  }

  // Check composition type is valid
  if (tokens.size() >= 3) {
    std::string comp_type = std::string(tokens[2]);
    if (comp_type != "single" && comp_type != "hier" && comp_type != "comp") {
      return false;
    }
  }

  // Validate bucket is numeric
  if (tokens.size() >= 5) {
    try {
      std::stoi(std::string(tokens[4]));
    } catch (const std::exception&) {
      return false;
    }
  }

  // Validate num_buckets is numeric
  if (tokens.size() >= 6) {
    try {
      std::stoul(std::string(tokens[5]));
    } catch (const std::exception&) {
      return false;
    }
  }

  return true;
}

arrow::Result<std::string> HashSegmentIdParser::ExtractBucket(
    const std::string& segment_id) const {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 5) {
    return arrow::Status::Invalid("Invalid hash segment ID format");
  }

  return std::string(tokens[4]);
}

arrow::Result<std::vector<std::string>> HashSegmentIdParser::ExtractColumnNames(
    const std::string& segment_id) const {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 4) {
    return arrow::Status::Invalid("Invalid hash segment ID format");
  }

  std::string comp_type = std::string(tokens[2]);
  std::string column_names_str = std::string(tokens[3]);

  if (comp_type == "single") {
    return std::vector<std::string>{column_names_str};
  } else if (comp_type == "hier") {
    auto column_tokens = Split(column_names_str, ">");
    std::vector<std::string> result;
    for (const auto& token : column_tokens) {
      result.push_back(std::string(token));
    }
    return result;
  } else if (comp_type == "comp") {
    auto column_tokens = Split(column_names_str, "+");
    std::vector<std::string> result;
    for (const auto& token : column_tokens) {
      result.push_back(std::string(token));
    }
    return result;
  }

  return arrow::Status::Invalid("Unknown composition type: ", comp_type);
}

// SegmentIdParserFactory Implementation
std::shared_ptr<SegmentIdParser> SegmentIdParserFactory::CreateParser(
    SegmentType type) {
  switch (type) {
    case SegmentType::kValue:
      return std::make_shared<ValueSegmentIdParser>();
    case SegmentType::kHash:
      return std::make_shared<HashSegmentIdParser>();
    default:
      return nullptr;
  }
}

std::shared_ptr<SegmentIdParser>
SegmentIdParserFactory::CreateParserFromSegmentId(
    const std::string& segment_id) {
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 2) {
    return nullptr;
  }

  std::string type = std::string(tokens[1]);
  if (type == "value") {
    return std::make_shared<ValueSegmentIdParser>();
  } else if (type == "hash") {
    return std::make_shared<HashSegmentIdParser>();
  }

  return nullptr;
}

// SegmentIdUtils Implementation
arrow::Result<std::unordered_map<std::string, std::string>>
SegmentIdUtils::ParseSegmentId(const std::string& segment_id,
                               const std::vector<std::string>& column_names,
                               SegmentType segment_type) {
  // Only allow _default_ as old-style segment id
  if (segment_id.find("::") == std::string::npos) {
    if (segment_id == kDefaultSegmentId) {
      return std::unordered_map<std::string, std::string>();
    }
    // Any other old-style segment id is invalid
    return arrow::Status::Invalid("Invalid old-style segment id: ", segment_id);
  }
  // Use new parser for new-style segment IDs
  auto parser = SegmentIdParserFactory::CreateParser(segment_type);
  if (!parser) {
    return arrow::Status::Invalid("Failed to create parser for segment type");
  }
  return parser->ParseSegmentId(segment_id, column_names);
}

arrow::Result<std::vector<std::string>> SegmentIdUtils::ExtractValues(
    const std::string& segment_id, const std::vector<std::string>& column_names,
    SegmentType segment_type) {
  ARROW_ASSIGN_OR_RAISE(auto mapping,
                        ParseSegmentId(segment_id, column_names, segment_type));

  std::vector<std::string> values;
  values.reserve(column_names.size());
  for (const auto& column_name : column_names) {
    auto it = mapping.find(column_name);
    if (it != mapping.end()) {
      values.push_back(it->second);
    } else {
      values.push_back("");
    }
  }

  return values;
}

arrow::Result<bool> SegmentIdUtils::ValidateSegmentId(
    const std::string& segment_id, SegmentType segment_type) {
  // Only allow _default_ as old-style segment id
  if (segment_id.find("::") == std::string::npos) {
    return segment_id == kDefaultSegmentId;
  }
  // Use new parser for new-style segment IDs
  auto parser = SegmentIdParserFactory::CreateParser(segment_type);
  if (!parser) {
    return arrow::Status::Invalid("Failed to create parser for segment type");
  }
  return parser->ValidateSegmentId(segment_id);
}

arrow::Result<SegmentType> SegmentIdUtils::GetSegmentType(
    const std::string& segment_id) {
  // Only allow _default_ as old-style segment id
  if (segment_id.find("::") == std::string::npos) {
    if (segment_id == kDefaultSegmentId) {
      return SegmentType::kUndefined;
    }
    return arrow::Status::Invalid("Invalid old-style segment id: ", segment_id);
  }
  auto tokens = Split(segment_id, "::");
  if (tokens.size() < 2) {
    return arrow::Status::Invalid("Invalid segment ID format");
  }
  std::string type = std::string(tokens[1]);
  if (type == "value") {
    return SegmentType::kValue;
  } else if (type == "hash") {
    return SegmentType::kHash;
  }
  return SegmentType::kUndefined;
}

}  // namespace vdb
