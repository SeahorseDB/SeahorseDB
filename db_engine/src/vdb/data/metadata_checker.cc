#include <string>
#include <arrow/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <nlohmann/json.hpp>

#include "vdb/common/json_validator.hh"
#include "vdb/common/util.hh"
#include "vdb/data/metadata.hh"

namespace vdb {

using json = nlohmann::json;

arrow::Status SegmentationInfoChecker::Check(
    const std::string& value, std::shared_ptr<arrow::Schema> schema) {
  try {
    auto segmentation_info = json::parse(value);
    JsonValidator validator(segmentation_info, "for \"segmentation_info\"");

    auto type_validator = [](const std::string& type) {
      return type == "value" || type == "hash";
    };
    ARROW_RETURN_NOT_OK(validator.RequireField("segment_type")
                            .ValidateString("segment_type", type_validator)
                            .GetStatus());

    auto key_validator = [schema](const std::string& column_name) {
      return schema->GetFieldIndex(column_name) != -1;
    };
    ARROW_RETURN_NOT_OK(validator.RequireField("segment_keys")
                            .ValidateStringArray("segment_keys", key_validator)
                            .GetStatus());

    auto hash_segmentation_validator = [](JsonValidator& v) -> JsonValidator& {
      auto num_buckets_validator = [](int32_t num) { return num > 0; };
      v.RequireField("num_buckets")
          .ValidateNumber<int32_t>("num_buckets", num_buckets_validator);

      auto key_composition_type_validator = [](const std::string& type) {
        // TODO: Support hierarchical, composite
        return type == "single";
      };
      v.RequireField("segment_key_composition_type")
          .ValidateString("segment_key_composition_type",
                          key_composition_type_validator);

      v.ValidateExactArraySize("segment_keys", 1);

      return v;
    };
    ARROW_RETURN_NOT_OK(
        validator
            .ValidateIf("segment_type", "hash", hash_segmentation_validator)
            .GetStatus());

    auto value_segmentation_validator = [](JsonValidator& v) -> JsonValidator& {
      auto key_composition_type_validator = [](const std::string& type) {
        return type == "single" || type == "composite";
      };
      v.RequireField("segment_key_composition_type")
          .ValidateString("segment_key_composition_type",
                          key_composition_type_validator);

      auto single_segment_validator = [](JsonValidator& v) -> JsonValidator& {
        v.RequireField("segment_keys")
            .ValidateExactArraySize("segment_keys", 1);
        return v;
      };
      v.ValidateIf("segment_keys", "single", single_segment_validator);

      auto multi_segment_validator = [](JsonValidator& v) -> JsonValidator& {
        v.RequireField("segment_keys")
            .ValidateArraySize("segment_keys", 2,
                               std::numeric_limits<size_t>::max());
        return v;
      };
      v.ValidateIf("segment_keys", "composite", multi_segment_validator);
      return v;
    };
    ARROW_RETURN_NOT_OK(
        validator
            .ValidateIf("segment_type", "value", value_segmentation_validator)
            .GetStatus());

    return arrow::Status::OK();
  } catch (const json::parse_error& e) {
    return arrow::Status::Invalid("Failed to parse segmentation info: ",
                                  e.what());
  } catch (const std::exception& e) {
    return arrow::Status::Invalid("Error processing segmentation info: ",
                                  e.what());
  }
}

arrow::Status TableNameChecker::Check(
    [[maybe_unused]] const std::string& value,
    [[maybe_unused]] std::shared_ptr<arrow::Schema> schema) {
  return arrow::Status::OK();
}

arrow::Status ActiveSetSizeLimitChecker::Check(
    const std::string& value,
    [[maybe_unused]] std::shared_ptr<arrow::Schema> schema) {
  ARROW_ASSIGN_OR_RAISE(auto active_set_size_limit, vdb::stoi32(value));
  if (active_set_size_limit <= 0) {
    return arrow::Status::Invalid(
        "active_set_size_limit must be greater than 0");
  }
  return arrow::Status::OK();
}

arrow::Status IndexInfoChecker::Check(
    [[maybe_unused]] const std::string& value,
    [[maybe_unused]] std::shared_ptr<arrow::Schema> schema) {
  return arrow::Status::OK();
}

arrow::Status MaxThreadsChecker::Check(
    const std::string& value,
    [[maybe_unused]] std::shared_ptr<arrow::Schema> schema) {
  ARROW_ASSIGN_OR_RAISE(auto max_threads, vdb::stoi32(value));
  if (max_threads <= 0) {
    return arrow::Status::Invalid("max_threads must be greater than 0");
  }
  return arrow::Status::OK();
}

}  // namespace vdb
