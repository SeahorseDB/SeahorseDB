#pragma once

#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace vdb {

class Segment;
struct Status;
class SegmentIdGenerator;
class SegmentPartsExtractor;

using json = nlohmann::json;

constexpr const char* kDefaultSegmentId = "_default_";

enum class SegmentType {
  kUndefined,
  kValue,
  kHash,
};

enum class SegmentKeyCompositionType {
  kSingle,
  kHierarchical,
  kComposite,
};

class SegmentationInfo {
 public:
  SegmentationInfo();

  SegmentationInfo(json&& segmentation_info, SegmentType segment_type,
                   std::shared_ptr<arrow::Schema> schema,
                   std::vector<std::string>&& segment_keys,
                   std::vector<uint32_t>&& segment_keys_indices);

  SegmentType GetSegmentType() const { return segment_type_; }

  arrow::Result<std::string> GetSegmentId(
      const SegmentPartsExtractor& extractor) const;

  const std::vector<uint32_t>& GetSegmentKeysIndices() const {
    return segment_keys_indices_;
  }

  const std::vector<std::string>& GetSegmentKeys() const {
    return segment_keys_;
  }

 private:
  json segmentation_info_;
  SegmentType segment_type_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::string> segment_keys_;
  std::vector<uint32_t> segment_keys_indices_;
  std::shared_ptr<SegmentIdGenerator> segment_id_generator_;
};

class SegmentationInfoBuilder {
 public:
  SegmentationInfoBuilder() = default;

  SegmentationInfoBuilder& SetSegmentationInfo(
      const std::string& segmentation_info_json_str) {
    segmentation_info_json_str_ = segmentation_info_json_str;
    return *this;
  }

  SegmentationInfoBuilder& SetSchema(std::shared_ptr<arrow::Schema> schema) {
    schema_ = schema;
    return *this;
  }

  arrow::Result<SegmentationInfo> Build();

 private:
  std::string segmentation_info_json_str_;
  std::shared_ptr<arrow::Schema> schema_;
};

class SegmentIdGenerator {
 public:
  virtual ~SegmentIdGenerator() = default;
  virtual arrow::Result<std::string> GenerateSegmentId(
      const std::vector<std::string>& segment_parts) const = 0;

  // Common setters and getters
  void SetVersion(const std::string& version) { version_ = version; }
  void SetSegmentKeys(const std::vector<std::string>& segment_keys) {
    segment_keys_ = segment_keys;
  }
  void SetSegmentKeyCompositionType(
      SegmentKeyCompositionType segment_key_composition_type) {
    segment_key_composition_type_ = segment_key_composition_type;
  }
  SegmentKeyCompositionType GetSegmentKeyCompositionType() const {
    return segment_key_composition_type_;
  }
  const std::string& GetVersion() const { return version_; }
  const std::string& GetMetadata() const { return metadata_; }

 protected:
  // Common members
  std::string version_;
  std::vector<std::string> segment_keys_;
  SegmentKeyCompositionType segment_key_composition_type_;
  std::string metadata_;
};

class ValueSegmentIdGenerator : public SegmentIdGenerator {
 public:
  ValueSegmentIdGenerator() {
    version_ = "v1";
    segment_key_composition_type_ = SegmentKeyCompositionType::kSingle;
    metadata_ = "";
  }

  arrow::Result<std::string> GenerateSegmentId(
      const std::vector<std::string>& segment_parts) const override;

  std::string GenerateColumnNamesForSegmentId() const;
};

class HashSegmentIdGenerator : public SegmentIdGenerator {
 public:
  HashSegmentIdGenerator() : num_buckets_(10) {
    version_ = "v1";
    segment_key_composition_type_ = SegmentKeyCompositionType::kSingle;
    metadata_ = "";
  }

  arrow::Result<std::string> GenerateSegmentId(
      const std::vector<std::string>& segment_parts) const override;

  void SetNumBuckets(size_t num_buckets) { num_buckets_ = num_buckets; }

  int32_t GetBucket(const std::vector<std::string>& segment_parts) const;

  size_t GetNumBuckets() const;

  std::string GenerateColumnNamesForSegmentId() const;

 private:
  size_t num_buckets_;
};

std::ostream& operator<<(std::ostream& os,
                         const SegmentKeyCompositionType& composition_type);

class SegmentIdGeneratorFactory {
 public:
  static std::shared_ptr<SegmentIdGenerator> CreateGenerator(SegmentType type);
};

class SegmentPartsExtractor {
 public:
  virtual ~SegmentPartsExtractor() = default;
  virtual arrow::Result<std::vector<std::string>> ExtractParts(
      const std::vector<uint32_t>& indices) const = 0;
};

class RecordViewPartsExtractor : public SegmentPartsExtractor {
 public:
  explicit RecordViewPartsExtractor(std::string_view record_view)
      : record_view_(record_view) {}

  arrow::Result<std::vector<std::string>> ExtractParts(
      const std::vector<uint32_t>& indices) const override;

 private:
  std::string_view record_view_;
};

class RecordBatchPartsExtractor : public SegmentPartsExtractor {
 public:
  explicit RecordBatchPartsExtractor(
      std::vector<std::shared_ptr<arrow::RecordBatch>>& rbs)
      : rbs_(rbs) {}

  arrow::Result<std::vector<std::string>> ExtractParts(
      const std::vector<uint32_t>& indices) const override;

 private:
  std::vector<std::shared_ptr<arrow::RecordBatch>>& rbs_;
};

class SegmentIdParser {
 public:
  virtual ~SegmentIdParser() = default;

  // Parse segment ID and extract values based on the specification format
  virtual arrow::Result<std::unordered_map<std::string, std::string>>
  ParseSegmentId(const std::string& segment_id,
                 const std::vector<std::string>& column_names) const = 0;

  // Extract specific field from segment ID
  virtual arrow::Result<std::string> ExtractField(
      const std::string& segment_id, const std::string& field_name) const = 0;

  // Validate if segment ID follows the specification format
  virtual arrow::Result<bool> ValidateSegmentId(
      const std::string& segment_id) const = 0;
};

class ValueSegmentIdParser : public SegmentIdParser {
 public:
  arrow::Result<std::unordered_map<std::string, std::string>> ParseSegmentId(
      const std::string& segment_id,
      const std::vector<std::string>& column_names) const override;

  arrow::Result<std::string> ExtractField(
      const std::string& segment_id,
      const std::string& field_name) const override;

  arrow::Result<bool> ValidateSegmentId(
      const std::string& segment_id) const override;

 private:
  // Helper method to parse JSON values from segment ID
  arrow::Result<std::vector<std::string>> ParseJsonValues(
      const std::string& segment_id) const;

  // Helper method to extract column names from segment ID
  arrow::Result<std::vector<std::string>> ExtractColumnNames(
      const std::string& segment_id) const;
};

class HashSegmentIdParser : public SegmentIdParser {
 public:
  arrow::Result<std::unordered_map<std::string, std::string>> ParseSegmentId(
      const std::string& segment_id,
      const std::vector<std::string>& column_names) const override;

  arrow::Result<std::string> ExtractField(
      const std::string& segment_id,
      const std::string& field_name) const override;

  arrow::Result<bool> ValidateSegmentId(
      const std::string& segment_id) const override;

 private:
  // Helper method to extract bucket information
  arrow::Result<std::string> ExtractBucket(const std::string& segment_id) const;

  // Helper method to extract column names from segment ID
  arrow::Result<std::vector<std::string>> ExtractColumnNames(
      const std::string& segment_id) const;
};

class SegmentIdParserFactory {
 public:
  static std::shared_ptr<SegmentIdParser> CreateParser(SegmentType type);

  // Utility method to create parser based on segment ID format
  static std::shared_ptr<SegmentIdParser> CreateParserFromSegmentId(
      const std::string& segment_id);
};

// Utility functions for segment ID parsing
namespace SegmentIdUtils {
// Parse segment ID and return column name to value mapping
arrow::Result<std::unordered_map<std::string, std::string>> ParseSegmentId(
    const std::string& segment_id, const std::vector<std::string>& column_names,
    SegmentType segment_type);

// Extract values from segment ID for specific columns
arrow::Result<std::vector<std::string>> ExtractValues(
    const std::string& segment_id, const std::vector<std::string>& column_names,
    SegmentType segment_type);

// Validate segment ID format
arrow::Result<bool> ValidateSegmentId(const std::string& segment_id,
                                      SegmentType segment_type);

// Get segment type from segment ID
arrow::Result<SegmentType> GetSegmentType(const std::string& segment_id);
}  // namespace SegmentIdUtils

}  // namespace vdb
