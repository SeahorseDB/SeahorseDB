#pragma once

#include <map>
#include <memory>

#include "vdb/common/status.hh"
#include "vdb/data/primary_key.hh"

namespace vdb {

class NumberKeySetContext;
struct PkScanResult;

enum class ContextType : uint8_t {
  kInsert = 0,
  kDelete = 1,
  kNoNeed = 2,
};

/* pure virtual class */
class NumberKeySet {
 public:
  NumberKeySet() : file_name_(""), size_(0) {}
  void SetFileName(std::string_view file_name) { file_name_ = file_name; }
  std::string_view GetFileName() const { return file_name_; }

  /* for save/load */
  NumberKeySet(uint8_t* buffer, uint64_t& buffer_offset);
  virtual Status SerializeInto(std::vector<uint8_t>& buffer,
                               uint64_t& buffer_offset) const = 0;

  /* Batch Unique Scan */
  virtual arrow::Result<PkScanResult> UniqueScan(
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
      ContextType context_type = ContextType::kNoNeed) = 0;
  /* Batch Insert */
  virtual Status Insert(const std::vector<std::pair<uint32_t, uint32_t>>&
                            number_offset_pairs) = 0;
  virtual Status InsertWithContext(
      std::shared_ptr<NumberKeySetContext> number_key_set_context) = 0;
  /* Batch Delete */
  virtual Status Delete(const std::vector<std::pair<uint32_t, uint32_t>>&
                            number_offset_pairs) = 0;
  virtual Status DeleteWithContext(
      std::shared_ptr<NumberKeySetContext> number_key_set_context) = 0;

  virtual uint32_t GetMaxNumber() const = 0;
  virtual uint32_t GetMinNumber() const = 0;

  virtual std::string ToString() const = 0;

 protected:
  std::string_view file_name_;
  uint64_t size_;
};

#if 0 
class NumberKeyBtree : public NumberKeySet {

};
#endif
/* TODO Flat -> Ranged */
class NumberKeyFlatArray : public NumberKeySet {
 public:
  NumberKeyFlatArray() : NumberKeySet() {}

  /* for save/load */
  NumberKeyFlatArray(uint8_t* buffer, uint64_t& buffer_offset);

  virtual ~NumberKeyFlatArray() = default;

  Status SerializeInto(std::vector<uint8_t>& buffer,
                       uint64_t& buffer_offset) const override;

  void Initialize(const uint32_t min_number, const uint32_t max_number);

  /* Batch Unique Scan */
  arrow::Result<PkScanResult> UniqueScan(
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
      ContextType context_type) override;

  size_t BinarySearch(
      const uint32_t number, const size_t start = 0,
      const size_t end = std::numeric_limits<size_t>::max()) const;

  bool IsFound(const uint32_t number, const size_t position) const;
  bool IsRangeStart(const size_t position) const;
  bool IsRangeEnd(const size_t position) const;

  /* Batch Insert */
  Status Insert(const std::vector<std::pair<uint32_t, uint32_t>>&
                    number_offset_pairs) override;
  Status InsertWithContext(
      std::shared_ptr<NumberKeySetContext> number_key_set_context) override;
  void Insert(uint32_t number, size_t position, int32_t& adjust);

  /* Batch Delete */
  Status Delete(const std::vector<std::pair<uint32_t, uint32_t>>&
                    number_offset_pairs) override;
  Status DeleteWithContext(
      std::shared_ptr<NumberKeySetContext> number_key_set_context) override;
  void Delete(uint32_t number, size_t position, int32_t& adjust);

  size_t GetRangeAt(size_t position,
                    std::pair<uint32_t, uint32_t>& range) const;

  uint32_t GetMaxNumber() const override { return number_array_.back(); }
  uint32_t GetMinNumber() const override { return number_array_.front(); }

  std::string ToString() const override;

 private:
  std::vector<bool> is_range_end_bitmap_;
  std::vector<uint32_t> number_array_;
};

/* pure virtual class */
class NumberKeySetContext {
 public:
  NumberKeySetContext(ContextType context_type)
      : numbers_(), context_type_(context_type) {}

  virtual ~NumberKeySetContext() = default;

  void PushNumberKey(const uint32_t number);
  std::vector<uint32_t>& GetNumbers() { return numbers_; }

  virtual std::string ToString() const = 0;

 protected:
  std::vector<uint32_t> numbers_;
  ContextType context_type_;
};

#if 0
class NumberKeyBtreeContext : public NumberKeySetContext {

};
#endif
class NumberKeyFlatArrayContext : public NumberKeySetContext {
 public:
  NumberKeyFlatArrayContext(ContextType context_type)
      : NumberKeySetContext(context_type) {}

  void PushNumberKey(const uint32_t number, const size_t position);
  std::vector<size_t>& GetPositions() { return positions_; }

  std::string ToString() const override;

 private:
  std::vector<size_t> positions_;
};

struct PkScanResult {
  static inline PkScanResult CreateEmpty() {
    PkScanResult scan_result;
    scan_result.number_key_set_context = nullptr;
    return scan_result;
  }
  static std::string PkScanResultToString(const PkScanResult& scan_result);

  std::vector<uint32_t> offsets;
  std::shared_ptr<NumberKeySetContext> number_key_set_context = nullptr;
};

class FileKeyMap {
 public:
  FileKeyMap();
  /* save/load */
  FileKeyMap(const std::string& file_path);
  Status Save(const std::string& file_path) const;

  /* Batch Unique Scan */
  arrow::Result<PkScanResult> UniqueScan(
      const std::string& file_name,
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
      ContextType context_type = ContextType::kNoNeed);
  /* Batch Insert */
  Status Insert(
      const std::string& file_name,
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs);
  Status InsertWithContext(
      const std::string& file_name,
      std::shared_ptr<NumberKeySetContext> number_key_set_context);
  /* Batch Delete */
  Status Delete(
      const std::string& file_name,
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs);
  Status DeleteWithContext(
      const std::string& file_name,
      std::shared_ptr<NumberKeySetContext> number_key_set_context);

  std::string ToString() const;

 private:
  vdb::map<std::string, std::shared_ptr<NumberKeySet>> number_key_sets_;
};

class PrimaryKeyIndex {
 public:
  PrimaryKeyIndex();

  /* save/load */
  explicit PrimaryKeyIndex(const std::string& file_path);
  Status Save(const std::string& file_path) const;

  /* Batch Unique Scan */
  arrow::Result<PkScanResult> UniqueScan(
      const std::string& file_name,
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
      ContextType context_type = ContextType::kNoNeed);
  /* Batch Insert */
  Status Insert(
      const std::string& file_name,
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs);
  Status InsertWithContext(
      const std::string& file_name,
      std::shared_ptr<NumberKeySetContext> number_key_set_context);
  /* Batch Delete */
  Status Delete(
      const std::string& file_name,
      const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs);
  Status DeleteWithContext(
      const std::string& file_name,
      std::shared_ptr<NumberKeySetContext> number_key_set_context);

  std::string ToString() const;

 protected:
  friend class PrimaryKeyHandle;

 private:
  std::shared_ptr<FileKeyMap> file_key_map_;
};

class NumberKeyHandle {
 public:
  NumberKeyHandle();
  Status PushNumberAndOffset(const uint32_t number, const uint32_t offset);

  void SetFileName(std::string_view file_name) { file_name_ = file_name; }
  void SetNumberKeySetContext(
      std::shared_ptr<NumberKeySetContext>& number_key_set_context);
  std::vector<std::pair<uint32_t, uint32_t>>& GetNumberOffsetPairs() {
    return number_offset_pairs_;
  }
  std::shared_ptr<NumberKeySetContext>& GetNumberKeySetContext() {
    return number_key_set_context_;
  }

  bool IsEmpty() const { return number_offset_pairs_.empty(); }

 private:
  std::string_view file_name_;
  std::vector<std::pair<uint32_t, uint32_t>> number_offset_pairs_;
  std::shared_ptr<NumberKeySetContext> number_key_set_context_ = nullptr;
};

class PrimaryKeyHandle {
 public:
  PrimaryKeyHandle() = delete;
  explicit PrimaryKeyHandle(std::shared_ptr<PrimaryKeyIndex> pk_index);

  bool IsEmpty() const;

  virtual Status ApplyToIndex() = 0;

 protected:
  Status ClassifyPrimaryKey(const uint32_t offset, const PrimaryKey& key);
  std::shared_ptr<PrimaryKeyIndex> pk_index_;
  std::map<std::string, NumberKeyHandle> number_key_handles_;
};

class PrimaryKeyHandleForInsert : public PrimaryKeyHandle {
 public:
  PrimaryKeyHandleForInsert() = delete;
  explicit PrimaryKeyHandleForInsert(std::shared_ptr<PrimaryKeyIndex> pk_index);

  /* Initialize function for batch insert */
  Status Initialize(std::shared_ptr<arrow::Array> composite_key_array);

  Status CheckUniqueViolation();

  std::pair<uint32_t, uint32_t> GetNextValidOffsetRange();

  static bool IsCompleted(const std::pair<uint32_t, uint32_t>& range);

  Status ApplyToIndex();

 private:
  Status Initialize(
      std::shared_ptr<arrow::LargeStringArray> composite_key_array);
  Status Initialize(std::shared_ptr<arrow::StringArray> composite_key_array);

  Status ApplyInvalidOffsets(std::vector<uint32_t>& invalid_offsets);

  NumberKeyFlatArray valid_offset_array_;
  size_t current_position_ = 0;
};

class PrimaryKeyHandleForDelete : public PrimaryKeyHandle {
 public:
  PrimaryKeyHandleForDelete() = delete;
  explicit PrimaryKeyHandleForDelete(std::shared_ptr<PrimaryKeyIndex> pk_index);

  /* Initialize function for batch delete */
  Status Initialize(std::shared_ptr<arrow::Array> composite_key_array,
                    arrow::Type::type composite_key_type,
                    std::shared_ptr<arrow::Array> filtered_indices);

  Status ApplyToIndex();

 private:
  Status CollectPrimaryKey(std::string_view& composite_key);
};

}  // namespace vdb