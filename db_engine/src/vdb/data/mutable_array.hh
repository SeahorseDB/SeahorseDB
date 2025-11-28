#pragma once

#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <unistd.h>

#include <arrow/api.h>

#include "vdb/common/defs.hh"
#include "vdb/common/status.hh"
#include "vdb/common/util.hh"

namespace vdb {
class MutableArray {
 public:
  MutableArray(size_t max_count) : size_{0} {
    size_t bit_array_size = ((max_count + 8 - 1) / 8);
    valid_.reserve(bit_array_size);
  }

  /* TODO constructor(std::shared_ptr<arrow::Array> arr)
   * mutable array -> arrow::Array is possible.
   * arrow::Array -> mutable array is not implemented, yet.
   * this direction will be needed when loading snapshot. */
  virtual ~MutableArray() = default;

  virtual std::string ToString() = 0;
  virtual std::shared_ptr<arrow::Array> ToArrowArray() = 0;
  virtual void Reset() = 0;
  virtual void TrimEnd(size_t count) = 0;

  /* need to deal with 'mutable array is full' case
   *
   * input array's all elements must be valid. this function is for batch
   * insertion. if you want to append array with some invalid elements, don't
   * use this function. */
  virtual Status Append(const arrow::Array *values) = 0;

  size_t Size() const { return size_; }

  inline bool IsAvailable(const size_t i) const {
    return (i < size_) && (valid_[i / 8] & BitPos[i % 8]);
  }

 protected:
  static int64_t EstimateInitialSpace(size_t max_count) {
    size_t bit_array_size = ((max_count + 8 - 1) / 8);
    /* [DAXE-939] add buffer size */
    bit_array_size += 64;  // buffer size for pointer, etc.
    return bit_array_size + sizeof(size_t);
  }

  Status AppendValidityBits(const arrow::Array *values) {
    auto bit_array = values->null_bitmap_data();
    int64_t bit_pos = values->offset();
    int64_t bit_count = values->length();
    if (bit_array == nullptr) {
      int64_t null_count = values->null_count();
      /* bit_array == null && bit_count != null_count => all valid */
      bool bit_value = (bit_count != null_count);
      return AppendUniformBitsTo(valid_, size_, bit_value, bit_count);
    } else {
      return AppendBitsTo(valid_, size_, bit_array, bit_pos, bit_count);
    }
  }

  void TrimValidBits(size_t new_size) {
    size_t new_valid_size = (new_size + 7) / 8;
    valid_.resize(new_valid_size);

    if (new_size > 0 && new_size % 8 != 0) {
      uint8_t mask = (1 << (new_size % 8)) - 1;
      valid_.back() &= mask;
    }
  }

  vdb::vector<uint8_t> valid_;
  size_t size_;
};

class BooleanArray : public MutableArray {
 public:
  BooleanArray(size_t max_count = 0) : MutableArray(max_count) {
    size_t bit_array_size = ((max_count + 8 - 1) / 8);
    data_.reserve(bit_array_size);
    Reset();
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 0; i < size_; i++) {
      if (IsAvailable(i)) {
        ss << "  ";
        if (data_[i / 8] & BitPos[i % 8])
          ss << "true";
        else
          ss << "false";
      } else {
        ss << " null";
      }
      if (i > 5) {
        ss << "  ...," << std::endl;
        break;
      } else if (i == size_ - 1) {
        ss << std::endl;
      } else {
        ss << "," << std::endl;
      }
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto bit_values = static_cast<const arrow::BooleanArray *>(values);
    auto bit_array = bit_values->values()->data();
    auto bit_pos = bit_values->offset();
    auto bit_count = bit_values->length();
    status = AppendBitsTo(data_, size_, bit_array, bit_pos, bit_count);
    size_ += values->length();
    return status;
  }

  Status Append(const bool value) {
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      if (value)
        data_.push_back(1);
      else
        data_.push_back(0);
      valid_.push_back(1);
    } else {
      if (value) data_.back() |= BitPos[bit_pos];
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    auto bit_pos = data_.size() % 8;
    if (bit_pos == 0) {
      data_.push_back(0);
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<bool> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      return data_[i / 8] & BitPos[i % 8];
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto data_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(bool)));
    return std::make_shared<arrow::BooleanArray>(size_, data_buf, validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    size_t new_byte_size = (size_ + 7) / 8;
    data_.resize(new_byte_size);

    if (size_ > 0 && size_ % 8 != 0) {
      uint8_t mask = (1 << (size_ % 8)) - 1;
      data_.back() &= mask;
    }

    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    size_t bit_array_size = ((max_count + 8 - 1) / 8);
    initial_space += bit_array_size;

    return initial_space;
  }

 private:
  vdb::vector<uint8_t> data_;
};

template <typename CType, typename ArrayType>
class NumericArray : public MutableArray {
 public:
  NumericArray(size_t max_count = 0) : MutableArray(max_count) {
    data_.reserve(max_count);
    Reset();
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 0; i < data_.size(); i++) {
      if (IsAvailable(i)) {
        ss << "  " << data_[i];
      } else {
        ss << " null";
      }
      if (i > 5) {
        ss << "  ...," << std::endl;
        break;
      } else if (i == data_.size() - 1) {
        ss << std::endl;
      } else {
        ss << "," << std::endl;
      }
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto numeric_values = static_cast<const ArrayType *>(values);
    /* offset computation is included in raw_values() */
    auto numeric_array = numeric_values->raw_values();
    auto numeric_count = numeric_values->length();
    status = AppendValuesTo(data_, numeric_array, numeric_count);

    size_ += values->length();
    return status;
  }

  Status Append(const CType value) {
    data_.push_back(value);
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    data_.emplace_back();
    auto bit_pos = data_.size() % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<CType> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      return data_[i];
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto data_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(CType)));
    return std::make_shared<ArrayType>(size_, data_buf, validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    data_.resize(size_);
    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += max_count * sizeof(CType);

    return initial_space;
  }

 private:
  vdb::vector<CType> data_;
};

class StringArray : public MutableArray {
 public:
  StringArray(size_t max_count = 0) : MutableArray(max_count) {
    offset_.reserve(max_count + 1);
    Reset();
    offset_[0] = 0;
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    int prev = 0;
    int cnt = 0;
    for (auto &offset : offset_) {
      if (offset == 0) continue;
      if (offset == prev) {
        ss << " null";
      } else {
        ss << " ";
        for (int i = prev; i < offset; i++) {
          ss << data_[i];
        }
      }
      if (cnt++ > 5) {
        ss << " ..." << std::endl;
      } else {
        ss << "," << std::endl;
      }
      prev = offset;
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto string_values = static_cast<const arrow::StringArray *>(values);
    /* append offset */
    auto offset_array = string_values->raw_value_offsets();
    auto string_count = string_values->length();
    status = AppendOffsets(offset_array, string_count);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto char_array = reinterpret_cast<const char *>(string_values->raw_data());
    auto char_pos = offset_array[0];
    auto char_count = string_values->total_values_length();
    status = AppendValuesTo(data_, char_array + char_pos, char_count);
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<char> &value) {
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status Append(const std::string_view &value) {
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status Append(const std::string &value) {
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::string_view> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      auto pos = offset_[i];
      auto len = offset_[i + 1] - offset_[i];
      return std::string_view(data_.data() + pos, len);
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto offset_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)offset_.data(), (int64_t)(offset_.size() * sizeof(int32_t)));
    auto data_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(char)));
    return std::make_shared<arrow::StringArray>(size_, offset_buf, data_buf,
                                                validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    offset_.resize(1);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    offset_.resize(size_ + 1);
    data_.resize(offset_.back());
    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += (max_count + 1) * sizeof(int32_t);

    return initial_space;
  }

  static int64_t EstimateExpandedSpace(const arrow::Array *values) {
    auto string_values = static_cast<const arrow::StringArray *>(values);
    auto char_count = string_values->total_values_length();
    int64_t expanded_size = char_count * sizeof(char);
    return expanded_size;
  }

  static int64_t EstimateExpandedSpace(const std::string &value) {
    int64_t expanded_size = value.length() * sizeof(char);
    return expanded_size;
  }

 private:
  Status AppendOffsets(const int32_t *offset_array,
                       const int64_t offset_count) {
    size_t prev_size = data_.size();
    int64_t prev_count = offset_.size();

    auto status = AppendValuesTo(offset_, offset_array + 1, offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (offset_array[0] != 0)) {
      for (int64_t i = 0; i < offset_count; i++) {
        offset_[prev_count + i] -= offset_array[0];
        offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  vdb::vector<int32_t> offset_;
  vdb::vector<char> data_;
};

class LargeStringArray : public MutableArray {
 public:
  LargeStringArray(size_t max_count = 0) : MutableArray(max_count) {
    offset_.reserve(max_count + 1);
    Reset();
    offset_[0] = 0;
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    int prev = 0;
    int cnt = 0;
    for (auto &offset : offset_) {
      if (offset == 0) continue;
      if (offset == prev) {
        ss << " null";
      } else {
        ss << " ";
        for (int i = prev; i < offset; i++) {
          ss << data_[i];
        }
      }
      if (cnt++ > 5) {
        ss << " ..." << std::endl;
      } else {
        ss << "," << std::endl;
      }
      prev = offset;
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto string_values = static_cast<const arrow::LargeStringArray *>(values);
    /* append offset */
    auto offset_array = string_values->raw_value_offsets();
    auto string_count = string_values->length();
    status = AppendOffsets(offset_array, string_count);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto char_array = reinterpret_cast<const char *>(string_values->raw_data());
    auto char_pos = offset_array[0];
    auto char_count = string_values->total_values_length();
    status = AppendValuesTo(data_, char_array + char_pos, char_count);
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<char> &value) {
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status Append(const std::string_view &value) {
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status Append(const std::string &value) {
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::string_view> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      auto pos = offset_[i];
      auto len = offset_[i + 1] - offset_[i];
      return std::string_view(data_.data() + pos, len);
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto offset_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)offset_.data(), (int64_t)(offset_.size() * sizeof(int64_t)));
    auto data_buf = std::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(char)));
    return std::make_shared<arrow::LargeStringArray>(size_, offset_buf,
                                                     data_buf, validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    offset_.resize(1);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    offset_.resize(size_ + 1);
    data_.resize(offset_.back());
    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += (max_count + 1) * sizeof(int64_t);

    return initial_space;
  }

  static int64_t EstimateExpandedSpace(const arrow::Array *values) {
    auto string_values = static_cast<const arrow::StringArray *>(values);
    auto char_count = string_values->total_values_length();
    int64_t expanded_size = char_count * sizeof(char);
    return expanded_size;
  }

  static int64_t EstimateExpandedSpace(const std::string &value) {
    int64_t expanded_size = value.length() * sizeof(char);
    return expanded_size;
  }

 private:
  Status AppendOffsets(const int64_t *offset_array,
                       const int64_t offset_count) {
    size_t prev_size = data_.size();
    int64_t prev_count = offset_.size();

    auto status = AppendValuesTo(offset_, offset_array + 1, offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (offset_array[0] != 0)) {
      for (int64_t i = 0; i < offset_count; i++) {
        offset_[prev_count + i] -= offset_array[0];
        offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  vdb::vector<int64_t> offset_;
  vdb::vector<char> data_;
};

template <typename CType, typename ChildType>
class ListArray : public MutableArray {
 public:
  ListArray(size_t max_count = 0) : MutableArray(max_count) {
    offset_.reserve(max_count + 1);
    Reset();
    offset_[0] = 0;
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    int prev = 0;
    int cnt = 0;
    for (auto &offset : offset_) {
      if (offset == 0) continue;
      if (offset == prev) {
        ss << "  null";
      } else {
        ss << "  [ ";
        for (int i = prev; i < offset; i++) {
          ss << data_[i];
          if (i != offset - 1) {
            ss << ", ";
          } else {
            ss << " ]";
          }
        }
      }
      if (cnt++ > 5) {
        ss << " ..." << std::endl;
      } else {
        ss << "," << std::endl;
      }
      prev = offset;
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto numeric_lists = static_cast<const arrow::ListArray *>(values);
    /* append offset */
    auto offset_array = numeric_lists->raw_value_offsets();
    auto list_count = numeric_lists->length();
    status = AppendOffsets(offset_array, list_count);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto numeric_array = numeric_lists->values()->data()->GetValues<CType>(1);
    auto numeric_pos = offset_array[0];
    auto numeric_count = offset_array[list_count] - offset_array[0];
    status = AppendValuesTo(data_, numeric_array + numeric_pos, numeric_count);
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<CType> &value) {
    /* if value is empty vector, no element is inserted */
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::vector<CType>> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      std::vector<CType> ret;
      for (int j = offset_[i]; j < offset_[i + 1]; j++) {
        ret.push_back(data_[j]);
      }
      return ret;
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto offset_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)offset_.data(), (int64_t)(offset_.size() * sizeof(int32_t)));
    auto data_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(CType)));
    auto data_arr = vdb::make_shared<ChildType>(data_.size(), data_buf);
    auto child_type = vdb::make_shared<typename ChildType::TypeClass>();
    auto type = vdb::make_shared<arrow::ListType>(child_type);
    return vdb::make_shared<arrow::ListArray>(type, size_, offset_buf, data_arr,
                                              validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    offset_.resize(1);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    offset_.resize(size_ + 1);
    data_.resize(offset_.back());
    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += (max_count + 1) * sizeof(int32_t);

    return initial_space;
  }

  static int64_t EstimateExpandedSpace(const arrow::Array *values) {
    auto numeric_lists = static_cast<const arrow::ListArray *>(values);
    auto offset_array = numeric_lists->raw_value_offsets();
    auto list_count = numeric_lists->length();
    auto numeric_count = offset_array[list_count] - offset_array[0];
    int64_t expanded_size = numeric_count * sizeof(CType);
    return expanded_size;
  }

  static int64_t EstimateExpandedSpace(const std::vector<CType> &value) {
    int64_t expanded_size = value.size() * sizeof(CType);
    return expanded_size;
  }

 private:
  Status AppendOffsets(const int32_t *offset_array,
                       const int64_t offset_count) {
    /* append offset listarray */
    size_t prev_size = data_.size();
    int64_t prev_count = offset_.size();

    auto status = AppendValuesTo(offset_, offset_array + 1, offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (offset_array[0] != 0)) {
      /* adjust offset */
      for (int64_t i = 0; i < offset_count; i++) {
        offset_[prev_count + i] -= offset_array[0];
        offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  vdb::vector<int32_t> offset_;
  vdb::vector<CType> data_;
};

class BooleanListArray : public MutableArray {
 public:
  BooleanListArray(size_t max_count = 0)
      : MutableArray(max_count), child_size_{0} {
    offset_.reserve(max_count + 1);
    Reset();
    offset_[0] = 0;
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    int prev = 0;
    int cnt = 0;
    for (auto &offset : offset_) {
      if (offset == 0) continue;
      if (offset == prev) {
        ss << "  null";
      } else {
        ss << "  [ ";
        for (int i = prev; i < offset; i++) {
          if (data_[i / 8] & BitPos[i % 8])
            ss << "true";
          else
            ss << "false";
          if (i != offset - 1) {
            ss << ", ";
          } else {
            ss << " ]";
          }
        }
      }
      if (cnt++ > 5) {
        ss << " ..." << std::endl;
      } else {
        ss << "," << std::endl;
      }
      prev = offset;
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto boolean_lists = static_cast<const arrow::ListArray *>(values);
    /* append offset */
    auto offset_array = boolean_lists->raw_value_offsets();
    auto list_count = boolean_lists->length();
    status = AppendOffsets(offset_array, list_count);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto bit_array = boolean_lists->values()->data()->GetValues<uint8_t>(1, 0);
    auto bit_pos = offset_array[0];
    auto bit_count = offset_array[list_count] - offset_array[0];
    status = AppendBitsTo(data_, child_size_, bit_array, bit_pos, bit_count);
    child_size_ += bit_count;
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<bool> &value) {
    /* if value is empty vector, no element is inserted */
    for (auto bit : value) {
      auto child_bit_pos = child_size_ % 8;
      if (child_bit_pos == 0) {
        if (bit)
          data_.push_back(1);
        else
          data_.push_back(0);
      } else {
        if (bit) data_.back() |= BitPos[child_bit_pos];
      }
      child_size_++;
    }
    offset_.push_back(child_size_);
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(1);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    offset_.push_back(child_size_);
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::vector<bool>> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      std::vector<bool> ret;
      for (int j = offset_[i]; j < offset_[i + 1]; j++) {
        ret.push_back(data_[j / 8] & BitPos[j % 8]);
      }
      return ret;
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto offset_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)offset_.data(), (int64_t)(offset_.size() * sizeof(int32_t)));
    auto data_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(uint8_t)));
    auto data_arr =
        vdb::make_shared<arrow::BooleanArray>(child_size_, data_buf);
    return vdb::make_shared<arrow::ListArray>(arrow::list(arrow::boolean()),
                                              size_, offset_buf, data_arr,
                                              validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    offset_.resize(1);
    data_.resize(0);
    size_ = 0;
    child_size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    offset_.resize(size_ + 1);
    child_size_ = offset_.back();

    size_t new_data_size = (child_size_ + 7) / 8;
    data_.resize(new_data_size);
    if (child_size_ > 0 && child_size_ % 8 != 0) {
      uint8_t mask = (1 << (child_size_ % 8)) - 1;
      data_.back() &= mask;
    }

    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += (max_count + 1) * sizeof(int32_t);
    initial_space += sizeof(size_t);

    return initial_space;
  }

  static int64_t EstimateExpandedSpace(const arrow::Array *values) {
    auto boolean_lists = static_cast<const arrow::ListArray *>(values);
    auto offset_array = boolean_lists->raw_value_offsets();
    auto list_count = boolean_lists->length();
    auto bit_count = offset_array[list_count] - offset_array[0];
    size_t expanded_bit_array_length = (bit_count + 8 - 1) / 8;
    int64_t expanded_size = expanded_bit_array_length * sizeof(uint8_t);
    return expanded_size;
  }

  static int64_t EstimateExpandedSpace(const std::vector<bool> &value) {
    size_t expanded_bit_array_length = (value.size() + 8 - 1) / 8;
    int64_t expanded_size = expanded_bit_array_length * sizeof(uint8_t);
    return expanded_size;
  }

 private:
  Status AppendOffsets(const int32_t *offset_array,
                       const int64_t offset_count) {
    /* append offset listarray */
    size_t prev_size = child_size_;
    int64_t prev_count = offset_.size();

    /* append offset array
     * start point is always zero, which is not needed to insert */
    auto status = AppendValuesTo(offset_, offset_array + 1, offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (offset_array[0] != 0)) {
      /* adjust offset */
      for (int64_t i = 0; i < offset_count; i++) {
        offset_[prev_count + i] -= offset_array[0];
        offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  vdb::vector<int32_t> offset_;
  vdb::vector<uint8_t> data_;
  size_t child_size_;
};

class StringListArray : public MutableArray {
 public:
  StringListArray(size_t max_count = 0) : MutableArray(max_count) {
    offset_.reserve(max_count + 1);
    child_offset_.reserve(1);
    Reset();
    offset_[0] = 0;
    child_offset_[0] = 0;
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 0; i < offset_.size() - 1; i++) {
      ss << "  [";
      for (int j = offset_[i]; j < offset_[i + 1]; j++) {
        ss << " ";
        for (int k = child_offset_[j]; k < child_offset_[j + 1]; k++) {
          ss << data_[k];
        }
        ss << ",";
      }
      ss << " ]," << std::endl;
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto string_lists = static_cast<const arrow::ListArray *>(values);
    /* append offset */
    auto offset_array = string_lists->raw_value_offsets();
    auto list_count = string_lists->length();
    status = AppendOffsets(offset_array, list_count);
    if (!status.ok()) {
      return status;
    }
    /* append child offset */
    auto string_values =
        static_cast<const arrow::StringArray *>(string_lists->values().get());
    /* append child offset array
     * starting point is always zero, which is not needed to insert */
    auto child_offset_array = string_values->raw_value_offsets();
    auto child_offset_pos = offset_array[0];
    auto child_value_count = offset_array[list_count] - offset_array[0];
    status = AppendChildOffsets(child_offset_array + child_offset_pos,
                                child_value_count);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto char_array = reinterpret_cast<const char *>(string_values->raw_data());
    auto char_pos = child_offset_array[child_offset_pos];
    auto char_count = child_offset_array[child_offset_pos + child_value_count] -
                      child_offset_array[child_offset_pos];
    status = AppendValuesTo(data_, char_array + char_pos, char_count);
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<std::string> &value) {
    /* if value is empty vector, no element is inserted */
    for (auto &elem : value) {
      std::copy(elem.begin(), elem.end(), std::back_inserter(data_));
      child_offset_.push_back(data_.size());
    }
    offset_.push_back(child_offset_.size() - 1);
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    offset_.push_back(child_offset_.size() - 1);
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::vector<std::string_view>> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      std::vector<std::string_view> ret;
      for (int j = offset_[i]; j < offset_[i + 1]; j++) {
        auto pos = child_offset_[j];
        auto len = child_offset_[j + 1] - child_offset_[j];
        ret.emplace_back(data_.data() + pos, len);
      }
      return ret;
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto offset_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)offset_.data(), (int64_t)(offset_.size() * sizeof(int32_t)));
    auto child_offset_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)child_offset_.data(),
        (int64_t)(child_offset_.size() * sizeof(int32_t)));
    auto data_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(char)));
    auto data_arr = vdb::make_shared<arrow::StringArray>(
        child_offset_.size() - 1, child_offset_buf, data_buf);
    auto type = arrow::list(arrow::utf8());
    return vdb::make_shared<arrow::ListArray>(type, size_, offset_buf, data_arr,
                                              validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    offset_.resize(1);
    child_offset_.resize(1);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    offset_.resize(size_ + 1);
    child_offset_.resize(offset_.back() + 1);
    data_.resize(child_offset_.back());
    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += (max_count + 1) * sizeof(int32_t);
    initial_space += 1 * sizeof(int32_t);

    return initial_space;
  }

  static int64_t EstimateExpandedSpace(const arrow::Array *values) {
    auto string_lists = static_cast<const arrow::ListArray *>(values);
    auto offset_array = string_lists->raw_value_offsets();
    auto list_count = string_lists->length();
    auto string_values =
        static_cast<const arrow::StringArray *>(string_lists->values().get());
    auto child_offset_array = string_values->raw_value_offsets();
    auto child_offset_pos = offset_array[0];
    auto child_value_count = offset_array[list_count] - offset_array[0];
    auto char_count = child_offset_array[child_offset_pos + child_value_count] -
                      child_offset_array[child_offset_pos];
    int64_t expanded_size = 0;
    expanded_size += child_value_count * sizeof(int32_t);
    expanded_size += char_count * sizeof(char);
    return expanded_size;
  }

  static int64_t EstimateExpandedSpace(const std::vector<std::string> &value) {
    auto string_count = value.size();
    auto char_count = 0;
    for (auto &str : value) {
      char_count += str.length();
    }
    int64_t expanded_size = 0;
    expanded_size += string_count * sizeof(int32_t);
    expanded_size += char_count * sizeof(char);
    return expanded_size;
  }

 private:
  Status AppendOffsets(const int32_t *offset_array,
                       const int64_t offset_count) {
    /* append offset listarray */
    size_t prev_size = child_offset_.size() - 1;
    int64_t prev_count = offset_.size();

    /* append offset array. first offset is starting point. */
    auto status = AppendValuesTo(offset_, offset_array + 1, offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (offset_array[0] != 0)) {
      /* adjust offset */
      for (int64_t i = 0; i < offset_count; i++) {
        offset_[prev_count + i] -= offset_array[0];
        offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  Status AppendChildOffsets(const int32_t *child_offset_array,
                            const int64_t child_offset_count) {
    /* append child offset listarray */
    size_t prev_size = data_.size();
    int64_t prev_count = child_offset_.size();

    auto status = AppendValuesTo(child_offset_, child_offset_array + 1,
                                 child_offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (child_offset_array[0] != 0)) {
      /* adjust child offset */
      for (int64_t i = 0; i < child_offset_count; i++) {
        child_offset_[prev_count + i] -= child_offset_array[0];
        child_offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  vdb::vector<int32_t> offset_;
  vdb::vector<int32_t> child_offset_;
  vdb::vector<char> data_;
};

template <typename CType, typename ChildType>
class FixedSizeListArray : public MutableArray {
 public:
  FixedSizeListArray(const size_t width, size_t max_count = 0)
      : MutableArray(max_count), width_{width} {
    data_.reserve(width * max_count);
    Reset();
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 0; i < size_; i++) {
      if (IsAvailable(i)) {
        ss << "  [ ";
        for (size_t j = 0; j < width_; j++) {
          if (j != width_ - 1) {
            ss << data_[i * width_ + j] << ", ";
          } else {
            ss << data_[i * width_ + j];
          }
        }
        ss << " ]," << std::endl;
      } else {
        if (i == size_ - 1)
          ss << "  null" << std::endl;
        else
          ss << "  null," << std::endl;
      }
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto fixed_size_lists =
        static_cast<const arrow::FixedSizeListArray *>(values);
    /* append data */
    /* offset computation is included in GetValues() */
    auto numeric_array =
        fixed_size_lists->values()->data()->GetValues<CType>(1);
    auto numeric_pos = fixed_size_lists->offset() * width_;
    auto numeric_count = fixed_size_lists->length() * width_;
    status = AppendValuesTo(data_, numeric_array + numeric_pos, numeric_count);
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<CType> &value) {
    if (value.size() != width_) {
      return Status::InvalidArgument(
          "FixedSizeListArray: number of elements is not matched with the "
          "width.");
    }
    std::copy(value.begin(), value.end(), std::back_inserter(data_));
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    for (size_t i = 0; i < width_; i++) data_.emplace_back();
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::vector<CType>> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      std::vector<CType> ret;
      for (size_t j = i * width_; j < (i + 1) * width_; j++) {
        ret.push_back(data_[j]);
      }
      return ret;
    } else {
      return {};
    }
  }

  const CType *GetRawValue(size_t i) const {
    if (IsAvailable(i)) {
      return &data_[i * width_];
    } else {
      return nullptr;
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto data_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(CType)));
    auto data_arr = vdb::make_shared<ChildType>(data_.size(), data_buf);
    auto child_type = vdb::make_shared<typename ChildType::TypeClass>();
    auto type = vdb::make_shared<arrow::FixedSizeListType>(child_type, width_);
    return vdb::make_shared<arrow::FixedSizeListArray>(type, size_, data_arr,
                                                       validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    data_.resize(size_ * width_);
    TrimValidBits(size_);
  }

  size_t GetWidth() const { return width_; }

  static int64_t EstimateInitialSpace(const size_t width, size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += width * max_count * sizeof(CType);
    initial_space += sizeof(size_t);

    return initial_space;
  }

 private:
  const size_t width_;
  vdb::vector<CType> data_;
};

class BooleanFixedSizeListArray : public MutableArray {
 public:
  BooleanFixedSizeListArray(const size_t width, size_t max_count = 0)
      : MutableArray(max_count), width_{width}, child_size_{0} {
    size_t bit_array_length = ((width * max_count) + 8 - 1) / 8;
    data_.reserve(bit_array_length);
    Reset();
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 0; i < size_; i++) {
      if (IsAvailable(i)) {
        ss << "  [ ";
        for (size_t j = 0; j < width_; j++) {
          if (j != width_ - 1) {
            if (data_[(i * width_ + j) / 8] & BitPos[(i * width_ + j) % 8])
              ss << "true, ";
            else
              ss << "false, ";
          } else {
            if (data_[(i * width_ + j) / 8] & BitPos[(i * width_ + j) % 8])
              ss << "true";
            else
              ss << "false";
          }
        }
        ss << " ]," << std::endl;
      } else {
        if (i == size_ - 1)
          ss << "  null" << std::endl;
        else
          ss << "  null," << std::endl;
      }
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto fixed_size_lists =
        static_cast<const arrow::FixedSizeListArray *>(values);
    /* append data */
    auto bit_array =
        fixed_size_lists->values()->data()->GetValues<uint8_t>(1, 0);
    auto bit_pos = fixed_size_lists->offset() * width_;
    auto bit_count = fixed_size_lists->length() * width_;
    status = AppendBitsTo(data_, child_size_, bit_array, bit_pos, bit_count);
    child_size_ += bit_count;
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<bool> &value) {
    if (value.size() != width_) {
      return Status::InvalidArgument(
          "FixedSizeListArray: number of elements is not matched with the "
          "width.");
    }
    for (auto val : value) {
      auto child_bit_pos = child_size_ % 8;
      if (child_bit_pos == 0) {
        if (val)
          data_.push_back(1);
        else
          data_.push_back(0);
      } else {
        if (val) data_.back() |= BitPos[child_bit_pos];
      }
      child_size_++;
    }
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(1);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    for (size_t i = 0; i < width_; i++) {
      if (child_size_ % 8 == 0) data_.push_back(0);
      child_size_++;
    }
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::vector<bool>> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      std::vector<bool> ret;
      for (size_t j = i * width_; j < (i + 1) * width_; j++) {
        ret.push_back(data_[j / 8] & BitPos[j % 8]);
      }
      return ret;
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto data_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(uint8_t)));
    auto data_arr =
        vdb::make_shared<arrow::BooleanArray>(child_size_, data_buf);
    auto type =
        vdb::make_shared<arrow::FixedSizeListType>(arrow::boolean(), width_);
    return vdb::make_shared<arrow::FixedSizeListArray>(type, size_, data_arr,
                                                       validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    data_.resize(0);
    size_ = 0;
    child_size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    child_size_ = size_ * width_;
    size_t new_data_size = (child_size_ + 7) / 8;
    data_.resize(new_data_size);

    if (child_size_ > 0 && child_size_ % 8 != 0) {
      uint8_t mask = (1 << (child_size_ % 8)) - 1;
      data_.back() &= mask;
    }

    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(const size_t width, size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    size_t bit_array_length = ((width * max_count) + 8 - 1) / 8;
    initial_space += bit_array_length * sizeof(uint8_t);
    initial_space += sizeof(size_t);
    initial_space += sizeof(size_t);

    return initial_space;
  }

 private:
  const size_t width_;
  vdb::vector<uint8_t> data_;
  size_t child_size_;
};

class StringFixedSizeListArray : public MutableArray {
 public:
  StringFixedSizeListArray(const size_t width, size_t max_count = 0)
      : MutableArray(max_count), width_{width} {
    child_offset_.reserve(max_count * width_ + 1);
    Reset();
    child_offset_[0] = 0;
  }

  std::string ToString() override {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 1; i < child_offset_.size(); i++) {
      if ((i - 1) % width_ == 0) ss << "  [ ";
      for (int j = child_offset_[i - 1]; j < child_offset_[i]; j++) {
        ss << data_[j];
      }
      ss << ", ";
      if ((i - 1) % width_ == width_ - 1) ss << "]," << std::endl;
    }
    ss << "]" << std::endl;
    return ss.str();
  }

  Status Append(const arrow::Array *values) override {
    /* append valid bit */
    auto status = AppendValidityBits(values);
    if (!status.ok()) {
      return status;
    }
    auto fixed_size_lists =
        static_cast<const arrow::FixedSizeListArray *>(values);
    auto string_array = static_cast<const arrow::StringArray *>(
        fixed_size_lists->values().get());
    /* append child offset */
    auto child_offset_array = string_array->raw_value_offsets();
    auto child_offset_pos = fixed_size_lists->offset() * width_;
    auto child_value_count = fixed_size_lists->length() * width_;
    status = AppendChildOffsets(child_offset_array + child_offset_pos,
                                child_value_count);
    if (!status.ok()) {
      return status;
    }
    /* append data */
    auto char_array = reinterpret_cast<const char *>(string_array->raw_data());
    auto char_pos = child_offset_array[child_offset_pos];
    auto char_count = child_offset_array[child_offset_pos + child_value_count] -
                      child_offset_array[child_offset_pos];
    status = AppendValuesTo(data_, char_array + char_pos, char_count);
    size_ += values->length();
    return status;
  }

  Status Append(const std::vector<std::string> &value) {
    if (value.size() != width_) {
      return Status::InvalidArgument(
          "FixedSizeListArray: number of elements is not matched with the "
          "width.");
    }
    for (auto &elem : value) {
      std::copy(elem.begin(), elem.end(), std::back_inserter(data_));
      child_offset_.push_back(data_.size());
    }
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(BitPos[0]);
    } else {
      valid_.back() |= BitPos[bit_pos];
    }
    size_++;
    return Status::Ok();
  }

  Status AppendNull() {
    for (size_t i = 0; i < width_; i++) child_offset_.push_back(data_.size());
    auto bit_pos = size_ % 8;
    if (bit_pos == 0) {
      valid_.push_back(0);
    }
    size_++;
    return Status::Ok();
  }

  std::optional<std::vector<std::string_view>> GetValue(size_t i) const {
    if (IsAvailable(i)) {
      std::vector<std::string_view> ret;
      for (size_t j = i * width_; j < (i + 1) * width_; j++) {
        auto pos = child_offset_[j];
        auto len = child_offset_[j + 1] - child_offset_[j];
        ret.emplace_back(data_.data() + pos, len);
      }
      return ret;
    } else {
      return {};
    }
  }

  std::shared_ptr<arrow::Array> ToArrowArray() override {
    auto validity_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)valid_.data(), (int64_t)(valid_.size()));
    auto child_offset_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)child_offset_.data(),
        (int64_t)(child_offset_.size() * sizeof(int32_t)));
    auto data_buf = vdb::make_shared<arrow::Buffer>(
        (uint8_t *)data_.data(), (int64_t)(data_.size() * sizeof(char)));
    auto data_arr = vdb::make_shared<arrow::StringArray>(
        child_offset_.size() - 1, child_offset_buf, data_buf);
    auto type = arrow::fixed_size_list(arrow::utf8(), width_);
    return vdb::make_shared<arrow::FixedSizeListArray>(type, size_, data_arr,
                                                       validity_buf);
  }

  void Reset() override {
    valid_.resize(0);
    child_offset_.resize(1);
    data_.resize(0);
    size_ = 0;
  }

  void TrimEnd(size_t count) override {
    if (count >= size_) {
      Reset();
      return;
    }

    size_ -= count;
    child_offset_.resize(size_ * width_ + 1);
    data_.resize(child_offset_.back());
    TrimValidBits(size_);
  }

  static int64_t EstimateInitialSpace(const size_t width, size_t max_count) {
    int64_t initial_space = MutableArray::EstimateInitialSpace(max_count);

    initial_space += (max_count * width + 1) * sizeof(int32_t);
    initial_space += sizeof(size_t);

    return initial_space;
  }

  static int64_t EstimateExpandedSpace(const arrow::Array *values) {
    auto fixed_size_lists =
        static_cast<const arrow::FixedSizeListArray *>(values);
    auto fixed_size_list_type =
        std::static_pointer_cast<arrow::FixedSizeListType>(
            fixed_size_lists->type());
    int32_t width = fixed_size_list_type->list_size();
    auto string_array = static_cast<const arrow::StringArray *>(
        fixed_size_lists->values().get());
    auto child_offset_array = string_array->raw_value_offsets();
    auto child_offset_pos = fixed_size_lists->offset() * width;
    auto child_value_count = fixed_size_lists->length() * width;
    auto char_count = child_offset_array[child_offset_pos + child_value_count] -
                      child_offset_array[child_offset_pos];
    int64_t expanded_size = char_count * sizeof(char);
    return expanded_size;
  }

  static int64_t EstimateExpandedSpace(const std::vector<std::string> &value) {
    auto char_count = 0;
    for (auto &str : value) {
      char_count += str.length();
    }
    int64_t expanded_size = char_count * sizeof(char);
    return expanded_size;
  }

 private:
  Status AppendChildOffsets(const int32_t *child_offset_array,
                            const int64_t child_offset_count) {
    /* append child offset listarray */
    size_t prev_size = data_.size();
    int64_t prev_count = child_offset_.size();

    /* append child offset array
     * start point is always zero, which is not needed to insert */
    auto status = AppendValuesTo(child_offset_, child_offset_array + 1,
                                 child_offset_count);
    if (!status.ok()) return status;
    if ((prev_size != 0) || (child_offset_array[0] != 0)) {
      /* adjust child_offset */
      for (int64_t i = 0; i < child_offset_count; i++) {
        child_offset_[prev_count + i] -= child_offset_array[0];
        child_offset_[prev_count + i] += prev_size;
      }
    }
    return Status::Ok();
  }

  const size_t width_;
  vdb::vector<int32_t> child_offset_;
  vdb::vector<char> data_;
};

using Int8Array = NumericArray<int8_t, arrow::Int8Array>;
using Int16Array = NumericArray<int16_t, arrow::Int16Array>;
using Int32Array = NumericArray<int32_t, arrow::Int32Array>;
using Int64Array = NumericArray<int64_t, arrow::Int64Array>;
using UInt8Array = NumericArray<uint8_t, arrow::UInt8Array>;
using UInt16Array = NumericArray<uint16_t, arrow::UInt16Array>;
using UInt32Array = NumericArray<uint32_t, arrow::UInt32Array>;
using UInt64Array = NumericArray<uint64_t, arrow::UInt64Array>;
using FloatArray = NumericArray<float, arrow::FloatArray>;
using DoubleArray = NumericArray<double, arrow::DoubleArray>;

using Int8ListArray = ListArray<int8_t, arrow::Int8Array>;
using Int16ListArray = ListArray<int16_t, arrow::Int16Array>;
using Int32ListArray = ListArray<int32_t, arrow::Int32Array>;
using Int64ListArray = ListArray<int64_t, arrow::Int64Array>;
using UInt8ListArray = ListArray<uint8_t, arrow::UInt8Array>;
using UInt16ListArray = ListArray<uint16_t, arrow::UInt16Array>;
using UInt32ListArray = ListArray<uint32_t, arrow::UInt32Array>;
using UInt64ListArray = ListArray<uint64_t, arrow::UInt64Array>;
using FloatListArray = ListArray<float, arrow::FloatArray>;
using DoubleListArray = ListArray<double, arrow::DoubleArray>;

using Int8FixedSizeListArray = FixedSizeListArray<int8_t, arrow::Int8Array>;
using Int16FixedSizeListArray = FixedSizeListArray<int16_t, arrow::Int16Array>;
using Int32FixedSizeListArray = FixedSizeListArray<int32_t, arrow::Int32Array>;
using Int64FixedSizeListArray = FixedSizeListArray<int64_t, arrow::Int64Array>;
using UInt8FixedSizeListArray = FixedSizeListArray<uint8_t, arrow::UInt8Array>;
using UInt16FixedSizeListArray =
    FixedSizeListArray<uint16_t, arrow::UInt16Array>;
using UInt32FixedSizeListArray =
    FixedSizeListArray<uint32_t, arrow::UInt32Array>;
using UInt64FixedSizeListArray =
    FixedSizeListArray<uint64_t, arrow::UInt64Array>;
using FloatFixedSizeListArray = FixedSizeListArray<float, arrow::FloatArray>;
using DoubleFixedSizeListArray = FixedSizeListArray<double, arrow::DoubleArray>;

using BoolArray = BooleanArray;
using BoolListArray = BooleanListArray;
using BoolFixedSizeListArray = BooleanFixedSizeListArray;
}  // namespace vdb
