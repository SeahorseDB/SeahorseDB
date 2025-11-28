#pragma once

#include <cassert>
#include <string>
#include <string_view>

#include "vdb/common/util.hh"

namespace vdb {
class PrimaryKey {
 public:
  PrimaryKey() = default;
  PrimaryKey(const std::string& file_name, uint64_t number);
  PrimaryKey(std::string_view file_name, uint64_t number);
  static arrow::Result<PrimaryKey> Build(const std::string_view& composite_key);

  const std::string& file_name() const { return file_name_; }
  uint64_t number() const { return number_; }

  void SetFileName(const std::string& file_name) { file_name_ = file_name; }
  void SetNumber(uint64_t number) { number_ = number; }

  /* prefix increment */
  PrimaryKey operator++() { return PrimaryKey(file_name_, number_ + 1); }

  /* postfix increment */
  PrimaryKey operator++(int) {
    PrimaryKey old = *this;
    ++(*this);
    return old;
  }

  PrimaryKey operator+(uint64_t n) const {
    return PrimaryKey(file_name_, number_ + n);
  }

  PrimaryKey operator-(uint64_t n) const {
    return PrimaryKey(file_name_, number_ - n);
  }

  uint64_t operator-(const PrimaryKey& other) const {
    assert(file_name_ == other.file_name_);
    return number_ - other.number_;
  }

  bool operator==(const PrimaryKey& other) const {
    return file_name_ == other.file_name_ && number_ == other.number_;
  }

  bool operator!=(const PrimaryKey& other) const { return !(*this == other); }

  bool operator<(const PrimaryKey& other) const {
    if (file_name_ != other.file_name_) {
      return file_name_ < other.file_name_;
    }
    return number_ < other.number_;
  }

  bool operator>(const PrimaryKey& other) const {
    if (file_name_ != other.file_name_) {
      return file_name_ > other.file_name_;
    }
    return number_ > other.number_;
  }

  bool operator<=(const PrimaryKey& other) const { return !(*this > other); }

  bool operator>=(const PrimaryKey& other) const { return !(*this < other); }

  std::string ToString() const {
    return file_name_ + ":" + std::to_string(number_);
  }

 private:
  std::string file_name_;
  uint64_t number_;
};
}  // namespace vdb