#pragma once

#include <string>
#include <arrow/status.h>

namespace vdb {

struct Status {
  enum class Code {
    kOk,
    kInvalidArgument,
    kAlreadyExists,
    kOutOfMemory,
    kUnknown,
  };

  Code status_;
  std::string string_;

  Status() : status_{Code::kUnknown} {}

  Status(Code status, const std::string &error_msg)
      : status_{status}, string_{error_msg} {}

  Status(Code status) : status_{status} {}

  Status(arrow::Status status)
      : status_{GetCodeFromArrowStatus(status)}, string_{status.ToString()} {}

  Status(const Status &other) = default;

  Status &operator=(const Status &other) = default;

  Status &operator=(arrow::Status status) {
    status_ = GetCodeFromArrowStatus(status);
    string_ = status.ToString();
    return *this;
  }

  static Status Ok() { return Status(Code::kOk); }

  static Status InvalidArgument(const std::string &msg) {
    return Status(Code::kInvalidArgument, msg);
  }

  static Status AlreadyExists(const std::string &msg) {
    return Status(Code::kAlreadyExists, msg);
  }

  static Status OutOfMemory(const std::string &msg) {
    return Status(Code::kOutOfMemory, msg);
  }

  static Status Unknown(const std::string &msg) {
    return Status(Code::kUnknown, msg);
  }

  static Status Unknown() { return Status(Code::kUnknown); }

  bool ok() const { return status_ == Code::kOk; }

  bool IsOutOfMemory() const { return status_ == Code::kOutOfMemory; }

  const std::string &ToString() const { return string_; }

  Code GetCodeFromArrowStatus(arrow::Status status) const;
};

}  // namespace vdb
