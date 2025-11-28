#pragma once

#include <string>
#include <functional>
#include <limits>

#include <arrow/status.h>
#include <nlohmann/json.hpp>

namespace vdb {

using json = nlohmann::json;

class JsonValidator {
 public:
  explicit JsonValidator(const json& j, const std::string& error_postfix = "");

  // check if required field exists
  JsonValidator& RequireField(const std::string& field_names);

  // string type validation
  JsonValidator& ValidateString(
      const std::string& field_name,
      const std::function<bool(const std::string&)>& validator = nullptr);

  // number type validation
  template <typename T>
  JsonValidator& ValidateNumber(
      const std::string& field_name,
      const std::function<bool(T)>& validator = nullptr);

  // array type validation
  JsonValidator& ValidateArray(
      const std::string& field_name,
      const std::function<bool(const json&)>& element_validator = nullptr);

  // string array validation (special case)
  JsonValidator& ValidateStringArray(
      const std::string& field_name,
      const std::function<bool(const std::string&)>& validator = nullptr);

  // array size validation
  JsonValidator& ValidateArraySize(
      const std::string& field_name, size_t min_size = 0,
      size_t max_size = std::numeric_limits<size_t>::max());

  // exact array size validation
  JsonValidator& ValidateExactArraySize(const std::string& field_name,
                                        size_t expected_size);

  // conditional validation
  JsonValidator& ValidateIf(
      const std::string& condition_field, const std::string& condition_value,
      const std::function<JsonValidator&(JsonValidator&)>& validation_func);

  // return result
  arrow::Status GetStatus() const { return status_; }
  bool IsValid() const { return status_.ok(); }

 private:
  const json& json_;
  arrow::Status status_ = arrow::Status::OK();
  std::string error_postfix_;

  // Helper method to append postfix to error messages
  arrow::Status AppendPostfix(const arrow::Status& status) const {
    if (status.ok() || error_postfix_.empty()) return status;
    // Remove "Invalid: " prefix from Arrow Status message
    std::string msg = status.message();
    if (msg.substr(0, 9) == "Invalid: ") {
      msg = msg.substr(9);
    }
    // Remove leading spaces from the message
    while (!msg.empty() && std::isspace(msg[0])) {
      msg = msg.substr(1);
    }
    return arrow::Status::Invalid(msg + " " + error_postfix_);
  }
};

}  // namespace vdb