#include "json_validator.hh"

namespace vdb {

JsonValidator::JsonValidator(const json& j, const std::string& error_postfix)
    : json_(j), error_postfix_(error_postfix) {}

JsonValidator& JsonValidator::RequireField(const std::string& field_name) {
  if (status_.ok() && !json_.contains(field_name)) {
    status_ = AppendPostfix(arrow::Status::Invalid(
        "Missing required field: \"" + field_name + "\""));
  }
  return *this;
}

JsonValidator& JsonValidator::ValidateString(
    const std::string& field_name,
    const std::function<bool(const std::string&)>& validator) {
  if (!status_.ok()) return *this;

  if (!json_.contains(field_name)) return *this;

  const auto& field = json_[field_name];
  if (!field.is_string()) {
    status_ = AppendPostfix(
        arrow::Status::Invalid("\"" + field_name + "\" must be a string"));
    return *this;
  }

  if (validator) {
    const std::string value = field.get<std::string>();
    if (!validator(value)) {
      status_ = AppendPostfix(arrow::Status::Invalid(
          "Invalid value for \"" + field_name + "\": " + value));
    }
  }
  return *this;
}

template <typename T>
JsonValidator& JsonValidator::ValidateNumber(
    const std::string& field_name, const std::function<bool(T)>& validator) {
  if (!status_.ok()) return *this;

  if (!json_.contains(field_name)) return *this;

  const auto& field = json_[field_name];
  if (!field.is_number()) {
    status_ = AppendPostfix(
        arrow::Status::Invalid("\"" + field_name + "\" must be a number"));
    return *this;
  }

  if (validator) {
    const T value = field.get<T>();
    if (!validator(value)) {
      status_ = AppendPostfix(
          arrow::Status::Invalid("Invalid value for \"" + field_name +
                                 "\": " + std::to_string(value)));
    }
  }
  return *this;
}

JsonValidator& JsonValidator::ValidateArray(
    const std::string& field_name,
    const std::function<bool(const json&)>& element_validator) {
  if (!status_.ok()) return *this;

  if (!json_.contains(field_name)) return *this;

  const auto& field = json_[field_name];
  if (!field.is_array()) {
    status_ = AppendPostfix(
        arrow::Status::Invalid("\"" + field_name + "\" must be an array"));
    return *this;
  }

  if (field.empty()) {
    status_ = AppendPostfix(
        arrow::Status::Invalid("\"" + field_name + "\" must not be empty"));
    return *this;
  }

  if (element_validator) {
    for (const auto& element : field) {
      if (!element_validator(element)) {
        status_ = AppendPostfix(arrow::Status::Invalid(
            "Invalid element in \"" + field_name + "\": " + element.dump()));
        break;
      }
    }
  }
  return *this;
}

JsonValidator& JsonValidator::ValidateStringArray(
    const std::string& field_name,
    const std::function<bool(const std::string&)>& validator) {
  return ValidateArray(field_name, [validator](const json& element) {
    if (!element.is_string()) return false;
    if (validator) {
      return validator(element.get<std::string>());
    }
    return true;
  });
}

JsonValidator& JsonValidator::ValidateArraySize(const std::string& field_name,
                                                size_t min_size,
                                                size_t max_size) {
  if (!status_.ok()) return *this;

  if (!json_.contains(field_name)) return *this;

  const auto& field = json_[field_name];
  if (!field.is_array()) {
    status_ = AppendPostfix(
        arrow::Status::Invalid("\"" + field_name + "\" must be an array"));
    return *this;
  }

  if (min_size == max_size) {
    if (field.size() != min_size) {
      status_ = AppendPostfix(arrow::Status::Invalid(
          "\"" + field_name + "\" array size must be exactly " +
          std::to_string(min_size) + ", but got " +
          std::to_string(field.size())));
      return *this;
    }
  }

  const size_t size = field.size();
  if (size < min_size) {
    status_ = AppendPostfix(arrow::Status::Invalid(
        "\"" + field_name + "\" array size must be at least " +
        std::to_string(min_size) + ", but got " + std::to_string(size)));
    return *this;
  }

  if (size > max_size) {
    status_ = AppendPostfix(arrow::Status::Invalid(
        "\"" + field_name + "\" array size must be at most " +
        std::to_string(max_size) + ", but got " + std::to_string(size)));
    return *this;
  }

  return *this;
}

JsonValidator& JsonValidator::ValidateExactArraySize(
    const std::string& field_name, size_t expected_size) {
  return ValidateArraySize(field_name, expected_size, expected_size);
}

JsonValidator& JsonValidator::ValidateIf(
    const std::string& condition_field, const std::string& condition_value,
    const std::function<JsonValidator&(JsonValidator&)>& validation_func) {
  if (!status_.ok()) return *this;

  if (json_.contains(condition_field) && json_[condition_field].is_string() &&
      json_[condition_field].get<std::string>() == condition_value) {
    validation_func(*this);
  }
  return *this;
}

// Explicit template instantiations for common number types
template JsonValidator& JsonValidator::ValidateNumber<int32_t>(
    const std::string&, const std::function<bool(int32_t)>&);
template JsonValidator& JsonValidator::ValidateNumber<int64_t>(
    const std::string&, const std::function<bool(int64_t)>&);
template JsonValidator& JsonValidator::ValidateNumber<double>(
    const std::string&, const std::function<bool(double)>&);

}  // namespace vdb
