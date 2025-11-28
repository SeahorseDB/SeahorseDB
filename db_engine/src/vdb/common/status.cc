#include "status.hh"

namespace vdb {

Status::Code Status::GetCodeFromArrowStatus(arrow::Status status) const {
  switch (status.code()) {
    case arrow::StatusCode::OK:
      return Code::kOk;
    case arrow::StatusCode::OutOfMemory:
      return Code::kOutOfMemory;
    case arrow::StatusCode::Invalid:
      return Code::kInvalidArgument;
    case arrow::StatusCode::AlreadyExists:
      return Code::kAlreadyExists;
    case arrow::StatusCode::UnknownError:
    default:
      return Code::kUnknown;
  }
}

}  // namespace vdb