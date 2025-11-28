#pragma once

#include <string>
#include <sstream>
#include <cstdint>

#include "vdb/common/status.hh"

namespace vdb {
class LabelInfo {
 public:
  /* TODO
   * It will be deprecated.
   * future hnsw structure does not has label, offset of vector will replace it.
   * label format
   * total 64 bits
   *
   * 16 bits - segment number   (MAXIMUM 2^16-1 = 65,535)
   * 16 bits - set number     (MAXIMUM 2^16-1 = 65,535)
   * 32 bits - record number    (MAXIMUM 2^32-1 = 4,294,967,295) */
  static constexpr uint16_t kSegmentMax = UINT16_MAX;
  static constexpr uint16_t kSetMax = UINT16_MAX;
  static constexpr uint32_t kRecordMax = UINT32_MAX;

  static uint64_t Build(const uint16_t &segment_number,
                        const uint16_t &set_number,
                        const uint32_t &record_number) {
    return (static_cast<uint64_t>(segment_number) << 48) |
           (static_cast<uint64_t>(set_number) << 32) | record_number;
  }

  static uint16_t GetSegmentNumber(const uint64_t &label) {
    return (label & (((1LLU << 16) - 1) << 48)) >> 48;
  }

  static uint16_t GetSetNumber(const uint64_t &label) {
    return (label & (((1LLU << 16) - 1) << 32)) >> 32;
  }

  static uint32_t GetRecordNumber(const uint64_t &label) {
    return (label & ((1LLU << 32) - 1));
  }

  static Status CheckLabel(const uint64_t &label,
                           const uint16_t &segment_number,
                           const uint16_t &set_number,
                           const uint32_t &record_number) {
    if (GetSegmentNumber(label) != segment_number)
      return Status::InvalidArgument("Segment Number is not Saved Correctly.");
    if (GetSetNumber(label) != set_number)
      return Status::InvalidArgument("Chunk Number is not Saved Correctly.");
    if (GetRecordNumber(label) != record_number)
      return Status::InvalidArgument("Record Number is not Saved Correctly.");
    return Status::Ok();
  }

  static std::string ToString(const uint64_t label) {
    std::stringstream ss;
    ss << "(segment=" << GetSegmentNumber(label) << ", ";
    ss << "set=" << GetSetNumber(label) << ", ";
    ss << "record=" << GetRecordNumber(label) << ")";
    return ss.str();
  }
};
}  // namespace vdb
