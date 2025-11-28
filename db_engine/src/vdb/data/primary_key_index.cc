#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include <filesystem>

#include "arrow/api.h"
#include "arrow/array/array_nested.h"

#include "vdb/common/defs.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/util.hh"
#include "vdb/data/primary_key_index.hh"

namespace vdb {

void NumberKeySetContext::PushNumberKey(const uint32_t number) {
  numbers_.push_back(number);
}

void NumberKeyFlatArrayContext::PushNumberKey(const uint32_t number,
                                              const size_t position) {
  NumberKeySetContext::PushNumberKey(number);
  positions_.push_back(position);
}

std::string NumberKeyFlatArrayContext::ToString() const {
  std::stringstream ss;
  ss << "numbers: ";
  for (auto number : numbers_) {
    ss << number << " ";
  }
  ss << "\n";
  ss << "positions: ";
  for (auto position : positions_) {
    ss << position << " ";
  }
  ss << "\n";
  return ss.str();
}

std::string PkScanResult::PkScanResultToString(
    const PkScanResult& scan_result) {
  std::stringstream ss;
  ss << "\noffsets: ";
  for (auto offset : scan_result.offsets) {
    ss << offset << " ";
  }
  ss << "\nnumber_key_set_context: " << std::endl
     << scan_result.number_key_set_context->ToString();
  return ss.str();
}

NumberKeyFlatArray::NumberKeyFlatArray(uint8_t* buffer,
                                       uint64_t& buffer_offset) {
  uint64_t count = 0;
  buffer_offset += GetLengthFrom(buffer + buffer_offset, count);
  number_array_.resize(count);
  is_range_end_bitmap_.resize(count);
  uint8_t* is_range_end_bitmap_buffer = buffer + buffer_offset;
  buffer_offset += count;
  uint32_t* number_array_buffer =
      reinterpret_cast<uint32_t*>(buffer + buffer_offset);
  buffer_offset += count * sizeof(uint32_t);
  for (size_t i = 0; i < count; i++) {
    is_range_end_bitmap_[i] = is_range_end_bitmap_buffer[i];
    number_array_[i] = number_array_buffer[i];
  }
}

Status NumberKeyFlatArray::SerializeInto(std::vector<uint8_t>& buffer,
                                         uint64_t& buffer_offset) const {
  uint32_t count = number_array_.size();
  uint64_t total_length = 0;
  total_length += ComputeBytesFor(count);
  total_length += count * sizeof(uint8_t);
  total_length += count * sizeof(uint32_t);
  buffer.resize(buffer.size() + total_length);
  /* serialize count of number_array_ */
  buffer_offset += PutLengthTo(buffer.data() + buffer_offset, count);
  /* serialize is_range_end_bitmap_ */
  std::vector<uint8_t> is_range_end_bitmap_buffer;
  for (size_t i = 0; i < count; i++) {
    is_range_end_bitmap_buffer.push_back(is_range_end_bitmap_[i]);
  }
  memcpy(buffer.data() + buffer_offset, is_range_end_bitmap_buffer.data(),
         is_range_end_bitmap_buffer.size());
  buffer_offset += is_range_end_bitmap_buffer.size();

  /* serialize number_array_ */
  memcpy(buffer.data() + buffer_offset, number_array_.data(),
         count * sizeof(uint32_t));
  buffer_offset += count * sizeof(uint32_t);
  return Status::Ok();
}

void NumberKeyFlatArray::Initialize(const uint32_t min_number,
                                    const uint32_t max_number) {
  number_array_.resize(2);
  is_range_end_bitmap_.resize(2);
  number_array_[0] = min_number;
  number_array_[1] = max_number;
  is_range_end_bitmap_[0] = false;
  is_range_end_bitmap_[1] = true;
}

size_t NumberKeyFlatArray::BinarySearch(const uint32_t number,
                                        const size_t start,
                                        const size_t end) const {
  auto start_it = number_array_.begin() + start;
  auto end_it = (end >= number_array_.size()) ? number_array_.end()
                                              : number_array_.begin() + end + 1;
  auto it = std::lower_bound(start_it, end_it, number);
  if (it == end_it) {
    return number_array_.size();
  }
  return it - number_array_.begin();
}

bool NumberKeyFlatArray::IsFound(const uint32_t number,
                                 const size_t position) const {
  bool is_found = false;
  /* if position is out of range, number is not found */
  if (position < number_array_.size()) {
    if (number_array_[position] == number) {
      /* if number is found at position, return true */
      is_found = true;
    } else if (is_range_end_bitmap_[position]) {
      /* if number is not found at position, but position is the end of a range,
       * return true */
      is_found = true;
    }
  }
  return is_found;
}

bool NumberKeyFlatArray::IsRangeStart(const size_t position) const {
  return (position < number_array_.size() - 1) &&
         is_range_end_bitmap_[position + 1];
}

bool NumberKeyFlatArray::IsRangeEnd(const size_t position) const {
  return (position < number_array_.size()) && is_range_end_bitmap_[position];
}

arrow::Result<PkScanResult> NumberKeyFlatArray::UniqueScan(
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
    ContextType context_type) {
  PkScanResult scan_result;
  auto& offsets = scan_result.offsets;
  std::shared_ptr<NumberKeyFlatArrayContext> context = nullptr;
  if (context_type != ContextType::kNoNeed) {
    context = std::make_shared<NumberKeyFlatArrayContext>(context_type);
  }
  scan_result.number_key_set_context = context;
  /* number_offset_pairs is sorted by number */
  size_t i;
  for (i = 0; i < number_offset_pairs.size();) {
    size_t consecutive_count = 1;
    while (i + consecutive_count < number_offset_pairs.size() &&
           number_offset_pairs[i + consecutive_count].first ==
               number_offset_pairs[i].first + consecutive_count) {
      consecutive_count++;
    }
    /* consecutive range: [i, i + consecutive_count - 1] */
    if (consecutive_count == 1) {
      uint32_t number = number_offset_pairs[i].first;
      uint32_t offset = number_offset_pairs[i].second;
      auto position = BinarySearch(number);
      if (IsFound(number, position)) {
        offsets.push_back(offset);
        if (context_type == ContextType::kDelete) {
          context->PushNumberKey(number, position);
        }
      } else if (context_type == ContextType::kInsert) {
        context->PushNumberKey(number, position);
      }
    } else {
      uint32_t start_number = number_offset_pairs[i].first;
      uint32_t end_number =
          number_offset_pairs[i + consecutive_count - 1].first;
      auto start_position = BinarySearch(start_number);
      auto end_position = BinarySearch(end_number, start_position);
      if (start_position == end_position) {
        for (size_t j = i; j < i + consecutive_count; j++) {
          auto number = number_offset_pairs[j].first;
          if (IsFound(number, start_position)) {
            offsets.push_back(number_offset_pairs[j].second);
            if (context_type == ContextType::kDelete) {
              context->PushNumberKey(number, start_position);
            }
          } else if (context_type == ContextType::kInsert) {
            context->PushNumberKey(number, start_position);
          }
        }
      } else {
        auto position = start_position;
        /* need to check findness of all numbers in the range */
        for (size_t j = i; j < i + consecutive_count; j++) {
          auto number = number_offset_pairs[j].first;
          position = BinarySearch(number, position, end_position);
          if (IsFound(number, position)) {
            offsets.push_back(number_offset_pairs[j].second);
            if (context_type == ContextType::kDelete) {
              context->PushNumberKey(number, position);
            }
          } else if (context_type == ContextType::kInsert) {
            context->PushNumberKey(number, position);
          }
        }
      }
    }
    i += consecutive_count;
  }

  return scan_result;
}

Status NumberKeyFlatArray::Insert(
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs) {
  ARROW_ASSIGN_OR_RAISE(auto scan_result,
                        UniqueScan(number_offset_pairs, ContextType::kInsert));
  return InsertWithContext(scan_result.number_key_set_context);
}

Status NumberKeyFlatArray::InsertWithContext(
    std::shared_ptr<NumberKeySetContext> number_key_set_context) {
  auto context = std::static_pointer_cast<NumberKeyFlatArrayContext>(
      number_key_set_context);
  auto& numbers = context->GetNumbers();
  auto& positions = context->GetPositions();
  int32_t adjust = 0;
  for (size_t i = 0; i < numbers.size(); i++) {
    auto number = numbers[i];
    auto position = positions[i] + adjust;
    Insert(number, position, adjust);
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
             "NumberKeyFlatArray::InsertWithContext: %d keys are inserted. "
             "array length is changed from %d to %d",
             numbers.size(), number_array_.size() - adjust,
             number_array_.size());
  return Status::Ok();
}

void NumberKeyFlatArray::Insert(uint32_t number, size_t position,
                                int32_t& adjust) {
  /* number must not exist in the array */
  constexpr uint8_t kConsecutiveWithPrev = 1;
  constexpr uint8_t kConsecutiveWithNext = 2;
  constexpr uint8_t kConsecutiveWithBoth = 3;
  uint8_t consecutive_with = 0;
  if (position > 0 && number_array_[position - 1] + 1 == number) {
    consecutive_with |= kConsecutiveWithPrev;
  }
  if (position < number_array_.size() &&
      number_array_[position] == number + 1) {
    consecutive_with |= kConsecutiveWithNext;
  }
  switch (consecutive_with) {
    case kConsecutiveWithBoth: {
      // Example: [1,5][7,10] -> insert 6 -> [1,10] (merge two ranges)
      // When inserting a number that connects two existing ranges, merge them
      constexpr uint8_t kPrevIsRange = 1;
      constexpr uint8_t kNextIsRange = 2;
      constexpr uint8_t kBothAreRange = 3;
      uint8_t is_range = 0;
      if (IsRangeEnd(position - 1)) {
        is_range |= kPrevIsRange;
      }
      if (IsRangeStart(position)) {
        is_range |= kNextIsRange;
      }
      switch (is_range) {
        case kBothAreRange: {
          // Example: [1,5][7,10] -> insert 6 -> [1,10] (remove both range
          // markers)
          // both range markers are removed, so adjust is decremented by 2
          number_array_.erase(number_array_.begin() + position - 1,
                              number_array_.begin() + position + 1);
          is_range_end_bitmap_.erase(
              is_range_end_bitmap_.begin() + position - 1,
              is_range_end_bitmap_.begin() + position + 1);
          adjust -= 2;
          break;
        }
        case kPrevIsRange: {
          // Example: [1,5] 7 -> insert 6 -> [1,7] (merge with previous range)
          // previous element(5) is removed, so adjust is decremented by 1
          number_array_[position - 1] = number_array_[position];
          number_array_.erase(number_array_.begin() + position);
          is_range_end_bitmap_.erase(is_range_end_bitmap_.begin() + position);
          adjust--;
          break;
        }
        case kNextIsRange: {
          // Example: 5 [7,10] -> insert 6 -> [5,10] (merge with next range)
          // next element(7) is removed, so adjust is decremented by 1
          number_array_[position] = number_array_[position - 1];
          number_array_.erase(number_array_.begin() + position - 1);
          is_range_end_bitmap_.erase(is_range_end_bitmap_.begin() + position -
                                     1);
          adjust--;
          break;
        }
        default: {
          // Example: 5 7 -> insert 6 -> [5,7] (create new range)
          // element count is not changed
          is_range_end_bitmap_[position] = true;
          break;
        }
      }
      break;
    }
    case kConsecutiveWithPrev: {
      if (IsRangeEnd(position - 1)) {
        // Example: [1,5] 10 -> insert 6 -> [1,6] 10 (extend previous range)
        // element count is not changed
        number_array_[position - 1] = number;
      } else {
        // Example: 1 5 -> insert 2 -> [1,2] 5 (create range with previous)
        // one element(2) is inserted, so adjust is incremented by 1
        number_array_.insert(number_array_.begin() + position, number);
        is_range_end_bitmap_.insert(is_range_end_bitmap_.begin() + position,
                                    true);
        adjust++;
      }
      break;
    }
    case kConsecutiveWithNext: {
      if (IsRangeStart(position)) {
        // Example: 1 [7,10] -> insert 6 -> 1 [6,10] (extend next range)
        // element count is not changed
        number_array_[position] = number;
      } else {
        // Example: 1 5 -> insert 4 -> 1 [4,5] (create range with next)
        // one element(4) is inserted, so adjust is incremented by 1
        number_array_.insert(number_array_.begin() + position, number);
        is_range_end_bitmap_.insert(is_range_end_bitmap_.begin() + position + 1,
                                    true);
        adjust++;
      }
      break;
    }
    default: {
      // Example: 1 5 10 -> insert 7 -> 1 5 7 10 (insert standalone number)
      // one element(7) is inserted, so adjust is incremented by 1
      number_array_.insert(number_array_.begin() + position, number);
      is_range_end_bitmap_.insert(is_range_end_bitmap_.begin() + position,
                                  false);
      adjust++;
    }
  }
}

Status NumberKeyFlatArray::Delete(
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs) {
  ARROW_ASSIGN_OR_RAISE(auto scan_result,
                        UniqueScan(number_offset_pairs, ContextType::kDelete));
  return DeleteWithContext(scan_result.number_key_set_context);
}

Status NumberKeyFlatArray::DeleteWithContext(
    std::shared_ptr<NumberKeySetContext> number_key_set_context) {
  auto context = std::static_pointer_cast<NumberKeyFlatArrayContext>(
      number_key_set_context);
  auto& numbers = context->GetNumbers();
  auto& positions = context->GetPositions();
  int32_t adjust = 0;
  for (size_t i = numbers.size() - 1; i < numbers.size(); i--) {
    auto number = numbers[i];
    auto position = positions[i];
    Delete(number, position, adjust);
  }
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
             "NumberKeyFlatArray::DeleteWithContext: "
             "%d keys are deleted. array length is changed from %d to %d",
             numbers.size(), number_array_.size() - adjust,
             number_array_.size());
  return Status::Ok();
}

void NumberKeyFlatArray::Delete(uint32_t number, size_t position,
                                int32_t& adjust) {
  /* number must exist in the array */
  if (IsRangeStart(position)) {
    /* number must be same as the start of the range */
    // Example: [1,5] -> delete 1 -> [2,5]
    // element count is not changed
    number_array_[position]++;
    if (number_array_[position] == number_array_[position + 1]) {
      // Example: [1,2] -> delete 1 -> [2,2] -> 2 (convert range to single
      // number) element count is decremented by 1
      number_array_.erase(number_array_.begin() + position + 1);
      is_range_end_bitmap_.erase(is_range_end_bitmap_.begin() + position + 1);
      adjust--;
    }
  } else if (IsRangeEnd(position)) {
    /* number is included in the range */
    if (number == number_array_[position]) {
      /* number is the end of the range */
      // Example: [1,5] -> delete 5 -> [1,4]
      // element count is not changed
      number_array_[position]--;
      if (number_array_[position] == number_array_[position - 1]) {
        // Example: [1,2] -> delete 2 -> [1,1] -> 1 (convert range to single
        // number) element count is decremented by 1
        number_array_.erase(number_array_.begin() + position);
        is_range_end_bitmap_.erase(is_range_end_bitmap_.begin() + position);
        adjust--;
      }
    } else {
      /* number is the middle of the range */
      /* split the range into two ranges */
      // Example: [1,5] -> delete 3 -> [1,2][4,5] (split range into two)
      constexpr uint8_t kPrevWillBeRange = 1;
      constexpr uint8_t kNextWillBeRange = 2;
      constexpr uint8_t kBothWillBeRange = 3;
      uint8_t will_be_range = 0;
      if (number_array_[position - 1] < number - 1) {
        will_be_range |= kPrevWillBeRange;
      }
      if (number_array_[position] > number + 1) {
        will_be_range |= kNextWillBeRange;
      }
      switch (will_be_range) {
        case kBothWillBeRange: {
          // Example: [1,5] -> delete 3 -> [1,2][4,5] (create two separate
          // ranges) Insert two new elements: (number-1) and (number+1)
          std::vector<uint32_t> insert_numbers(2);
          std::vector<bool> insert_is_range_end_bitmap(2);
          insert_numbers[0] = number - 1;
          insert_numbers[1] = number + 1;
          insert_is_range_end_bitmap[0] = true;
          insert_is_range_end_bitmap[1] = false;
          number_array_.insert(number_array_.begin() + position,
                               insert_numbers.begin(), insert_numbers.end());
          is_range_end_bitmap_.insert(is_range_end_bitmap_.begin() + position,
                                      insert_is_range_end_bitmap.begin(),
                                      insert_is_range_end_bitmap.end());
          adjust += 2;
          break;
        }
        case kPrevWillBeRange: {
          // Example: [1,5] -> delete 4 -> [1,3] 5
          // Insert one new element: (number-1)
          number_array_.insert(number_array_.begin() + position, number - 1);
          is_range_end_bitmap_.insert(
              is_range_end_bitmap_.begin() + position + 1, false);
          adjust++;
          break;
        }
        case kNextWillBeRange: {
          // Example: [1,5] -> delete 2 -> 1 [3,5]
          // Insert one new element: (number+1)
          number_array_.insert(number_array_.begin() + position, number + 1);
          is_range_end_bitmap_.insert(is_range_end_bitmap_.begin() + position,
                                      false);
          adjust++;
          break;
        }
        default: {
          // Example: [1,3] -> delete 2 -> 1 3 (convert range to two single
          // numbers) element count is not changed
          is_range_end_bitmap_[position] = false;
          break;
        }
      }
    }
  } else {
    /* number must be same as the standalone number */
    // Example: 1 5 10 -> delete 5 -> 1 10
    // element count is decremented by 1
    number_array_.erase(number_array_.begin() + position);
    is_range_end_bitmap_.erase(is_range_end_bitmap_.begin() + position);
    adjust--;
  }
}

size_t NumberKeyFlatArray::GetRangeAt(
    size_t position, std::pair<uint32_t, uint32_t>& range) const {
  if (position >= number_array_.size()) {
    /* special case: end of the array */
    range.first = UINT32_MAX;
    range.second = UINT32_MAX;
    return position;
  } else if (position < number_array_.size() - 1 &&
             is_range_end_bitmap_[position + 1]) {
    /* case: range */
    range.first = number_array_[position];
    range.second =
        number_array_[position + 1] + 1; /* +1 because end is exclusive */
    return position + 2;
  }
  /* case: single number */
  range.first = number_array_[position];
  range.second = number_array_[position] + 1; /* +1 because end is exclusive */
  return position + 1;
}

std::string NumberKeyFlatArray::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < number_array_.size();) {
    if (i < number_array_.size() - 1) {
      if (is_range_end_bitmap_[i + 1]) {
        ss << "[" << number_array_[i] << ", " << number_array_[i + 1] << "]";
        i += 2;
      } else {
        ss << number_array_[i];
        i++;
      }
    } else {
      ss << number_array_[i];
      i++;
    }
    if (i < number_array_.size()) {
      ss << ", ";
    }
  }
  return ss.str();
}

FileKeyMap::FileKeyMap() : number_key_sets_() {}

FileKeyMap::FileKeyMap(const std::string& file_path) : number_key_sets_() {
  uint8_t* buffer;
  size_t file_size;
  file_size = std::filesystem::file_size(file_path);
  ARROW_CAST_OR_NULL(buffer, uint8_t*, AllocateAligned(64, file_size));
  if (buffer == nullptr) {
    throw std::invalid_argument("Failed to allocate memory");
  }
  auto status = ReadFrom(file_path, buffer, file_size, 0);
  if (!status.ok()) {
    throw std::invalid_argument(status.ToString());
  }

  uint64_t buffer_offset = 0;
  while (buffer_offset < file_size) {
    uint64_t filename_length;
    buffer_offset += GetLengthFrom(buffer + buffer_offset, filename_length);
    std::string file_name(reinterpret_cast<char*>(buffer + buffer_offset),
                          filename_length);
    buffer_offset += filename_length;
    auto number_key_set =
        std::make_shared<NumberKeyFlatArray>(buffer, buffer_offset);
    number_key_sets_.emplace(file_name, number_key_set);
  }
  DeallocateAligned(buffer, file_size);
}

Status FileKeyMap::Save(const std::string& file_path) const {
  std::vector<uint8_t> buffer;
  uint64_t buffer_offset = 0;
  /* format
   * filename | serialized number_key_set | filename | serialized number_key_set
   * | ... */
  for (auto& [file_name, number_key_set] : number_key_sets_) {
    uint64_t filename_length = file_name.length();
    uint64_t length_bytes = ComputeBytesFor(filename_length);
    buffer.resize(buffer.size() + length_bytes + filename_length);
    buffer_offset +=
        PutLengthTo(buffer.data() + buffer_offset, filename_length);
    memcpy(buffer.data() + buffer_offset, file_name.data(), filename_length);
    buffer_offset += filename_length;
    auto status = number_key_set->SerializeInto(buffer, buffer_offset);
    if (!status.ok()) {
      return status;
    }
  }
  /* save buffer to file */
  auto status = WriteTo(file_path, buffer.data(), buffer_offset, 0);
  if (!status.ok()) {
    return status;
  }

  return status;
}

arrow::Result<PkScanResult> FileKeyMap::UniqueScan(
    const std::string& file_name,
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
    ContextType context_type) {
  auto iter = number_key_sets_.find(file_name);
  if (iter == number_key_sets_.end()) {
    number_key_sets_.emplace(file_name, std::make_shared<NumberKeyFlatArray>());
  }
  auto& number_key_set = number_key_sets_[file_name];
  return number_key_set->UniqueScan(number_offset_pairs, context_type);
}

Status FileKeyMap::Insert(
    const std::string& file_name,
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs) {
  auto iter = number_key_sets_.find(file_name);
  if (iter == number_key_sets_.end()) {
    number_key_sets_.emplace(file_name, std::make_shared<NumberKeyFlatArray>());
  }
  auto& number_key_set = number_key_sets_[file_name];
  return number_key_set->Insert(number_offset_pairs);
}

Status FileKeyMap::InsertWithContext(
    const std::string& file_name,
    std::shared_ptr<NumberKeySetContext> number_key_set_context) {
  auto iter = number_key_sets_.find(file_name);
  if (iter == number_key_sets_.end()) {
    number_key_sets_.emplace(file_name, std::make_shared<NumberKeyFlatArray>());
  }
  auto& number_key_set = number_key_sets_[file_name];
  return number_key_set->InsertWithContext(number_key_set_context);
}

Status FileKeyMap::Delete(
    const std::string& file_name,
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs) {
  auto iter = number_key_sets_.find(file_name);
  if (iter == number_key_sets_.end()) {
    return Status::Ok();
  }
  auto& number_key_set = number_key_sets_[file_name];
  return number_key_set->Delete(number_offset_pairs);
}

Status FileKeyMap::DeleteWithContext(
    const std::string& file_name,
    std::shared_ptr<NumberKeySetContext> number_key_set_context) {
  auto iter = number_key_sets_.find(file_name);
  if (iter == number_key_sets_.end()) {
    number_key_sets_.emplace(file_name, std::make_shared<NumberKeyFlatArray>());
  }
  auto& number_key_set = number_key_sets_[file_name];
  return number_key_set->DeleteWithContext(number_key_set_context);
}

std::string FileKeyMap::ToString() const {
  std::stringstream ss;
  for (auto& [file_name, number_key_set] : number_key_sets_) {
    ss << file_name << ": " << number_key_set->ToString() << std::endl;
  }
  return ss.str();
}

PrimaryKeyIndex::PrimaryKeyIndex()
    : file_key_map_(std::make_shared<FileKeyMap>()) {}

PrimaryKeyIndex::PrimaryKeyIndex(const std::string& file_path)
    : file_key_map_(std::make_shared<FileKeyMap>(file_path)) {}

Status PrimaryKeyIndex::Save(const std::string& file_path) const {
  return file_key_map_->Save(file_path);
}

arrow::Result<PkScanResult> PrimaryKeyIndex::UniqueScan(
    const std::string& file_name,
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs,
    ContextType context_type) {
  return file_key_map_->UniqueScan(file_name, number_offset_pairs,
                                   context_type);
}

Status PrimaryKeyIndex::Insert(
    const std::string& file_name,
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs) {
  return file_key_map_->Insert(file_name, number_offset_pairs);
}

Status PrimaryKeyIndex::InsertWithContext(
    const std::string& file_name,
    std::shared_ptr<NumberKeySetContext> number_key_set_context) {
  auto status =
      file_key_map_->InsertWithContext(file_name, number_key_set_context);

  return status;
}

Status PrimaryKeyIndex::Delete(
    const std::string& file_name,
    const std::vector<std::pair<uint32_t, uint32_t>>& number_offset_pairs) {
  return file_key_map_->Delete(file_name, number_offset_pairs);
}

Status PrimaryKeyIndex::DeleteWithContext(
    const std::string& file_name,
    std::shared_ptr<NumberKeySetContext> number_key_set_context) {
  return file_key_map_->DeleteWithContext(file_name, number_key_set_context);
}

std::string PrimaryKeyIndex::ToString() const {
  return file_key_map_->ToString();
}

NumberKeyHandle::NumberKeyHandle()
    : number_offset_pairs_(), number_key_set_context_(nullptr) {}

Status NumberKeyHandle::PushNumberAndOffset(const uint32_t number,
                                            const uint32_t offset) {
  if (!number_offset_pairs_.empty()) {
    auto& last_number_offset_pair = number_offset_pairs_.back();
    if (last_number_offset_pair.first == number ||
        last_number_offset_pair.second == offset) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "NumberKeyHandle::PushNumberAndOffset: "
                 "number %d or offset %d already exists. "
                 "last number: %d, last offset: %d. "
                 "This is NOT expected behavior!",
                 number, offset, last_number_offset_pair.first,
                 last_number_offset_pair.second);
      if (number == last_number_offset_pair.first) {
        std::stringstream ss;
        ss << "Inserting same primary key at once is not allowed. "
           << "file name: " << file_name_ << ", number: " << number;
        return Status::InvalidArgument(ss.str());
      } else {
        /* must not be here. */
        std::stringstream ss;
        ss << "Internal Error: same offsets are found during collecting "
              "primary keys. "
           << "file name: " << file_name_ << ", number: " << number
           << ", offset: " << offset;
        return Status::InvalidArgument(ss.str());
      }
    }
  }
  number_offset_pairs_.emplace_back(number, offset);
  return Status::Ok();
}

void NumberKeyHandle::SetNumberKeySetContext(
    std::shared_ptr<NumberKeySetContext>& number_key_set_context) {
  number_key_set_context_ = number_key_set_context;
}

PrimaryKeyHandle::PrimaryKeyHandle(std::shared_ptr<PrimaryKeyIndex> pk_index)
    : pk_index_(pk_index) {}

bool PrimaryKeyHandle::IsEmpty() const { return number_key_handles_.empty(); }

Status PrimaryKeyHandle::ClassifyPrimaryKey(const uint32_t offset,
                                            const PrimaryKey& key) {
  auto [iterator, created] = number_key_handles_.try_emplace(key.file_name());
  auto& [file_name, number_key_handle] = *iterator;
  if (created) {
    number_key_handle.SetFileName(file_name);
  }

  auto status = number_key_handle.PushNumberAndOffset(key.number(), offset);
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

PrimaryKeyHandleForInsert::PrimaryKeyHandleForInsert(
    std::shared_ptr<PrimaryKeyIndex> pk_index)
    : PrimaryKeyHandle(pk_index) {}

Status PrimaryKeyHandleForInsert::Initialize(
    std::shared_ptr<arrow::Array> composite_key_array) {
  if (composite_key_array->type_id() == arrow::Type::STRING) {
    return Initialize(
        std::static_pointer_cast<arrow::StringArray>(composite_key_array));
  } else if (composite_key_array->type_id() == arrow::Type::LARGE_STRING) {
    return Initialize(
        std::static_pointer_cast<arrow::LargeStringArray>(composite_key_array));
  } else {
    return Status::InvalidArgument(
        "primary key column input array type must be string or large string: " +
        composite_key_array->type()->ToString());
  }
}

Status PrimaryKeyHandleForInsert::Initialize(
    std::shared_ptr<arrow::LargeStringArray> composite_key_array) {
  auto length = composite_key_array->length();
  /* build input ranges */
  for (int i = 0; i < composite_key_array->length(); i++) {
    auto composite_key = composite_key_array->Value(i);
    ARROW_ASSIGN_OR_RAISE(auto primary_key, PrimaryKey::Build(composite_key));
    auto status = ClassifyPrimaryKey(i, primary_key);
    if (!status.ok()) {
      return status;
    }
  }
  valid_offset_array_.Initialize(0, length - 1);
  return Status::Ok();
}

Status PrimaryKeyHandleForInsert::Initialize(
    std::shared_ptr<arrow::StringArray> composite_key_array) {
  auto length = composite_key_array->length();
  /* build input ranges */
  for (int i = 0; i < composite_key_array->length(); i++) {
    auto composite_key = composite_key_array->Value(i);
    ARROW_ASSIGN_OR_RAISE(auto primary_key, PrimaryKey::Build(composite_key));
    auto status = ClassifyPrimaryKey(i, primary_key);
    if (!status.ok()) {
      return status;
    }
  }
  valid_offset_array_.Initialize(0, length - 1);
  return Status::Ok();
}

Status PrimaryKeyHandleForInsert::ApplyInvalidOffsets(
    std::vector<uint32_t>& invalid_offsets) {
  int32_t dummy_adjust = 0;
  for (size_t i = 0; i < invalid_offsets.size(); i++) {
    auto offset = invalid_offsets[i];
    auto position = valid_offset_array_.BinarySearch(offset);
    valid_offset_array_.Delete(offset, position, dummy_adjust);
  }
  return Status::Ok();
}

Status PrimaryKeyHandleForInsert::CheckUniqueViolation() {
  for (auto& [file_name, number_key_handle] : number_key_handles_) {
    auto& number_offset_pairs = number_key_handle.GetNumberOffsetPairs();
    std::sort(number_offset_pairs.begin(), number_offset_pairs.end(),
              [](const std::pair<uint32_t, uint32_t>& a,
                 const std::pair<uint32_t, uint32_t>& b) {
                return a.first < b.first;
              });
    ARROW_ASSIGN_OR_RAISE(auto pk_scan_result,
                          pk_index_->UniqueScan(file_name, number_offset_pairs,
                                                ContextType::kInsert));
    number_key_handle.SetNumberKeySetContext(
        pk_scan_result.number_key_set_context);
    if (!pk_scan_result.offsets.empty()) {
      auto status = ApplyInvalidOffsets(pk_scan_result.offsets);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

Status PrimaryKeyHandleForInsert::ApplyToIndex() {
  for (auto& [file_name, number_key_handle] : number_key_handles_) {
    auto& number_key_set_context = number_key_handle.GetNumberKeySetContext();
    auto status =
        pk_index_->InsertWithContext(file_name, number_key_set_context);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

std::pair<uint32_t, uint32_t>
PrimaryKeyHandleForInsert::GetNextValidOffsetRange() {
  std::pair<uint32_t, uint32_t> range;
  current_position_ = valid_offset_array_.GetRangeAt(current_position_, range);
  return range;
}

bool PrimaryKeyHandleForInsert::IsCompleted(
    const std::pair<uint32_t, uint32_t>& range) {
  return range.first == UINT32_MAX;
}

PrimaryKeyHandleForDelete::PrimaryKeyHandleForDelete(
    std::shared_ptr<PrimaryKeyIndex> pk_index)
    : PrimaryKeyHandle(pk_index) {}

Status PrimaryKeyHandleForDelete::Initialize(
    std::shared_ptr<arrow::Array> composite_key_array,
    arrow::Type::type composite_key_type,
    std::shared_ptr<arrow::Array> filtered_indices) {
  auto filtered_indices_int64 =
      std::static_pointer_cast<arrow::Int64Array>(filtered_indices);
  std::string_view primary_key;
  constexpr uint8_t kLargeStringType = 1;
  constexpr uint8_t kStringType = 2;
  constexpr uint8_t kUnknown = 0;
  uint8_t pk_type = kUnknown;
  if (composite_key_type == arrow::Type::LARGE_STRING) {
    pk_type = kLargeStringType;
  } else if (composite_key_type == arrow::Type::STRING) {
    pk_type = kStringType;
  }
  if (pk_type == kUnknown) {
    return arrow::Status::Invalid(
        "Primary key type must be STRING or LARGE_STRING.");
  }

  switch (pk_type) {
    case kLargeStringType: {
      auto large_string_array =
          std::static_pointer_cast<arrow::LargeStringArray>(
              composite_key_array);
      for (int64_t i = 0; i < filtered_indices->length(); ++i) {
        primary_key =
            large_string_array->Value(filtered_indices_int64->Value(i));
        CollectPrimaryKey(primary_key);
      }
      break;
    }
    case kStringType: {
      auto string_array =
          std::static_pointer_cast<arrow::StringArray>(composite_key_array);
      for (int64_t i = 0; i < filtered_indices->length(); ++i) {
        primary_key = string_array->Value(filtered_indices_int64->Value(i));
        CollectPrimaryKey(primary_key);
      }
      break;
    }
  }
  return Status::Ok();
}

Status PrimaryKeyHandleForDelete::CollectPrimaryKey(
    std::string_view& composite_key) {
  ARROW_ASSIGN_OR_RAISE(auto primary_key, PrimaryKey::Build(composite_key));
  /* offset is dummy value */
  auto status = ClassifyPrimaryKey(primary_key.number(), primary_key);
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

Status PrimaryKeyHandleForDelete::ApplyToIndex() {
  for (auto& [file_name, number_key_handle] : number_key_handles_) {
    auto& number_offset_pairs = number_key_handle.GetNumberOffsetPairs();
    std::sort(number_offset_pairs.begin(), number_offset_pairs.end(),
              [](const std::pair<uint32_t, uint32_t>& a,
                 const std::pair<uint32_t, uint32_t>& b) {
                return a.first < b.first;
              });
    auto status = pk_index_->Delete(file_name, number_offset_pairs);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}
}  // namespace vdb
