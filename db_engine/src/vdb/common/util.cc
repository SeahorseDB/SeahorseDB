#include <cstdlib>
#include <filesystem>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <arpa/inet.h>

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/ipc/api.h>
#include <arrow/io/api.h>

#include "vdb/common/defs.hh"
#include "vdb/common/fd_manager.hh"
#include "vdb/common/status.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/util.hh"
#include "vdb/metrics/metrics_api.hh"

#ifdef __cplusplus
extern "C" {
#include "endianconv.h"
#include "server.h"
}
#endif

namespace vdb {

std::string Trim(std::string_view sv) {
  auto start = sv.find_first_not_of(" \t\n\r");
  if (start == std::string_view::npos) return {};
  auto end = sv.find_last_not_of(" \t\n\r");
  return std::string(sv.substr(start, end - start + 1));
}

std::vector<std::string_view> Tokenize(const std::string_view& target_string,
                                       const std::string_view& delimiter) {
  std::vector<std::string_view> tokens;
  auto prev = 0;
  auto pos = target_string.find(delimiter, prev);

  while (pos != std::string::npos) {
    auto token = target_string.substr(prev, pos - prev);
    if (token != "") {  // Ignore empty string
      tokens.push_back(token);
    }
    prev = pos + delimiter.length();
    pos = target_string.find(delimiter, prev);
  }

  auto last_token = target_string.substr(prev, std::string::npos);
  if (last_token != "") {
    tokens.push_back(last_token);
  }
  return tokens;
}

std::vector<std::string_view> Tokenize(const std::string_view& target_string,
                                       const char delimiter) {
  return Tokenize(std::string_view{target_string},
                  std::string_view(&delimiter, 1));
}

std::vector<std::string_view> Tokenize(const std::string& target_string,
                                       const char delimiter) {
  return Tokenize(std::string_view{target_string}, delimiter);
}

std::vector<std::string_view> Tokenize(const std::string& target_string,
                                       const std::string_view& delimiter) {
  return Tokenize(std::string_view{target_string}, delimiter);
}

std::vector<std::string_view> Tokenize(const std::string& target_string,
                                       const std::string& delimiter) {
  return Tokenize(std::string_view{target_string}, std::string_view{delimiter});
}

std::vector<std::string_view> Tokenize(const std::string_view& target_string) {
  return Tokenize(target_string, ' ');
}

std::vector<std::string_view> Tokenize(const std::string& target_string) {
  return Tokenize(std::string_view{target_string});
}

std::string_view GetTokenFrom(const std::string_view& target_string,
                              const char delimiter, const int id) {
  auto count = 0;
  auto prev = 0;
  auto pos = target_string.find(delimiter, prev);

  while (pos != std::string::npos) {
    if (count == id) return target_string.substr(prev, pos - prev);
    prev = pos + 1;
    pos = target_string.find(delimiter, prev);
    count++;
  }
  if (count == id) {
    return target_string.substr(prev, pos - prev);
  }

  return target_string.substr(0, 0);
}

std::vector<std::string_view> Split(std::string_view str,
                                    std::string_view del) {
  if (del.empty()) {
    return {str};
  }

  std::vector<std::string_view> tokens;
  tokens.reserve(std::count(str.begin(), str.end(), del[0]) + 1);

  size_t prev = 0;
  size_t pos = str.find(del, prev);

  while (pos != std::string::npos) {
    tokens.emplace_back(str.substr(prev, pos - prev));
    prev = pos + del.length();
    pos = str.find(del, prev);
  }

  tokens.emplace_back(str.substr(prev));
  return tokens;
}

std::vector<std::string_view> Split(std::string_view str, const char del) {
  return Split(str, std::string_view(&del, 1));
}

std::string Join(const std::vector<std::string>& strings, const char delimter) {
  size_t size = 0;
  for (auto& str : strings) {
    size += str.size() + 1;
  }
  std::string joined_string;
  joined_string.reserve(size);

  for (auto iter = strings.begin(); iter != strings.end(); iter++) {
    joined_string += *iter;
    if (iter != strings.end() - 1) joined_string += delimter;
  }

  return joined_string;
}

std::string Join(const std::vector<std::string_view>& strings,
                 const char delimter) {
  size_t size = 0;
  for (auto& str : strings) {
    size += str.size() + 1;
  }
  std::string joined_string;
  joined_string.reserve(size);

  for (auto iter = strings.begin(); iter != strings.end(); iter++) {
    joined_string += *iter;
    if (iter != strings.end() - 1) joined_string += delimter;
  }

  return joined_string;
}

std::string Join(const std::vector<std::string>& strings,
                 std::string_view delimiter) {
  size_t size = 0;
  for (const auto& str : strings) {
    size += str.size() + delimiter.size();
  }
  std::string joined_string;
  joined_string.reserve(size);

  for (auto iter = strings.begin(); iter != strings.end(); ++iter) {
    joined_string += *iter;
    if (std::next(iter) != strings.end()) joined_string += delimiter;
  }

  return joined_string;
}

std::string Join(const std::vector<std::string_view>& strings,
                 std::string_view delimiter) {
  size_t size = 0;
  for (const auto& str : strings) {
    size += str.size() + delimiter.size();
  }
  std::string joined_string;
  joined_string.reserve(size);

  for (auto iter = strings.begin(); iter != strings.end(); ++iter) {
    joined_string += *iter;
    if (std::next(iter) != strings.end()) joined_string += delimiter;
  }

  return joined_string;
}

Status AppendBitsTo(vdb::vector<uint8_t>& sink, const size_t& sink_bit_count,
                    const uint8_t* source, const int64_t& source_bit_count) {
  /* is there are more efficient way to bitwise shift and copy? */
  int8_t source_bit_count_in_last_byte = ((source_bit_count - 1) % 8) + 1;
  int64_t source_byte_count =
      (source_bit_count - source_bit_count_in_last_byte) / 8 +
      /* last byte */ 1;
  auto bit_pos = sink_bit_count % 8;
  if (bit_pos == 0) {
    return AppendValuesTo(sink, source, source_byte_count);
  } else {
    /* calculate next array length and reserve array size */
    size_t new_sink_bit_count = sink_bit_count + source_bit_count;
    size_t new_sink_byte_count =
        new_sink_bit_count / 8 + ((new_sink_bit_count % 8) > 0 ? 1 : 0);

    /* reserve 50% more spaces than needed */
    if (new_sink_byte_count > sink.capacity())
      sink.reserve(new_sink_byte_count * 3 / 2);

    /* determine how input byte is divided */
    uint8_t lower_part_size = 8 - bit_pos;
    uint8_t upper_part_size = bit_pos;

    /* if ALL BITS in LAST BYTES are included in LOWER part,
     * there is NO BITS in UPPER part to append. */
    bool append_last_upper_part =
        (source_bit_count_in_last_byte > lower_part_size);

    for (int64_t i = 0; i < source_byte_count; i++) {
      auto input_byte = source[i];
      /* append input_byte's LOWER part into array byte's UPPER position */
      uint8_t lower_part = (input_byte << (8 - lower_part_size));
      sink.back() |= lower_part;

      if ((i < source_byte_count - 1) || append_last_upper_part) {
        /* append input_byte's UPPER part into array byte's LOWER position */
        uint8_t upper_part = (input_byte >> (8 - upper_part_size));
        sink.emplace_back(upper_part);
      }
    }
  }
  return Status::Ok();
}

Status AppendBitsTo(vdb::vector<uint8_t>& sink, const size_t& sink_bit_count,
                    const uint8_t* source, const int64_t source_bit_pos,
                    const int64_t& source_bit_count) {
  int64_t source_byte_pos = source_bit_pos / 8;
  int64_t source_bit_count_in_first_byte = 8 - (source_bit_pos % 8);
  if (source_bit_count_in_first_byte < 8) {
    uint8_t first_byte =
        source[source_byte_pos] >> (8 - source_bit_count_in_first_byte);
    auto status = AppendBitsTo(sink, sink_bit_count, &first_byte,
                               source_bit_count_in_first_byte);
    if (!status.ok()) return status;

    int64_t remain_source_bit_count =
        source_bit_count - source_bit_count_in_first_byte;
    if (remain_source_bit_count > 0) {
      int64_t new_sink_bit_count =
          sink_bit_count + source_bit_count_in_first_byte;
      return AppendBitsTo(sink, new_sink_bit_count,
                          source + (source_byte_pos) + 1,
                          remain_source_bit_count);
    } else {
      return Status::Ok();
    }
  } else {
    return AppendBitsTo(sink, sink_bit_count, source + (source_byte_pos),
                        source_bit_count);
  }
}

Status AppendUniformBitsTo(vdb::vector<uint8_t>& sink,
                           const int64_t& sink_bit_count, const bool bit,
                           int64_t bit_count) {
  auto bit_pos = sink_bit_count % 8;
  if (bit_pos != 0) {
    if (bit) {
      sink.back() |= BitMask[bit_pos];
    }
    bit_count -= (8 - bit_pos);
    if (bit_count <= 0) {
      return Status::Ok();
    }
  }
  size_t remain_bit_count = bit_count % 8;
  size_t added_byte_count = bit_count / 8 + (remain_bit_count > 0 ? 1 : 0);
  uint8_t mask = (bit) ? UINT8_MAX : 0;
  sink.resize(sink.size() + added_byte_count, mask);
  if (bit && remain_bit_count > 0) {
    sink.back() &= ~BitMask[remain_bit_count];
  }
  return Status::Ok();
}

// Type string parser. Supported types should be listed here.
// Returns the shared pointer of arrow::DataType with the appropriate
// `type_string`. If not, returns nullptr.
std::shared_ptr<arrow::DataType> GetDataTypeFrom(
    const std::string_view& type_string_token) {
  auto type_string = Trim(type_string_token);

  if (type_string == "bool") return arrow::boolean();

  if (type_string == "int8") return arrow::int8();
  if (type_string == "int16") return arrow::int16();
  if (type_string == "int32") return arrow::int32();
  if (type_string == "int64") return arrow::int64();

  if (type_string == "uint8") return arrow::uint8();
  if (type_string == "uint16") return arrow::uint16();
  if (type_string == "uint32") return arrow::uint32();
  if (type_string == "uint64") return arrow::uint64();

  if (type_string == "float32") return arrow::float32();
  if (type_string == "float64") return arrow::float64();

  if (type_string == "string") return arrow::utf8();
  if (type_string == "large_string") return arrow::large_utf8();
  if (type_string == "char") return arrow::utf8();
  if (type_string == "utf8") return arrow::utf8();

  // "list[int8]" or "list[     int8  ]"
  if (type_string.size() > 5 && type_string.substr(0, 5) == "list[" &&
      type_string.substr(type_string.size() - 1, 1) == "]") {
    auto subtype_pos = 5;
    while (std::isspace(type_string[subtype_pos])) subtype_pos++;
    auto subtype_end_pos = subtype_pos + 1;
    while (std::isalnum(type_string[subtype_end_pos])) subtype_end_pos++;
    auto subtype_str =
        Trim(type_string.substr(subtype_pos, subtype_end_pos - subtype_pos));
    auto subtype = GetDataTypeFrom(subtype_str);
    if (subtype != nullptr) return arrow::list(subtype);
  }

  // "fixed_size_list[1024,float32]" or "fixed_size_list[  1024,    float32   ]"
  if (type_string.size() > 16 &&
      type_string.substr(0, 16) == "fixed_size_list[" &&
      type_string.substr(type_string.size() - 1, 1) == "]") {
    auto list_type_pos = type_string.find(',') + 1;
    if (list_type_pos == std::string::npos &&
        list_type_pos >= type_string.size()) {
      std::cerr << "Failed to parse fixed_size_list type: " << type_string
                << std::endl;
      return nullptr;
    }

    auto list_size_str = Trim(type_string.substr(16, list_type_pos - 16));
    if (!list_size_str.empty() && list_size_str.back() == ',') {
      list_size_str = list_size_str.substr(0, list_size_str.size() - 1);
      list_size_str = Trim(list_size_str);
    }
    auto list_size_result = vdb::stoi64(list_size_str);
    if (!list_size_result.ok()) {
      std::cerr << "Failed to parse fixed_size_list, type string: '"
                << type_string << "', list_size_str: '" << list_size_str << "'"
                << std::endl;
      return nullptr;
    }
    auto list_size = list_size_result.ValueUnsafe();
    while (std::isspace(type_string[list_type_pos])) list_type_pos++;
    auto list_type_end_pos = list_type_pos + 1;
    while (std::isalnum(type_string[list_type_end_pos])) list_type_end_pos++;
    auto list_type_str = Trim(
        type_string.substr(list_type_pos, list_type_end_pos - list_type_pos));
    auto subtype = GetDataTypeFrom(list_type_str);
    if (subtype != nullptr) return arrow::fixed_size_list(subtype, list_size);
  }

  std::cout << "returning nullptr" << std::endl;
  return nullptr;
}

// Returns the shared pointer of arrow::Schema with appropriate `schema_string`.
// e.g.) "ID Int32, Name String, Height Float32"
// e.g.) "ID String, Vector Fixed_Size_List[ 1024,  Float32  ], Attribute List[
// String ]" If not, returns nullptr.
std::shared_ptr<arrow::Schema> ParseSchemaFrom(std::string& schema_string) {
  std::transform(schema_string.begin(), schema_string.end(),
                 schema_string.begin(), tolower);
  std::vector<std::string_view> tokens;
  char delimiter = ',';
  size_t i = 0;
  while (i < schema_string.size()) {
    size_t j = i;
    bool subtype_flag = false;
    for (; j < schema_string.size(); j++) {
      if (schema_string[j] == '[') subtype_flag = true;
      if (schema_string[j] == ']') subtype_flag = false;
      if (!subtype_flag && schema_string[j] == delimiter) {
        break;
      }
    }
    tokens.emplace_back(&schema_string[i], j - i);
    i = j + 1;
  }

  arrow::FieldVector fields;
  for (auto& token : tokens) {
    size_t begin_pos = 0;
    while (begin_pos < token.size()) {
      if (isspace(token[begin_pos]))
        begin_pos++;
      else
        break;
    }
    auto delim_pos = token.find(' ', begin_pos);
    if (delim_pos == std::string::npos) return nullptr;
    auto name = token.substr(begin_pos, delim_pos - begin_pos);

    // Find the position of the "not null" specifier, if present
    auto not_null_pos = token.find("not null", delim_pos + 1);
    auto type_end_pos = (not_null_pos != std::string::npos) ? not_null_pos - 1
                                                            : std::string::npos;

    auto type =
        GetDataTypeFrom(token.substr(delim_pos + 1, type_end_pos - delim_pos));
    if (type == nullptr) return nullptr;

    // Create the field with nullable set to false if "not null" is specified
    bool nullable = (not_null_pos == std::string::npos);
    fields.push_back(
        std::make_shared<arrow::Field>(std::string{name}, type, nullable));
  }

  auto schema = std::make_shared<arrow::Schema>(fields);

  return schema;
}

std::string TransformToLower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str;
}

std::string TransformToUpper(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return str;
}

int32_t ComputeBytesFor(uint64_t length) {
  int32_t length_bytes;
  if (length < (1 << 6)) {
    length_bytes = 1;
  } else if (length < (1 << 14)) {
    length_bytes = 2;
  } else if (length <= UINT32_MAX) {
    length_bytes = 1 + 4;
  } else {
    length_bytes = 1 + 8;
  }
  return length_bytes;
}

int32_t PutLengthTo(uint8_t* buffer, uint64_t length) {
  int32_t length_bytes;

  if (length < (1 << 6)) {
    /* Save a 6 bit len */
    buffer[0] = (length & 0xFF) | (k6BitLength << 6);
    length_bytes = 1;
  } else if (length < (1 << 14)) {
    /* Save a 14 bit len */
    buffer[0] = ((length >> 8) & 0xFF) | (k14BitLength << 6);
    buffer[1] = length & 0xFF;
    length_bytes = 2;
  } else if (length <= UINT32_MAX) {
    /* Save a 32 bit len */
    buffer[0] = k32BitLength;
    uint32_t len32 = htonl(length);
    memcpy(&buffer[1], &len32, 4);
    length_bytes = 1 + 4;
  } else {
    /* Save a 64 bit len */
    buffer[0] = k64BitLength;
    length = htonu64(length);
    memcpy(&buffer[1], &length, 8);
    length_bytes = 1 + 8;
  }

  return length_bytes;
}

int32_t GetLengthFrom(uint8_t* buffer, uint64_t& length) {
  int type;
  int32_t length_bytes;

  type = (buffer[0] & 0xC0) >> 6;
  if (type == k6BitLength) {
    /* Read a 6 bit len. */
    length = buffer[0] & 0x3F;
    length_bytes = 1;
  } else if (type == k14BitLength) {
    /* Read a 14 bit len. */
    length = ((buffer[0] & 0x3F) << 8) | buffer[1];
    length_bytes = 2;
  } else if (buffer[0] == k32BitLength) {
    /* Read a 32 bit len. */
    uint32_t raw_length;
    memcpy(&raw_length, &buffer[1], 4);
    length = ntohl(raw_length);
    length_bytes = 1 + 4;
  } else if (buffer[0] == k64BitLength) {
    /* Read a 64 bit len. */
    uint64_t raw_length;
    memcpy(&raw_length, &buffer[1], 8);
    length = ntohu64(raw_length);
    length_bytes = 1 + 8;
  } else {
    length_bytes = -1;
    SYSTEM_LOG(
        vdb::LogTopic::Unknown, LogLevel::kLogAlways,
        "Wrong Length Type is Found! %d "
        "(Correct Cases: (1 bytes)%d (2 bytes)%d (1+4 bytes)%d (1+9 bytes)%d)",
        (int)buffer[0], k6BitLength, k14BitLength, k32BitLength, k64BitLength);
  }

  return length_bytes;
}

Status WriteTo(const std::string& file_path, const uint8_t* buffer,
               size_t bytes, size_t file_offset) {
  metrics::ScopedDurationMetric write_io_latency(
      vdb::metrics::MetricIndex::WriteIoLatency);

  auto fd_handle = FdHandle(file_path, O_RDWR | O_CREAT, 0644);
  int fd = fd_handle.Get();
  if (fd < 0) {
    return Status::InvalidArgument(file_path + " Write Error - " +
                                   strerror(errno));
  }

  ssize_t written_size = pwrite(fd, buffer, bytes, file_offset);
  if (written_size < 0) {
    return Status::InvalidArgument(file_path + " Write Error - " +
                                   strerror(errno));
  }

  if (static_cast<size_t>(written_size) != bytes) {
    return Status::InvalidArgument(
        file_path +
        " Write Error - Written bytes is different from requested bytes.");
  }

  metrics::CollectValue(vdb::metrics::MetricIndex::WriteIoSize, bytes);

  return Status::Ok();
}

Status ReadFrom(const std::string& file_path, uint8_t* buffer, size_t bytes,
                size_t file_offset) {
  std::error_code ec;
  if (!std::filesystem::exists(file_path, ec)) {
    return Status::InvalidArgument(file_path + " Read Error - " + ec.message());
  }

  metrics::ScopedDurationMetric read_io_latency(
      vdb::metrics::MetricIndex::ReadIoLatency);
  auto fd_handle = FdHandle(file_path, O_RDONLY);
  int fd = fd_handle.Get();
  if (fd < 0) {
    return Status::InvalidArgument(file_path + " Read Error - " +
                                   strerror(errno));
  }

  ssize_t read_size = pread(fd, buffer, bytes, file_offset);
  if (read_size < 0) {
    return Status::InvalidArgument(file_path + " Read Error - " +
                                   strerror(errno));
  }

  if (static_cast<size_t>(read_size) != bytes) {
    return Status::InvalidArgument(
        file_path +
        " Read Error - Read bytes is different with requested bytes.");
  }
  metrics::CollectValue(vdb::metrics::MetricIndex::ReadIoSize, bytes);

  return Status::Ok();
}

// TODO: Refactor this function to use the FdHandle class

Status ReadLineFrom(const std::string& file_path, uint32_t line_number,
                    std::string& line) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return Status::InvalidArgument("Failed to open file: " + file_path);
  }

  // Move to target line
  for (uint32_t i = 0; i < line_number; i++) {
    if (!file.ignore(INT_MAX, '\n')) {
      return Status::InvalidArgument("Line number out of range");
    }
  }

  // Read the line
  if (!std::getline(file, line)) {
    return Status::InvalidArgument("Failed to read line");
  }

  return Status::Ok();
}

Status ReadLinesFrom(const std::string& file_path, uint32_t start_line,
                     uint32_t count, std::vector<std::string>& lines) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return Status::InvalidArgument("Failed to open file: " + file_path);
  }

  lines.clear();
  lines.reserve(count);

  // Move to start line
  for (uint32_t i = 0; i < start_line; i++) {
    if (!file.ignore(INT_MAX, '\n')) {
      return Status::InvalidArgument("Start line number out of range");
    }
  }

  // Read count lines
  std::string line;
  for (uint32_t i = 0; i < count; i++) {
    if (!std::getline(file, line)) {
      return Status::InvalidArgument("Failed to read line");
    }
    lines.push_back(line);
  }

  return Status::Ok();
}

Status ReadLinesFrom(const std::string& file_path,
                     const std::vector<uint32_t>& line_numbers,
                     std::vector<std::string>& lines) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return Status::InvalidArgument("Failed to open file: " + file_path);
  }

  lines.clear();
  lines.reserve(line_numbers.size());

  for (uint32_t line_number : line_numbers) {
    // Move to target line
    for (uint32_t i = 0; i < line_number; i++) {
      if (!file.ignore(INT_MAX, '\n')) {
        return Status::InvalidArgument("Line number out of range: " +
                                       std::to_string(line_number));
      }
    }

    // Read the line
    std::string line;
    if (!std::getline(file, line)) {
      return Status::InvalidArgument("Failed to read line");
    }
    lines.push_back(line);

    // Reset file pointer to beginning
    file.clear();
    file.seekg(0);
  }

  return Status::Ok();
}

Status WriteLinesTo(const std::string& file_path, uint32_t start_line,
                    const std::vector<std::string>& lines_to_write) {
  std::vector<std::string> lines;
  std::ifstream infile(file_path);
  if (infile.is_open()) {
    std::string line;
    while (std::getline(infile, line)) {
      lines.push_back(std::move(line));
    }
    infile.close();
  }

  if (lines.size() < start_line + lines_to_write.size()) {
    lines.resize(start_line + lines_to_write.size(), "");
  }

  for (size_t i = 0; i < lines_to_write.size(); ++i) {
    lines[start_line + i] = lines_to_write[i];
  }

  std::ofstream outfile(file_path, std::ios::trunc);
  if (!outfile.is_open()) {
    return Status::InvalidArgument("Failed to open file for writing: " +
                                   file_path);
  }

  for (const auto& l : lines) {
    outfile << l << '\n';
  }

  return Status::Ok();
}

std::shared_ptr<arrow::Buffer> MakeBufferFromSds(const sds s) {
  return MakeBuffer(s, sdslen(s));
}

std::shared_ptr<arrow::Buffer> MakeBuffer(const char* data, size_t size) {
  auto buffer = std::make_shared<arrow::Buffer>(
      reinterpret_cast<const uint8_t*>(data), size * sizeof(uint32_t));
  return buffer;
}

arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
MakeRecordBatchesFromSds(const sds s) {
  ARROW_ASSIGN_OR_RAISE(
      auto reader,
      arrow::ipc::RecordBatchStreamReader::Open(
          std::make_shared<arrow::io::BufferReader>(MakeBufferFromSds(s))));
  ARROW_ASSIGN_OR_RAISE(auto rbs, reader->ToRecordBatches());
  return rbs;
}

bool IsSupportedType(arrow::Type::type type_id) {
  return is_primitive(type_id) || is_string(type_id) || is_decimal(type_id) ||
         is_list(type_id);
}

// Type trait to map C++ types to Arrow types
template <typename T>
struct ArrowTypeMap;

template <>
struct ArrowTypeMap<int32_t> {
  using ArrowType = arrow::Int32Type;
  using ArrowScalar = arrow::Int32Scalar;
};

template <>
struct ArrowTypeMap<uint32_t> {
  using ArrowType = arrow::UInt32Type;
  using ArrowScalar = arrow::UInt32Scalar;
};

template <>
struct ArrowTypeMap<int64_t> {
  using ArrowType = arrow::Int64Type;
  using ArrowScalar = arrow::Int64Scalar;
};

template <>
struct ArrowTypeMap<uint64_t> {
  using ArrowType = arrow::UInt64Type;
  using ArrowScalar = arrow::UInt64Scalar;
};

template <>
struct ArrowTypeMap<float> {
  using ArrowType = arrow::FloatType;
  using ArrowScalar = arrow::FloatScalar;
};

template <>
struct ArrowTypeMap<double> {
  using ArrowType = arrow::DoubleType;
  using ArrowScalar = arrow::DoubleScalar;
};

// Template function implementation using type trait
template <typename T>
arrow::Result<T> string_to_number(std::string_view sv) {
  using ArrowType = typename ArrowTypeMap<T>::ArrowType;
  using ArrowScalar = typename ArrowTypeMap<T>::ArrowScalar;

  auto maybe_scalar = arrow::Scalar::Parse(std::make_shared<ArrowType>(), sv);
  if (!maybe_scalar.ok()) {
    return arrow::Status::Invalid("Failed to parse string to " +
                                  std::string(ArrowType::type_name()) + ": " +
                                  maybe_scalar.status().message());
  }

  auto scalar = maybe_scalar.ValueUnsafe();
  auto casted = std::dynamic_pointer_cast<ArrowScalar>(scalar);
  if (!casted) {
    return arrow::Status::Invalid(
        "Failed to parse string to " + std::string(ArrowType::type_name()) +
        ": " + "Because failed to cast scalar to appropriate type");
  }
  return static_cast<T>(casted->value);
}

arrow::Result<bool> stobool(std::string_view sv) {
  if (sv == "true" || sv == "1") {
    return true;
  } else if (sv == "false" || sv == "0") {
    return false;
  } else {
    return arrow::Status::Invalid("Failed to parse string to bool: " +
                                  std::string(sv));
  }
}

// Refactor existing functions to use the template
arrow::Result<int32_t> stoi32(std::string_view sv) {
  return string_to_number<int32_t>(sv);
}

arrow::Result<uint32_t> stoui32(std::string_view sv) {
  return string_to_number<uint32_t>(sv);
}

arrow::Result<int64_t> stoi64(std::string_view sv) {
  return string_to_number<int64_t>(sv);
}

arrow::Result<uint64_t> stoui64(std::string_view sv) {
  return string_to_number<uint64_t>(sv);
}

arrow::Result<float> stof(std::string_view sv) {
  return string_to_number<float>(sv);
}

arrow::Result<double> stod(std::string_view sv) {
  return string_to_number<double>(sv);
}

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>> MakeDeepCopy(
    const std::shared_ptr<arrow::FixedSizeListArray>& array) {
  auto memory_manager = arrow::CPUDevice::memory_manager(&vdb::arrow_pool);
  ARROW_ASSIGN_OR_RAISE(auto copied_array, array->CopyTo(memory_manager));
  return std::static_pointer_cast<arrow::FixedSizeListArray>(copied_array);
}
arrow::Result<std::shared_ptr<arrow::RecordBatch>> FilterInternalColumns(
    const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  if (record_batch == nullptr) {
    return record_batch;
  }

  auto schema = record_batch->schema();
  int n = schema->num_fields();

  // Quick check: if the last field is an internal column, filter all
  // internal columns
  if (n > 0 && IsInternalColumn(schema->field(n - 1)->name())) {
    std::vector<int> indices;
    // Collect indices of non-internal columns
    for (int i = 0; i < n; ++i) {
      if (!IsInternalColumn(schema->field(i)->name())) {
        indices.push_back(i);
      }
    }
    return record_batch->SelectColumns(indices);
  }

  // No internal columns, return original record batch
  return record_batch;
}

arrow::Result<std::shared_ptr<arrow::Schema>> FilterInternalColumnsFromSchema(
    const std::shared_ptr<arrow::Schema>& schema) {
  if (schema == nullptr) {
    return schema;
  }

  int n = schema->num_fields();

  // Quick check: if the last field is an internal column, filter all
  // internal columns
  if (n > 0 && IsInternalColumn(schema->field(n - 1)->name())) {
    std::vector<std::shared_ptr<arrow::Field>> visible_fields;
    // Collect non-internal fields
    for (int i = 0; i < n; ++i) {
      if (!IsInternalColumn(schema->field(i)->name())) {
        visible_fields.push_back(schema->field(i));
      }
    }
    return arrow::schema(visible_fields, schema->metadata());
  }

  // No internal columns, return original schema
  return schema;
}

std::shared_ptr<arrow::RecordBatch> RebuildRecordBatchWithoutInternalColumns(
    const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  if (record_batch == nullptr) {
    return record_batch;
  }

  auto schema = record_batch->schema();

  // Quick check: if the last field is an internal column, filter all
  // internal columns
  if (schema->num_fields() > 0 &&
      IsInternalColumn(schema->field(schema->num_fields() - 1)->name())) {
    std::vector<std::shared_ptr<arrow::Field>> visible_fields;
    std::vector<std::shared_ptr<arrow::Array>> visible_columns;

    for (int i = 0; i < schema->num_fields(); i++) {
      if (!IsInternalColumn(schema->field(i)->name())) {
        visible_fields.push_back(schema->field(i));
        visible_columns.push_back(record_batch->column(i));
      }
    }

    auto visible_schema =
        std::make_shared<arrow::Schema>(visible_fields, schema->metadata());
    return arrow::RecordBatch::Make(visible_schema, record_batch->num_rows(),
                                    visible_columns);
  }

  // No internal columns, return original record batch
  return record_batch;
}

}  // namespace vdb
