#pragma once

#include <string_view>
#include <vector>

#include <arrow/api.h>

#include "vdb/common/status.hh"
#include "vdb/common/memory_allocator.hh"

using sds = char *;

namespace vdb {

std::string Trim(std::string_view sv);

// Tokenize doesn't return empty token
std::vector<std::string_view> Tokenize(const std::string_view &target_string,
                                       const std::string_view &delimiter);
std::vector<std::string_view> Tokenize(const std::string &target_string,
                                       const char delimiter);
std::vector<std::string_view> Tokenize(const std::string_view &target_string,
                                       const char delimiter);
std::vector<std::string_view> Tokenize(const std::string &target_string,
                                       const std::string &delimiter);
std::vector<std::string_view> Tokenize(const std::string &target_string);
std::vector<std::string_view> Tokenize(const std::string_view &target_string);

std::string_view GetTokenFrom(const std::string_view &target_string,
                              const char delimiter, const int id);

// Split returns empty token also
std::vector<std::string_view> Split(std::string_view str, std::string_view del);

std::vector<std::string_view> Split(std::string_view str, const char del);

std::string Join(const std::vector<std::string> &strings, const char delimiter);

std::string Join(const std::vector<std::string_view> &strings,
                 const char delimiter);

std::string Join(const std::vector<std::string> &strings,
                 std::string_view delimiter);

std::string Join(const std::vector<std::string_view> &strings,
                 std::string_view delimiter);

Status AppendBitsTo(vdb::vector<uint8_t> &sink, const size_t &sink_bit_count,
                    const uint8_t *source, const int64_t &source_bit_count);

Status AppendBitsTo(vdb::vector<uint8_t> &sink, const size_t &sink_bit_count,
                    const uint8_t *source, const int64_t source_bit_start,
                    const int64_t &source_bit_count);

template <typename CType>
Status AppendValuesTo(vdb::vector<CType> &sink, const CType *source,
                      const int64_t source_count) {
  sink.resize(sink.size() + source_count);
  std::copy(source, source + source_count, sink.end() - source_count);
  return Status::Ok();
}

Status AppendUniformBitsTo(vdb::vector<uint8_t> &sink,
                           const int64_t &sink_bit_count, const bool bit,
                           int64_t bit_count);
std::shared_ptr<arrow::Schema> ParseSchemaFrom(std::string &schema_str);

std::string TransformToLower(std::string str);
std::string TransformToUpper(std::string str);

constexpr uint32_t k6BitLength = 0;
constexpr uint32_t k14BitLength = 1;
constexpr uint32_t k32BitLength = 0x80;
constexpr uint32_t k64BitLength = 0x81;

/* distance function type */
template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

/* return value is bytes of length */
int32_t ComputeBytesFor(uint64_t length);
/* return value is bytes of length */
int32_t PutLengthTo(uint8_t *buffer, uint64_t length);
/* return value is bytes of length */
int32_t GetLengthFrom(uint8_t *buffer, uint64_t &length);

Status WriteTo(const std::string &file_path, const uint8_t *buffer,
               size_t bytes, size_t file_offset);
Status ReadFrom(const std::string &file_path, uint8_t *buffer, size_t bytes,
                size_t file_offset);
std::shared_ptr<arrow::Buffer> MakeBufferFromSds(const sds s);
std::shared_ptr<arrow::Buffer> MakeBuffer(const char *data, size_t size);
arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
MakeRecordBatchesFromSds(const sds s);

bool IsSupportedType(arrow::Type::type type_id);

inline size_t GetSearchSize(size_t top_k_size, size_t ef_search) {
  /* set k as max(k, ef_search). in hnsw algorithm, k is used for resizing final
   * result. so, set k as max(k, ef_search) and resizing it to k in caller
   * function. */
  return std::max(top_k_size, ef_search);
}

arrow::Result<bool> stobool(std::string_view sv);
arrow::Result<int32_t> stoi32(std::string_view sv);
arrow::Result<uint32_t> stoui32(std::string_view sv);
arrow::Result<int64_t> stoi64(std::string_view sv);
arrow::Result<uint64_t> stoui64(std::string_view sv);
arrow::Result<float> stof(std::string_view sv);
arrow::Result<double> stod(std::string_view sv);

arrow::Result<std::shared_ptr<arrow::FixedSizeListArray>> MakeDeepCopy(
    const std::shared_ptr<arrow::FixedSizeListArray> &array);

Status ReadLineFrom(const std::string &file_path, uint32_t line_number,
                    std::string &line);
Status ReadLinesFrom(const std::string &file_path, uint32_t start_line,
                     uint32_t count, std::vector<std::string> &lines);
Status ReadLinesFrom(const std::string &file_path,
                     const std::vector<uint32_t> &line_numbers,
                     std::vector<std::string> &lines);

Status WriteLinesTo(const std::string &file_path, uint32_t start_line,
                    const std::vector<std::string> &lines_to_write);

// Filter out internal columns from RecordBatch (keeps only user-visible
// columns).
arrow::Result<std::shared_ptr<arrow::RecordBatch>> FilterInternalColumns(
    const std::shared_ptr<arrow::RecordBatch> &record_batch);

// Filter out internal columns from a schema (keeps only user-visible fields).
arrow::Result<std::shared_ptr<arrow::Schema>> FilterInternalColumnsFromSchema(
    const std::shared_ptr<arrow::Schema> &schema);

// Rebuild a RecordBatch without internal columns by constructing a new Schema
std::shared_ptr<arrow::RecordBatch> RebuildRecordBatchWithoutInternalColumns(
    const std::shared_ptr<arrow::RecordBatch> &record_batch);

}  // namespace vdb
