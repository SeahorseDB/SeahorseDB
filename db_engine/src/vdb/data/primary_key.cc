#include <stdexcept>

#include "vdb/common/defs.hh"
#include "vdb/common/util.hh"
#include "vdb/data/primary_key.hh"

namespace vdb {

PrimaryKey::PrimaryKey(const std::string& file_name, uint64_t number)
    : file_name_(file_name), number_(number) {}

PrimaryKey::PrimaryKey(std::string_view file_name, uint64_t number)
    : file_name_(std::string(file_name)), number_(number) {}

arrow::Result<PrimaryKey> PrimaryKey::Build(
    const std::string_view& composite_key) {
  size_t pos = composite_key.find(kRS);
  if (pos == std::string::npos) {
    return arrow::Status::Invalid("Invalid composite key format: " +
                                  std::string(composite_key));
  }
  std::string file_name;
  file_name = std::string(composite_key.substr(0, pos));
  ARROW_ASSIGN_OR_RAISE(uint64_t number,
                        stoui64(std::string(composite_key.substr(pos + 1))));
  return PrimaryKey(file_name, number);
}

}  // namespace vdb
