#pragma once

#include <string_view>

#include <arrow/api.h>

#include "vdb/data/checker_registry.hh"

namespace vdb {

class MetadataChecker {
 public:
  MetadataChecker() = default;

  virtual ~MetadataChecker() = default;

  virtual const char* GetKey() const = 0;

  virtual arrow::Status Check(const std::string& value,
                              std::shared_ptr<arrow::Schema> schema) = 0;
};

/*
 * Macro to define metadata/checker and automatically register a MetadataChecker
 * class
 *
 * It also defines the static MetadataCheckerRegistrar instance
 *
 * During the program startup, when MetadataCheckerRegistrar is initialized,
 * it will register the checker to the registry with the given key and factory
 * function
 *
 * Usage: DEFINE_METADATA(KeyName, KeyString) { implementation... };
 */
#define DEFINE_METADATA(KeyName, KeyString)                              \
  constexpr const char* k##KeyName##Key = KeyString;                     \
  constexpr const char* Get##KeyName##Key() { return k##KeyName##Key; }  \
                                                                         \
  class KeyName##Checker : public MetadataChecker {                      \
   public:                                                               \
    KeyName##Checker() = default;                                        \
    virtual ~KeyName##Checker() = default;                               \
                                                                         \
    const char* GetKey() const override { return KeyString; }            \
                                                                         \
    arrow::Status Check(const std::string& value,                        \
                        std::shared_ptr<arrow::Schema> schema) override; \
  };                                                                     \
                                                                         \
  namespace {                                                            \
                                                                         \
  static MetadataCheckerRegistrar KeyName##_registrar(KeyString, []() {  \
    return std::make_shared<KeyName##Checker>();                         \
  });                                                                    \
  }                                                                      \
                                                                         \
/* Implementation will be defined in metadata_checker.cc */

// Define existing checker classes using the macro
DEFINE_METADATA(SegmentationInfo, "segmentation_info")
DEFINE_METADATA(TableName, "table name")
DEFINE_METADATA(ActiveSetSizeLimit, "active_set_size_limit")
DEFINE_METADATA(IndexInfo, "index_info")
DEFINE_METADATA(MaxThreads, "max_threads")

// To add a new metadata and checker, simply use the macro:
// DEFINE_METADATA(NewCheckerName, "new_key_string")

[[maybe_unused]] static bool IsReservedMetadataKey(std::string_view key) {
  return vdb::GetCheckerRegistry().find(std::string(key)) !=
         vdb::GetCheckerRegistry().end();
}

[[maybe_unused]] static bool IsImmutable(std::string_view key) {
  return key == vdb::GetActiveSetSizeLimitKey();
}

}  // namespace vdb