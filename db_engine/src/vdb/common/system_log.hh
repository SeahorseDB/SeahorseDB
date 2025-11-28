#pragma once

#include <cstdint>
#include <string_view>

#include "vdb/common/server_configuration.hh"

#ifdef __cplusplus
extern "C" {
#endif
void _serverLog(int level, const char* file, int line, const char* fmt, ...);
#ifdef __cplusplus
}
#endif

namespace vdb {
/* system log level
 *
 * MemDebug: Logs memory allocation and deallocation details.
 *
 * Debug: Used for frequent loop logging, may impact performance if it logs
 * system-level details without memory operation details.

 * Verbose: Logs detailed steps of a process, potentially
 * generating a large volume of logs for a single request.

 * Notice: (Default)
 * Logs warnings when errors occur. As the default, minimize use in frequently
 * executed code.

 * Always: Logs essential information that should not be turned
 * off, even when other logs are disabled, for critical issues. */

enum class LogLevel : uint8_t {
  kLogMemDebug = 0,
  kLogDebug,
  kLogVerbose,
  kLogNotice,
  kLogAlways,
  kLogLevelMax
};

constexpr const char* LogLevelName[] = {"MemDebug", "Debug", "Verbose",
                                        "Notice", "Always"};

enum class LogTopicOrder : uint8_t {
#define LogTopicMacro(topic_name, description) topic_name,
#include "input/system_log_topic.input"
#undef LogTopicMacro
  kLogTopicMax
};

enum class LogTopic : uint64_t {
#define LogTopicMacro(topic_name, description) \
  topic_name = (1ULL << static_cast<uint64_t>(LogTopicOrder::topic_name)),
#include "input/system_log_topic.input"
#undef LogTopicMacro
  kLogTopicMax = UINT64_MAX
};

inline uint64_t operator|(const LogTopic& a, const LogTopic& b) {
  return static_cast<uint64_t>(a) | static_cast<uint64_t>(b);
}

constexpr const char* LogTopicName[] = {
#define LogTopicMacro(topic_name, description) #topic_name,
#include "input/system_log_topic.input"
#undef LogTopicMacro
    "NotUsed"};

#define SYSTEM_LOG(topic, level, ...)                                \
  do {                                                               \
    if (level < vdb::LogLevel::kLogNotice &&                         \
        !(vdb::ServerConfiguration::GetEnabledLogTopics() &          \
          static_cast<uint64_t>(topic)))                             \
      break;                                                         \
    const int level_int = static_cast<int>(level);                   \
    if (level_int < vdb::ServerConfiguration::GetVerbosity()) break; \
    _serverLog(level_int, __FILE__, __LINE__, __VA_ARGS__);          \
  } while (0)

#define SYSTEM_LOG_WITH_PATH(topic, level, file, line, ...)          \
  do {                                                               \
    if (level < vdb::LogLevel::kLogNotice &&                         \
        !(vdb::ServerConfiguration::GetEnabledLogTopics() &          \
          static_cast<uint64_t>(topic)))                             \
      break;                                                         \
    const int level_int = static_cast<int>(level);                   \
    if (level_int < vdb::ServerConfiguration::GetVerbosity()) break; \
    _serverLog(level_int, file, line, __VA_ARGS__);                  \
  } while (0)

/* system log template for memory usage
 * below include is required
 * #include "vdb/common/memory_allocator.hh"
 *
 * redis resident size, redis allocated size, vdb allocated size are logged.
 */
#define MEMORY_USAGE_LOG(topic, level, msg)                                \
  SYSTEM_LOG(                                                              \
      topic | vdb::LogTopic::Memory, level,                                \
      "%s: seahorse_resident:%s, seahorse_allocated:%s, vdb_allocated:%s", \
      msg, vdb::BytesToHumanReadable(vdb::GetRedisResidentSize()).c_str(), \
      vdb::BytesToHumanReadable(vdb::GetRedisAllocatedSize()).c_str(),     \
      vdb::BytesToHumanReadable(vdb::GetVdbAllocatedSize()).c_str())

bool CheckLogTopicsString(std::string_view topic_list);

bool SetLogTopics(std::string_view topic_list);
bool SetLogTopics(uint64_t topics);
bool SetDefaultLogTopics();
bool SetAllLogTopics();

bool ResetLogTopics(std::string_view topic_list);
bool ResetLogTopics(uint64_t topics);
bool ResetAllLogTopics();

std::string GetEnabledLogTopics();

bool IsLogTopicEnabled(LogTopic topic);
bool IsLogTopicEnabled(std::string_view topic);

}  // namespace vdb
