#include <iostream>

#include "vdb/common/defs.hh"
#include "vdb/common/system_log.hh"
#include "vdb/common/util.hh"

#ifdef __cplusplus
extern "C" {
#include "server.h"
}
#endif

namespace vdb {
bool CheckLogTopicsString(std::string_view topic_list) {
  for (char c : topic_list) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ',' ||
          c == ' ' || c == '\t')) {
      return false;
    }
  }
  return true;
}

bool SetLogTopics(uint64_t topics) {
  server.enabled_log_topics |= topics;
  return true;
}

bool ResetLogTopics(uint64_t topics) {
  server.enabled_log_topics &= ~topics;
  return true;
}

void RemoveAllSpaces(std::string& str) {
  str.erase(std::remove_if(str.begin(), str.end(),
                           [](unsigned char c) { return std::isspace(c); }),
            str.end());
}

uint64_t GetLogTopicIdsFromCommaSeparatedString(std::string_view topics) {
  uint64_t result = 0;

  std::string processed = std::string(topics);
  RemoveAllSpaces(processed);
  processed = TransformToLower(processed);

  auto tokens = Tokenize(processed, ',');

  for (const auto& token : tokens) {
    for (uint8_t topic_id = 0;
         topic_id < static_cast<uint8_t>(LogTopic::kLogTopicMax); topic_id++) {
      std::string lowercase_topic_name =
          TransformToLower(LogTopicName[topic_id]);
      if (token == "all") {
        result |= UINT64_MAX;
        return result;
      }
      if (std::string(token) == lowercase_topic_name) {
        result |= BitPos64[topic_id];
        break;
      }
    }
  }
  return result;
}

bool SetLogTopics(std::string_view topic_list) {
  uint64_t topics = GetLogTopicIdsFromCommaSeparatedString(topic_list);
  return SetLogTopics(topics);
}

bool SetDefaultLogTopics() {
  return SetLogTopics(std::string(server.default_log_topics));
}

bool SetAllLogTopics() { return SetLogTopics(UINT64_MAX); }

bool ResetLogTopics(std::string_view topic_list) {
  uint64_t topics = GetLogTopicIdsFromCommaSeparatedString(topic_list);
  return ResetLogTopics(topics);
}

bool ResetAllLogTopics() { return ResetLogTopics(UINT64_MAX); }

std::string GetEnabledLogTopics() {
  auto enabled_log_topics = ServerConfiguration::GetEnabledLogTopics();
  std::string result = "";
  for (uint8_t topic_id = 0;
       topic_id < static_cast<uint8_t>(LogTopic::kLogTopicMax); topic_id++) {
    if (enabled_log_topics & BitPos64[topic_id]) {
      if (result != "") {
        result += ",";
      }
      result += LogTopicName[topic_id];
    }
  }
  return result;
}

bool IsLogTopicEnabled(LogTopic topic) {
  auto enabled_log_topics = ServerConfiguration::GetEnabledLogTopics();
  return !!(enabled_log_topics & static_cast<uint64_t>(topic));
}

bool IsLogTopicEnabled(std::string_view topic) {
  return IsLogTopicEnabled(
      static_cast<LogTopic>(GetLogTopicIdsFromCommaSeparatedString(topic)));
}

}  // namespace vdb