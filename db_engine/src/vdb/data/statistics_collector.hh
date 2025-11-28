#pragma once

#include "vdb/data/table.hh"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace vdb {

class SegmentStatisticsCollector {
 public:
  struct SegmentStatisticsEntry {
    std::string table_name;
    std::string segment_id;
    size_t row_count;
    std::map<std::string, size_t>
        indexed_row_count;  // column name -> indexed count
    size_t deleted_row_count;
    size_t active_set_row_count;
    size_t active_set_size_limit;
    size_t inactive_set_count;

    // JSON serialization
    json ToJson() const {
      json j;
      j["table_name"] = table_name;
      j["segment_id"] = segment_id;
      j["row_count"] = row_count;

      // Convert indexed_row_count map to JSON object
      json indexed_json = json::object();
      for (const auto& [column_name, count] : indexed_row_count) {
        indexed_json[column_name] = count;
      }
      j["indexed_row_count"] = indexed_json;

      j["deleted_row_count"] = deleted_row_count;
      j["active_set_row_count"] = active_set_row_count;
      j["active_set_size_limit"] = active_set_size_limit;
      j["inactive_set_count"] = inactive_set_count;
      return j;
    }

    // JSON deserialization
    static SegmentStatisticsEntry FromJson(const json& j) {
      SegmentStatisticsEntry entry;
      entry.table_name = j["table_name"].get<std::string>();
      entry.segment_id = j["segment_id"].get<std::string>();
      entry.row_count = j["row_count"].get<size_t>();

      // Convert JSON object to indexed_row_count map
      const auto& indexed_row_count_json = j["indexed_row_count"];
      for (const auto& [key, value] : indexed_row_count_json.items()) {
        entry.indexed_row_count[key] = value.get<size_t>();
      }

      entry.deleted_row_count = j["deleted_row_count"].get<size_t>();
      entry.active_set_row_count = j["active_set_row_count"].get<size_t>();
      entry.active_set_size_limit = j["active_set_size_limit"].get<size_t>();
      entry.inactive_set_count = j["inactive_set_count"].get<size_t>();
      return entry;
    }
  };

  SegmentStatisticsCollector(
      const vdb::map<std::string, std::shared_ptr<vdb::Table>>*
          table_dictionary);

  arrow::Result<std::vector<SegmentStatisticsEntry>> CollectAll();

  arrow::Result<std::vector<SegmentStatisticsEntry>> CollectByTable(
      const std::string& table_name);

  arrow::Result<std::vector<SegmentStatisticsEntry>> CollectByTables(
      const std::vector<std::string>& table_names);

  // JSON serialization for the entire collection
  json ToJson() const {
    json obj = json::object();
    json j = json::array();
    for (const auto& entry : collected_statistics_) {
      j.push_back(entry.ToJson());
    }
    obj["segment_statistics"] = j;
    return obj;
  }

  // JSON deserialization for the entire collection
  static std::vector<SegmentStatisticsEntry> FromJson(const json& obj) {
    std::vector<SegmentStatisticsEntry> entries;
    for (const auto& item : obj["segment_statistics"]) {
      entries.push_back(SegmentStatisticsEntry::FromJson(item));
    }
    return entries;
  }

  std::string ToJsonString() const { return ToJson().dump(); }

 private:
  const vdb::map<std::string, std::shared_ptr<vdb::Table>>* table_dictionary_;
  std::vector<SegmentStatisticsEntry> collected_statistics_;
};

}  // namespace vdb