#include "vdb/data/statistics_collector.hh"
#include "vdb/data/table.hh"
#include "vdb/data/index_handler.hh"
#include "vdb/common/defs.hh"

namespace vdb {

SegmentStatisticsCollector::SegmentStatisticsCollector(
    const vdb::map<std::string, std::shared_ptr<vdb::Table>>* table_dictionary)
    : table_dictionary_(table_dictionary) {}

arrow::Result<std::vector<SegmentStatisticsCollector::SegmentStatisticsEntry>>
SegmentStatisticsCollector::CollectAll() {
  std::vector<std::string> table_names;
  for (const auto& [table_name, _] : *table_dictionary_) {
    table_names.push_back(table_name);
  }
  return CollectByTables(table_names);
}

arrow::Result<std::vector<SegmentStatisticsCollector::SegmentStatisticsEntry>>
SegmentStatisticsCollector::CollectByTable(const std::string& table_name) {
  return CollectByTables({table_name});
}

arrow::Result<std::vector<SegmentStatisticsCollector::SegmentStatisticsEntry>>
SegmentStatisticsCollector::CollectByTables(
    const std::vector<std::string>& table_names) {
  collected_statistics_.clear();

  for (const auto& table_name : table_names) {
    auto table_it = table_dictionary_->find(table_name);
    if (table_it == table_dictionary_->end()) {
      return arrow::Status::Invalid("Table not found: ", table_name);
    }

    const auto& table = table_it->second;
    const auto& segments = table->GetSegments();

    // Skip tables with no segments
    if (segments.empty()) {
      continue;
    }

    const auto& index_infos = table->GetIndexInfos();

    for (const auto& [segment_id, segment] : segments) {
      SegmentStatisticsEntry info;
      info.table_name = table_name;
      // Replace kRS with "\\" string in segment_id
      std::string segment_id_str = std::string(segment_id);
      std::string replacement = "\\\\";
      size_t pos = 0;
      while ((pos = segment_id_str.find(kRS, pos)) != std::string::npos) {
        segment_id_str.replace(pos, 1, replacement);
        pos += replacement.length();
      }
      info.segment_id = segment_id_str;

      // Get total row count
      info.row_count = segment->Size();

      // Get deleted row count
      info.deleted_row_count = segment->DeletedSize();

      // Get indexed row count for each column
      if (segment->HasIndex()) {
        auto index_handler = segment->IndexHandler();
        for (const auto& index_info : *index_infos) {
          auto column_id = index_info.GetColumnId();
          auto column_name = table->GetSchema()->field(column_id)->name();
          info.indexed_row_count[column_name] =
              index_handler->CountIndexedElements(column_id);
        }
      }

      // Get active set information
      info.active_set_row_count = segment->ActiveSetRecordCount();
      info.active_set_size_limit = table->GetActiveSetSizeLimit();

      // Get inactive set count
      info.inactive_set_count = segment->InactiveSetCount();

      collected_statistics_.push_back(std::move(info));
    }
  }

  return collected_statistics_;
}

}  // namespace vdb