#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <queue>
#include <sched.h>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <optional>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "BS_thread_pool.hpp"

#include "vdb/data/index_handler.hh"
#include "vdb/vdb.hh"
#include "vdb/vdb_api.hh"
#include "vdb/common/common_guards.hh"
#include "vdb/common/fd_manager.hh"
#include "vdb/common/memory_allocator.hh"
#include "vdb/common/util.hh"
#include "vdb/common/spinlock.hh"
#include "vdb/common/status.hh"
#include "vdb/common/system_log.hh"
#include "vdb/compute/execution.hh"
#include "vdb/data/checker.hh"
#include "vdb/data/checker_registry.hh"
#include "vdb/data/table.hh"

#include "vdb/metrics/performance_monitor.hh"
#include "vdb/metrics/metrics_collection.hh"
#include "vdb/metrics/metrics_api.hh"
#include "vdb/metrics/timer.hh"

#ifdef __cplusplus
extern "C" {
#endif
#include "server.h"
#ifdef __cplusplus
}
#endif

/* =======================NOTICE======================== */
/* TODO: To prevent read-write conflicts in using record batch,
 *       we use per-table atomic integer as a read-write lock.
 *       But to achieve this locking mechanism, we need to check
 *       table_dictionary twice (outside of main command impl, and
 *       inside of main command impl). We have to refactor this
 *       to achieve better performance. */
/* TODO: SeahorseDB's per-table locking mechanism is predicated on
 *       the assumption that each command accesses only one table.
 *       Consequently, if the behavior is modified to allow
 *       a single command to access multiple tables,
 *       a corresponding adjustment to the locking mechanism
 *       will also be necessary. */
/* ======================================================= */

static inline void addReplyErrorSanitized(client *c, const std::string &msg) {
  std::string err = msg;
  for (char &ch : err) {
    if (ch == '\r' || ch == '\n') ch = ' ';
  }
  addReplyError(c, err.c_str());
}

void TotalRecordCountCommand(client *c) {
  if (c->argc < 2) {
    addReplyErrorArity(c);
    return;
  }

  auto maybe_result = vdb::_TotalRecordCountCommand();
  if (maybe_result.ok()) {
    std::string reply_string = "*2\r\n";
    auto [total_count, deleted_count] = maybe_result.ValueUnsafe();
    reply_string += ":" + std::to_string(total_count) + vdb::kCRLF;
    reply_string += ":" + std::to_string(deleted_count) + vdb::kCRLF;
    addReplyProto(c, reply_string.c_str(), reply_string.size());
  } else {
    addReplyError(c, maybe_result.status().ToString().c_str());
  }
}

void TotalTableCountCommand(client *c) {
  if (c->argc < 2) {
    addReplyErrorArity(c);
    return;
  }

  auto maybe_result = vdb::_TotalTableCountCommand();
  if (maybe_result.ok()) {
    std::string reply_string = "*2\r\n";
    auto [total_count, unloaded_count] = maybe_result.ValueUnsafe();
    reply_string += ":" + std::to_string(total_count) + vdb::kCRLF;
    reply_string += ":" + std::to_string(unloaded_count) + vdb::kCRLF;
    addReplyProto(c, reply_string.c_str(), reply_string.size());
  } else {
    addReplyError(c, maybe_result.status().ToString().c_str());
  }
}

void ListTableCommand(client *c) {
  if (c->argc > 2) {
    addReplyErrorArity(c);
    return;
  }

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c]() {
        vdb::metrics::CollectDuration(vdb::metrics::MetricIndex::ListTable,
                                      vdb::metrics::TimerAction::kStart);
        auto table_list = vdb::_ListTableCommand();
        std::string reply_string;
        size_t reply_string_len = 10;  // reserve array length

        for (auto &table_name : table_list) {
          reply_string_len += table_name.size() + 3;  // 3 = +, kCRLF
        }
        reply_string.reserve(reply_string_len);
        reply_string += "*";  // Array
        if (!table_list.size())
          reply_string += "-1";
        else
          reply_string += std::to_string(table_list.size());
        reply_string += vdb::kCRLF;

        for (auto &table_name : table_list) {
          reply_string += '+';  // Simple string
          reply_string += table_name;
          reply_string += vdb::kCRLF;
        }
        vdb::metrics::CollectDuration(vdb::metrics::MetricIndex::ListTable,
                                      vdb::metrics::TimerAction::kEnd);

        addReplyProtoAndWriteDirectly(c, reply_string.c_str(),
                                      reply_string.size());
      });
}

void CreateTableCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  sds serialized_schema = static_cast<sds>(c->argv[2]->ptr);

  vdb::Status status = vdb::Status::Unknown();
  try {
    vdb::metrics::CollectDuration(vdb::metrics::MetricIndex::CreateTable,
                                  vdb::metrics::TimerAction::kStart);
    status = vdb::_CreateTableCommand(serialized_schema);
    vdb::metrics::CollectDuration(vdb::metrics::MetricIndex::CreateTable,
                                  vdb::metrics::TimerAction::kEnd);
  } catch (std::runtime_error &e) {
    status = vdb::Status::OutOfMemory(e.what());
  }

  if (status.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    addReply(c, shared.ok);
  } else {
    vdb::metrics::CollectValue(vdb::metrics::MetricIndex::CreateTable, 1);
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               status.ToString().data());
    addReplyErrorSanitized(c, status.ToString());
  }
}

void InsertCommand(client *c) {
  if (c->argc < 4) {
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds record_sds = static_cast<sds>(c->argv[3]->ptr);

  std::string table_name{table_sds, sdslen(table_sds)};
  std::string_view record_view{record_sds, sdslen(record_sds)};

  vdb::Status status = vdb::Status::Unknown();

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    while (table->CheckRunningActiveReadCommands()) {
      if (vdb::ServerConfiguration::GetShutdownAsap()) {
        addReplyError(c, "Server is shutting down");
        return;
      }
      sched_yield();
    }
  }

  try {
    status = vdb::_InsertCommand(table_name, record_view);
  } catch (std::runtime_error &e) {
    status = vdb::Status::OutOfMemory(e.what());
  }

  if (status.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    addReply(c, shared.ok);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               status.ToString().data());
    addReplyErrorSanitized(c, status.ToString());
  }
}

void BatchInsertCommand(client *c) {
  if (c->argc < 4) {
    addReplyErrorArity(c);
    return;
  }
  vdb::metrics::ScopedDurationMetric batch_insert_latency(
      vdb::metrics::MetricIndex::BatchInsertCommandLatency);

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds serialized_rb = static_cast<sds>(c->argv[3]->ptr);

  std::string table_name{table_sds, sdslen(table_sds)};

  vdb::Status status = vdb::Status::Unknown();

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    while (table->CheckRunningActiveReadCommands()) {
      if (vdb::ServerConfiguration::GetShutdownAsap()) {
        addReplyError(c, "Server is shutting down");
        return;
      }
      sched_yield();
    }
  }

  try {
    status = vdb::_BatchInsertCommand(table_name, serialized_rb);
  } catch (std::runtime_error &e) {
    status = vdb::Status::OutOfMemory(e.what());
  }

  if (status.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    addReply(c, shared.ok);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               status.ToString().data());
    addReplyError(c, status.ToString().c_str());
  }
}

void DescribeTableCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  // Check if this is a DESCRIBE_INTERNAL command by normalizing argv[1] to
  // upper-case
  sds command_sds = static_cast<sds>(c->argv[1]->ptr);
  std::string_view command_name{command_sds, sdslen(command_sds)};
  std::string command_upper = vdb::TransformToUpper(std::string(command_name));
  bool include_internal_columns = (command_upper == "DESCRIBE_INTERNAL");

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, table_name, include_internal_columns]() {
        auto maybe_serialized_schema =
            vdb::_DescribeTableCommand(table_name, include_internal_columns);
        if (maybe_serialized_schema.ok()) {
          auto serialized_schema = maybe_serialized_schema.ValueUnsafe();
          std::string reply_string = "";
          reply_string.reserve(20 + serialized_schema->size());
          reply_string +=
              "$" + std::to_string(serialized_schema->size()) + vdb::kCRLF;
          reply_string += std::string_view((char *)serialized_schema->data(),
                                           serialized_schema->size());
          reply_string += vdb::kCRLF;
          addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                        reply_string.size());
        } else {
          SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogNotice, "%s",
                     maybe_serialized_schema.status().ToString().data());
          addReplyError(c, maybe_serialized_schema.status().ToString().c_str());
        }
      });
}

void DebugScanCommand(client *c) {
  if (c->argc < 5) {
    addReplyErrorArity(c);
    return;
  }
  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds projection_sds = sdsdup(static_cast<sds>(c->argv[3]->ptr));
  sds filter_sds = sdsdup(static_cast<sds>(c->argv[4]->ptr));
  std::string table_name{table_sds, sdslen(table_sds)};

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    auto guard_ptr = std::make_shared<vdb::AtomicCounterGuard<uint64_t>>(
        table->GetActiveReadCommands());

    static_cast<BS::light_thread_pool *>(
        vdb::ServerResources::GetReadCommandPool())
        ->detach_task([c, table_name, table, projection_sds, filter_sds,
                       guard_ptr]() {
          std::string_view projection_view{projection_sds,
                                           sdslen(projection_sds)};
          std::string_view filter_view{filter_sds, sdslen(filter_sds)};
          auto maybe_rb_string =
              vdb::_DebugScanCommand(table_name, projection_view, filter_view);
          if (maybe_rb_string.ok()) {
            auto reply_string = maybe_rb_string.ValueUnsafe();
            reply_string = "$" + std::to_string(reply_string.size()) +
                           vdb::kCRLF + reply_string + vdb::kCRLF;
            addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                          reply_string.size());
          } else {
            SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
                       maybe_rb_string.status().ToString().data());
            addReplyErrorSanitized(c, maybe_rb_string.status().ToString());
          }
          sdsfree(projection_sds);
          sdsfree(filter_sds);
        });
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "DebugScanCommand is Failed: Invalid table name.");
    addReplyErrorSanitized(c,
                           "DebugScanCommand is Failed: Invalid table name.");
  }
}

void ScanCommand(client *c) {
  if (c->argc < 5) {
    addReplyErrorArity(c);
    return;
  }

  // Check if this is a SCAN_INTERNAL command by normalizing argv[1] to
  // upper-case
  sds command_sds = static_cast<sds>(c->argv[1]->ptr);
  std::string_view command_name{command_sds, sdslen(command_sds)};
  std::string command_upper = vdb::TransformToUpper(std::string(command_name));
  bool include_internal_columns = (command_upper == "SCAN_INTERNAL");

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds projection_sds = sdsdup(static_cast<sds>(c->argv[3]->ptr));
  sds filter_sds = sdsdup(static_cast<sds>(c->argv[4]->ptr));

  std::string table_name{table_sds, sdslen(table_sds)};
  sds limit_sds = nullptr;
  if (c->argc > 5) {
    limit_sds = sdsdup(static_cast<sds>(c->argv[5]->ptr));
  }

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    auto guard_ptr = std::make_shared<vdb::AtomicCounterGuard<uint64_t>>(
        table->GetActiveReadCommands());

    static_cast<BS::light_thread_pool *>(
        vdb::ServerResources::GetReadCommandPool())
        ->detach_task([c, table_name, table, projection_sds, filter_sds,
                       limit_sds, include_internal_columns, guard_ptr]() {
          std::string_view projection_view{projection_sds,
                                           sdslen(projection_sds)};
          std::string_view filter_view{filter_sds, sdslen(filter_sds)};
          std::string_view limit_view;
          if (limit_sds != nullptr) {
            limit_view = std::string_view{limit_sds, sdslen(limit_sds)};
          } else {
            limit_view = std::string_view{"0"};
          }
          SYSTEM_LOG(
              vdb::LogTopic::Command, vdb::LogLevel::kLogDebug,
              "ScanCommand: table_name=%s, projection=%s, filter=%s, limit=%s, "
              "include_internal_columns=%s",
              table_name.c_str(), projection_view.data(), filter_view.data(),
              limit_view.data(), include_internal_columns ? "true" : "false");

          auto maybe_serialized_rb =
              vdb::_ScanCommand(table_name, projection_view, filter_view,
                                limit_view, include_internal_columns);
          if (maybe_serialized_rb.ok()) {
            auto serialized_rb = maybe_serialized_rb.ValueUnsafe();
            std::string reply_string;
            if (serialized_rb->size() != 0) {
              /* bulk string */
              reply_string +=
                  '$' + std::to_string(serialized_rb->size()) + vdb::kCRLF;
              reply_string += std::string_view((char *)serialized_rb->data(),
                                               serialized_rb->size());
              reply_string += vdb::kCRLF;
            } else {
              reply_string = "*0";
              reply_string += vdb::kCRLF;
            }
            addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                          reply_string.size());
          } else {
            SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogNotice, "%s",
                       maybe_serialized_rb.status().ToString().data());
            addReplyErrorSanitized(c, maybe_serialized_rb.status().ToString());
          }
          sdsfree(projection_sds);
          sdsfree(filter_sds);
          if (limit_sds != nullptr) {
            sdsfree(limit_sds);
          }
        });
  } else {
    SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogNotice,
               "ScanCommand is Failed: Invalid table name: %s",
               table_name.c_str());
    addReplyError(c, "ScanCommand is Failed: Invalid table name.");
  }
}

void ScanOpenCommand(client *c) {
  if (c->argc < 6) {
    addReplyErrorArity(c);
    return;
  }

  // Check if this is a SCANOPEN_INTERNAL command by normalizing argv[1] to
  // upper-case
  sds command_sds = static_cast<sds>(c->argv[1]->ptr);
  std::string_view command_name{command_sds, sdslen(command_sds)};
  std::string command_upper = vdb::TransformToUpper(std::string(command_name));
  bool include_internal_columns = (command_upper == "SCANOPEN_INTERNAL");

  sds uuid_sds = sdsdup(static_cast<sds>(c->argv[2]->ptr));
  sds table_sds = static_cast<sds>(c->argv[3]->ptr);
  sds projection_sds = sdsdup(static_cast<sds>(c->argv[4]->ptr));
  sds filter_sds = sdsdup(static_cast<sds>(c->argv[5]->ptr));

  std::string table_name{table_sds, sdslen(table_sds)};
  sds limit_sds = nullptr;
  if (c->argc > 6) {
    limit_sds = sdsdup(static_cast<sds>(c->argv[6]->ptr));
  }

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    auto guard_ptr = std::make_shared<vdb::AtomicCounterGuard<uint64_t>>(
        table->GetActiveReadCommands());

    static_cast<BS::light_thread_pool *>(
        vdb::ServerResources::GetReadCommandPool())
        ->detach_task([c, uuid_sds, table_name, table, projection_sds,
                       filter_sds, limit_sds, include_internal_columns,
                       guard_ptr]() {
          std::string_view uuid_view{uuid_sds, sdslen(uuid_sds)};
          std::string_view projection_view{projection_sds,
                                           sdslen(projection_sds)};
          std::string_view filter_view{filter_sds, sdslen(filter_sds)};
          std::string_view limit_view;
          if (limit_sds != nullptr) {
            limit_view = std::string_view{limit_sds, sdslen(limit_sds)};
          } else {
            limit_view = std::string_view{"0"};
          }

          SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogDebug,
                     "ScanOpenCommand: uuid=%s, table_name=%s, projection=%s, "
                     "filter=%s, limit=%s, include_internal_columns=%s",
                     uuid_view.data(), table_name.c_str(),
                     projection_view.data(), filter_view.data(),
                     limit_view.data(),
                     include_internal_columns ? "true" : "false");

          auto maybe_tuple = vdb::_ScanOpenCommand(
              uuid_view, table_name, projection_view, filter_view, limit_view,
              include_internal_columns);
          if (maybe_tuple.ok()) {
            auto [serialized_rb, has_next] = maybe_tuple.ValueUnsafe();

            std::string reply_string = "*2\r\n";
            reply_string +=
                "$" + std::to_string(serialized_rb->size()) + vdb::kCRLF;
            reply_string += std::string_view((char *)serialized_rb->data(),
                                             serialized_rb->size());
            reply_string += vdb::kCRLF;
            std::string has_next_str = has_next ? "1" : "0";
            reply_string += ":" + has_next_str + vdb::kCRLF;
            addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                          reply_string.size());
          } else {
            SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogNotice, "%s",
                       maybe_tuple.status().ToString().data());
            addReplyErrorSanitized(c, maybe_tuple.status().ToString());
          }
          sdsfree(uuid_sds);
          sdsfree(projection_sds);
          sdsfree(filter_sds);
          if (limit_sds != nullptr) {
            sdsfree(limit_sds);
          }
        });
  } else {
    SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogNotice,
               "ScanOpenCommand is Failed: Invalid table name.");
    addReplyErrorSanitized(c, "ScanOpenCommand is Failed: Invalid table name.");
  }
}

void FetchNextCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }
  sds uuid_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string uuid{uuid_sds, sdslen(uuid_sds)};

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, uuid]() {
        SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                   "FetchNextCommand: uuid=%s", uuid.c_str());

        auto maybe_tuple = vdb::_FetchNextCommand(uuid);
        if (maybe_tuple.ok()) {
          auto [serialized_rb, has_next] = maybe_tuple.ValueUnsafe();

          std::string reply_string = "*2\r\n";
          reply_string +=
              "$" + std::to_string(serialized_rb->size()) + vdb::kCRLF;
          reply_string += std::string_view((char *)serialized_rb->data(),
                                           serialized_rb->size());
          reply_string += vdb::kCRLF;
          std::string has_next_str = has_next ? "1" : "0";
          reply_string += ":" + has_next_str + vdb::kCRLF;
          addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                        reply_string.size());
        } else {
          SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
                     maybe_tuple.status().ToString().data());
          addReplyError(c, maybe_tuple.status().ToString().c_str());
        }
      });
}

// Internal helper function for vector search
static void _InternalVectorSearchCommand(client *c, bool is_batch_query) {
  if (c->argc < 6) {
    addReplyErrorArity(c);
    return;
  }

  bool projection_sds_copied = false;
  bool filter_sds_copied = false;

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds column_name_sds = static_cast<sds>(c->argv[3]->ptr);
  sds query_sds = sdsdup(static_cast<sds>(c->argv[5]->ptr));

  std::string table_name{table_sds, sdslen(table_sds)};
  auto table_dictionary = vdb::GetTableDictionary();
  auto iter = table_dictionary->find(table_name);
  if (iter == table_dictionary->end()) {
    std::string error_msg =
        is_batch_query
            ? "BatchVectorSearchCommand is Failed: Invalid table name."
            : "VectorSearchCommand is Failed: Invalid table name.";
    addReplyError(c, error_msg.c_str());
    sdsfree(query_sds);
    return;
  }
  auto table = iter->second;

  auto index_column_id = table->GetSchema()->GetFieldIndex(column_name_sds);
  auto index_infos = table->GetIndexInfos();
  // Parse k parameter (number of nearest neighbors)
  auto parsed_k_result = vdb::stoui64(static_cast<sds>(c->argv[4]->ptr));
  if (!parsed_k_result.ok()) {
    addReplyError(c, parsed_k_result.status().ToString().c_str());
    sdsfree(query_sds);
    return;
  }
  size_t k = parsed_k_result.ValueUnsafe();

  // Set ef_search parameter (defaults to k if not provided)
  size_t ef_search = k;
  if (c->argc > 6) {
    sds ef_search_sds = static_cast<sds>(c->argv[6]->ptr);
    auto parsed_ef_search_result = vdb::stoui64(ef_search_sds);
    if (!parsed_ef_search_result.ok()) {
      addReplyError(c, parsed_ef_search_result.status().ToString().c_str());
      sdsfree(query_sds);
      return;
    }
    ef_search = parsed_ef_search_result.ValueUnsafe();
  }

  std::string_view projection_view;
  sds projection_sds = nullptr;
  if (c->argc > 7) {
    projection_sds = sdsdup(static_cast<sds>(c->argv[7]->ptr));
    projection_view = {projection_sds, sdslen(projection_sds)};
    projection_sds_copied = true;
  } else {
    projection_view = "*";
  }

  std::string_view filter_view;
  sds filter_sds = nullptr;
  if (c->argc > 8) {
    filter_sds = sdsdup(static_cast<sds>(c->argv[8]->ptr));
    filter_view = {filter_sds, sdslen(filter_sds)};
    filter_sds_copied = true;
  } else {
    filter_view = "";
  }

  auto guard_ptr = std::make_shared<vdb::AtomicCounterGuard<uint64_t>>(
      table->GetActiveReadCommands());

  VdbMainBeforeAddTask(c);

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, table_name, table, index_column_id, k, ef_search,
                     projection_view, filter_view, query_sds,
                     projection_sds_copied, filter_sds_copied, projection_sds,
                     filter_sds, is_batch_query, guard_ptr]() {
        VdbThreadPoolBeforeTask(c);

        if (is_batch_query) {
          // Batch query - returns array of results
          auto maybe_serialized_rbs =
              vdb::_BatchAnnCommand(table_name, index_column_id, k, query_sds,
                                    ef_search, projection_view, filter_view);

          if (maybe_serialized_rbs.ok()) {
            auto serialized_rbs = maybe_serialized_rbs.ValueUnsafe();
            std::string reply_string;
            if (serialized_rbs.size() != 0) {
              SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                         "Returned %lu results", serialized_rbs.size());
              reply_string = "*";
              reply_string += std::to_string(serialized_rbs.size());
              reply_string += vdb::kCRLF;
              for (auto &serialized_rb : serialized_rbs) {
                reply_string +=
                    '$' + std::to_string(serialized_rb->size()) + vdb::kCRLF;
                reply_string += std::string_view((char *)serialized_rb->data(),
                                                 serialized_rb->size());
                reply_string += vdb::kCRLF;
              }
            } else {
              SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                         "No results found");
              reply_string = "*0";
              reply_string += vdb::kCRLF;
            }
            VdbThreadPoolAfterTask(c);
            addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                          reply_string.size());
          } else {
            SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
                       maybe_serialized_rbs.status().ToString().data());
            VdbThreadPoolAfterTask(c);
            addReplyError(c, maybe_serialized_rbs.status().ToString().c_str());
          }
        } else {
          // Single query - returns single result
          auto maybe_serialized_rb =
              vdb::_AnnCommand(table_name, index_column_id, k, query_sds,
                               ef_search, projection_view, filter_view);

          if (maybe_serialized_rb.ok()) {
            auto serialized_rb = maybe_serialized_rb.ValueUnsafe();
            std::string reply_string;
            if (serialized_rb != nullptr) {
              reply_string =
                  '$' + std::to_string(serialized_rb->size()) + vdb::kCRLF;
              reply_string += std::string_view((char *)serialized_rb->data(),
                                               serialized_rb->size());
              reply_string += vdb::kCRLF;
            } else {
              reply_string = "$0";
              reply_string += vdb::kCRLF;
              reply_string += vdb::kCRLF;
            }
            VdbThreadPoolAfterTask(c);
            addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                          reply_string.size());
          } else {
            SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
                       maybe_serialized_rb.status().ToString().data());
            VdbThreadPoolAfterTask(c);
            addReplyError(c, maybe_serialized_rb.status().ToString().c_str());
          }
        }

        sdsfree(query_sds);
        if (projection_sds_copied) {
          sdsfree(projection_sds);
        }
        if (filter_sds_copied) {
          sdsfree(filter_sds);
        }
      });
}

void VectorSearchCommand(client *c) { _InternalVectorSearchCommand(c, false); }

void BatchVectorSearchCommand(client *c) {
  _InternalVectorSearchCommand(c, true);
}

void DeleteCommand(client *c) {
  if (c->argc < 4) {
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds filter_sds = static_cast<sds>(c->argv[3]->ptr);

  std::string table_name{table_sds, sdslen(table_sds)};
  std::string_view filter_view{filter_sds, sdslen(filter_sds)};

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    while (table->CheckRunningActiveReadCommands()) {
      if (vdb::ServerConfiguration::GetShutdownAsap()) {
        addReplyError(c, "Server is shutting down");
        return;
      }
      sched_yield();
    }
  }

  auto result = vdb::_DeleteCommand(table_name, filter_view);
  if (result.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    uint32_t deleted_count = result.ValueUnsafe();
    std::string reply_string = ":" + std::to_string(deleted_count) + vdb::kCRLF;
    addReplyProto(c, reply_string.data(), reply_string.size());
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               result.status().ToString().data());

    addReplyErrorSanitized(c, result.status().ToString());
  }
}

void UpdateCommand(client *c) {
  if (c->argc < 5) {
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  sds update_sds = static_cast<sds>(c->argv[3]->ptr);
  sds filter_sds = static_cast<sds>(c->argv[4]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};
  std::string_view update_view{update_sds, sdslen(update_sds)};
  std::string_view filter_view{filter_sds, sdslen(filter_sds)};

  auto table_dictionary = vdb::GetTableDictionary();

  auto iter = table_dictionary->find(table_name);
  if (iter != table_dictionary->end()) {
    auto table = iter->second;
    while (table->CheckRunningActiveReadCommands()) {
      if (vdb::ServerConfiguration::GetShutdownAsap()) {
        addReplyError(c, "Server is shutting down");
        return;
      }
      sched_yield();
    }
  }

  auto result = vdb::_UpdateCommand(table_name, update_view, filter_view);
  if (result.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    uint32_t updated_count = result.ValueUnsafe();
    std::string reply_string = ":" + std::to_string(updated_count) + vdb::kCRLF;
    addReplyProto(c, reply_string.data(), reply_string.size());
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               result.status().ToString().data());
    addReplyErrorSanitized(c, result.status().ToString());
  }
}

void CheckIndexingCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, table_name]() {
        auto maybe_result = vdb::_CheckIndexingCommand(table_name);
        if (maybe_result.ok()) {
          addReplyBool(c, maybe_result.ValueUnsafe());
        } else {
          addReplyError(c, maybe_result.status().ToString().c_str());
        }
      });
}

void CountRecordsCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, table_name]() {
        auto maybe_result = vdb::_CountRecordsCommand(table_name);
        if (maybe_result.ok()) {
          std::string reply_string =
              ":" + std::to_string(maybe_result.ValueUnsafe()) + vdb::kCRLF;
          addReplyProtoAndWriteDirectly(c, reply_string.c_str(),
                                        reply_string.size());
        } else {
          addReplyError(c, maybe_result.status().ToString().c_str());
        }
      });
}

void CountIndexedElementsCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, table_name]() {
        auto maybe_result = vdb::_CountIndexedElementsCommand(table_name);
        if (maybe_result.ok()) {
          auto [total_count, indexed_count_infos] = maybe_result.ValueUnsafe();
          std::string reply_string =
              "*" + std::to_string(2 + (indexed_count_infos.size() * 3)) +
              vdb::kCRLF;
          reply_string += ":" + std::to_string(total_count) + vdb::kCRLF;
          reply_string +=
              ":" + std::to_string(indexed_count_infos.size()) + vdb::kCRLF;
          for (auto &indexed_count_info : indexed_count_infos) {
            reply_string +=
                "$" + std::to_string(indexed_count_info.column_name.length()) +
                vdb::kCRLF;
            reply_string += indexed_count_info.column_name + vdb::kCRLF;

            reply_string +=
                "$" + std::to_string(indexed_count_info.index_type.length()) +
                vdb::kCRLF;
            reply_string += indexed_count_info.index_type + vdb::kCRLF;

            reply_string += ":" +
                            std::to_string(indexed_count_info.indexed_count) +
                            vdb::kCRLF;
          }
          addReplyProtoAndWriteDirectly(c, reply_string.c_str(),
                                        reply_string.size());
        } else {
          addReplyError(c, maybe_result.status().ToString().c_str());
        }
      });
}

void CountAofRewritesCommand(client *c) {
  std::string reply_string =
      ":" + std::to_string(vdb::ServerConfiguration::GetAofRewriteCount()) +
      vdb::kCRLF;
  addReplyProto(c, reply_string.c_str(), reply_string.size());
}

void DropTableCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }
  sds table_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};
  auto status = vdb::_DropTableCommand(table_name);

  if (status.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    addReply(c, shared.ok);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               status.ToString().data());
    addReplyError(c, status.ToString().c_str());
  }
}

void LoadCommand(client *c) {
  if (c->argc < 2) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "LoadCommand: Arity is %d", c->argc);
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[1]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};

  std::string directory_path = "";
  if (c->argc > 2) {
    sds directory_sds = static_cast<sds>(c->argv[2]->ptr);
    directory_path = {directory_sds, sdslen(directory_sds)};
  }

  vdb::Status result = vdb::Status::Unknown();
  result = vdb::_LoadCommand(table_name, directory_path);

  if (result.ok()) {
    addReply(c, shared.ok);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               result.ToString().data());
    addReplyError(c, result.ToString().c_str());
  }
}

void UnloadCommand(client *c) {
  if (c->argc < 2) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "UnloadCommand: Arity is %d", c->argc);
    addReplyErrorArity(c);
    return;
  }

  sds table_sds = static_cast<sds>(c->argv[1]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};
  vdb::Status result = vdb::Status::Unknown();
  if (c->argc > 2) {
    sds directory_sds = static_cast<sds>(c->argv[2]->ptr);
    std::string directory_path{directory_sds, sdslen(directory_sds)};
    result = vdb::_UnloadCommand(table_name, directory_path);
  } else {
    result = vdb::_UnloadCommand(table_name, "");
  }

  if (result.ok()) {
    vdb::ServerConfiguration::IncrementServerDirty();
    addReply(c, shared.ok);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               result.ToString().data());
    addReplyError(c, result.ToString().c_str());
  }
}

void SaveCommand(client *c) {
  if (c->argc < 2) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "SaveCommand: Arity is %d", c->argc);
    addReplyErrorArity(c);
    return;
  }

  vdb::Status result = vdb::Status::Unknown();

  sds table_sds = static_cast<sds>(c->argv[1]->ptr);
  std::string table_name{table_sds, sdslen(table_sds)};
  if (c->argc > 2) {
    sds directory_sds = static_cast<sds>(c->argv[2]->ptr);
    std::string directory_path{directory_sds, sdslen(directory_sds)};
    result = vdb::_SaveCommand(table_name, directory_path);
  } else {
    result = vdb::_SaveCommand(table_name, "");
  }

  if (result.ok()) {
    addReply(c, shared.ok);
  } else {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               result.ToString().data());
    addReplyError(c, result.ToString().c_str());
  }
}

void MetricsPrintCommand(client *c) {
  auto report = vdb::metrics::performance_monitor->GetPerformanceReport();
  std::cout << report << std::endl;
  // TODO: add reply
  addReply(c, shared.ok);
}

void MetricsClearCommand(client *c) {
  vdb::metrics::performance_monitor->ResetCollection();
  addReply(c, shared.ok);
}

void SegmentStatisticsCommand(client *c) {
  std::optional<std::string> table_name;
  if (c->argc > 2) {
    sds table_sds = static_cast<sds>(c->argv[2]->ptr);
    table_name = std::string{table_sds, sdslen(table_sds)};
  }

  static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool())
      ->detach_task([c, table_name]() {
        auto maybe_result = vdb::_SegmentStatisticsCommand(table_name);

        if (maybe_result.ok()) {
          auto reply_string = maybe_result.ValueUnsafe();
          reply_string = "$" + std::to_string(reply_string.size()) +
                         vdb::kCRLF + reply_string + vdb::kCRLF;
          addReplyProtoAndWriteDirectly(c, reply_string.data(),
                                        reply_string.size());
        } else {
          SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
                     maybe_result.status().ToString().data());
          addReplyError(c, maybe_result.status().ToString().c_str());
        }
      });
}

void LogSetTopicsCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  sds topics_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string topics{topics_sds, sdslen(topics_sds)};

  bool is_valid = vdb::CheckLogTopicsString(topics);
  if (is_valid) {
    vdb::SetLogTopics(topics);
    addReply(c, shared.ok);
  } else {
    std::string reply_string =
        "Invalid log topics string. Only alphabets, comma, and space are "
        "allowed.";
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               reply_string.c_str());
    addReplyError(c, reply_string.c_str());
  }
}

void LogSetDefaultTopicsCommand(client *c) {
  vdb::ResetAllLogTopics();
  vdb::SetDefaultLogTopics();
  addReply(c, shared.ok);
}

void LogSetAllTopicsCommand(client *c) {
  vdb::SetAllLogTopics();
  addReply(c, shared.ok);
}

void LogResetTopicsCommand(client *c) {
  if (c->argc < 3) {
    addReplyErrorArity(c);
    return;
  }

  sds topics_sds = static_cast<sds>(c->argv[2]->ptr);
  std::string topics{topics_sds, sdslen(topics_sds)};

  bool is_valid = vdb::CheckLogTopicsString(topics);
  if (is_valid) {
    vdb::ResetLogTopics(topics);
    addReply(c, shared.ok);
  } else {
    std::string reply_string =
        "Invalid log topics string. Only alphabets, comma, and space are "
        "allowed.";
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice, "%s",
               reply_string.c_str());
    addReplyError(c, reply_string.c_str());
  }
}

void LogResetAllTopicsCommand(client *c) {
  vdb::ResetAllLogTopics();
  addReply(c, shared.ok);
}

void LogEnabledTopicsCommand(client *c) {
  std::string enabled_topics = vdb::GetEnabledLogTopics();
  enabled_topics = "+" + enabled_topics + vdb::kCRLF;
  SYSTEM_LOG(vdb::LogTopic::Command, vdb::LogLevel::kLogNotice,
             "enabled_topics: %s", enabled_topics.c_str());
  addReplyProto(c, enabled_topics.c_str(), enabled_topics.size());
}

void LogTestCommand(client *c) {
  for (uint8_t level_id = 0;
       level_id < static_cast<uint8_t>(vdb::LogLevel::kLogLevelMax);
       level_id++) {
    auto level = static_cast<vdb::LogLevel>(level_id);
    std::string level_name = vdb::LogLevelName[level_id];

    for (uint8_t topic_id = 0;
         topic_id < static_cast<uint8_t>(vdb::LogTopic::kLogTopicMax);
         topic_id++) {
      auto topic = static_cast<vdb::LogTopic>(topic_id);
      std::string topic_name = vdb::LogTopicName[topic_id];

      SYSTEM_LOG(topic, level, "Test log from %s with level %s",
                 topic_name.c_str(), level_name.c_str());
    }
  }

  addReply(c, shared.ok);
}

void LogSetDefaultTopics() { vdb::SetDefaultLogTopics(); }

void InitializeReadCommandThreads(size_t num_threads) {
  vdb::ServerResources::SetReadCommandPool(
      static_cast<void *>(new BS::light_thread_pool(num_threads)));
}

void TerminateReadCommandThreads() {
  auto read_command_pool = static_cast<BS::light_thread_pool *>(
      vdb::ServerResources::GetReadCommandPool());
  delete read_command_pool;
  vdb::ServerResources::SetReadCommandPool(nullptr);
}

void AllocateTableDictionary() {
  vdb::ServerResources::SetTableDictionary(static_cast<void *>(
      new vdb::map<std::string, std::shared_ptr<vdb::Table>>()));
}
void DeallocateTableDictionary() {
  auto table_dictionary =
      static_cast<vdb::map<std::string, std::shared_ptr<vdb::Table>> *>(
          vdb::ServerResources::GetTableDictionary());
  delete table_dictionary;
  vdb::ServerResources::SetTableDictionary(nullptr);
}

void AllocatePerformanceMonitor() {
  vdb::metrics::performance_monitor =
      vdb::make_shared<vdb::metrics::PerformanceMonitor>();
}
void DeallocatePerformanceMonitor() {
  vdb::metrics::performance_monitor = nullptr;
}

void AllocateFdManager() { vdb::FdManager::GetInstance(); }

void AllocateMetadataCheckers() {
  // Allocate the map itself
  vdb::ServerResources::SetMetadataCheckers(static_cast<void *>(
      new std::map<std::string, std::shared_ptr<vdb::MetadataChecker>>()));

  auto metadata_checkers = static_cast<
      std::map<std::string, std::shared_ptr<vdb::MetadataChecker>> *>(
      vdb::ServerResources::GetMetadataCheckers());

  // Get the registered factories from the registry
  const auto &factories = vdb::GetRegisteredCheckerFactories();

  // Populate the map by calling the factories
  for (const auto &[key, factory] : factories) {
    if (factory) {  // Ensure factory is valid
      metadata_checkers->insert({key, factory()});
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                 "Allocated metadata checker: %s", key.c_str());
    }
  }

  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
             "Allocated %zu metadata checkers.", metadata_checkers->size());
}

void DeallocateMetadataCheckers() {
  auto metadata_checkers = static_cast<
      std::map<std::string, std::shared_ptr<vdb::MetadataChecker>> *>(
      vdb::ServerResources::GetMetadataCheckers());
  delete metadata_checkers;
  vdb::ServerResources::SetMetadataCheckers(nullptr);
}

/* snapshot api */
vdb::Spinlock snapshot_lock;

void CheckVdbSnapshot() {
  bool entered = false;
  while (snapshot_lock.IsLocked()) {
    if (!entered) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Building Index is blocked by Snapshot. "
                 "Waits until Creating Snapshot is Done.");
      entered = true;
    }
    sched_yield();
  }
  if (entered) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Building Index is Resumed.");
  }
}

static bool ConvertActiveSetToInactiveForTable(
    std::shared_ptr<vdb::Table> &table) {
  auto segments = table->GetSegments();

  for (auto &[segment_id, segment] : segments) {
    if (segment->ActiveSetRecordCount() > 0) {
      vdb::InsertionGuard guard(segment.get());
      auto status = segment->MakeInactive(guard);
      if (!status.ok()) {
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "Failed to convert active set to inactive set. "
                   "table_name(%s), segment(%s), set_id(%u), status(%s)",
                   table->GetTableName().data(), segment_id.data(),
                   segment->ActiveSetId(), status.ToString().data());
        return false;
      } else {
        guard.Success();
        SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                   "Prepared Snapshot: Active set converted to inactive "
                   "set. table_name(%s), segment(%s), set_id(%u)",
                   table->GetTableName().data(), segment_id.data(),
                   segment->ActiveSetId() - 1);
      }
    }
  }

  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
             "All active sets are converted to inactive sets for table (%s).",
             table->GetTableName().data());
  return true;
}

bool PrepareVdbSnapshot() {
  /* Check all sets of the tables whether they are building indexes now. */
  auto table_dictionary = vdb::GetTableDictionary();

  snapshot_lock.Lock();
  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
             "Creating Snapshot is Requested. Snapshot Lock is held.");

  /* SDDEV-105:
     To prevent duplicate data loading during recovery,
     the active set is converted into an inactive set at the time of VDB
     snapshot creation. However, for table snapshots (not VDB snapshot),
     duplicate loading is not an issue, so the active set is not converted into
     an inactive set when creating a table snapshot. */
  for (auto &[table_name, table] : *table_dictionary) {
    if (!ConvertActiveSetToInactiveForTable(table)) {
      SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
                 "Failed to prepare snapshot: active set conversion failed for "
                 "table (%s).",
                 table_name.data());
      snapshot_lock.Unlock();
      return false;
    }
  }

  vdb::vector<std::shared_ptr<vdb::VectorIndex>> uncompleted_indexes;

  for (auto &[table_name, table] : *table_dictionary) {
    auto segments = table->GetSegments();
    auto index_infos = table->GetIndexInfos();
    if (table->HasIndex()) {
      for (auto &[segment_id, segment] : segments) {
        std::shared_ptr<vdb::IndexHandler> index_handler =
            segment->IndexHandler();
        for (auto &index_info : *index_infos) {
          auto column_id = index_info.GetColumnId();
          size_t num_index = index_handler->Size(column_id);
          for (size_t i = 0; i < num_index; i++) {
            auto index = index_handler->Index(column_id, i);
            // MAYBE just wait at this line instead of using vector
            if (index->Size() != index->CompleteSize()) {
              uncompleted_indexes.push_back(index);
            }
          }
        }
      }

      while (!uncompleted_indexes.empty()) {
        auto index = uncompleted_indexes.back();
        if (index->Size() == index->CompleteSize()) {
          uncompleted_indexes.pop_back();
        } else {
          sched_yield();
        }
      }
    }
  }
  SYSTEM_LOG(vdb::LogTopic::Table, vdb::LogLevel::kLogNotice,
             "Creating Snapshot is Started. All active sets converted to "
             "inactive sets and there are no building indexes now.");
  return true;
}

void PostVdbSnapshot() {
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
             "Creating Snapshot is Done. Snapshot Lock is released.");
  snapshot_lock.Unlock();
}

bool SaveVdbSnapshot(char *directory_path_) {
  std::string_view directory_path(directory_path_);
  /* create directory */
  if (!std::filesystem::create_directory(directory_path)) {
    return false;
  }
  /* save snapshot */
  auto table_dictionary = vdb::GetTableDictionary();

  vdb::vector<uint8_t> buffer;
  int buffer_offset = 0;
  /* save table count */
  int table_count = table_dictionary->size();
  int32_t needed_bytes = vdb::ComputeBytesFor(table_count);
  buffer.resize(needed_bytes);
  buffer_offset += vdb::PutLengthTo(buffer.data() + buffer_offset, table_count);

  /* save tables */
  for (auto &[table_name, table] : *table_dictionary) {
    auto status = table->Save(directory_path);
    if (!status.ok()) {
      return false;
    }
    /* save table name */
    int32_t needed_bytes = vdb::ComputeBytesFor(table_name.length());
    buffer.resize(buffer.size() + needed_bytes + table_name.length());
    buffer_offset +=
        vdb::PutLengthTo(buffer.data() + buffer_offset, table_name.length());
    memcpy(buffer.data() + buffer_offset, table_name.data(),
           table_name.length());
    buffer_offset += table_name.length();
  }
  /* save manifest (# of tables, table names) */
  std::string manifest_file_path(directory_path);
  manifest_file_path.append("/manifest");
  auto status =
      vdb::WriteTo(manifest_file_path, buffer.data(), buffer_offset, 0);
  if (!status.ok()) {
    return false;
  }
  return true;
}

bool PrepareTableSnapshot(const char *table_name) {
  snapshot_lock.Lock();
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
             "Creating Snapshot is Requested. Snapshot Lock is held.");

  auto maybe_table = vdb::GetTable(std::string(table_name));
  if (!maybe_table.ok()) {
    return false;
  }
  auto table = maybe_table.ValueUnsafe();

  vdb::vector<std::shared_ptr<vdb::VectorIndex>> uncompleted_indexes;

  auto segments = table->GetSegments();
  auto index_infos = table->GetIndexInfos();
  if (table->HasIndex()) {
    for (auto &[segment_id, segment] : segments) {
      std::shared_ptr<vdb::IndexHandler> index_handler =
          segment->IndexHandler();
      for (auto &index_info : *index_infos) {
        auto column_id = index_info.GetColumnId();
        size_t num_index = index_handler->Size(column_id);
        for (size_t i = 0; i < num_index; i++) {
          auto index = index_handler->Index(column_id, i);
          if (index->Size() != index->CompleteSize()) {
            uncompleted_indexes.push_back(index);
          }
        }
      }
    }

    while (!uncompleted_indexes.empty()) {
      auto index = uncompleted_indexes.back();
      if (index->Size() == index->CompleteSize()) {
        uncompleted_indexes.pop_back();
      } else {
        sched_yield();
      }
    }
  }
  SYSTEM_LOG(
      vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
      "Creating Snapshot is Started. There are no building indexes now.");

  return true;
}

void PostTableSnapshot() {
  SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
             "Creating Snapshot is Done. Snapshot Lock is released.");
  snapshot_lock.Unlock();
}

bool SaveTableSnapshot(const char *table_name, const char *directory_path_) {
  std::string_view directory_path(directory_path_);
  std::optional<std::string> backup_directory_path;

  if (std::filesystem::exists(directory_path)) {
    // check if it is a directory
    if (!std::filesystem::is_directory(directory_path)) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Failed to save table snapshot: %s is not a directory",
                 directory_path.data());
      return false;
    }

    // check permission to write
    if (access(directory_path_, W_OK) != 0) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
                 "Failed to save table snapshot: %s is not writable",
                 directory_path.data());
      return false;
    }
    vdb::FdManager::GetInstance().CleanupFdsInDirectory(std::string(table_name),
                                                        true);

    // rename directory with backup suffix
    backup_directory_path = std::make_optional(std::string(directory_path));
    backup_directory_path->append(".backup");
    std::filesystem::rename(directory_path, backup_directory_path.value());
  }

  /* create directory */
  if (!std::filesystem::create_directory(directory_path)) {
    // restore the backup directory if it exists
    if (backup_directory_path) {
      std::filesystem::rename(backup_directory_path.value(), directory_path);
    }

    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Failed to create directory: %s", directory_path.data());
    return false;
  }

  /* save snapshot */
  auto maybe_table = vdb::GetTable(std::string(table_name));
  if (!maybe_table.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Failed to get table: %s", table_name);

    // remove the created directory
    std::filesystem::remove_all(directory_path);
    // restore the backup directory if it exists
    if (backup_directory_path) {
      std::filesystem::rename(backup_directory_path.value(), directory_path);
    }
    return false;
  }

  auto table = maybe_table.ValueUnsafe();

  /* save tables */
  auto status = table->Save(directory_path);
  if (!status.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogNotice,
               "Failed to save table snapshot: %s", status.ToString().data());

    // remove the created directory
    std::filesystem::remove_all(directory_path);

    // restore the backup directory if it exists
    if (backup_directory_path) {
      std::filesystem::rename(backup_directory_path.value(), directory_path);
    }
    return false;
  }

  // remove the backup directory if it exists
  if (backup_directory_path) {
    std::filesystem::remove_all(backup_directory_path.value());
  }

  return true;
}

/* load table snapshot */
bool LoadTableSnapshot(const char *table_name_, const char *directory_path_) {
  assert(directory_path_ != nullptr);
  assert(table_name_ != nullptr);

  std::string_view directory_path(directory_path_);
  /* check existence of snapshot */
  if (!std::filesystem::exists(directory_path)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Directory(%s) does not exist.", directory_path.data());
    return false;
  }

  if (!std::filesystem::is_directory(directory_path)) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Path(%s) is not a directory.", directory_path.data());
    return false;
  }

  auto table_dictionary = vdb::GetTableDictionary();
  auto iter = table_dictionary->find(table_name_);
  assert(iter == table_dictionary->end());

  auto table_directory_path = std::string(directory_path_);
  table_directory_path.append("/");
  table_directory_path.append(table_name_);

  /* load the table */
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder(
      std::move(options.SetTableName(table_name_)
                    .SetTableDirectoryPath(table_directory_path)));
  auto maybe_table = builder.Build();
  if (!maybe_table.ok()) {
    SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
               "Failed to load table snapshot: %s",
               maybe_table.status().ToString().data());
    return false;
  }

  auto table = maybe_table.ValueUnsafe();
  table_dictionary->insert({table_name_, table});

  return true;
}

static bool LoadManifest(std::string &file_path, uint64_t &table_count,
                         std::vector<std::string> &table_names) {
  size_t manifest_size = std::filesystem::file_size(file_path);
  ARROW_CAST_OR_NULL(auto buffered_manifest, uint8_t *,
                     AllocateAlignedUseThrow(64, manifest_size));
  auto status = vdb::ReadFrom(file_path, buffered_manifest, manifest_size, 0);
  if (!status.ok()) {
    DeallocateAligned(buffered_manifest, manifest_size);
    return false;
  }

  uint64_t offset = 0;
  /* get table count */
  offset += vdb::GetLengthFrom(buffered_manifest, table_count);

  for (uint64_t i = 0; i < table_count; i++) {
    /* get table names */
    uint64_t table_name_length;
    offset += vdb::GetLengthFrom(buffered_manifest + offset, table_name_length);
    table_names.emplace_back(
        reinterpret_cast<char *>(buffered_manifest + offset),
        table_name_length);
    offset += table_name_length;
  }
  DeallocateAligned(buffered_manifest, manifest_size);
  return true;
}

/* load snapshot */
bool LoadVdbSnapshot(char *directory_path_) {
  std::string_view directory_path(directory_path_);
  /* check existence of snapshot */
  if (!std::filesystem::exists(directory_path)) {
    return false;
  }
  /* check completeness of snapshot
   * if snapshot.manifest file exists, snapshot of all tables was completely
   * saved. */
  std::string manifest_file_path(directory_path);
  manifest_file_path.append("/manifest");
  if (!std::filesystem::exists(manifest_file_path)) {
    return false;
  }
  uint64_t table_count;
  std::vector<std::string> table_names;
  if (!LoadManifest(manifest_file_path, table_count, table_names)) {
    return false;
  }

  /* load all tables */
  for (auto table_name : table_names) {
    if (!LoadTableSnapshot(table_name.c_str(), directory_path_)) {
      SYSTEM_LOG(vdb::LogTopic::Unknown, vdb::LogLevel::kLogDebug,
                 "Failed to load table snapshot: %s", table_name.c_str());
      return false;
    }
  }

  return true;
}

/* garbage collect snapshot */
bool RemoveDirectory(char *directory_path) {
  try {
    if (!std::filesystem::remove_all(directory_path)) {
      return false;
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << __PRETTY_FUNCTION__ << " - Filesystem error: " << e.what()
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << __PRETTY_FUNCTION__ << " - General error: " << e.what()
              << std::endl;
  }
  return true;
}

// Helper function to sync a single file or directory
static bool SyncSingleFile(const std::string &path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    int open_errno = errno;
    // Don't fail for permission denied or other non-critical errors
    if (open_errno == EACCES || open_errno == EPERM) {
      SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogDebug,
                 "Permission denied for %s: errno=%d (%s)", path.c_str(),
                 open_errno, strerror(open_errno));
    } else {
      SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
                 "Failed to open %s: errno=%d (%s)", path.c_str(), open_errno,
                 strerror(open_errno));
    }
    return false;
  }

  // Perform fsync
  bool sync_success = true;
  int fsync_result = fsync(fd);
  if (fsync_result != 0) {
    int fsync_errno = errno;
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Failed to fsync %s: errno=%d (%s)", path.c_str(), fsync_errno,
               strerror(fsync_errno));
    sync_success = false;
  }

  // Close file descriptor
  int close_result = close(fd);
  if (close_result != 0) {
    int close_errno = errno;
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Failed to close %s: errno=%d (%s)", path.c_str(), close_errno,
               strerror(close_errno));
    /* if close is failed, sync can be still successful */
    return sync_success;
  }

  return sync_success;
}

/* Sync all files in a directory
 * return synced file count if all files are synced successfully
 * return 0 if any file is not synced successfully */
int SyncAllFilesInDirectory(const char *directory_path) {
  // Input validation
  if (!directory_path || strlen(directory_path) == 0) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Invalid directory path: null or empty");
    return 0;
  }

  // Check if directory exists and is accessible
  if (!std::filesystem::exists(directory_path)) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Directory does not exist: %s", directory_path);
    return 0;
  }

  if (!std::filesystem::is_directory(directory_path)) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Path is not a directory: %s", directory_path);
    return 0;
  }

  try {
    std::queue<std::string> directory_queue;
    directory_queue.push(std::string(directory_path));

    bool is_all_synced = true;

    // Performance statistics
    int files_synced = 0;
    int directories_processed = 0;

    while (!directory_queue.empty()) {
      std::string current_dir = directory_queue.front();
      directory_queue.pop();

      // First, sync the directory itself
      if (!SyncSingleFile(current_dir)) {
        SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
                   "Failed to sync directory: %s", current_dir.c_str());
        is_all_synced = false;
        break;
      }

      bool is_all_synced_in_directory = true;
      // Iterate through all entries in the current directory
      std::error_code ec;
      auto dir_iterator = std::filesystem::directory_iterator(current_dir, ec);

      // Check if directory iteration had errors
      if (ec) {
        SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
                   "Directory iteration failed for %s: %s", current_dir.c_str(),
                   ec.message().c_str());
        is_all_synced = false;
        break;
      }

      for (const auto &entry : dir_iterator) {
        const std::string file_path = entry.path().string();

        // Skip symbolic links to avoid infinite loops
        if (entry.is_symlink()) {
          SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
                     "Skipping symbolic link: %s", file_path.c_str());
          continue;
        }

        // Add directories to queue for further processing
        if (entry.is_directory()) {
          directory_queue.push(file_path);
          continue;
        }

        // For files: perform open, fsync, close sequence
        if (!SyncSingleFile(file_path)) {
          SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
                     "Failed to sync file: %s", file_path.c_str());
          is_all_synced_in_directory = false;
          break;
        }

        files_synced++;
      }

      if (!is_all_synced_in_directory) {
        is_all_synced = false;
        break;
      }

      directories_processed++;
    }

    if (!is_all_synced) {
      return 0;
    }

    // Log completion statistics
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Sync completed: %d files synced, %d directories processed",
               files_synced, directories_processed);

    return files_synced + directories_processed;
  } catch (const std::filesystem::filesystem_error &e) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "Filesystem error: %s", e.what());
    return 0;
  } catch (const std::exception &e) {
    SYSTEM_LOG(vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
               "General error: %s", e.what());
    return 0;
  }
}

int SyncAllFilesInEmbeddingStore() {
  const char *embedding_store_path =
      vdb::ServerConfiguration::GetEmbeddingStoreRootDirname();

  // Check if directory exists
  if (!std::filesystem::exists(embedding_store_path)) {
    SYSTEM_LOG(
        vdb::LogTopic::Disk, vdb::LogLevel::kLogNotice,
        "EmbeddingStore directory is not created, so syncing is skipped: %s",
        embedding_store_path);
    return 1;
  }

  return SyncAllFilesInDirectory(embedding_store_path);
}

thread_local uint64_t thread_add_task_time = 0;

void VdbMainBeforeAddTask(client *c) {
  c->need_vdb_task_after_reply = 1;
  c->vdb_task_allocation_time = vdb::metrics::Timer::GetCurrentTime();
  if (thread_add_task_time > 0) {
    vdb::metrics::CollectTime(
        vdb::metrics::ThreadPoolAddTaskInterval,
        c->vdb_task_allocation_time - thread_add_task_time);
    if (c->vdb_task_allocation_time - thread_add_task_time < 1000000000) {
      vdb::metrics::CollectTime(
          vdb::metrics::ThreadPoolAddTaskIntervalInBusy,
          c->vdb_task_allocation_time - thread_add_task_time);
    }
  }
  thread_add_task_time = c->vdb_task_allocation_time;
}

std::atomic<uint64_t> active_thread_count(0);
void VdbThreadPoolBeforeTask(client *c) {
  active_thread_count++;
  auto now = vdb::metrics::Timer::GetCurrentTime();
  vdb::metrics::CollectTime(vdb::metrics::ThreadPoolTaskWakeupLatency,
                            now - c->vdb_task_allocation_time);
}

void VdbThreadPoolAfterTask(client *c) {
  c->vdb_task_completion_time = vdb::metrics::Timer::GetCurrentTime();
  auto active_thread_count_ = active_thread_count.fetch_sub(1);
  vdb::metrics::CollectValue(vdb::metrics::ThreadPoolActiveThreadCountAtTaskEnd,
                             active_thread_count_);
  if (active_thread_count_ > 1) {
    vdb::metrics::CollectValue(
        vdb::metrics::ThreadPoolActiveThreadCountAtTaskEndExceptOne,
        active_thread_count_);
  }
}

void VdbTaskAfterReply(client *c) {
  if (c->need_vdb_task_after_reply) {
    auto now = vdb::metrics::Timer::GetCurrentTime();
    vdb::metrics::CollectTime(vdb::metrics::ThreadPoolResponseDeliveryLatency,
                              now - c->vdb_task_completion_time);
  }
  c->need_vdb_task_after_reply = 0;
  c->vdb_task_allocation_time = 0;
  c->vdb_task_completion_time = 0;
}

void AlterSuffix(char *str, const char *suffix) {
  int length = strlen(str);
  int suffix_length = strlen(suffix);
  for (int i = 0; i < suffix_length; i++) {
    str[length - suffix_length + i] = suffix[i];
  }
}
