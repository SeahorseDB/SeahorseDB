#include <string.h>
#include <stdio.h>

#include "vdb/common/server_configuration.hh"

#ifdef __cplusplus
extern "C" {
#include "server.h"
}
#endif

namespace vdb {

namespace ServerConfiguration {

int64_t GetActiveSetSizeLimit() { return server.vdb_active_set_size_limit; }

bool GetAllowBgIndexThread() { return server.allow_bg_index_thread != 0; }

bool GetEnableInFilter() { return server.enable_in_filter != 0; }

int64_t GetBgReadCommandThread() { return server.read_command_thread_num; }

char* GetEmbeddingStoreRootDirname() {
  return server.embedding_store_root_dirname;
}

char* GetLoadDataPath() { return server.load_data_path; }

char* GetUnloadDataPath() { return server.unload_data_path; }

char* GetSaveDataPath() { return server.save_data_path; }

int64_t GetVerbosity() { return server.verbosity; }

uint64_t GetEnabledLogTopics() { return server.enabled_log_topics; }

bool GetEnableHnswMtSingleSearch() {
  return server.enable_hnsw_mt_single_search != 0;
}

int64_t GetIndexThreadNum() { return server.index_thread_num; }

bool GetLoading() { return server.loading != 0; }

int64_t GetMaxMemoryRetryCount() { return server.max_memory_retry_count; }

uint64_t GetMaxMemoryRetryInterval() {
  return server.max_memory_retry_interval;
}

uint64_t GetMaxMemory() { return server.maxmemory; }

bool GetShutdownAsap() { return server.shutdown_asap; }

int64_t GetAofRewriteCount() { return server.stat_aof_rewrites; }

bool IsAofChild() { return server.child_type == CHILD_TYPE_AOF; }

void SetAofRewriteScheduled(int aof_rewrite_scheduled) {
  server.aof_rewrite_scheduled = aof_rewrite_scheduled;
}

void IncrementServerDirty() { server.dirty++; }

int GetHiddenCheckMemoryAvailability() {
  return server.hidden_check_memory_availability;
}

void SetTemporaryOomBlocker(int temporary_oom_blocker) {
  server.temporary_oom_blocker = temporary_oom_blocker;
}

}  // namespace ServerConfiguration

namespace ServerResources {

void* GetMetadataCheckers() { return server.metadata_checkers; }

void SetMetadataCheckers(void* metadata_checkers) {
  server.metadata_checkers = metadata_checkers;
}

void* GetReadCommandPool() { return server.read_command_pool; }

void SetReadCommandPool(void* read_command_pool) {
  server.read_command_pool = read_command_pool;
}

void* GetTableDictionary() { return server.table_dictionary; }

void SetTableDictionary(void* table_dictionary) {
  server.table_dictionary = table_dictionary;
}

}  // namespace ServerResources

}  // namespace vdb
