#pragma once

#include <cstdint>

namespace vdb {

namespace ServerConfiguration {
/* Getters */
int64_t GetActiveSetSizeLimit();
bool GetAllowBgIndexThread();
bool GetEnableInFilter();
char* GetEmbeddingStoreRootDirname();
char* GetLoadDataPath();
char* GetUnloadDataPath();
char* GetSaveDataPath();
int64_t GetVerbosity();
uint64_t GetEnabledLogTopics();
bool GetEnableHnswMtSingleSearch();
int64_t GetIndexThreadNum();
bool GetLoading();
int64_t GetMaxMemoryRetryCount();
uint64_t GetMaxMemoryRetryInterval();
uint64_t GetMaxMemory();
bool GetShutdownAsap();
int64_t GetAofRewriteCount();
int64_t GetBgReadCommandThread();
int GetHiddenCheckMemoryAvailability();
void SetTemporaryOomBlocker(int temporary_oom_blocker);
/* Checkers */
bool IsAofChild();

/* Setters */
void SetAofRewriteScheduled(int aof_rewrite_scheduled);
void IncrementServerDirty();
}  // namespace ServerConfiguration

namespace ServerResources {
/* Getters */
void* GetMetadataCheckers();
void* GetReadCommandPool();
void* GetTableDictionary();

/* Setters */
void SetMetadataCheckers(void* metadata_checkers);
void SetReadCommandPool(void* read_command_pool);
void SetTableDictionary(void* table_dictionary);
}  // namespace ServerResources
}  // namespace vdb
