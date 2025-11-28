#pragma once
#include <unistd.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct client;

// Read command thread pool functions
void InitializeReadCommandThreads(size_t num_threads);
void TerminateReadCommandThreads();

void TotalTableCountCommand(client *c);
void TotalRecordCountCommand(client *c);

void ListTableCommand(client *c);
void CreateTableCommand(client *c);
void DropTableCommand(client *c);
void DescribeTableCommand(client *c);

void InsertCommand(client *c);
void BatchInsertCommand(client *c);
void DebugScanCommand(client *c);
void ScanCommand(client *c);
void ScanOpenCommand(client *c);
void FetchNextCommand(client *c);
void VectorSearchCommand(client *c);
void BatchVectorSearchCommand(client *c);
void DeleteCommand(client *c);
void UpdateCommand(client *c);
void CheckIndexingCommand(client *c);
void CountRecordsCommand(client *c);
void CountIndexedElementsCommand(client *c);
void CountAofRewritesCommand(client *c);
void SegmentStatisticsCommand(client *c);
void LoadCommand(client *c);
void UnloadCommand(client *c);
void SaveCommand(client *c);

/* metrics container commands */
void MetricsPrintCommand(client *c);
void MetricsClearCommand(client *c);

/* log container commands */
void LogSetTopicsCommand(client *c);
void LogSetDefaultTopicsCommand(client *c);
void LogSetAllTopicsCommand(client *c);
void LogResetTopicsCommand(client *c);
void LogResetAllTopicsCommand(client *c);
void LogEnabledTopicsCommand(client *c);
void LogTestCommand(client *c);

void LogSetDefaultTopics();

/* snapshot apis */
void CheckVdbSnapshot();
bool PrepareVdbSnapshot();
bool SaveVdbSnapshot(char *directory_path_);
bool LoadVdbSnapshot(char *directory_path_);
void PostVdbSnapshot();
bool RemoveDirectory(char *directory_path);

int SyncAllFilesInDirectory(const char *directory_path);
int SyncAllFilesInEmbeddingStore();

/* table snapshot apis */

bool PrepareTableSnapshot(const char *table_name);
void PostTableSnapshot();
bool SaveTableSnapshot(const char *table_name, const char *directory_path_);
bool LoadTableSnapshot(const char *table_name, const char *directory_path_);
// Utility functions
void AllocateTableDictionary();
void DeallocateTableDictionary();

// Utility functions
void AllocatePerformanceMonitor();
void DeallocatePerformanceMonitor();

void AllocateFdManager();

void AllocateMetadataCheckers();
void DeallocateMetadataCheckers();

void VdbMainBeforeAddTask(client *c);
void VdbThreadPoolBeforeTask(client *c);
void VdbThreadPoolAfterTask(client *c);
void VdbTaskAfterReply(client *c);

void AlterSuffix(char *str, const char *suffix);
#ifdef __cplusplus
}
#endif
