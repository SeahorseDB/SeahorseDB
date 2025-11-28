#pragma once

#include <string>
#include <filesystem>
#include <gtest/gtest.h>

#include "vdb/common/fd_manager.hh"
#include "vdb/common/system_log.hh"
#include "vdb/tests/util_for_test.hh"

namespace vdb {
/* test root - test suite - test case - test */
std::string test_root_directory_path = "__vdb_test_root__";

extern std::string test_suite_directory_path;

extern std::atomic_uint64_t vdb_allocated_size;

uint64_t initial_vdb_allocated_size = 0;
uint64_t initial_redis_allocated_size = 0;

void ResetCommonRedisConfiguration() {
  /* redis server configuration */
#ifdef _DEBUG_GTEST
  server.verbosity = LL_DEBUG;
#else
  server.verbosity = LL_NOTHING;
#endif
  server.logfile = empty_string.data();
  server.allow_bg_index_thread = false;
  server.enable_in_filter = false;
  server.maxmemory = 0; /* reset max memory */
  server.vdb_active_set_size_limit = 10000;

  static std::string default_log_topics = "all";
  server.default_log_topics = default_log_topics.data();
  server.enabled_log_topics = UINT64_MAX;  // all topics are set

  server.hidden_check_memory_availability =
      1;  // Used with 0 only in MemoryCheckRollbackTest.
  server.hidden_enable_oom_testing =
      0;  // Used with 1 only in MemoryCheckRollbackTest.
}

class BaseEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!std::filesystem::exists(test_root_directory_path)) {
      std::filesystem::create_directory(test_root_directory_path);
    }
    std::filesystem::remove_all(test_suite_directory_path);
    std::filesystem::create_directory(test_suite_directory_path);

    ResetCommonRedisConfiguration();

    initial_vdb_allocated_size = vdb::GetVdbAllocatedSize();
    initial_redis_allocated_size = vdb::GetRedisAllocatedSize();

    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug,
               "Test Suite Directory is re-created: %s",
               test_suite_directory_path.c_str());
  }

  void TearDown() override {
#if _REMOVE_ALL_AFTER_TEST == ON
    std::filesystem::remove_all(test_root_directory_path);
#endif
  }
};

class BaseTestCore {
 protected:
  static void SetUpTestCaseCore(bool is_parameterized_test) {
    std::string full_test_case_name =
        testing::UnitTest::GetInstance()->current_test_case()->name();

    std::string prefix, test_case_name;
    if (is_parameterized_test) {
      // full_test_name: "Prefix/TestCase"
      size_t slash_pos = full_test_case_name.find('/');
      prefix = full_test_case_name.substr(0, slash_pos);           // "Prefix"
      test_case_name = full_test_case_name.substr(slash_pos + 1);  // "TestCase"
    } else {
      // full_test_name: "TestCase"
      prefix = "";
      test_case_name = full_test_case_name;
    }

    /* __vdb_test_root__/TestSuite */
    std::string test_case_directory_path = test_suite_directory_path;
    if (!prefix.empty()) {
      /* __vdb_test_root__/TestSuite/Prefix */
      test_case_directory_path += "/" + prefix;
      std::filesystem::remove_all(test_case_directory_path);
      std::filesystem::create_directory(test_case_directory_path);
    }
    /* __vdb_test_root__/TestSuite/Prefix/TestCase */
    test_case_directory_path += "/" + test_case_name + "Case";

    std::filesystem::remove_all(test_case_directory_path);
    std::filesystem::create_directory(test_case_directory_path);

    ResetCommonRedisConfiguration();

    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug,
               "Test Suite Directory is re-created: %s",
               test_suite_directory_path.c_str());
    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug,
               "Test Case Directory is re-created: %s",
               test_case_directory_path.c_str());
  }

  void SetUpCore(bool is_parameterized_test) {
    initial_vdb_allocated_size = vdb::GetVdbAllocatedSize();
    initial_redis_allocated_size = vdb::GetRedisAllocatedSize();

    AllocateTableDictionary();
    AllocateMetadataCheckers();
    // Note: ScanRegistry is now used instead of ResultSetDictionary
    AllocatePerformanceMonitor();

    ResetCommonRedisConfiguration();

    const testing::TestInfo *test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    std::string full_test_case_name = test_info->test_case_name();
    std::string full_test_name = test_info->name();

    std::string prefix;
    std::string test_case_name;
    std::string test_name;
    std::string test_part_number;
    if (is_parameterized_test) {
      // full_test_case_name: "Prefix/TestCase"
      // full_test_name: "TestName/PartNumber"
      size_t slash_pos = full_test_case_name.find('/');
      test_case_name = full_test_case_name.substr(slash_pos + 1);
      prefix = full_test_case_name.substr(0, slash_pos);

      slash_pos = full_test_name.find('/');
      test_name = full_test_name.substr(0, slash_pos);
      test_part_number = full_test_name.substr(slash_pos + 1);
    } else {
      test_case_name = full_test_case_name;
      test_name = full_test_name;
    }

    /* __vdb_test_root__/TestSuite */
    test_case_directory_path_ = test_suite_directory_path;
    if (!prefix.empty()) {
      /* __vdb_test_root__/TestSuite/Prefix */
      test_case_directory_path_ += "/" + prefix;
    }
    /* __vdb_test_root__/TestSuite/Prefix/TestCase */
    test_case_directory_path_ += "/" + test_case_name + "Case";

    /* __vdb_test_root__/TestSuite/Prefix/TestCase/TestName */
    test_directory_path_ = test_case_directory_path_ + "/" + test_name;

    if (!std::filesystem::exists(test_directory_path_)) {
      std::filesystem::create_directory(test_directory_path_);
    }

    if (!test_part_number.empty()) {
      /* __vdb_test_root__/TestSuite/Prefix/TestCase/TestName/TestPartNumber */
      test_directory_path_ += "/" + test_part_number;
      std::filesystem::remove_all(test_directory_path_);
      std::filesystem::create_directory(test_directory_path_);
    }

    server.aof_filename = snapshot_directory_name_.data();
    server.aof_dirname = test_directory_path_.data();

    /* __vdb_test_root__/TestSuite/Prefix/TestCase/TestName/embedding_store */
    embedding_store_directory_path_ =
        test_directory_path_ + "/" + embedding_store_directory_name_;
    server.embedding_store_root_dirname =
        embedding_store_directory_path_.data();

    std::filesystem::create_directory(embedding_store_directory_path_);

    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug,
               "Test Directory is re-created: %s",
               test_directory_path_.c_str());
    SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug,
               "Embedding Store Directory is re-created: %s",
               embedding_store_directory_path_.c_str());
  }

  void TearDownCore(bool is_parameterized_test) {
    DeallocateTableDictionary();
    DeallocateMetadataCheckers();
    DeallocatePerformanceMonitor();

    FdManager::GetInstance().CleanupAllFds();
    /* check if all fds are closed. if some fds are not closed, it means that
     * some files are not released. */
    ASSERT_EQ(FdManager::GetInstance().GetFdCount(), 0);

    /* memory usage must be same as initial value after finish of test case */
    ASSERT_EQ(vdb::GetVdbAllocatedSize(), initial_vdb_allocated_size)
        << "difference in vdb allocated size: "
        << (vdb::GetVdbAllocatedSize() >= initial_vdb_allocated_size
                ? vdb::GetVdbAllocatedSize() - initial_vdb_allocated_size
                : initial_vdb_allocated_size - vdb::GetVdbAllocatedSize());
    ASSERT_EQ(vdb::GetRedisAllocatedSize(), initial_redis_allocated_size)
        << "difference in redis allocated size: "
        << (vdb::GetRedisAllocatedSize() >= initial_redis_allocated_size
                ? vdb::GetRedisAllocatedSize() - initial_redis_allocated_size
                : initial_redis_allocated_size - vdb::GetRedisAllocatedSize());
  }

  const std::string &TestDirectoryPath() const { return test_directory_path_; }
  const std::string &TestCaseDirectoryPath() const {
    return test_case_directory_path_;
  }
  const std::string &EmbeddingStoreDirectoryPath() const {
    return embedding_store_directory_path_;
  }

 private:
  std::string test_directory_path_;
  std::string test_case_directory_path_;
  std::string embedding_store_directory_path_;
  std::string embedding_store_directory_name_ = "embedding_store";
  std::string snapshot_directory_name_ = "snapshot";
};

class BaseTestSuite : public ::testing::Test, protected BaseTestCore {
 protected:
  static void SetUpTestCase() { SetUpTestCaseCore(false); }

  void SetUp() override { SetUpCore(false); }
  void TearDown() override { TearDownCore(false); }

  using BaseTestCore::EmbeddingStoreDirectoryPath;
  using BaseTestCore::TestCaseDirectoryPath;
  using BaseTestCore::TestDirectoryPath;
};

template <typename T>
class BaseTestSuiteWithParam : public ::testing::TestWithParam<T>,
                               protected BaseTestCore {
 protected:
  static void SetUpTestCase() { SetUpTestCaseCore(true); }

  void SetUp() override { SetUpCore(true); }
  void TearDown() override { TearDownCore(true); }

  using BaseTestCore::EmbeddingStoreDirectoryPath;
  using BaseTestCore::TestCaseDirectoryPath;
  using BaseTestCore::TestDirectoryPath;
};
}  // namespace vdb