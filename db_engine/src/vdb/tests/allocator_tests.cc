
#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <iostream>

#include "vdb/vdb.hh"
#include "vdb/vdb_api.hh"
#include "vdb/tests/base_environment.hh"
#include "vdb/common/memory_allocator.hh"
#include "zmalloc.h"

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/AllocatorTestSuite";

class AllocatorTestSuite : public BaseTestSuite {
 public:
  void AssertMemoryUsageMatchesInitial(bool expect_equal = true) {
    if (expect_equal) {
      ASSERT_EQ(vdb::GetVdbAllocatedSize(), initial_vdb_allocated_size);
      ASSERT_EQ(vdb::GetRedisAllocatedSize(), initial_redis_allocated_size);
    } else {
      ASSERT_NE(vdb::GetVdbAllocatedSize(), initial_vdb_allocated_size);
      ASSERT_NE(vdb::GetRedisAllocatedSize(), initial_redis_allocated_size);
    }
  }

  void AssertMemoryUsageIncreasedBy(size_t expected) {
    ASSERT_EQ(vdb::GetVdbAllocatedSize(),
              initial_vdb_allocated_size + expected);
    ASSERT_EQ(vdb::GetRedisAllocatedSize(),
              initial_redis_allocated_size + expected);
  }

  void AssertMemoryUsageIncreaseWithin(size_t max_increase) {
    ASSERT_LE(vdb::GetVdbAllocatedSize(),
              initial_vdb_allocated_size + max_increase);
    ASSERT_LE(vdb::GetRedisAllocatedSize(),
              initial_redis_allocated_size + max_increase);
  }

 protected:
  void SetUp() override {
    server.maxmemory = 0;
    BaseTestSuite::SetUp();
    DeallocatePerformanceMonitor();
  }
};

class CommonTest : public AllocatorTestSuite {};
class MakeSharedTest : public AllocatorTestSuite {};
class StdWithAllocatorTest : public AllocatorTestSuite {};
class ArrowMemoryPoolTest : public AllocatorTestSuite {};
class MemoryLimitTest : public AllocatorTestSuite {
 protected:
  void SetUp() override {
    AllocatorTestSuite::SetUp();
    /* all memory must be freed before starting test case */
    server.hidden_enable_oom_testing = true;
    server.maxmemory = initial_vdb_allocated_size + 1024 * 1024; /* 1MB */
  }
  void TearDown() override {
    server.hidden_enable_oom_testing = false;
    AllocatorTestSuite::TearDown();
  }
};

TEST_F(CommonTest, NormalVariableTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *, AllocateNoPrefix(sizeof(char)));
    value[0] = 'a';
    /* sizeof(char) (1) */
    expected += 1;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, sizeof(char));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int16_t *,
                       AllocateNoPrefix(sizeof(int16_t)));
    value[0] = 8;
    /* sizeof(int16_t) (2) */
    expected += 2;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, sizeof(int16_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int32_t *,
                       AllocateNoPrefix(sizeof(int32_t)));
    value[0] = 8;
    /* sizeof(int32_t) (4) */
    expected += 4;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, sizeof(int32_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int64_t *,
                       AllocateNoPrefix(sizeof(int64_t)));
    value[0] = 8;
    /* sizeof(int64_t) (8) */
    expected += 8;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, sizeof(int64_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, double *, AllocateNoPrefix(sizeof(double)));
    value[0] = 8.0;
    /* sizeof(double) (8) */
    expected += 8;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, sizeof(double));
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(CommonTest, NormalArrayTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *, AllocateNoPrefix(7 * sizeof(char)));
    value[0] = 'a';
    /* 7 * sizeof(char) (7) */
    expected += 7;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, 7 * sizeof(char));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int16_t *,
                       AllocateNoPrefix(7 * sizeof(int16_t)));
    value[0] = 8;
    /* 7 * sizeof(int16_t) (14) */
    expected += 14;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, 7 * sizeof(int16_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int32_t *,
                       AllocateNoPrefix(7 * sizeof(int32_t)));
    value[0] = 8;
    /* 7 * sizeof(int32_t) (28) */
    expected += 28;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, 7 * sizeof(int32_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int64_t *,
                       AllocateNoPrefix(7 * sizeof(int64_t)));
    value[0] = 8;
    /* 7 * sizeof(int64_t) (56) */
    expected += 56;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, 7 * sizeof(int64_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, double *,
                       AllocateNoPrefix(7 * sizeof(double)));
    value[0] = 8.0;
    /* 7 * sizeof(double) (56) */
    expected += 56;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, 7 * sizeof(double));
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(CommonTest, SizeAwareVariableTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *, AllocateWithPrefix(sizeof(char)));
    value[0] = 'a';
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int16_t *,
                       AllocateWithPrefix(sizeof(int16_t)));
    value[0] = 8;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int32_t *,
                       AllocateWithPrefix(sizeof(int32_t)));
    value[0] = 8;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int64_t *,
                       AllocateWithPrefix(sizeof(int64_t)));
    value[0] = 8;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, double *,
                       AllocateWithPrefix(sizeof(double)));
    value[0] = 8.0;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(CommonTest, SizeAwareArrayTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *,
                       AllocateWithPrefix(7 * sizeof(char)));
    value[0] = 'a';
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int16_t *,
                       AllocateWithPrefix(7 * sizeof(int16_t)));
    value[0] = 8;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int32_t *,
                       AllocateWithPrefix(7 * sizeof(int32_t)));
    value[0] = 8;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int64_t *,
                       AllocateWithPrefix(7 * sizeof(int64_t)));
    value[0] = 8;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, double *,
                       AllocateWithPrefix(7 * sizeof(double)));
    value[0] = 8.0;
    expected += zmalloc_size(value);
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateWithPrefix(value);
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(CommonTest, AlignedVariableTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *, AllocateAligned(64, sizeof(char)));
    value[0] = 'a';
    /* sizeof(char) (1) */
    expected += 1;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, sizeof(char));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int16_t *,
                       AllocateAligned(64, sizeof(int16_t)));
    value[0] = 8;
    /* sizeof(char) (2) */
    expected += 2;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, sizeof(int16_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int32_t *,
                       AllocateAligned(64, sizeof(int32_t)));
    value[0] = 8;
    /* sizeof(char) (4) */
    expected += 4;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, sizeof(int32_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int64_t *,
                       AllocateAligned(64, sizeof(int64_t)));
    value[0] = 8;
    /* sizeof(char) (8) */
    expected += 8;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, sizeof(int64_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, double *,
                       AllocateAligned(64, sizeof(double)));
    value[0] = 8.0;
    /* sizeof(char) (8) */
    expected += 8;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, sizeof(double));
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(CommonTest, AlignedArrayTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *,
                       AllocateAligned(64, 7 * sizeof(char)));
    value[0] = 'a';
    /* 7 * sizeof(char) (7) */
    expected += 7;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, 7 * sizeof(char));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int16_t *,
                       AllocateAligned(64, 7 * sizeof(int16_t)));
    value[0] = 8;
    /* 7 * sizeof(int16_t) (14) */
    expected += 14;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, 7 * sizeof(int16_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int32_t *,
                       AllocateAligned(64, 7 * sizeof(int32_t)));
    value[0] = 8;
    /* 7 * sizeof(int32_t) (28) */
    expected += 28;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, 7 * sizeof(int32_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, int64_t *,
                       AllocateAligned(64, 7 * sizeof(int64_t)));
    value[0] = 8;
    /* 7 * sizeof(int64_t) (56) */
    expected += 56;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, 7 * sizeof(int64_t));
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, double *,
                       AllocateAligned(64, 7 * sizeof(double)));
    value[0] = 8.0;
    /* 7 * sizeof(double) (56) */
    expected += 56;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateAligned(value, 7 * sizeof(double));
  }
  AssertMemoryUsageMatchesInitial();
}
TEST_F(MakeSharedTest, NormalVariableTest) {
  {
    size_t expected = 0;
    auto value = vdb::make_shared<char>('a');
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_shared<int16_t>(8);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_shared<int32_t>(8);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_shared<int64_t>(8);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_shared<double>(8.0);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(MakeSharedTest, SizeAwareVariableTest) {
  {
    size_t expected = 0;
    auto value = vdb::make_size_aware_shared<char>('a');
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_size_aware_shared<int16_t>(8);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_size_aware_shared<int32_t>(8);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_size_aware_shared<int64_t>(8);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
  {
    size_t expected = 0;
    auto value = vdb::make_size_aware_shared<double>(8.0);
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  AssertMemoryUsageMatchesInitial();
}

TEST_F(MakeSharedTest, AlignedVariableTest) {
  {
    size_t expected = 0;
    auto value = vdb::make_aligned_shared<char>(8, 'a');
    /* alignment bytes in allocator (8) */
    expected += 8;
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  {
    size_t expected = 0;
    auto value = vdb::make_aligned_shared<int16_t>(8, 8);
    /* alignment bytes in allocator (8) */
    expected += 8;
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  {
    size_t expected = 0;
    auto value = vdb::make_aligned_shared<int32_t>(8, 8);
    /* alignment bytes in allocator (8) */
    expected += 8;
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  {
    size_t expected = 0;
    auto value = vdb::make_aligned_shared<int64_t>(8, 8);
    /* alignment bytes in allocator (8) */
    expected += 8;
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
  {
    size_t expected = 0;
    auto value = vdb::make_aligned_shared<double>(8, 8.0);
    /* alignment bytes in allocator (8) */
    expected += 8;
    /* aligned bytes (8) */
    expected += 8;
#ifdef __APPLE__
    /* control block size of shared pointer (24) */
    expected += 24;
#else
    /* control block size of shared pointer (16) */
    expected += 16;
#endif
    AssertMemoryUsageIncreasedBy(expected);
  }
}

TEST_F(StdWithAllocatorTest, VectorTest) {
  size_t expected = 0;
  vdb::vector<int> vector;
  AssertMemoryUsageMatchesInitial();
  vector.insert(vector.begin(), 10);
  for (auto val : vector) {
    ASSERT_EQ(val, 10);
  }
  /* sizeof(int) (4) */
  expected += 4;
  AssertMemoryUsageIncreasedBy(expected);
  vector.reserve(10);
  /* sizeof(int) * 9 (36) */
  expected += 36;
  auto vector_shared = vdb::make_shared<vdb::vector<int>>();
  expected += sizeof(vdb::vector<int>);
#ifdef __APPLE__
  /* control block size of shared pointer (24) */
  expected += 24;
#else
  /* control block size of shared pointer (16) */
  expected += 16;
#endif
  AssertMemoryUsageIncreasedBy(expected);
  vector_shared->insert(vector_shared->begin(), 20);
  /* sizeof(int) (4) */
  expected += 4;
  AssertMemoryUsageIncreasedBy(expected);
  vector_shared->insert(vector_shared->begin(), 30);
  /* sizeof(int) (4) */
  expected += 4;
  AssertMemoryUsageIncreasedBy(expected);
}

TEST_F(StdWithAllocatorTest, StringTest) {
  size_t expected = 0;
  vdb::string str;
  AssertMemoryUsageMatchesInitial();
  str.reserve(100);
  str.append("hello world");
  str.append("hello world");
  str.append("hello world");
  /* reserve save (100) */
  expected += 100;
#ifdef __APPLE__
  expected += 4;
#else
  expected += 1;
#endif

  AssertMemoryUsageIncreasedBy(expected);
  auto str_shared = vdb::make_shared<vdb::string>();
  expected += sizeof(vdb::string);
#ifdef __APPLE__
  /* pre-allocated space in string (8) */
  expected += 8;
  /* control block size of shared pointer (24) */
  expected += 24;
#else
  /* control block size of shared pointer (16) */
  expected += 16;
#endif
  AssertMemoryUsageIncreasedBy(expected);
  str_shared->reserve(100);
  str_shared->append("goodbye world");
  str_shared->append("goodbye world");
  str_shared->append("goodbye world");
  expected += 100;
#ifdef __APPLE__
  expected += 4;
#else
  expected += 1;
#endif
  AssertMemoryUsageIncreasedBy(expected);
}

TEST_F(StdWithAllocatorTest, MapTest) {
  size_t expected = 0;
  vdb::map<int, int> map;
  AssertMemoryUsageMatchesInitial();
  map.insert(std::pair<int, int>(1, 2));
  expected += 40;
  AssertMemoryUsageIncreasedBy(expected);
  map.insert(std::pair<int, int>(2, 3));
  expected += 40;
  AssertMemoryUsageIncreasedBy(expected);
  auto map_shared = vdb::make_shared<vdb::map<int, int>>();
#ifdef __APPLE__
  expected += 48;
#else
  expected += 64;
#endif
  AssertMemoryUsageIncreasedBy(expected);
  map_shared->insert(std::pair<int, int>(1, 2));
  expected += 40;
  AssertMemoryUsageIncreasedBy(expected);
}

TEST_F(ArrowMemoryPoolTest, BuilderTest) {
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[256, Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  AssertMemoryUsageMatchesInitial();
  auto maybe_rb = GenerateRecordBatch(schema, 9700, 256);
  AssertMemoryUsageMatchesInitial(false);
}

TEST_F(MemoryLimitTest, SimpleTest) {
  {
    size_t expected = 0;
    ASSERT_OK_AND_CAST(auto value, char *, AllocateNoPrefix(sizeof(char)));
    value[0] = 'a';
    /* sizeof(char) (1) */
    expected += 1;
    AssertMemoryUsageIncreasedBy(expected);
    DeallocateNoPrefix(value, sizeof(char));
  }
  {
    /* return arrow::Status::OutOfMemory */
    size_t expected = 0;
    ARROW_CAST_OR_NULL(auto value, char *,
                       AllocateNoPrefix(1024 * 1024 * 2 * sizeof(char)));
    ASSERT_EQ(value, nullptr);
    AssertMemoryUsageIncreasedBy(expected);
  }
  {
    /* throw std::runtime_error */
    size_t expected = 0;
    bool throw_catched = false;
    try {
      auto result = AllocateNoPrefixUseThrow(1024 * 1024 * 2 * sizeof(char));
      ASSERT_TRUE(false);
    } catch (std::runtime_error &e) {
      throw_catched = true;
    }
    ASSERT_TRUE(throw_catched);
    AssertMemoryUsageIncreasedBy(expected);
  }
}

TEST_F(MemoryLimitTest, StdStructureTest) {
  {
    size_t expected = 0;
    vdb::vector<uint8_t> vec;
    vec.reserve(1);
    expected += 1;
    AssertMemoryUsageIncreasedBy(expected);
  }
  {
    /* throw std::runtime_error */
    size_t expected = 0;
    bool throw_catched = false;
    vdb::vector<uint8_t> vec;
    try {
      vec.reserve(2 * 1024 * 1024);
      ASSERT_TRUE(false);
    } catch (std::runtime_error &e) {
      throw_catched = true;
    }
    ASSERT_TRUE(throw_catched);
    AssertMemoryUsageIncreasedBy(expected);
  }
}

TEST_F(MemoryLimitTest, ArrowMemoryPoolTest) {
  std::string table_schema_string =
      "Id int32 not null, Name String, Height float32, feature "
      "Fixed_Size_List[256, Float32 ]";
  auto schema = vdb::ParseSchemaFrom(table_schema_string);
  {
    AssertMemoryUsageMatchesInitial();
    auto maybe_rb = GenerateRecordBatch(schema, 10, 256);
    AssertMemoryUsageMatchesInitial(false);
  }
  {
    AssertMemoryUsageMatchesInitial();
    auto maybe_rb = GenerateRecordBatch(schema, 9700, 1024);
    AssertMemoryUsageIncreaseWithin(1024 * 1024);
  }
}

}  // namespace vdb

int main(int argc, char **argv) {
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
