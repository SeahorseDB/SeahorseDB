#include <algorithm>
#include <filesystem>

#include <arrow/api.h>
#include <arrow/testing/gtest_util.h>

#include <gtest/gtest.h>

#include "vdb/vdb_api.hh"
#include "vdb/common/defs.hh"
#include "vdb/common/util.hh"
#include "vdb/common/status.hh"
#include "vdb/data/expression.hh"
#include "vdb/data/metadata.hh"

#include "vdb/tests/base_environment.hh"

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/UtilTestSuite";

class UtilTestSuite : public BaseTestSuite {
 protected:
  void SetUp() override {
    BaseTestSuite::SetUp();
    server.vdb_active_set_size_limit = 1000;
  }
};

class UtilityTest : public UtilTestSuite {};
class ArrowRecordbatchTest : public UtilTestSuite {
 public:
  std::shared_ptr<arrow::Schema> schema;
  void SetUp() override {
    UtilTestSuite::SetUp();
    auto nested_type =
        arrow::fixed_size_list(arrow::field("item", arrow::float32()), 1024);
    schema = arrow::schema({arrow::field("large_list", nested_type)});
  }
};

class FileSystemTest : public UtilTestSuite {};

TEST_F(UtilityTest, StringTokenizerTest) {
  std::string test_string =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
      "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
      "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
      "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
      "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
      "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
      "mollit anim id est laborum.";

  auto token = vdb::GetTokenFrom(test_string, ' ', 0);
  EXPECT_EQ(token, "Lorem");

  token = vdb::GetTokenFrom(test_string, ' ', 4);
  EXPECT_EQ(token, "amet,");

  token = vdb::GetTokenFrom(test_string, ' ', -1);
  EXPECT_EQ(token, "");

  token = vdb::GetTokenFrom(test_string, ' ', 128);
  EXPECT_EQ(token, "");

  auto tokens = vdb::Tokenize(test_string);

  EXPECT_EQ(tokens.size(), 69);

  tokens = vdb::Tokenize(test_string, '.');

  EXPECT_EQ(tokens.size(), 4);

  tokens = vdb::Split(test_string, ".");

  EXPECT_EQ(tokens.size(), 5);

  std::string_view test_view{test_string};

  tokens = vdb::Tokenize(test_view);

  EXPECT_EQ(tokens.size(), 69);

  tokens = vdb::Tokenize(test_view, '.');

  EXPECT_EQ(tokens.size(), 4);

  tokens = vdb::Split(test_view, ".");

  EXPECT_EQ(tokens.size(), 5);

  // Note: test_string.clear() doens't change the content, but the size.
  // test_view doesn't changed.
  std::transform(test_string.begin(), test_string.end(), test_string.begin(),
                 [](char c) {
                   if (c == ' ')
                     return '.';
                   else
                     return c;
                 });

  tokens = vdb::Tokenize(test_view);

  EXPECT_EQ(tokens.size(), 1);

  tokens = vdb::Tokenize(test_view, '.');

  EXPECT_EQ(tokens.size(), 69);

  tokens = vdb::Split(test_view, ".");

  EXPECT_EQ(tokens.size(), 73);
}

TEST_F(UtilityTest, JoinTest) {
  std::vector<std::string> tokens = {"Hello", "World", "Nice", "Weather"};

  auto str = vdb::Join(tokens, ' ');

  EXPECT_EQ(str, "Hello World Nice Weather");

  str = vdb::Join(tokens, '\u001e');

  EXPECT_EQ(str, "Hello\u001eWorld\u001eNice\u001eWeather");
}

TEST_F(UtilityTest, MetadataKeyTest) {
  EXPECT_EQ(vdb::GetCheckerRegistry().size(),
            5);  // Check that we have 9 registered metadata checkers

  // Tests for IsReservedMetadataKey
  EXPECT_TRUE(IsReservedMetadataKey("segmentation_info"));
  EXPECT_TRUE(IsReservedMetadataKey("table name"));
  EXPECT_TRUE(IsReservedMetadataKey("active_set_size_limit"));
  EXPECT_TRUE(IsReservedMetadataKey("index_info"));
  EXPECT_TRUE(IsReservedMetadataKey("max_threads"));

  EXPECT_FALSE(IsReservedMetadataKey("non_reserved_key"));
  EXPECT_FALSE(IsReservedMetadataKey(""));
  EXPECT_FALSE(IsReservedMetadataKey(
      " segmentation_info "));  // Check surrounding spaces

  // Tests for IsImmutable
  EXPECT_TRUE(IsImmutable("active_set_size_limit"));

  EXPECT_FALSE(IsImmutable("segmentation_info"));
  EXPECT_FALSE(IsImmutable("table name"));
  EXPECT_FALSE(IsImmutable("non_reserved_key"));
  EXPECT_FALSE(IsImmutable(""));
  EXPECT_FALSE(
      IsImmutable(" active_set_size_limit "));  // Check surrounding spaces
}

TEST_F(UtilityTest, ExpressionParserTest) {
  auto schema = arrow::schema(
      {arrow::field("a", arrow::int32()), arrow::field("b", arrow::int32()),
       arrow::field("c", arrow::int32()), arrow::field("d", arrow::int32()),
       arrow::field("e", arrow::utf8())});

  vdb::expression::ExpressionBuilder builder(schema);

  std::vector<std::pair<std::string, std::string>> test_filter_cases = {
      {"(a = 1 aND b !=2) OR (c = 3 AnD d>=4 anD e='abc')",
       "((a = 1 AND b != 2) OR (c = 3 AND d >= 4 AND e = 'abc'))"},
      {"a = 1 AND b!= 2 OR c = 3 AND d = 4 AND e not LIKE 'abc'",
       "((a = 1 AND b != 2) OR (c = 3 AND d = 4 AND e NOT LIKE 'abc'))"},
      {"a = 1 AND b = 2 oR nOT (c = 3 AND d = 4 AND e='abc')",
       "((a = 1 AND b = 2) OR NOT (c = 3 AND d = 4 AND e = 'abc'))"},
      {"a = 1 AND b = 2 Or not (c = 3 OR d = 4 AND e='abc')",
       "((a = 1 AND b = 2) OR NOT (c = 3 OR (d = 4 AND e = 'abc')))"},
      {"a in (1, 2, 3) AND b IS null", "(a IN [1, 2, 3] AND b IS NULL)"},
      {R"(e Like '%pattern%' OR b is nOT nuLL)",
       R"del((e LIKE '%pattern%' OR b IS NOT NULL))del"},
      {"(a=1 and b !=2 or (c>=1 and d != 50 and (e like '70' AND a > 10)))",
       "((a = 1 AND b != 2) OR (c >= 1 AND d != 50 AND (e LIKE '70' AND a > "
       "10)))"}};
  for (const auto &[test_filter, expected_expr] : test_filter_cases) {
    std::cout << "Testing: " << test_filter << std::endl;
    auto expr = builder.ParseFilter(test_filter);
    if (!expr.ok()) {
      std::cerr << expr.status().message() << std::endl;
    }
    ASSERT_TRUE(expr.ok());
    EXPECT_STREQ(expr.ValueOrDie()->ToString().c_str(), expected_expr.c_str());
  }
}

TEST_F(ArrowRecordbatchTest, SerializeDeserializeUnder8GB) {
  arrow::FixedSizeListBuilder builder(arrow::default_memory_pool(),
                                      std::make_shared<arrow::FloatBuilder>(),
                                      1024);

  for (int i = 0; i < 1000; ++i) {
    ASSERT_OK(builder.Append());
    for (int j = 0; j < 1024; ++j) {
      auto *float_builder =
          dynamic_cast<arrow::FloatBuilder *>(builder.value_builder());
      ASSERT_NE(float_builder, nullptr);
      for (int j = 0; j < 1024; ++j) {
        ASSERT_OK(float_builder->Append(float(j)));
      }
    }
  }

  std::shared_ptr<arrow::Array> array;
  ASSERT_OK(builder.Finish(&array));

  auto record_batch = arrow::RecordBatch::Make(schema, 1000, {array});
  auto test_file_path = TestDirectoryPath() + "/RbUnder8GB.bin";
  auto status = vdb::_SaveRecordBatchTo(test_file_path, record_batch);
  ASSERT_TRUE(status.ok()) << status.ToString() << std::endl;

  auto result = vdb::_LoadRecordBatchFrom(test_file_path, schema);
  ASSERT_TRUE(result.ok());

  auto [loaded_record_batch, buffer] = result.ValueOrDie();

  ASSERT_TRUE(record_batch->Equals(*loaded_record_batch))
      << record_batch->ToString() << std::endl
      << loaded_record_batch->ToString() << std::endl;
}
TEST_F(ArrowRecordbatchTest, SerializeDeserializeOver8GB) {
  int64_t list_size = 1024 * sizeof(float);
  int64_t num_lists = (10L * 1024 * 1024 * 1024) / list_size;

  arrow::FixedSizeListBuilder builder(arrow::default_memory_pool(),
                                      std::make_shared<arrow::FloatBuilder>(),
                                      1024);

  for (int64_t i = 0; i < num_lists; ++i) {
    ASSERT_OK(builder.Append());
    auto *float_builder =
        dynamic_cast<arrow::FloatBuilder *>(builder.value_builder());
    ASSERT_NE(float_builder, nullptr);
    for (int j = 0; j < 1024; ++j) {
      ASSERT_OK(float_builder->Append(float(j)));
    }
  }

  std::shared_ptr<arrow::Array> array;
  ASSERT_OK(builder.Finish(&array));

  auto record_batch = arrow::RecordBatch::Make(schema, num_lists, {array});
  auto test_file_path = TestDirectoryPath() + "/RbOver8GB.bin";
  auto status = vdb::_SaveRecordBatchTo(test_file_path, record_batch);
  ASSERT_TRUE(status.ok()) << status.ToString() << std::endl;

  auto result = vdb::_LoadRecordBatchFrom(test_file_path, schema);
  ASSERT_TRUE(result.ok());
  auto [loaded_record_batch, buffer] = result.ValueOrDie();

  ASSERT_TRUE(record_batch->Equals(*loaded_record_batch));
}

TEST_F(UtilityTest, StringToNumberConversion) {
  // Test stoi
  {
    // Valid cases
    int result;
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("123"));
    EXPECT_EQ(result, 123);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("-123"));
    EXPECT_EQ(result, -123);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("2147483647"));
    EXPECT_EQ(result, 2147483647);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("-2147483648"));
    EXPECT_EQ(result, -2147483648);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("0"));
    EXPECT_EQ(result, 0);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("0x123"));
    EXPECT_EQ(result, 0x123);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("0x123abc"));
    EXPECT_EQ(result, 0x123abc);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi32("0x123abc"));
    EXPECT_EQ(result, 0x123abc);

    // Invalid cases
    EXPECT_FALSE(vdb::stoi32("").ok());
    EXPECT_FALSE(vdb::stoi32("+123").ok());
    EXPECT_FALSE(vdb::stoi32(" 123 ").ok());
    EXPECT_FALSE(vdb::stoi32("\t123\n").ok());
    EXPECT_FALSE(vdb::stoi32("123abc").ok());
    EXPECT_FALSE(vdb::stoi32("1.23").ok());
    EXPECT_FALSE(vdb::stoi32("1 2 3").ok());

    EXPECT_FALSE(vdb::stoi32("1e3").ok());
    EXPECT_FALSE(vdb::stoi32("1,234").ok());
    EXPECT_FALSE(vdb::stoi32("  -123  ").ok());
    EXPECT_FALSE(vdb::stoi32("  +123  ").ok());
    EXPECT_FALSE(vdb::stoi32("abc").ok());
    EXPECT_FALSE(vdb::stoi32("12345678901234567890").ok());   // overflow
    EXPECT_FALSE(vdb::stoi32("-12345678901234567890").ok());  // underflow
    EXPECT_FALSE(vdb::stoi32("--123").ok());                  // double minus
    EXPECT_FALSE(vdb::stoi32("++123").ok());
    EXPECT_FALSE(vdb::stoi32("１２３").ok());  // double plus
  }

  // Test stoll
  {
    // Valid cases
    long long result;
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi64("1234567890123456789"));
    EXPECT_EQ(result, 1234567890123456789LL);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi64("-1234567890123456789"));
    EXPECT_EQ(result, -1234567890123456789LL);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoi64("0"));
    EXPECT_EQ(result, 0LL);

    // Invalid cases
    EXPECT_FALSE(vdb::stoi64("").ok());
    EXPECT_FALSE(vdb::stoi64("abc").ok());
    EXPECT_FALSE(vdb::stoi64("1234567890123456789012345678901234567890").ok());
    EXPECT_FALSE(vdb::stoi64("+1234567890123456789").ok());
    EXPECT_FALSE(vdb::stoi64(" 1234567890123456789 ").ok());
    EXPECT_FALSE(vdb::stoi64("--1234567890123456789").ok());
    EXPECT_FALSE(vdb::stoi64("++1234567890123456789").ok());
    EXPECT_FALSE(vdb::stoi64("0x1234567890123456789").ok());
    EXPECT_FALSE(vdb::stoi64("\t1234567890123456789\n").ok());
    EXPECT_FALSE(vdb::stoi64("1e19").ok());
    EXPECT_FALSE(vdb::stoi64("1,234").ok());
    EXPECT_FALSE(vdb::stoi64("1.23").ok());
    EXPECT_FALSE(vdb::stoi64("1 2 3").ok());
    EXPECT_FALSE(vdb::stoi64("１２３４５６７８９０１２３４５６７８９").ok());
  }

  // Test stoull
  {
    // Valid cases
    unsigned long long result;
    ASSERT_OK_AND_ASSIGN(result, vdb::stoui64("12345678901234567890"));
    EXPECT_EQ(result, 12345678901234567890ULL);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoui64("18446744073709551615"));
    EXPECT_EQ(result, 18446744073709551615ULL);
    ASSERT_OK_AND_ASSIGN(result, vdb::stoui64("0"));
    EXPECT_EQ(result, 0ULL);

    // Invalid casesx
    EXPECT_FALSE(vdb::stoui64("").ok());
    EXPECT_FALSE(vdb::stoui64("abc").ok());
    EXPECT_FALSE(vdb::stoui64("-123").ok());
    EXPECT_FALSE(vdb::stoui64("1.23").ok());
    EXPECT_FALSE(vdb::stoui64(" 12345678901234567890 ").ok());
    EXPECT_FALSE(vdb::stoui64("\t12345678901234567890\n").ok());
    EXPECT_FALSE(vdb::stoui64("+12345678901234567890").ok());
    EXPECT_FALSE(vdb::stoui64("1234567890123456789012345678901234567890").ok());
    EXPECT_FALSE(vdb::stoui64("--12345678901234567890").ok());
    EXPECT_FALSE(vdb::stoui64("++12345678901234567890").ok());
    EXPECT_FALSE(vdb::stoui64("1 2 3 4 5 6 7 8 9 0").ok());
    EXPECT_FALSE(vdb::stoui64("0x12345678901234567890").ok());
    EXPECT_FALSE(vdb::stoui64("1e19").ok());
    EXPECT_FALSE(vdb::stoui64("1,234,567,890,123,456,789,0").ok());
    EXPECT_FALSE(vdb::stoui64("１２３４５６７８９０１２３４５６７８９０").ok());
  }

  // Test stof
  {
    // Valid cases
    float result;
    ASSERT_OK_AND_ASSIGN(result, vdb::stof("123.456"));
    EXPECT_FLOAT_EQ(result, 123.456f);
    ASSERT_OK_AND_ASSIGN(result, vdb::stof("-123.456"));
    EXPECT_FLOAT_EQ(result, -123.456f);
    ASSERT_OK_AND_ASSIGN(result, vdb::stof("0"));
    EXPECT_FLOAT_EQ(result, 0.0f);
    ASSERT_OK_AND_ASSIGN(result, vdb::stof("+123.456"));
    EXPECT_FLOAT_EQ(result, 123.456f);
    ASSERT_OK_AND_ASSIGN(result, vdb::stof("1.23e-4"));
    EXPECT_FLOAT_EQ(result, 1.23e-4f);

    // Invalid cases
    EXPECT_FALSE(vdb::stof("").ok());
    EXPECT_FALSE(vdb::stof("abc").ok());
    EXPECT_FALSE(vdb::stof("123.456abc").ok());
    EXPECT_FALSE(vdb::stof("\t123.456\n").ok());
    EXPECT_FALSE(vdb::stof("  +123.456  ").ok());
    EXPECT_FALSE(vdb::stof("  -123.456  ").ok());
    EXPECT_FALSE(vdb::stof(" 123.456 ").ok());
    EXPECT_FALSE(vdb::stof("1.2.3").ok());
    EXPECT_FALSE(vdb::stof("--123.456").ok());
    EXPECT_FALSE(vdb::stof("++123.456").ok());
    EXPECT_FALSE(vdb::stof("1 2 3.4 5 6").ok());
    EXPECT_FALSE(vdb::stof("0x123.456").ok());
    EXPECT_FALSE(vdb::stof("1,234.567").ok());
    EXPECT_FALSE(vdb::stof("１２３.４５６").ok());
  }

  // Test stod
  {
    // Valid cases
    double result;
    ASSERT_OK_AND_ASSIGN(result, vdb::stod("123.456"));
    EXPECT_DOUBLE_EQ(result, 123.456);
    ASSERT_OK_AND_ASSIGN(result, vdb::stod("-123.456"));
    EXPECT_DOUBLE_EQ(result, -123.456);
    ASSERT_OK_AND_ASSIGN(result, vdb::stod("0"));
    EXPECT_DOUBLE_EQ(result, 0.0);
    ASSERT_OK_AND_ASSIGN(result, vdb::stod("+123.456"));
    EXPECT_DOUBLE_EQ(result, 123.456);
    ASSERT_OK_AND_ASSIGN(result, vdb::stod("1.23e-4"));
    EXPECT_DOUBLE_EQ(result, 1.23e-4);

    // Invalid cases
    EXPECT_FALSE(vdb::stod("").ok());
    EXPECT_FALSE(vdb::stod("abc").ok());
    EXPECT_FALSE(vdb::stod("123.456abc").ok());
    EXPECT_FALSE(vdb::stod("\t123.456\n").ok());
    EXPECT_FALSE(vdb::stod("  +123.456  ").ok());
    EXPECT_FALSE(vdb::stod("  -123.456  ").ok());
    EXPECT_FALSE(vdb::stod(" 123.456 ").ok());
    EXPECT_FALSE(vdb::stod("1.2.3").ok());
    EXPECT_FALSE(vdb::stod("--123.456").ok());
    EXPECT_FALSE(vdb::stod("++123.456").ok());
    EXPECT_FALSE(vdb::stod("1 2 3.4 5 6").ok());
    EXPECT_FALSE(vdb::stod("0x123.456").ok());
    EXPECT_FALSE(vdb::stod("1,234.567").ok());
    EXPECT_FALSE(vdb::stod("１２３.４５６").ok());
  }

  // Test stobool
  {
    // Valid cases
    bool result;
    ASSERT_OK_AND_ASSIGN(result, vdb::stobool("true"));
    EXPECT_TRUE(result);
    ASSERT_OK_AND_ASSIGN(result, vdb::stobool("false"));
    EXPECT_FALSE(result);
    ASSERT_OK_AND_ASSIGN(result, vdb::stobool("1"));
    EXPECT_TRUE(result);
    ASSERT_OK_AND_ASSIGN(result, vdb::stobool("0"));
    EXPECT_FALSE(result);

    // Invalid cases
    EXPECT_FALSE(vdb::stobool("").ok());
    EXPECT_FALSE(vdb::stobool("abc").ok());
    EXPECT_FALSE(vdb::stobool("123").ok());
    EXPECT_FALSE(vdb::stobool("0x123").ok());
    EXPECT_FALSE(vdb::stobool("1.23").ok());
    EXPECT_FALSE(vdb::stobool("1 2 3").ok());
    EXPECT_FALSE(vdb::stobool("0x123.456").ok());
    EXPECT_FALSE(vdb::stobool("1,234.567").ok());
    EXPECT_FALSE(vdb::stobool("１２３.４５６").ok());
  }
}

// Basic functionality test
TEST_F(FileSystemTest, SyncAllFilesInDirectoryBasicTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_basic_test";
  std::filesystem::create_directories(test_dir);

  // Create some files and subdirectories
  std::ofstream file1(test_dir + "/file1.txt");
  file1 << "Hello World" << std::endl;
  file1.close();

  std::string subdir = test_dir + "/subdir";
  std::filesystem::create_directories(subdir);

  std::ofstream file2(subdir + "/file2.txt");
  file2 << "Hello Subdir" << std::endl;
  file2.close();

  // Test should succeed
  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 4 files 1 directory */
  EXPECT_EQ(result, 4);
}

// Test with empty directory
TEST_F(FileSystemTest, SyncAllFilesInDirectoryEmptyTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_empty_test";
  std::filesystem::create_directories(test_dir);

  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 0 files 1 directory */
  EXPECT_EQ(result, 1);
}

// Test with nested directories
TEST_F(FileSystemTest, SyncAllFilesInDirectoryNestedTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_nested_test";
  std::filesystem::create_directories(test_dir);

  // Create deeply nested structure
  std::string level1 = test_dir + "/level1";
  std::string level2 = level1 + "/level2";
  std::string level3 = level2 + "/level3";

  std::filesystem::create_directories(level3);

  // Add files at each level
  std::ofstream(test_dir + "/root.txt") << "root" << std::endl;
  std::ofstream(level1 + "/level1.txt") << "level1" << std::endl;
  std::ofstream(level2 + "/level2.txt") << "level2" << std::endl;
  std::ofstream(level3 + "/level3.txt") << "level3" << std::endl;

  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 4 files 4 directories */
  EXPECT_EQ(result, 8);
}

// Test with permission denied scenario
TEST_F(FileSystemTest, SyncAllFilesInDirectoryPermissionTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_permission_test";
  std::filesystem::create_directories(test_dir);

  // Create a file and subdirectory
  std::ofstream accessible_file(test_dir + "/accessible.txt");
  accessible_file << "accessible content" << std::endl;
  accessible_file.close();

  std::string restricted_dir = test_dir + "/restricted";
  std::filesystem::create_directories(restricted_dir);

  std::ofstream restricted_file(restricted_dir + "/file.txt");
  restricted_file << "restricted content" << std::endl;
  restricted_file.close();

  // Create directory with read-only permission
  std::filesystem::permissions(restricted_dir,
                               std::filesystem::perms::owner_read);

  // If the directory is created with write permission or user has root
  // permission, it can be written to
  auto perms = std::filesystem::status(restricted_dir).permissions();
  bool is_read_only = (perms & std::filesystem::perms::owner_write) ==
                      std::filesystem::perms::none;

  bool is_root = (geteuid() == 0);

  if (is_read_only && !is_root) {
    // Test should fail when trying to sync restricted directory
    int result = SyncAllFilesInDirectory(test_dir.c_str());
    EXPECT_EQ(result, 0);

    // Restore permissions for cleanup
    std::filesystem::permissions(restricted_dir,
                                 std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::add);

    // Now it should succeed
    result = SyncAllFilesInDirectory(test_dir.c_str());
    /* 3 files 1 directory */
    EXPECT_EQ(result, 4);
  }
}

// Test with symbolic links (should be skipped)
TEST_F(FileSystemTest, SyncAllFilesInDirectorySymlinkTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_symlink_test";
  std::filesystem::create_directories(test_dir);

  // Create a regular file
  std::ofstream regular_file(test_dir + "/regular.txt");
  regular_file << "regular file content" << std::endl;
  regular_file.close();

  // Create a symbolic link to the file
  std::filesystem::create_symlink(test_dir + "/regular.txt",
                                  test_dir + "/symlink.txt");

  // Create a symbolic link to a directory
  std::string target_dir = test_dir + "/target_dir";
  std::filesystem::create_directories(target_dir);
  std::filesystem::create_directory_symlink(target_dir,
                                            test_dir + "/symlink_dir");

  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 1 file 2 directory */
  EXPECT_EQ(result, 3);
}

// Test with invalid input parameters
TEST_F(FileSystemTest, SyncAllFilesInDirectoryInvalidInputTest) {
  // Test with null pointer
  int result1 = SyncAllFilesInDirectory(nullptr);
  EXPECT_EQ(result1, 0);

  // Test with empty string
  int result2 = SyncAllFilesInDirectory("");
  EXPECT_EQ(result2, 0);

  // Test with non-existent directory
  int result3 = SyncAllFilesInDirectory("/path/that/does/not/exist");
  EXPECT_EQ(result3, 0);

  // Test with a file path instead of directory
  std::string test_dir = TestDirectoryPath() + "/sync_invalid_test";
  std::filesystem::create_directories(test_dir);

  std::string file_path = test_dir + "/not_a_directory.txt";
  std::ofstream file(file_path);
  file << "I'm a file, not a directory" << std::endl;
  file.close();

  int result4 = SyncAllFilesInDirectory(file_path.c_str());
  EXPECT_EQ(result4, 0);
}

// Test with large number of files (stress test)
TEST_F(FileSystemTest, SyncAllFilesInDirectoryStressTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_stress_test";
  std::filesystem::create_directories(test_dir);

  // Create multiple directories with files
  const int num_dirs = 5;
  const int files_per_dir = 10;

  for (int dir_idx = 0; dir_idx < num_dirs; ++dir_idx) {
    std::string sub_dir = test_dir + "/dir_" + std::to_string(dir_idx);
    std::filesystem::create_directories(sub_dir);

    for (int file_idx = 0; file_idx < files_per_dir; ++file_idx) {
      std::string file_path =
          sub_dir + "/file_" + std::to_string(file_idx) + ".txt";
      std::ofstream file(file_path);
      file << "Content for file " << file_idx << " in directory " << dir_idx
           << std::endl;
      file.close();
    }
  }

  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 50 files 6 directories */
  EXPECT_EQ(result, 56);
}

// Test with special characters in filenames
TEST_F(FileSystemTest, SyncAllFilesInDirectorySpecialCharsTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_special_test";
  std::filesystem::create_directories(test_dir);

  // Create files with special characters (safe ones for filesystem)
  std::vector<std::string> special_names = {
      "file with spaces.txt", "file_with_underscores.txt",
      "file-with-dashes.txt", "file.with.dots.txt", "file123numbers.txt"};

  for (const auto &name : special_names) {
    std::ofstream file(test_dir + "/" + name);
    file << "Content for " << name << std::endl;
    file.close();
  }

  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 5 files 1 directory */
  EXPECT_EQ(result, 6);
}

// Test with read-only files
TEST_F(FileSystemTest, SyncAllFilesInDirectoryReadOnlyTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_readonly_test";
  std::filesystem::create_directories(test_dir);

  // Create a regular file
  std::string readonly_file = test_dir + "/readonly.txt";
  std::ofstream file(readonly_file);
  file << "This file will be read-only" << std::endl;
  file.close();

  // Make the file read-only
  std::filesystem::permissions(readonly_file,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::group_read |
                                   std::filesystem::perms::others_read);

  int result = SyncAllFilesInDirectory(test_dir.c_str());

  // Restore write permissions for cleanup
  std::filesystem::permissions(readonly_file,
                               std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);

  // read-only can be synced
  /* 1 file 1 directory */
  EXPECT_EQ(result, 2);
}

// Test with very deep directory structure (potential stack overflow if
// recursive)
TEST_F(FileSystemTest, SyncAllFilesInDirectoryDeepNestingTest) {
  std::string test_dir = TestDirectoryPath() + "/sync_deep_test";
  std::filesystem::create_directories(test_dir);

  // Create a very deep directory structure
  std::string current_path = test_dir;
  const int max_depth = 20;  // Not too deep to avoid filesystem limits

  for (int depth = 0; depth < max_depth; ++depth) {
    current_path += "/depth_" + std::to_string(depth);
    std::filesystem::create_directories(current_path);

    // Add a file at each level
    std::ofstream file(current_path + "/file_at_depth_" +
                       std::to_string(depth) + ".txt");
    file << "File at depth " << depth << std::endl;
    file.close();
  }

  int result = SyncAllFilesInDirectory(test_dir.c_str());
  /* 21 files 20 directories */
  EXPECT_EQ(result, 41);
}

TEST_F(FileSystemTest, SyncEmbeddingStoreTest) {
  /* existing embedding store */
  int result = SyncAllFilesInEmbeddingStore();
  EXPECT_EQ(result, 1);

  std::filesystem::remove_all(EmbeddingStoreDirectoryPath());

  /* non-existing embedding store */
  result = SyncAllFilesInEmbeddingStore();
  EXPECT_EQ(result, 1);
}

}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
