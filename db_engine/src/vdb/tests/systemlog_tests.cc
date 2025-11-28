#include "vdb/tests/base_environment.hh"
#include "vdb/common/system_log.hh"
#include <fstream>
#include <sstream>

namespace vdb {
std::string test_suite_directory_path =
    test_root_directory_path + "/SystemLogTestSuite";

// Count lines in log file that contain the specified string
int CountLinesInLogFile(const std::string& search_string) {
  std::ifstream log_file(server.logfile);
  if (!log_file.is_open()) {
    return -1;  // Cannot open file
  }

  std::string line;
  int count = 0;
  while (std::getline(log_file, line)) {
    if (line.find(search_string) != std::string::npos) {
      count++;
    }
  }
  return count;
}

class SystemLogTestSuiteEnvironment : public BaseEnvironment {};

class SystemLogTestSuite : public BaseTestSuite {};

class SystemLogTest : public SystemLogTestSuite {
 public:
  void SetUp() override {
    BaseTestSuite::SetUp();
    server.verbosity = LL_DEBUG;
    ResetAllLogTopics();
    log_file_path = TestDirectoryPath() + "/systemlog.log";
    server.logfile = log_file_path.data();
  }
  void TearDown() override {
    server.logfile = empty_string.data();
    server.verbosity = LL_NOTHING;
    BaseTestSuite::TearDown();
  }

  std::string GetLogFilePath() const { return log_file_path; }

 private:
  std::string log_file_path;
};

TEST_F(SystemLogTest, SystemLogSetAndResetTopicsTest) {
  int must_print_count = 0;
  std::string topics = "all";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;

  ResetAllLogTopics();
  topics = "table";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "DEBUG: MUST NOT PRINT");
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  ResetAllLogTopics();
  topics = "index";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "DEBUG: MUST NOT PRINT");
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: MUST NOT PRINT");
  ResetAllLogTopics();
  topics = "index,table";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "DEBUG: MUST NOT PRINT");
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;

  ResetAllLogTopics();
  topics = "index,table,unknown";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;

  ResetAllLogTopics();
  topics = "index,table,unknown,all";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug, "DEBUG: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug,
             "DEBUG: MUST PRINT is printed %d times", must_print_count);
  must_print_count++;

  int debug_must_print_count = must_print_count;

  must_print_count = 0;
  topics = "all";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;

  ResetAllLogTopics();
  topics = "table";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  ResetAllLogTopics();
  topics = "index";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  ResetAllLogTopics();
  topics = "index,table";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;

  ResetAllLogTopics();
  topics = "index,table,unknown";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;

  ResetAllLogTopics();
  topics = "index,table,unknown,all";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Unknown, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogNotice, "NOTICE: MUST PRINT %d",
             must_print_count);
  must_print_count++;
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogNotice,
             "NOTICE: MUST PRINT is printed %d times", must_print_count);
  must_print_count++;

  int notice_must_print_count = must_print_count;
  // Check if "MUST PRINT" is properly written to log file
  int debug_must_print_count_in_file = CountLinesInLogFile("DEBUG: MUST PRINT");
  int notice_must_print_count_in_file =
      CountLinesInLogFile("NOTICE: MUST PRINT");

  // "MUST PRINT" should be printed 8 times at DEBUG level
  EXPECT_EQ(debug_must_print_count_in_file, debug_must_print_count);
  // "MUST PRINT" should be printed 8 times at NOTICE level
  EXPECT_EQ(notice_must_print_count_in_file, notice_must_print_count);
}

TEST_F(SystemLogTest, SingleLogMultiTopicTest) {
  std::string topics = "index";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug, "DEBUG: INDEX MUST PRINT");
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug,
             "DEBUG: TABLE MUST NOT PRINT");
  SYSTEM_LOG(LogTopic::Table | LogTopic::Index, LogLevel::kLogDebug,
             "DEBUG: TABLE | INDEX MUST PRINT");

  ResetAllLogTopics();
  topics = "table";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug,
             "DEBUG: INDEX MUST NOT PRINT");
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: TABLE MUST PRINT");
  SYSTEM_LOG(LogTopic::Table | LogTopic::Index, LogLevel::kLogDebug,
             "DEBUG: TABLE | INDEX MUST PRINT");

  ResetAllLogTopics();
  topics = "index,table";
  SetLogTopics(topics);
  SYSTEM_LOG(LogTopic::Index, LogLevel::kLogDebug, "DEBUG: INDEX MUST PRINT");
  SYSTEM_LOG(LogTopic::Table, LogLevel::kLogDebug, "DEBUG: TABLE MUST PRINT");
  SYSTEM_LOG(LogTopic::Table | LogTopic::Index, LogLevel::kLogDebug,
             "DEBUG: TABLE | INDEX MUST PRINT");

  int index_must_print_count = CountLinesInLogFile("DEBUG: INDEX MUST PRINT");
  int table_must_print_count = CountLinesInLogFile("DEBUG: TABLE MUST PRINT");
  int table_index_must_print_count =
      CountLinesInLogFile("DEBUG: TABLE | INDEX MUST PRINT");
  EXPECT_EQ(index_must_print_count, 2);
  EXPECT_EQ(table_must_print_count, 2);
  EXPECT_EQ(table_index_must_print_count, 3);
}
}  // namespace vdb

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::SystemLogTestSuiteEnvironment);
  return RUN_ALL_TESTS();
}
