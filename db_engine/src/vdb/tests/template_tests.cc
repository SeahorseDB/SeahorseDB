#include "vdb/tests/base_environment.hh"

namespace vdb {
/* TODO: change this to actual test suite name
 * test_suite_directory_path must not be same as other test suite's one */
std::string test_suite_directory_path =
    test_root_directory_path + "/TemplateTestSuite";

class TemplateTestSuiteEnvironment : public BaseEnvironment {};

class TemplateTestSuite : public BaseTestSuite {};

class SampleTest : public TemplateTestSuite {};

TEST_F(SampleTest, SampleTest) { std::cout << "SampleTest" << std::endl; }

}  // namespace vdb

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::TemplateTestSuiteEnvironment);
  return RUN_ALL_TESTS();
}
