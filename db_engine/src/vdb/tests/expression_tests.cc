#include <gtest/gtest.h>
#include <arrow/testing/gtest_util.h>
#include "vdb/data/expression.hh"
#include "vdb/common/util.hh"

#include "vdb/tests/base_environment.hh"

std::string vdb::test_suite_directory_path =
    test_root_directory_path + "/ExpressionTestSuite";
namespace vdb::expression {

class ExpressionTest : public BaseTestSuite {
 protected:
  void SetUp() override {
    BaseTestSuite::SetUp();
    schema = arrow::schema({arrow::field("id", arrow::int32()),
                            arrow::field("name", arrow::utf8()),
                            arrow::field("age", arrow::int32()),
                            arrow::field("score", arrow::float32()),
                            arrow::field("is_active", arrow::boolean()),
                            arrow::field("tags", arrow::list(arrow::utf8()))});
  }

  std::shared_ptr<arrow::Schema> schema;
};

// A predicate that always returns an error (to test failure policy)
class ErrorPredicate : public Predicate {
 public:
  ErrorPredicate() : Predicate(PredicateType::kComparison) {}

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override {
    return arrow::compute::literal(true);
  }

  std::string ToString() const override { return "ERROR_PRED"; }

  arrow::Result<bool> EvaluateSegment(
      const std::unordered_map<std::string, std::string> &) override {
    return arrow::Status::Invalid("forced error");
  }

  arrow::Result<bool> EvaluateRecord(std::shared_ptr<arrow::RecordBatch>,
                                     const size_t) override {
    return arrow::Status::Invalid("forced error");
  }

  std::vector<std::string> GetReferencedColumns() const override { return {}; }
};

TEST_F(ExpressionTest, ExpressionParserTest) {
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
      {"a = 1 AND b = 2 oR nOT (c = 3 AND d = 4 AND e='1')",
       "((a = 1 AND b = 2) OR NOT (c = 3 AND d = 4 AND e = '1'))"},
      {"a = 1 AND b = 2 Or not (c = 3 OR d = 4 AND e='1')",
       "((a = 1 AND b = 2) OR NOT (c = 3 OR (d = 4 AND e = '1')))"},
      {"a in (1, 2, 3) AND b IS null", "(a IN [1, 2, 3] AND b IS NULL)"},
      {R"(e Like '%pattern%' OR b is nOT nuLL)",
       R"del((e LIKE '%pattern%' OR b IS NOT NULL))del"},
      {"(a=1 and b !=2 or (c>=1 and d != 50 and (e > '70' AND a > 10)))",
       "((a = 1 AND b != 2) OR (c >= 1 AND d != 50 AND (e > '70' AND a > "
       "10)))"}};
  for (const auto &[test_filter, expected_expr] : test_filter_cases) {
    std::cout << "Testing: " << test_filter << std::endl;
    ASSERT_OK_AND_ASSIGN(auto expr, builder.ParseFilter(test_filter));
    EXPECT_STREQ(expr->ToString().c_str(), expected_expr.c_str());
  }
}

TEST_F(ExpressionTest, AndEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {{"id", "1"},
                                                                 {"age", "30"}};

  auto expr1 = std::make_shared<Comparison>(
      ComparisonOperation::kEqual,
      std::make_shared<Column>("id", arrow::int32()),
      std::make_shared<Literal>("1", arrow::int32()));
  auto expr2 = std::make_shared<Comparison>(
      ComparisonOperation::kGreaterThan,
      std::make_shared<Column>("age", arrow::int32()),
      std::make_shared<Literal>("25", arrow::int32()));

  And and_expr({expr1, expr2});

  ASSERT_OK_AND_ASSIGN(auto result, and_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["age"] = "20";
  ASSERT_OK_AND_ASSIGN(result, and_expr.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, OrEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {{"id", "1"},
                                                                 {"age", "20"}};

  auto expr1 = std::make_shared<Comparison>(
      ComparisonOperation::kEqual,
      std::make_shared<Column>("id", arrow::int32()),
      std::make_shared<Literal>("1", arrow::int32()));
  auto expr2 = std::make_shared<Comparison>(
      ComparisonOperation::kGreaterThan,
      std::make_shared<Column>("age", arrow::int32()),
      std::make_shared<Literal>("25", arrow::int32()));

  Or or_expr({expr1, expr2});

  ASSERT_OK_AND_ASSIGN(auto result, or_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["id"] = "2";
  ASSERT_OK_AND_ASSIGN(result, or_expr.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, ComparisonEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {{"age", "30"}};

  Comparison comp(ComparisonOperation::kGreaterThan,
                  std::make_shared<Column>("age", arrow::int32()),
                  std::make_shared<Literal>("25", arrow::int32()));

  ASSERT_OK_AND_ASSIGN(auto result, comp.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["age"] = "20";
  ASSERT_OK_AND_ASSIGN(result, comp.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, IsNullEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {{"name", ""}};

  IsNull is_null(std::make_shared<Column>("name"));

  ASSERT_OK_AND_ASSIGN(auto result, is_null.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["name"] = "John";
  ASSERT_OK_AND_ASSIGN(result, is_null.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);
}

TEST_F(ExpressionTest, InEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {{"id", "2"}};

  In in_expr(std::make_shared<Column>("id", arrow::int32()), {"1", "2", "3"});

  ASSERT_OK_AND_ASSIGN(auto result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["id"] = "4";
  ASSERT_OK_AND_ASSIGN(result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, InExpressionWithStringType) {
  // Test IN expression with string type values
  std::unordered_map<std::string, std::string> segment_id_map = {
      {"name", "John"}};

  In in_expr(std::make_shared<Column>("name", arrow::utf8()),
             {"Alice", "Bob", "John"});

  ASSERT_OK_AND_ASSIGN(auto result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["name"] = "Jane";
  ASSERT_OK_AND_ASSIGN(result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);

  // Test with BuildArrowExpression
  ASSERT_OK_AND_ASSIGN(auto arrow_expr, in_expr.BuildArrowExpression());
  EXPECT_EQ(arrow_expr.ToString(),
            "is_in(name, {value_set=string:[\n  \"Alice\",\n  \"Bob\",\n  "
            "\"John\"\n], null_matching_behavior=MATCH})");
}

TEST_F(ExpressionTest, InExpressionWithSpecialCharacters) {
  // Test IN expression with special characters (control characters)
  // This tests the fix for JSON parsing error with strings containing
  // special characters like \u001e (Record Separator)

  std::string value_with_control_char =
      "34ff11398acee6c89a18ddb47d276612\u001e00000001";
  std::string another_value = "test\u001evalue";
  std::string normal_value = "normalvalue";

  std::unordered_map<std::string, std::string> segment_id_map = {
      {"id", value_with_control_char}};

  In in_expr(std::make_shared<Column>("id", arrow::utf8()),
             {value_with_control_char, another_value, normal_value});

  // Test EvaluateSegment with value containing control character
  ASSERT_OK_AND_ASSIGN(auto result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  // Test with a different value
  segment_id_map["id"] = another_value;
  ASSERT_OK_AND_ASSIGN(result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  // Test with normal value
  segment_id_map["id"] = normal_value;
  ASSERT_OK_AND_ASSIGN(result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  // Test with value not in the list
  segment_id_map["id"] = "not_in_list";
  ASSERT_OK_AND_ASSIGN(result, in_expr.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);

  // Test BuildArrowExpression to ensure proper JSON escaping
  ASSERT_OK_AND_ASSIGN(auto arrow_expr, in_expr.BuildArrowExpression());
  std::string expected_pattern = "is_in(id, {value_set=string:[";
  EXPECT_TRUE(arrow_expr.ToString().find(expected_pattern) != std::string::npos)
      << "Expected to find '" << expected_pattern
      << "' in: " << arrow_expr.ToString();
}

TEST_F(ExpressionTest, InExpressionParseFilterWithSpecialCharacters) {
  // Test parsing a filter string with IN clause containing special characters
  auto test_schema = arrow::schema({arrow::field("id", arrow::utf8()),
                                    arrow::field("value", arrow::int32())});

  ExpressionBuilder builder(test_schema);

  // Create a filter with special character in IN clause
  std::string value1 = "test\u001evalue1";
  std::string value2 = "test\u001evalue2";
  std::string filter = "id in ('" + value1 + "', '" + value2 + "')";

  ASSERT_OK_AND_ASSIGN(auto expr, builder.ParseFilter(filter));
  EXPECT_EQ(expr->ToString(), "id IN ['test\u001evalue1', 'test\u001evalue2']");

  // Test BuildArrowExpression - should not throw JSON parse error
  ASSERT_OK_AND_ASSIGN(auto arrow_expr, expr->BuildArrowExpression());
  EXPECT_TRUE(arrow_expr.ToString().find("is_in(id") != std::string::npos);
}

TEST_F(ExpressionTest, InExpressionWithVariousControlCharacters) {
  // Test various control characters
  std::vector<std::string> test_values = {
      "value\u0001",   // SOH (Start of Heading)
      "value\u001e",   // RS (Record Separator)
      "value\u001f",   // US (Unit Separator)
      "value\ttab",    // Tab
      "value\nline",   // Newline
      "value\"quote",  // Double quote
      "value\\back"    // Backslash
  };

  for (const auto &test_value : test_values) {
    std::unordered_map<std::string, std::string> segment_id_map = {
        {"id", test_value}};

    In in_expr(std::make_shared<Column>("id", arrow::utf8()),
               {test_value, "other_value"});

    // Should not fail with JSON parse error
    ASSERT_OK_AND_ASSIGN(auto result, in_expr.EvaluateSegment(segment_id_map));
    EXPECT_TRUE(result) << "Failed for value: " << test_value;

    // Should build Arrow expression without error
    ASSERT_OK_AND_ASSIGN(auto arrow_expr, in_expr.BuildArrowExpression());
    EXPECT_TRUE(!arrow_expr.ToString().empty())
        << "Failed to build expression for value: " << test_value;
  }
}

TEST_F(ExpressionTest, LikeEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {
      {"name", "John Doe"}};

  std::vector<std::string> patterns = {
      "John Doe",   "%Doe%",    "%John%",   "John%",    "John Doe%", "John%Doe",
      "%Jo_n%",     "John%",    "%hn%oe",   "J%Doe%",   "%John%Doe", "Jo%Doe%",
      "%John Doe%", "_ohn Doe", "J_hn%Doe", "John _oe", "J_hn _oe",  "J_hn%oe"};

  int i = 0;
  for (const auto &pattern : patterns) {
    Like like_expr(std::make_shared<Column>("name", arrow::utf8()), pattern);
    ASSERT_OK_AND_ASSIGN(auto result,
                         like_expr.EvaluateSegment(segment_id_map));
    EXPECT_TRUE(result) << "At case " << i << ": " << like_expr.ToString();
    ++i;
  }

  // false cases
  std::vector<std::string> false_patterns = {
      "Jane",      "JohnxDoe",   "Johns%",    "%Jane%",   "Deo",
      "John%Jane", "%John%Jane", "Joh%DoeXX", "JohnXDoe", "_John Doe",
      "Jahn Doe",  "John De999", "Jahn De",   "Jyhn%D_e", "John DoeX"};

  i = 0;
  for (const auto &pattern : false_patterns) {
    Like like_expr(std::make_shared<Column>("name", arrow::utf8()), pattern);
    ASSERT_OK_AND_ASSIGN(auto result,
                         like_expr.EvaluateSegment(segment_id_map));
    EXPECT_FALSE(result) << "At case " << i << ": " << like_expr.ToString();
    ++i;
  }
}

TEST_F(ExpressionTest, NotEvaluateSegment) {
  std::unordered_map<std::string, std::string> segment_id_map = {{"age", "30"}};

  auto inner_expr = std::make_shared<Comparison>(
      ComparisonOperation::kLessThan,
      std::make_shared<Column>("age", arrow::int32()),
      std::make_shared<Literal>("25", arrow::int32()));

  Not not_expr(inner_expr);

  ASSERT_OK_AND_ASSIGN(auto result, not_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["age"] = "20";
  ASSERT_OK_AND_ASSIGN(result, not_expr.EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, AndEvaluateRecord) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::uint32(), false),
      arrow::field("age", arrow::uint32())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  auto id_arr = std::shared_ptr<arrow::Array>();
  auto age_arr = std::shared_ptr<arrow::Array>();

  arrow::UInt32Builder id_builder;
  arrow::UInt32Builder age_builder;

  ARROW_EXPECT_OK(id_builder.Append(0));
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(id_builder.Append(2));

  ARROW_EXPECT_OK(age_builder.Append(40));
  ARROW_EXPECT_OK(age_builder.Append(30));
  ARROW_EXPECT_OK(age_builder.Append(20));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(age_builder.Finish(&age_arr));

  auto rb = arrow::RecordBatch::Make(schema, 3, {id_arr, age_arr});

  auto expr1 = std::make_shared<Comparison>(
      ComparisonOperation::kEqual,
      std::make_shared<Column>("id", arrow::int32()),
      std::make_shared<Literal>("1", arrow::int32()));
  auto expr2 = std::make_shared<Comparison>(
      ComparisonOperation::kGreaterThan,
      std::make_shared<Column>("age", arrow::int32()),
      std::make_shared<Literal>("25", arrow::int32()));

  And and_expr({expr1, expr2});

  ASSERT_OK_AND_ASSIGN(auto result, and_expr.EvaluateRecord(rb, 1));
  EXPECT_TRUE(result);

  ASSERT_OK_AND_ASSIGN(result, and_expr.EvaluateRecord(rb, 2));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, OrEvaluateRecord) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::uint32(), false),
      arrow::field("age", arrow::uint32())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto id_arr = std::shared_ptr<arrow::Array>();
  auto age_arr = std::shared_ptr<arrow::Array>();

  arrow::UInt32Builder id_builder;
  arrow::UInt32Builder age_builder;

  ARROW_EXPECT_OK(id_builder.Append(0));
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(id_builder.Append(2));

  ARROW_EXPECT_OK(age_builder.Append(30));
  ARROW_EXPECT_OK(age_builder.Append(20));
  ARROW_EXPECT_OK(age_builder.Append(10));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(age_builder.Finish(&age_arr));

  auto rb = arrow::RecordBatch::Make(schema, 3, {id_arr, age_arr});

  auto expr1 = std::make_shared<Comparison>(
      ComparisonOperation::kEqual,
      std::make_shared<Column>("id", arrow::int32()),
      std::make_shared<Literal>("1", arrow::int32()));
  auto expr2 = std::make_shared<Comparison>(
      ComparisonOperation::kGreaterThan,
      std::make_shared<Column>("age", arrow::int32()),
      std::make_shared<Literal>("25", arrow::int32()));

  Or or_expr({expr1, expr2});

  ASSERT_OK_AND_ASSIGN(auto result, or_expr.EvaluateRecord(rb, 1));
  EXPECT_TRUE(result);

  ASSERT_OK_AND_ASSIGN(result, or_expr.EvaluateRecord(rb, 2));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, AndEvaluateSegment_ConservativeOnError) {
  // err (unknown) AND true -> conservative keep (true)
  auto err = std::make_shared<ErrorPredicate>();
  auto comp_true = std::make_shared<Comparison>(
      ComparisonOperation::kEqual,
      std::make_shared<Column>("id", arrow::int32()),
      std::make_shared<Literal>("1", arrow::int32()));

  And and_expr({err, comp_true});

  std::unordered_map<std::string, std::string> segment_id_map = {{"id", "1"}};
  ASSERT_OK_AND_ASSIGN(auto result, and_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);
}

TEST_F(ExpressionTest, OrEvaluateSegment_ConservativeOnError) {
  // err (unknown) OR false -> conservative keep (true)
  auto err = std::make_shared<ErrorPredicate>();
  auto comp_false = std::make_shared<Comparison>(
      ComparisonOperation::kEqual,
      std::make_shared<Column>("id", arrow::int32()),
      std::make_shared<Literal>("2", arrow::int32()));

  Or or_expr({err, comp_false});
  std::unordered_map<std::string, std::string> segment_id_map = {{"id", "1"}};
  ASSERT_OK_AND_ASSIGN(auto result, or_expr.EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);
}

TEST_F(ExpressionTest, AndOrEvaluateRecord_PropagateError) {
  auto err = std::make_shared<ErrorPredicate>();
  And and_expr({err});
  Or or_expr({err});

  // Build a minimal RecordBatch; the error predicate ignores it and fails
  auto schema = arrow::schema({arrow::field("id", arrow::int32())});
  std::shared_ptr<arrow::Array> id_arr;
  arrow::Int32Builder b;
  ARROW_EXPECT_OK(b.Append(1));
  ARROW_EXPECT_OK(b.Finish(&id_arr));
  auto rb = arrow::RecordBatch::Make(schema, 1, {id_arr});

  auto and_res = and_expr.EvaluateRecord(rb, 0);
  EXPECT_FALSE(and_res.ok());
  auto or_res = or_expr.EvaluateRecord(rb, 0);
  EXPECT_FALSE(or_res.ok());
}

TEST_F(ExpressionTest, ComparisonEvaluateRecord) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::uint32(), false),
      arrow::field("age", arrow::uint32())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto id_arr = std::shared_ptr<arrow::Array>();
  auto age_arr = std::shared_ptr<arrow::Array>();

  arrow::UInt32Builder id_builder;
  arrow::UInt32Builder age_builder;

  ARROW_EXPECT_OK(id_builder.Append(0));
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(id_builder.Append(2));

  ARROW_EXPECT_OK(age_builder.Append(30));
  ARROW_EXPECT_OK(age_builder.Append(20));
  ARROW_EXPECT_OK(age_builder.Append(10));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(age_builder.Finish(&age_arr));

  auto rb = arrow::RecordBatch::Make(schema, 3, {id_arr, age_arr});

  Comparison comp(ComparisonOperation::kGreaterThan,
                  std::make_shared<Column>("age", arrow::int32()),
                  std::make_shared<Literal>("25", arrow::int32()));

  ASSERT_OK_AND_ASSIGN(auto result, comp.EvaluateRecord(rb, 0));
  EXPECT_TRUE(result);

  ASSERT_OK_AND_ASSIGN(result, comp.EvaluateRecord(rb, 2));
  EXPECT_FALSE(result);
}

TEST_F(ExpressionTest, ComparisonEvaluateRecord_OutOfBoundsReturnsInvalid) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::uint32(), false),
      arrow::field("age", arrow::uint32())};

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  auto id_arr = std::shared_ptr<arrow::Array>();
  auto age_arr = std::shared_ptr<arrow::Array>();

  arrow::UInt32Builder id_builder;
  arrow::UInt32Builder age_builder;

  ARROW_EXPECT_OK(id_builder.Append(0));
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(id_builder.Append(2));

  ARROW_EXPECT_OK(age_builder.Append(30));
  ARROW_EXPECT_OK(age_builder.Append(20));
  ARROW_EXPECT_OK(age_builder.Append(10));

  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));
  ARROW_EXPECT_OK(age_builder.Finish(&age_arr));

  auto rb = arrow::RecordBatch::Make(schema, 3, {id_arr, age_arr});

  Comparison comp(ComparisonOperation::kGreaterThan,
                  std::make_shared<Column>("age", arrow::int32()),
                  std::make_shared<Literal>("25", arrow::int32()));

  auto res = comp.EvaluateRecord(rb, 3);  // out-of-bounds index
  EXPECT_FALSE(res.ok());
  EXPECT_TRUE(res.status().IsIndexError() || res.status().IsInvalid());
}

TEST_F(ExpressionTest, IsNullEvaluateRecord) {
  // Schema: name string
  auto schema = arrow::schema({arrow::field("name", arrow::utf8())});

  arrow::StringBuilder name_builder;
  ARROW_EXPECT_OK(name_builder.Append("John"));
  ARROW_EXPECT_OK(name_builder.AppendNull());
  ARROW_EXPECT_OK(name_builder.Append(""));
  std::shared_ptr<arrow::Array> name_arr;
  ARROW_EXPECT_OK(name_builder.Finish(&name_arr));

  auto rb = arrow::RecordBatch::Make(schema, 3, {name_arr});

  IsNull is_null(std::make_shared<Column>("name", arrow::utf8()));
  ASSERT_OK_AND_ASSIGN(auto res0, is_null.EvaluateRecord(rb, 0));
  EXPECT_FALSE(res0);  // "John" is not null
  ASSERT_OK_AND_ASSIGN(auto res1, is_null.EvaluateRecord(rb, 1));
  EXPECT_TRUE(res1);  // null
  ASSERT_OK_AND_ASSIGN(auto res2, is_null.EvaluateRecord(rb, 2));
  EXPECT_FALSE(res2);  // empty string != null

  IsNull is_not_null(std::make_shared<Column>("name", arrow::utf8()), true);
  ASSERT_OK_AND_ASSIGN(auto res1_not, is_not_null.EvaluateRecord(rb, 1));
  EXPECT_FALSE(res1_not);  // NOT NULL on null -> false
}

TEST_F(ExpressionTest, InEvaluateRecord) {
  // Schema: id uint32
  auto schema = arrow::schema({arrow::field("id", arrow::uint32())});

  arrow::UInt32Builder id_builder;
  ARROW_EXPECT_OK(id_builder.Append(0));
  ARROW_EXPECT_OK(id_builder.Append(1));
  ARROW_EXPECT_OK(id_builder.Append(2));
  std::shared_ptr<arrow::Array> id_arr;
  ARROW_EXPECT_OK(id_builder.Finish(&id_arr));

  auto rb = arrow::RecordBatch::Make(schema, 3, {id_arr});

  In in_expr(std::make_shared<Column>("id", arrow::int32()), {"1", "2", "3"});

  ASSERT_OK_AND_ASSIGN(auto r0, in_expr.EvaluateRecord(rb, 0));
  EXPECT_FALSE(r0);
  ASSERT_OK_AND_ASSIGN(auto r1, in_expr.EvaluateRecord(rb, 1));
  EXPECT_TRUE(r1);
  ASSERT_OK_AND_ASSIGN(auto r2, in_expr.EvaluateRecord(rb, 2));
  EXPECT_TRUE(r2);
}

TEST_F(ExpressionTest, LikeEvaluateRecord) {
  // Schema: name string
  auto schema = arrow::schema({arrow::field("name", arrow::utf8())});

  arrow::StringBuilder name_builder;
  ARROW_EXPECT_OK(name_builder.Append("John Doe"));     // idx 0
  ARROW_EXPECT_OK(name_builder.Append("Jane"));         // idx 1
  ARROW_EXPECT_OK(name_builder.Append("Super John"));   // idx 2
  ARROW_EXPECT_OK(name_builder.Append("foo ' ' bar"));  // idx 3, contains "' '"
  std::shared_ptr<arrow::Array> name_arr;
  ARROW_EXPECT_OK(name_builder.Finish(&name_arr));

  auto rb = arrow::RecordBatch::Make(schema, 4, {name_arr});

  // Pattern: John% (starts with "John")
  Like like_start(std::make_shared<Column>("name", arrow::utf8()), "John%");
  ASSERT_OK_AND_ASSIGN(auto res0_start, like_start.EvaluateRecord(rb, 0));
  EXPECT_TRUE(res0_start);  // "John Doe"
  ASSERT_OK_AND_ASSIGN(auto res1_start, like_start.EvaluateRecord(rb, 1));
  EXPECT_FALSE(res1_start);  // "Jane"
  ASSERT_OK_AND_ASSIGN(auto res2_start, like_start.EvaluateRecord(rb, 2));
  EXPECT_FALSE(res2_start);  // "Super John"
  ASSERT_OK_AND_ASSIGN(auto res3_start, like_start.EvaluateRecord(rb, 3));
  EXPECT_FALSE(res3_start);  // "foo ' ' bar"

  // Pattern: %John (ends with "John")
  Like like_end(std::make_shared<Column>("name", arrow::utf8()), "%John");
  ASSERT_OK_AND_ASSIGN(auto res0_end, like_end.EvaluateRecord(rb, 0));
  EXPECT_FALSE(res0_end);
  ASSERT_OK_AND_ASSIGN(auto res1_end, like_end.EvaluateRecord(rb, 1));
  EXPECT_FALSE(res1_end);
  ASSERT_OK_AND_ASSIGN(auto res2_end, like_end.EvaluateRecord(rb, 2));
  EXPECT_TRUE(res2_end);
  ASSERT_OK_AND_ASSIGN(auto res3_end, like_end.EvaluateRecord(rb, 3));
  EXPECT_FALSE(res3_end);

  // Pattern: %John% (contains "John")
  Like like_contains(std::make_shared<Column>("name", arrow::utf8()), "%John%");
  ASSERT_OK_AND_ASSIGN(auto res0_contains, like_contains.EvaluateRecord(rb, 0));
  EXPECT_TRUE(res0_contains);
  ASSERT_OK_AND_ASSIGN(auto res1_contains, like_contains.EvaluateRecord(rb, 1));
  EXPECT_FALSE(res1_contains);
  ASSERT_OK_AND_ASSIGN(auto res2_contains, like_contains.EvaluateRecord(rb, 2));
  EXPECT_TRUE(res2_contains);
  ASSERT_OK_AND_ASSIGN(auto res3_contains, like_contains.EvaluateRecord(rb, 3));
  EXPECT_FALSE(res3_contains);

  // Pattern: %' '% (contains the literal substring "' '")
  Like like_quote_space(std::make_shared<Column>("name", arrow::utf8()),
                        "%' '%");
  ASSERT_OK_AND_ASSIGN(auto res0_qs, like_quote_space.EvaluateRecord(rb, 0));
  EXPECT_FALSE(res0_qs);
  ASSERT_OK_AND_ASSIGN(auto res1_qs, like_quote_space.EvaluateRecord(rb, 1));
  EXPECT_FALSE(res1_qs);
  ASSERT_OK_AND_ASSIGN(auto res2_qs, like_quote_space.EvaluateRecord(rb, 2));
  EXPECT_FALSE(res2_qs);
  ASSERT_OK_AND_ASSIGN(auto res3_qs, like_quote_space.EvaluateRecord(rb, 3));
  EXPECT_TRUE(res3_qs);
}

TEST_F(ExpressionTest, NotEvaluateRecord) {
  // Schema: age uint32
  auto schema = arrow::schema({arrow::field("age", arrow::uint32())});

  arrow::UInt32Builder age_builder;
  ARROW_EXPECT_OK(age_builder.Append(30));
  ARROW_EXPECT_OK(age_builder.Append(10));
  std::shared_ptr<arrow::Array> age_arr;
  ARROW_EXPECT_OK(age_builder.Finish(&age_arr));

  auto rb = arrow::RecordBatch::Make(schema, 2, {age_arr});

  auto inner = std::make_shared<Comparison>(
      ComparisonOperation::kLessThan,
      std::make_shared<Column>("age", arrow::int32()),
      std::make_shared<Literal>("20", arrow::int32()));

  Not not_expr(inner);
  ASSERT_OK_AND_ASSIGN(auto res0, not_expr.EvaluateRecord(rb, 0));
  EXPECT_TRUE(res0);  // 30 < 20 is false, NOT false -> true
  ASSERT_OK_AND_ASSIGN(auto res1, not_expr.EvaluateRecord(rb, 1));
  EXPECT_FALSE(res1);  // 10 < 20 is true, NOT true -> false
}

TEST_F(ExpressionTest, ExpressionBuilderParseFilter) {
  ExpressionBuilder builder(schema);

  ASSERT_OK_AND_ASSIGN(
      auto expr,
      builder.ParseFilter("age > 25 AND (name LIKE '%John%' OR score < 3.5)"));

  std::unordered_map<std::string, std::string> segment_id_map = {
      {"age", "30"}, {"name", "John Doe"}, {"score", "4.0"}};

  auto predicate = std::dynamic_pointer_cast<Predicate>(expr);
  ASSERT_NE(predicate, nullptr);
  ASSERT_OK_AND_ASSIGN(auto result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  ASSERT_OK_AND_ASSIGN(auto expr2, builder.ParseFilter("name LIKE '%John%'"));

  auto predicate2 = std::dynamic_pointer_cast<Predicate>(expr2);
  ASSERT_NE(predicate2, nullptr);
  ASSERT_OK_AND_ASSIGN(result, predicate2->EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["age"] = "20";
  ASSERT_OK_AND_ASSIGN(result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);

  segment_id_map["age"] = "30";
  segment_id_map["name"] = "Jane Smith";
  segment_id_map["score"] = "3.0";
  ASSERT_OK_AND_ASSIGN(result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);
}

TEST_F(ExpressionTest, ComplexExpressionBuilderParseFilter) {
  ExpressionBuilder builder(schema);

  ASSERT_OK_AND_ASSIGN(
      auto expr,
      builder.ParseFilter("(age > 25 AND name LIKE '%John%') OR "
                          "(score >= 4.5 AND is_active = true) OR "
                          "(id IN (1, 2, 3) AND tags IS NOT NULL) AND "
                          "NOT (name LIKE 'Test%' OR age < 18)"));

  std::unordered_map<std::string, std::string> segment_id_map = {
      {"id", "2"},      {"name", "John Doe"},  {"age", "30"},
      {"score", "4.7"}, {"is_active", "true"}, {"tags", "['tag1', 'tag2']"}};

  auto predicate = std::dynamic_pointer_cast<Predicate>(expr);
  ASSERT_NE(predicate, nullptr);
  ASSERT_OK_AND_ASSIGN(auto result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["age"] = "20";
  segment_id_map["name"] = "Jane Smith";
  segment_id_map["score"] = "4.8";
  ASSERT_OK_AND_ASSIGN(result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);

  segment_id_map["score"] = "4.0";
  segment_id_map["is_active"] = "false";
  segment_id_map["age"] = "10";
  segment_id_map["name"] = "ABC";
  segment_id_map["id"] = "5";
  ASSERT_OK_AND_ASSIGN(result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_FALSE(result);

  segment_id_map["score"] = "5.0";
  segment_id_map["is_active"] = "true";
  ASSERT_OK_AND_ASSIGN(result, predicate->EvaluateSegment(segment_id_map));
  EXPECT_TRUE(result);
}

TEST_F(ExpressionTest, BuildArrowExpression) {
  ExpressionBuilder builder(schema);

  // Test AND expression
  ASSERT_OK_AND_ASSIGN(auto and_expr,
                       builder.ParseFilter("age > 25 AND score < 4.5"));
  ASSERT_OK_AND_ASSIGN(auto arrow_and_expr, and_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_and_expr.ToString(), "((age > 25) and (score < 4.5))");

  // Test OR expression
  ASSERT_OK_AND_ASSIGN(
      auto or_expr,
      builder.ParseFilter("name LIKE 'John%' OR is_active = true"));
  ASSERT_OK_AND_ASSIGN(auto arrow_or_expr, or_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_or_expr.ToString(),
            "(match_like(name, {pattern=\"John%\", ignore_case=false}) or "
            "(is_active == true))");

  // Test NOT expression
  ASSERT_OK_AND_ASSIGN(auto not_expr, builder.ParseFilter("NOT (age < 18)"));
  ASSERT_OK_AND_ASSIGN(auto arrow_not_expr, not_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_not_expr.ToString(), "invert((age < 18))");

  // Test IN expression
  ASSERT_OK_AND_ASSIGN(auto in_expr, builder.ParseFilter("id IN (1, 2, 3)"));
  ASSERT_OK_AND_ASSIGN(auto arrow_in_expr, in_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_in_expr.ToString(),
            "is_in(id, {value_set=int32:[\n  1,\n  2,\n  3\n], "
            "null_matching_behavior=MATCH})");

  // Test complex expression
  ASSERT_OK_AND_ASSIGN(
      auto complex_expr,
      builder.ParseFilter("(age > 25 AND name LIKE '%John%') OR (score >= 4.5 "
                          "AND is_active = true)"));
  ASSERT_OK_AND_ASSIGN(auto arrow_complex_expr,
                       complex_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_complex_expr.ToString(),
            "(((age > 25) and match_like(name, {pattern=\"%John%\", "
            "ignore_case=false})) or ((score >= 4.5) and (is_active == "
            "true)))");
}

TEST_F(ExpressionTest, BuildArrowExpressionWithIsNull) {
  ExpressionBuilder builder(schema);

  ASSERT_OK_AND_ASSIGN(auto is_null_expr, builder.ParseFilter("tags IS NULL"));
  ASSERT_OK_AND_ASSIGN(auto arrow_is_null_expr,
                       is_null_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_is_null_expr.ToString(),
            "is_null(tags, {nan_is_null=false})");

  ASSERT_OK_AND_ASSIGN(auto is_not_null_expr,
                       builder.ParseFilter("tags IS NOT NULL"));
  ASSERT_OK_AND_ASSIGN(auto arrow_is_not_null_expr,
                       is_not_null_expr->BuildArrowExpression());
  EXPECT_EQ(arrow_is_not_null_expr.ToString(),
            "invert(is_null(tags, {nan_is_null=false}))");
}

TEST_F(ExpressionTest, ParseFilterUnexpectedToken) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("id = 1 junk");
  EXPECT_FALSE(res.ok());
  EXPECT_TRUE(res.status().IsInvalid());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Unexpected token"), std::string::npos);
  EXPECT_NE(msg.find("'junk'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterNotEqualAlternateOperator) {
  ExpressionBuilder builder(schema);
  ASSERT_OK_AND_ASSIGN(auto predicate, builder.ParseFilter("age <> 10"));
  EXPECT_EQ(predicate->ToString(), "age != 10");
}

TEST_F(ExpressionTest, ParseFilterStringMustBeQuoted) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name = John");
  EXPECT_FALSE(res.ok());
  EXPECT_TRUE(res.status().IsInvalid());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Column 'name' is of string type"), std::string::npos);
  EXPECT_NE(msg.find("single quotes"), std::string::npos);
  EXPECT_NE(msg.find("Near 'John'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
  EXPECT_NE(msg.find("string type"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterInListStringMustBeQuoted) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name in (John, 'Jane')");
  EXPECT_FALSE(res.ok());
  EXPECT_TRUE(res.status().IsInvalid());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Column 'name' is of string type"), std::string::npos);
  EXPECT_NE(msg.find("IN list"), std::string::npos);
  EXPECT_NE(msg.find("Near 'John'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
  EXPECT_NE(msg.find("string type"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterStringColumnNumericUnquoted) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name = 1000");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("string type"), std::string::npos);
  EXPECT_NE(msg.find("single quotes"), std::string::npos);
  EXPECT_NE(msg.find("Near '1000'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterLikeOnlyOnStringColumns) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("id like '1%'");
  EXPECT_FALSE(res.ok());
  EXPECT_TRUE(res.status().IsInvalid());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("for column 'id'"), std::string::npos);
  EXPECT_NE(msg.find("LIKE operator"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterColumnNotFoundIncludesName) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("nae = 1");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Column not found: nae"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterUnexpectedEndAfterColumnNameIncludesName) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("after column name"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterInvalidOperatorIncludesColumn) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("age ~~ 10");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Invalid operator"), std::string::npos);
  EXPECT_NE(msg.find("Near '~~'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterLikePatternMustBeQuotedIncludesColumn) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name like John%");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("LIKE pattern for column 'name'"), std::string::npos);
  EXPECT_NE(msg.find("single quotes"), std::string::npos);
  EXPECT_NE(msg.find("Near 'John%'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterExpectedOpenParenAfterIn) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name in 'John'");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Expected opening parenthesis after IN"),
            std::string::npos);
  EXPECT_NE(msg.find("Near 'IN'"), std::string::npos);
  EXPECT_NE(msg.find("pos"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterExpectedCloseParenAfterInList) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("name in ('John', 'Jane'");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Expected closing parenthesis after IN list"),
            std::string::npos);
}

TEST_F(ExpressionTest,
       ParseFilterUnexpectedEndAfterOperatorIncludesColumnAndOp) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("age >");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("after operator '>' for column 'age'"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterMismatchedParentheses) {
  ExpressionBuilder builder(schema);
  auto res = builder.ParseFilter("(age > 1");
  EXPECT_FALSE(res.ok());
  auto msg = res.status().ToString();
  EXPECT_NE(msg.find("Mismatched parentheses"), std::string::npos);
}

TEST_F(ExpressionTest, ParseFilterCaseInsensitiveKeywords_AllLowerSimple) {
  ExpressionBuilder builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto expr, builder.ParseFilter("id in (1, 2, 3) and name like 'a%'"));
  EXPECT_EQ(expr->ToString(), "(id IN [1, 2, 3] AND name LIKE 'a%')");
}

TEST_F(ExpressionTest, ParseFilterCaseInsensitiveKeywords_AllLowerComposite) {
  ExpressionBuilder builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto expr, builder.ParseFilter("(age > 25 and name like '%john%') or "
                                     "(score >= 4.5 and is_active = true)"));
  EXPECT_EQ(expr->ToString(),
            "((age > 25 AND name LIKE '%john%') OR (score >= 4.5 AND is_active "
            "= true))");
}

TEST_F(ExpressionTest, ParseFilterCaseInsensitiveKeywords_NotLikeAndIsNotNull) {
  ExpressionBuilder builder(schema);
  ASSERT_OK_AND_ASSIGN(auto expr1, builder.ParseFilter("name not like 'x%'"));
  EXPECT_EQ(expr1->ToString(), "name NOT LIKE 'x%'");

  ASSERT_OK_AND_ASSIGN(auto expr2, builder.ParseFilter("tags is not null"));
  EXPECT_EQ(expr2->ToString(), "tags IS NOT NULL");
}

TEST_F(ExpressionTest, ParseFilterCaseInsensitiveKeywords_LowerNotWithParens) {
  ExpressionBuilder builder(schema);
  ASSERT_OK_AND_ASSIGN(auto expr, builder.ParseFilter("not (age < 18)"));
  EXPECT_EQ(expr->ToString(), "NOT age < 18");
}

TEST_F(ExpressionTest, BasicFilterTest) {
  std::string test_schema_string =
      "Id int32 not null, Name string, height float32";
  auto schema = vdb::ParseSchemaFrom(test_schema_string);
  expression::ExpressionBuilder builder(schema);

  ASSERT_OK_AND_ASSIGN(auto predicate, builder.ParseFilter("id <= 2"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kComparison);
  EXPECT_EQ(
      dynamic_cast<expression::Comparison *>(predicate.get())->GetOperation(),
      expression::ComparisonOperation::kLessThanEqual);
  EXPECT_EQ(predicate->ToString(), "id <= 2");

  ASSERT_OK_AND_ASSIGN(predicate,
                       builder.ParseFilter("id <= 2 and height > 150"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kAnd);
  EXPECT_EQ(predicate->ToString(), "(id <= 2 AND height > 150)");

  ASSERT_OK_AND_ASSIGN(predicate,
                       builder.ParseFilter("not (id <= 2 AND height > 150)"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kNot);
  EXPECT_EQ(predicate->ToString(), "NOT (id <= 2 AND height > 150)");

  ASSERT_OK_AND_ASSIGN(
      predicate,
      builder.ParseFilter(
          "not (id <= 2 and height > 150) and name in ('John', 'Jane')"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kAnd);
  EXPECT_EQ(predicate->ToString(),
            "(NOT (id <= 2 AND height > 150) AND name IN ['John', 'Jane'])");
}

TEST_F(ExpressionTest, SegmentFilterTest) {
  std::string test_schema_string =
      "id int32 not null, pid uint32 not null, name string, height float32";
  auto schema = vdb::ParseSchemaFrom(test_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys_in = "id";
  std::string segment_key_composition_type = "composite";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys_in, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  expression::ExpressionBuilder expr_builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto predicate,
      expr_builder.ParseFilter("(id < 2 and id > 0) and name like 'T%'"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kAnd);
  EXPECT_EQ(predicate->ToString(), "((id < 2 AND id > 0) AND name LIKE 'T%')");

  auto segmentation_info =
      schema->metadata()->Get("segmentation_info").ValueOr("");
  auto segmentation_info_json = json::parse(segmentation_info);
  auto segment_keys = segmentation_info_json["segment_keys"];
  std::vector<uint32_t> segment_keys_column_indexes;
  std::vector<std::string> segment_keys_column_names;
  for (auto &column : segment_keys) {
    auto column_name = column.get<std::string>();
    auto column_index = schema->GetFieldIndex(column_name);
    segment_keys_column_indexes.push_back(column_index);
    segment_keys_column_names.push_back(column_name);
  }

  std::vector<std::string> records = {
      "0\u001e1\u001eJohn\u001e180.3", "0\u001e1\u001eJane\u001e170.1",
      "1\u001e2\u001eTom\u001e163.0",  "1\u001e2\u001eDaniel\u001e169.3",
      "2\u001e2\u001eFord\u001e182.7", "2\u001e3\u001eLeo\u001e172.8",
      "0\u001e3\u001eLead\u001e179.5"};

  for (auto &record : records) {
    std::vector<std::string> key_values;
    for (auto &column_index : segment_keys_column_indexes) {
      key_values.push_back(
          std::string(vdb::GetTokenFrom(record, vdb::kRS, column_index)));
    }
    std::string col_names = vdb::Join(segment_keys_column_names, "+");
    std::string json_values = "[\"" + vdb::Join(key_values, "\",\"") + "\"]";
    std::string segment_id =
        "v3::value::comp::" + col_names + "::" + json_values + "::::";
    auto segment = table->GetSegment(segment_id);
    if (segment == nullptr) {
      segment = table->AddSegment(table, segment_id);
    }
    auto status = segment->AppendRecord(record);
    if (!status.ok()) {
      EXPECT_TRUE(status.ok());
      std::cout << status.ToString() << std::endl;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto filtered_segments,
                       table->GetFilteredSegments(predicate, schema));
  EXPECT_EQ(filtered_segments.size(), 1);
}

TEST_F(ExpressionTest, DataFilterTest) {
  std::string test_schema_string =
      "id int32 not null, name string, height float32";
  auto schema = vdb::ParseSchemaFrom(test_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  expression::ExpressionBuilder expr_builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto predicate,
      expr_builder.ParseFilter(
          "id < 2 and id > 0 and name like 'T%' and id is not null"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kAnd);
  EXPECT_EQ(predicate->ToString(),
            "(id < 2 AND id > 0 AND name LIKE 'T%' AND id IS NOT NULL)");

  auto segment = std::make_shared<vdb::Segment>(table, "test_segment", 0);
  std::vector<std::string> records = {
      "0\u001eTom\u001e180.3", "1\u001eJane\u001e190.1",
      "1\u001eTyson\u001e197.9", "2\u001eJohn\u001e168.7"};

  for (auto &record : records) {
    auto status = segment->AppendRecord(record);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      EXPECT_TRUE(status.ok());
    }
  }

  ASSERT_OK_AND_ASSIGN(auto filtered_result,
                       segment->GetFilteredRecordbatches(predicate));
  auto filtered_rbs = filtered_result.first;
  EXPECT_EQ(filtered_rbs.size(), 1);
}

TEST_F(ExpressionTest, MixedFilterTest1) {
  std::string test_schema_string = "id int32 not null, val int32, tag string";
  auto schema = vdb::ParseSchemaFrom(test_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  expression::ExpressionBuilder expr_builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto predicate,
      expr_builder.ParseFilter("(id < 4 and val > 10) and (tag like '1%')"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kAnd);
  EXPECT_EQ(predicate->ToString(), "((id < 4 AND val > 10) AND tag LIKE '1%')");

  std::vector<std::string> records;
  for (int i = 0; i < 100; i++) {
    std::string str = "";
    // id
    str += std::to_string(i % 10);
    str += "\u001e";
    // val
    str += std::to_string(i % 20);
    str += "\u001e";
    // tag
    str += std::to_string(i);
    str += "th SampleTags";
    records.push_back(str);
  }

  for (auto &record : records) {
    std::string id_value = std::string(vdb::GetTokenFrom(record, vdb::kRS, 0));
    std::string segment_id = "v3::value::comp::id::[\"" + id_value + "\"]::::";
    auto segment = table->GetSegment(segment_id);
    if (segment == nullptr) {
      segment = table->AddSegment(table, segment_id);
    }
    auto status = segment->AppendRecord(record);
    if (!status.ok()) {
      EXPECT_TRUE(status.ok());
      std::cout << status.ToString() << std::endl;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto filtered_segments,
                       table->GetFilteredSegments(predicate, schema));
  EXPECT_EQ(filtered_segments.size(), 4);
  for (auto &segment : filtered_segments) {
    ASSERT_OK_AND_ASSIGN(auto filtered_result,
                         segment->GetFilteredRecordbatches(predicate));
    auto filtered_rbs = filtered_result.first;
    for (auto rb : filtered_rbs) {
      EXPECT_EQ(rb->num_rows(), 1);
      auto id_column = std::static_pointer_cast<arrow::Int32Array>(
          rb->GetColumnByName("id"));
      EXPECT_NE(id_column, nullptr);

      if (!id_column->IsNull(0)) {
        int32_t id = id_column->Value(0);
        if (1 <= id && id <= 3) {
          EXPECT_EQ(rb->num_rows(), 1);
        } else {
          ASSERT_TRUE(false);
        }
      }
    }
  }
}

TEST_F(ExpressionTest, MixedFilterTest2) {
  std::string test_schema_string = "id int32 not null, val int32, tag string";
  auto schema = vdb::ParseSchemaFrom(test_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  expression::ExpressionBuilder expr_builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto predicate,
      expr_builder.ParseFilter("(id < 4 and val > 10) or (tag like '1%')"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kOr);
  EXPECT_EQ(predicate->ToString(), "((id < 4 AND val > 10) OR tag LIKE '1%')");

  std::vector<std::string> records;
  for (int i = 0; i < 100; i++) {
    std::string str = "";
    // id
    str += std::to_string(i % 10);
    str += "\u001e";
    // val
    str += std::to_string(i % 20);
    str += "\u001e";
    // tag
    str += std::to_string(i);
    str += "th SampleTags";
    records.push_back(str);
  }

  for (auto &record : records) {
    std::string id_value = std::string(vdb::GetTokenFrom(record, vdb::kRS, 0));
    std::string segment_id = "v3::value::comp::id::[\"" + id_value + "\"]::::";
    auto segment = table->GetSegment(segment_id);
    if (segment == nullptr) {
      segment = table->AddSegment(table, segment_id);
    }
    auto status = segment->AppendRecord(record);
    if (!status.ok()) {
      EXPECT_TRUE(status.ok());
      std::cout << status.ToString() << std::endl;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto filtered_segments,
                       table->GetFilteredSegments(predicate, schema));
  EXPECT_EQ(filtered_segments.size(), 10);
  for (auto &segment : filtered_segments) {
    ASSERT_OK_AND_ASSIGN(auto filtered_result,
                         segment->GetFilteredRecordbatches(predicate));
    auto filtered_rbs = filtered_result.first;
    for (auto rb : filtered_rbs) {
      auto id_column = std::static_pointer_cast<arrow::Int32Array>(
          rb->GetColumnByName("id"));
      ASSERT_NE(id_column, nullptr);
      if (!id_column->IsNull(0)) {
        int32_t id = id_column->Value(0);
        if (id == 1) {
          EXPECT_EQ(rb->num_rows(), 6);
        } else if (2 <= id && id <= 3) {
          EXPECT_EQ(rb->num_rows(), 5);
        } else if (0 == id || id <= 9) {
          // id = 0, 4, 5, 6, 7, 8, 9
          EXPECT_EQ(rb->num_rows(), 1);
        } else {
          // id < 0 or id >= 10
          ASSERT_TRUE(false);
        }
      }
    }
  }
}

TEST_F(ExpressionTest, MixedFilterTest3) {
  // All segments are filtered out
  std::string test_schema_string = "id int32 not null, val int32, tag string";
  auto schema = vdb::ParseSchemaFrom(test_schema_string);
  std::string table_name = "test_table";
  std::string segment_type = "value";
  std::string segment_keys = "id";
  std::string segment_key_composition_type = "single";
  std::string segmentation_info_str = MakeSegmentationInfoString(
      segment_type, segment_keys, segment_key_composition_type);
  auto metadata = std::make_shared<arrow::KeyValueMetadata>(
      std::unordered_map<std::string, std::string>(
          {{"segmentation_info", segmentation_info_str},
           {"table name", table_name},
           {"active_set_size_limit",
            std::to_string(server.vdb_active_set_size_limit)}}));
  schema = schema->WithMetadata(metadata);
  vdb::TableBuilderOptions options;
  vdb::TableBuilder builder{
      std::move(options.SetTableName(table_name).SetSchema(schema))};
  ASSERT_OK_AND_ASSIGN(auto table, builder.Build());

  expression::ExpressionBuilder expr_builder(schema);
  ASSERT_OK_AND_ASSIGN(
      auto predicate,
      expr_builder.ParseFilter("(id > 10 and val > 10) and (tag like '1%')"));
  EXPECT_EQ(predicate->GetType(), expression::PredicateType::kAnd);
  EXPECT_EQ(predicate->ToString(),
            "((id > 10 AND val > 10) AND tag LIKE '1%')");

  std::vector<std::string> records;
  for (int i = 0; i < 100; i++) {
    std::string str = "";
    // id
    str += std::to_string(i % 10);
    str += "\u001e";
    // val
    str += std::to_string(i % 20);
    str += "\u001e";
    // tag
    str += std::to_string(i);
    str += "th SampleTags";
    records.push_back(str);
  }

  for (auto &record : records) {
    std::string id_value = std::string(vdb::GetTokenFrom(record, vdb::kRS, 0));
    std::string segment_id = "v3::value::comp::id::[\"" + id_value + "\"]::::";
    auto segment = table->GetSegment(segment_id);
    if (segment == nullptr) {
      segment = table->AddSegment(table, segment_id);
    }
    auto status = segment->AppendRecord(record);
    if (!status.ok()) {
      EXPECT_TRUE(status.ok());
      std::cout << status.ToString() << std::endl;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto filtered_segments,
                       table->GetFilteredSegments(predicate, schema));
  EXPECT_TRUE(filtered_segments.empty());
}

}  // namespace vdb::expression

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new vdb::BaseEnvironment);
  return RUN_ALL_TESTS();
}
