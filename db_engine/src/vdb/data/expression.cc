#include "vdb/data/expression.hh"

#include <memory>
#include <set>
#include <sstream>

#include <arrow/compute/api.h>
#include <arrow/json/from_string.h>
#include <arrow/ipc/api.h>
#include <arrow/scalar.h>
#include <nlohmann/json.hpp>
#include <string>
#include <strings.h>
#include <vector>

#include "vdb/common/system_log.hh"
#include "vdb/data/metadata.hh"
#include "vdb/data/segmentation.hh"
#include "vdb/data/table.hh"
#include "vdb/common/util.hh"

namespace vdb::expression {

static inline std::shared_ptr<Expression> UnwrapProjection(
    std::shared_ptr<Expression> expr) {
  while (expr) {
    auto proj = std::dynamic_pointer_cast<Projection>(expr);
    if (!proj) break;
    expr = proj->GetInnerExpression();
  }
  return expr;
}

static inline std::shared_ptr<arrow::DataType> GetExpressionDataType(
    const std::shared_ptr<Expression>& expr) {
  auto e = UnwrapProjection(expr);
  if (!e) return nullptr;
  if (auto col = std::dynamic_pointer_cast<Column>(e)) {
    return col->GetDataType();
  }
  if (auto lit = std::dynamic_pointer_cast<Literal>(e)) {
    return lit->GetDataType();
  }
  return nullptr;
}

// Helper: Fetch a scalar value for a given column and record index from a
// RecordBatch
[[maybe_unused]] static inline arrow::Result<std::shared_ptr<arrow::Scalar>>
GetRecordScalarByName(const std::shared_ptr<arrow::RecordBatch>& rb,
                      const std::string& column_name,
                      const size_t record_number) {
  if (!rb) {
    return arrow::Status::Invalid("Null RecordBatch");
  }

  auto array = rb->GetColumnByName(column_name);
  if (!array) {
    std::stringstream ss;
    ss << "Column not found: " << column_name;
    return arrow::Status::Invalid(ss.str());
  }

  if (record_number >= static_cast<size_t>(array->length())) {
    std::stringstream ss;
    ss << "Record number out of bounds: " << record_number << " for column '"
       << column_name << "' with length " << array->length();
    return arrow::Status::Invalid(ss.str());
  }

  return array->GetScalar(record_number);
}

// Overload for Column expression
[[maybe_unused]] static inline arrow::Result<std::shared_ptr<arrow::Scalar>>
GetRecordScalar(const std::shared_ptr<arrow::RecordBatch>& rb,
                const std::shared_ptr<Column>& column,
                const size_t record_number) {
  if (!column) {
    return arrow::Status::Invalid("Null Column expression");
  }
  return GetRecordScalarByName(rb, column->ToString(), record_number);
}

// Generalized: fetch scalar for a left-side Expression (Column or Literal)
static inline arrow::Result<std::shared_ptr<arrow::Scalar>>
GetRecordScalarFromExpr(const std::shared_ptr<arrow::RecordBatch>& rb,
                        const std::shared_ptr<Expression>& expr,
                        const size_t record_number) {
  auto e = UnwrapProjection(expr);
  if (!e) {
    return arrow::Status::Invalid(
        "Null expression after unwrapping projection");
  }
  if (auto col = std::dynamic_pointer_cast<Column>(e)) {
    return GetRecordScalar(rb, col, record_number);
  }
  if (auto lit = std::dynamic_pointer_cast<Literal>(e)) {
    return lit->GetScalar();
  }
  // Fallback: evaluate arbitrary Arrow expression on the whole batch,
  // then take the scalar at record_number
  ARROW_ASSIGN_OR_RAISE(auto arrow_expr, e->BuildArrowExpression());
  ARROW_ASSIGN_OR_RAISE(auto bound_expr, arrow_expr.Bind(*(rb->schema())));
  auto eb = arrow::compute::ExecBatch(*rb);
  ARROW_ASSIGN_OR_RAISE(
      auto output, arrow::compute::ExecuteScalarExpression(bound_expr, eb));
  if (output.is_scalar()) {
    return output.scalar();
  }
  if (output.is_array()) {
    auto arr = output.make_array();
    if (record_number >= static_cast<size_t>(arr->length())) {
      return arrow::Status::Invalid(
          "Record number out of bounds for evaluated expression");
    }
    return arr->GetScalar(static_cast<int64_t>(record_number));
  }
  return arrow::Status::Invalid(
      "Unsupported evaluated datum kind for record evaluation");
}

// Helper: Fetch a scalar value from segment id map by column name and datatype
static inline arrow::Result<std::shared_ptr<arrow::Scalar>>
GetSegmentScalarByName(
    const std::unordered_map<std::string, std::string>& segment_id_map,
    const std::string& column_name,
    const std::shared_ptr<arrow::DataType>& datatype) {
  auto it = segment_id_map.find(column_name);
  if (it == segment_id_map.end()) {
    std::stringstream ss;
    ss << "Column not found in segment_id_map: " << column_name;
    return arrow::Status::Invalid(ss.str());
  }
  if (!datatype) {
    std::stringstream ss;
    ss << "No datatype for column '" << column_name << "'";
    return arrow::Status::Invalid(ss.str());
  }
  return arrow::Scalar::Parse(datatype, it->second);
}

// Overload for Column expression
static inline arrow::Result<std::shared_ptr<arrow::Scalar>> GetSegmentScalar(
    const std::unordered_map<std::string, std::string>& segment_id_map,
    const std::shared_ptr<Column>& column) {
  if (!column) {
    return arrow::Status::Invalid("Null Column expression");
  }
  return GetSegmentScalarByName(segment_id_map, column->ToString(),
                                column->GetDataType());
}

// Generalized: fetch scalar for a left-side Expression (Column or Literal)
static inline arrow::Result<std::shared_ptr<arrow::Scalar>>
GetSegmentScalarFromExpr(
    const std::unordered_map<std::string, std::string>& segment_id_map,
    const std::shared_ptr<Expression>& expr) {
  if (auto col = std::dynamic_pointer_cast<Column>(expr)) {
    return GetSegmentScalar(segment_id_map, col);
  }
  // Non-column expressions can't be resolved from segment metadata
  return arrow::Status::Invalid(
      "Unsupported left expression type for segment evaluation");
}

arrow::Result<arrow::compute::Expression> Column::BuildArrowExpression() const {
  return arrow::compute::field_ref(name_);
}

arrow::Result<arrow::compute::Expression> Projection::BuildArrowExpression()
    const {
  return expr_->BuildArrowExpression();
}

arrow::Result<arrow::compute::Expression> Literal::BuildArrowExpression()
    const {
  if (scalar_ != nullptr) {
    return arrow::compute::literal(scalar_);
  }

  return arrow::compute::literal(scalar_string_);
}

arrow::Result<arrow::compute::Expression> And::BuildArrowExpression() const {
  auto exprs = std::vector<arrow::compute::Expression>();
  for (const auto& expr : exprs_) {
    ARROW_ASSIGN_OR_RAISE(auto e, expr->BuildArrowExpression());
    exprs.push_back(e);
  }
  return arrow::compute::and_(exprs);
}

std::string And::ToString() const {
  std::vector<std::string> strs;
  for (const auto& expr : exprs_) {
    strs.push_back(expr->ToString());
  }
  return "(" + vdb::Join(strs, " AND ") + ")";
}

arrow::Result<arrow::compute::Expression> Or::BuildArrowExpression() const {
  auto exprs = std::vector<arrow::compute::Expression>();
  for (const auto& expr : exprs_) {
    ARROW_ASSIGN_OR_RAISE(auto e, expr->BuildArrowExpression());
    exprs.push_back(e);
  }
  return arrow::compute::or_(exprs);
}

std::string Or::ToString() const {
  std::vector<std::string> strs;
  for (const auto& expr : exprs_) {
    strs.push_back(expr->ToString());
  }
  return "(" + vdb::Join(strs, " OR ") + ")";
}

std::string Comparison::ToString() const {
  const char* op_str;
  switch (operation_) {
    case ComparisonOperation::kEqual:
      op_str = "=";
      break;
    case ComparisonOperation::kNotEqual:
      op_str = "!=";
      break;
    case ComparisonOperation::kGreaterThan:
      op_str = ">";
      break;
    case ComparisonOperation::kGreaterThanEqual:
      op_str = ">=";
      break;
    case ComparisonOperation::kLessThan:
      op_str = "<";
      break;
    case ComparisonOperation::kLessThanEqual:
      op_str = "<=";
      break;
    default:
      std::invalid_argument("Must not reach here.");
  }
  return left_->ToString() + " " + op_str + " " + right_->ToString();
}

std::string IsNull::ToString() const {
  const std::string name = expr_->ToString();
  return is_negated_ ? name + " IS NOT NULL" : name + " IS NULL";
}

std::string In::ToString() const {
  std::string joined_values;
  auto type = GetExpressionDataType(LeftExpression());
  if (type && is_string(type->id())) {
    std::vector<std::string> quoted_values;
    for (const auto& value : values_) {
      quoted_values.push_back("'" + value + "'");
    }
    joined_values = vdb::Join(quoted_values, ", ");
  } else {
    joined_values = vdb::Join(values_, ", ");
  }
  return is_negated_
             ? expr_->ToString() + " NOT IN " + "[" + joined_values + "]"
             : expr_->ToString() + " IN " + "[" + joined_values + "]";
}

std::string Like::ToString() const {
  return is_negated_ ? expr_->ToString() + " NOT LIKE '" + pattern_ + "'"
                     : expr_->ToString() + " LIKE '" + pattern_ + "'";
}

std::string Not::ToString() const { return "NOT " + expr_->ToString(); }

arrow::Result<arrow::compute::Expression> Comparison::BuildArrowExpression()
    const {
  switch (operation_) {
    case ComparisonOperation::kEqual: {
      ARROW_ASSIGN_OR_RAISE(auto lhs, left_->BuildArrowExpression());
      ARROW_ASSIGN_OR_RAISE(auto rhs, right_->BuildArrowExpression());
      return arrow::compute::equal(lhs, rhs);
    }
    case ComparisonOperation::kNotEqual: {
      ARROW_ASSIGN_OR_RAISE(auto lhs, left_->BuildArrowExpression());
      ARROW_ASSIGN_OR_RAISE(auto rhs, right_->BuildArrowExpression());
      return arrow::compute::not_equal(lhs, rhs);
    }
    case ComparisonOperation::kGreaterThan: {
      ARROW_ASSIGN_OR_RAISE(auto lhs, left_->BuildArrowExpression());
      ARROW_ASSIGN_OR_RAISE(auto rhs, right_->BuildArrowExpression());
      return arrow::compute::greater(lhs, rhs);
    }
    case ComparisonOperation::kGreaterThanEqual: {
      ARROW_ASSIGN_OR_RAISE(auto lhs, left_->BuildArrowExpression());
      ARROW_ASSIGN_OR_RAISE(auto rhs, right_->BuildArrowExpression());
      return arrow::compute::greater_equal(lhs, rhs);
    }
    case ComparisonOperation::kLessThan: {
      ARROW_ASSIGN_OR_RAISE(auto lhs, left_->BuildArrowExpression());
      ARROW_ASSIGN_OR_RAISE(auto rhs, right_->BuildArrowExpression());
      return arrow::compute::less(lhs, rhs);
    }
    case ComparisonOperation::kLessThanEqual: {
      ARROW_ASSIGN_OR_RAISE(auto lhs, left_->BuildArrowExpression());
      ARROW_ASSIGN_OR_RAISE(auto rhs, right_->BuildArrowExpression());
      return arrow::compute::less_equal(lhs, rhs);
    }
    default:
      return arrow::Status::Invalid("Invalid comparison operation");
  }
}

std::string In::BuildJsonArrayString() const {
  auto type = GetExpressionDataType(LeftExpression());

  // For string types, we need to properly escape values for JSON
  if (type && is_string(type->id())) {
    // Use nlohmann::json to properly escape string values
    nlohmann::json json_array = nlohmann::json::array();
    for (const auto& value : values_) {
      json_array.push_back(value);
    }
    return json_array.dump();
  } else {
    return "[" + vdb::Join(values_, ',') + "]";
  }
}

arrow::Result<arrow::compute::Expression> In::BuildArrowExpression() const {
  auto lhs_expr = LeftExpression();
  if (!lhs_expr) return arrow::Status::Invalid("Left side is null for IN");
  ARROW_ASSIGN_OR_RAISE(auto lhs, lhs_expr->BuildArrowExpression());
  auto type = GetExpressionDataType(lhs_expr);
  if (!type)
    return arrow::Status::Invalid("Unsupported left expression type for IN");

  std::string set = BuildJsonArrayString();

  ARROW_ASSIGN_OR_RAISE(auto setarr_result,
                        arrow::json::ArrayFromJSONString(type, set));
  auto rhs = arrow::compute::SetLookupOptions(setarr_result);
  auto in_expr = arrow::compute::call("is_in", {lhs}, rhs);
  if (is_negated_) {
    return arrow::compute::not_(in_expr);
  }
  return in_expr;
}

arrow::Result<arrow::compute::Expression> IsNull::BuildArrowExpression() const {
  ARROW_ASSIGN_OR_RAISE(auto expr, expr_->BuildArrowExpression());
  auto is_null_expr = arrow::compute::is_null(expr);
  if (is_negated_) {
    return arrow::compute::not_(is_null_expr);
  }
  return is_null_expr;
}

arrow::Result<arrow::compute::Expression> Like::BuildArrowExpression() const {
  auto lhs_expr = LeftExpression();
  if (!lhs_expr) return arrow::Status::Invalid("Left side is null for LIKE");
  ARROW_ASSIGN_OR_RAISE(auto expr, lhs_expr->BuildArrowExpression());
  auto like_expr = arrow::compute::call(
      "match_like", {expr},
      arrow::compute::MatchSubstringOptions(pattern_, false));
  if (is_negated_) {
    return arrow::compute::not_(like_expr);
  }
  return like_expr;
}

arrow::Result<arrow::compute::Expression> Not::BuildArrowExpression() const {
  ARROW_ASSIGN_OR_RAISE(auto expr, expr_->BuildArrowExpression());
  auto not_expr = arrow::compute::not_(expr);
  return not_expr;
}

arrow::Result<std::vector<std::shared_ptr<vdb::Segment>>>
Predicate::PruneSegments(
    const vdb::map<std::string_view, std::shared_ptr<vdb::Segment>>& segments,
    const std::shared_ptr<arrow::Schema>& schema) {
  auto segmentation_info_str =
      schema->metadata()->Get(kSegmentationInfoKey).ValueOr("");
  SegmentationInfoBuilder segmentation_info_builder;
  segmentation_info_builder.SetSegmentationInfo(segmentation_info_str);
  segmentation_info_builder.SetSchema(schema);
  ARROW_ASSIGN_OR_RAISE(auto segmentation_info,
                        segmentation_info_builder.Build());
  if (segmentation_info.GetSegmentType() == SegmentType::kUndefined) {
    assert(segments.size() <= 1);
    assert(segments.size() == 0 || segments.begin()->first == "_default_");
  }

  if (segmentation_info.GetSegmentType() == SegmentType::kUndefined ||
      segmentation_info.GetSegmentType() == SegmentType::kHash) {
    std::vector<std::shared_ptr<vdb::Segment>> result;
    result.reserve(segments.size());
    for (const auto& [_, segment] : segments) {
      result.push_back(segment);
    }
    return result;
  }

  const auto& segment_keys_column_names = segmentation_info.GetSegmentKeys();
  std::vector<std::shared_ptr<vdb::Segment>> result;
  for (const auto& segment : segments) {
    ARROW_ASSIGN_OR_RAISE(
        auto segment_id_map,
        SegmentIdUtils::ParseSegmentId(std::string(segment.first),
                                       segment_keys_column_names,
                                       segmentation_info.GetSegmentType()));

    ARROW_ASSIGN_OR_RAISE(auto evaluation_result,
                          EvaluateSegment(segment_id_map));
    if (evaluation_result) {
      result.push_back(segment.second);
    }
  }
  return result;
}

arrow::Result<bool> And::EvaluateSegment(
    const std::unordered_map<std::string, std::string>& segment_id_map) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug, "EvaluateSegment: %s",
             ToString().c_str());
  for (const auto& expr : exprs_) {
    if (!expr) continue;
    auto res = expr->EvaluateSegment(segment_id_map);
    if (!res.ok()) {
      continue;
    }
    if (!res.ValueUnsafe()) {
      return false;  // Short-circuit AND on explicit false
    }
  }
  // Conservative keep for unknowns at segment-level
  return true;
}

arrow::Result<bool> Or::EvaluateSegment(
    const std::unordered_map<std::string, std::string>& segment_id_map) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug, "EvaluateSegment: %s",
             ToString().c_str());
  bool has_unknown = false;
  for (const auto& expr : exprs_) {
    if (!expr) continue;
    auto res = expr->EvaluateSegment(segment_id_map);
    if (!res.ok()) {
      has_unknown = true;
      continue;
    }
    if (res.ValueUnsafe()) {
      return true;  // Short-circuit OR on explicit true
    }
  }
  // If all were explicit false but any unknown existed, conservative keep
  return has_unknown ? true : false;
}

arrow::Result<bool> ScalarPredicateBase::EvaluateRecord(
    std::shared_ptr<arrow::RecordBatch> rb, const size_t record_number) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug, "EvaluateRecord: %s",
             ToString().c_str());
  ARROW_ASSIGN_OR_RAISE(auto left, FetchLeftScalarRecord(rb, record_number));
  ARROW_ASSIGN_OR_RAISE(auto eval, Evaluate(left));
  return eval;
}

arrow::Result<bool> ScalarPredicateBase::EvaluateSegment(
    const std::unordered_map<std::string, std::string>& segment_id_map) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug, "EvaluateSegment: %s",
             ToString().c_str());
  auto left_res = FetchLeftScalarSegment(segment_id_map);
  if (!left_res.ok()) {
    // Segment-level conservative policy: keep on failure
    return true;
  }
  auto eval_res = Evaluate(left_res.ValueUnsafe());
  if (!eval_res.ok()) {
    return true;
  }
  return eval_res.ValueUnsafe();
}

arrow::Result<std::shared_ptr<arrow::Scalar>>
ScalarPredicateBase::FetchLeftScalarRecord(
    std::shared_ptr<arrow::RecordBatch> rb, size_t record_number) const {
  return GetRecordScalarFromExpr(rb, LeftExpression(), record_number);
}

arrow::Result<std::shared_ptr<arrow::Scalar>>
ScalarPredicateBase::FetchLeftScalarSegment(
    const std::unordered_map<std::string, std::string>& segment_id_map) const {
  return GetSegmentScalarFromExpr(segment_id_map, LeftExpression());
}

arrow::Result<bool> Comparison::EvaluateComparison(
    const std::shared_ptr<arrow::Scalar>& left_value,
    const std::shared_ptr<arrow::Scalar>& right_value,
    const ComparisonOperation operation) const {
  if (!left_value || !right_value) {
    return arrow::Status::Invalid("Null scalar value");
  }

  if (!left_value->is_valid || !right_value->is_valid) {
    return arrow::Status::Invalid("Invalid scalar value");
  }

  const char* function_name = nullptr;
  switch (operation) {
    case ComparisonOperation::kEqual:
      function_name = "equal";
      break;
    case ComparisonOperation::kNotEqual:
      function_name = "not_equal";
      break;
    case ComparisonOperation::kLessThan:
      function_name = "less";
      break;
    case ComparisonOperation::kLessThanEqual:
      function_name = "less_equal";
      break;
    case ComparisonOperation::kGreaterThan:
      function_name = "greater";
      break;
    case ComparisonOperation::kGreaterThanEqual:
      function_name = "greater_equal";
      break;
    default:
      return arrow::Status::Invalid("Unknown comparison operation");
  }

  ARROW_ASSIGN_OR_RAISE(
      auto result,
      arrow::compute::CallFunction(function_name, {left_value, right_value}));
  auto result_scalar = result.scalar_as<arrow::BooleanScalar>();
  return result_scalar.value;
}

arrow::Result<bool> Not::EvaluateSegment(
    const std::unordered_map<std::string, std::string>& segment_id_map) {
  SYSTEM_LOG(vdb::LogTopic::Unknown, LogLevel::kLogDebug, "EvaluateSegment: %s",
             ToString().c_str());

  auto pred = std::dynamic_pointer_cast<Predicate>(expr_);
  if (!pred) {
    return arrow::Status::Invalid("Expression is not a predicate");
  }
  ARROW_ASSIGN_OR_RAISE(auto evaluation_result,
                        pred->EvaluateSegment(segment_id_map));
  return !evaluation_result;
}

arrow::Result<bool> Or::EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                       const size_t record_number) {
  for (const auto& expr : exprs_) {
    if (!expr) continue;
    auto res = expr->EvaluateRecord(rb, record_number);
    if (!res.ok()) return res.status();
    if (res.ValueUnsafe()) return true;  // Short-circuit on true
  }
  return false;
}

arrow::Result<bool> And::EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                        const size_t record_number) {
  for (const auto& expr : exprs_) {
    if (!expr) continue;
    auto res = expr->EvaluateRecord(rb, record_number);
    if (!res.ok()) return res.status();
    if (!res.ValueUnsafe()) return false;  // Short-circuit on false
  }
  return true;
}

// Comparison column provider and evaluator
std::shared_ptr<Column> Comparison::LeftColumn() const {
  return std::dynamic_pointer_cast<Column>(left_);
}

arrow::Result<bool> Comparison::Evaluate(
    const std::shared_ptr<arrow::Scalar>& left) const {
  auto right_literal = std::dynamic_pointer_cast<Literal>(right_);
  if (!right_literal) {
    return arrow::Status::Invalid("Right is not a literal");
  }
  ARROW_ASSIGN_OR_RAISE(auto right_value, right_literal->GetScalar());
  return EvaluateComparison(left, right_value, operation_);
}

arrow::Result<bool> IsNull::Evaluate(
    const std::shared_ptr<arrow::Scalar>& left) const {
  bool is_null = !left || !left->is_valid;
  return is_negated_ ? !is_null : is_null;
}

// In column provider and evaluator
std::shared_ptr<Column> In::LeftColumn() const {
  return std::dynamic_pointer_cast<Column>(expr_);
}

arrow::Result<bool> In::Evaluate(
    const std::shared_ptr<arrow::Scalar>& left) const {
  auto options_type = [&]() -> std::shared_ptr<arrow::DataType> {
    if (auto col = In::LeftColumn()) return col->GetDataType();
    if (auto lit = std::dynamic_pointer_cast<Literal>(LeftExpression()))
      return lit->GetDataType();
    if (left && left->type) return left->type;
    return nullptr;
  }();
  if (!options_type) {
    return arrow::Status::Invalid(
        "Unsupported left expression type for IN predicate");
  }
  auto lhs = left;
  if (!lhs || lhs->type->id() != options_type->id()) {
    ARROW_ASSIGN_OR_RAISE(
        auto casted, arrow::compute::Cast(arrow::Datum(lhs), options_type));
    lhs = casted.scalar();
  }
  std::string set = BuildJsonArrayString();
  ARROW_ASSIGN_OR_RAISE(auto setarr_result,
                        arrow::json::ArrayFromJSONString(options_type, set));
  auto options = arrow::compute::SetLookupOptions(setarr_result);
  ARROW_ASSIGN_OR_RAISE(auto result,
                        arrow::compute::CallFunction("is_in", {lhs}, &options));
  auto result_scalar = result.scalar_as<arrow::BooleanScalar>();
  return is_negated_ ? !result_scalar.value : result_scalar.value;
}

// Like column provider and evaluator
std::shared_ptr<Column> Like::LeftColumn() const {
  return std::dynamic_pointer_cast<Column>(expr_);
}

arrow::Result<bool> Like::Evaluate(
    const std::shared_ptr<arrow::Scalar>& left) const {
  auto options = arrow::compute::MatchSubstringOptions(pattern_, false);
  ARROW_ASSIGN_OR_RAISE(auto result, arrow::compute::CallFunction(
                                         "match_like", {left}, &options));
  auto result_scalar = result.scalar_as<arrow::BooleanScalar>();
  return is_negated_ ? !result_scalar.value : result_scalar.value;
}

arrow::Result<bool> Not::EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                        const size_t record_number) {
  auto pred = std::dynamic_pointer_cast<Predicate>(expr_);
  if (!pred) {
    return arrow::Status::Invalid("Expression is not a predicate for NOT");
  }
  ARROW_ASSIGN_OR_RAISE(auto evaluation_result,
                        pred->EvaluateRecord(rb, record_number));
  return !evaluation_result;
}

static inline std::vector<std::string> CollectReferencedColumns(
    const std::vector<std::shared_ptr<Predicate>>& exprs) {
  std::vector<std::string> columns;
  for (const auto& expr : exprs) {
    if (expr) {
      auto expr_columns = expr->GetReferencedColumns();
      columns.insert(columns.end(), expr_columns.begin(), expr_columns.end());
    }
  }
  std::sort(columns.begin(), columns.end());
  columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
  return columns;
}

std::vector<std::string> And::GetReferencedColumns() const {
  return CollectReferencedColumns(exprs_);
}

std::vector<std::string> Or::GetReferencedColumns() const {
  return CollectReferencedColumns(exprs_);
}

std::vector<std::string> Comparison::GetReferencedColumns() const {
  auto collect =
      [](const std::shared_ptr<Expression>& expr) -> std::vector<std::string> {
    std::vector<std::string> out;
    auto e = UnwrapProjection(expr);
    if (auto col = std::dynamic_pointer_cast<Column>(e)) {
      out.push_back(col->ToString());
    }
    return out;
  };
  auto lcols = collect(left_);
  auto rcols = collect(right_);
  lcols.insert(lcols.end(), rcols.begin(), rcols.end());
  return lcols;
}

std::vector<std::string> IsNull::GetReferencedColumns() const {
  if (auto col = std::dynamic_pointer_cast<Column>(
          UnwrapProjection(LeftExpression()))) {
    return {col->ToString()};
  }
  return {};
}

std::vector<std::string> In::GetReferencedColumns() const {
  if (auto col = std::dynamic_pointer_cast<Column>(
          UnwrapProjection(LeftExpression()))) {
    return {col->ToString()};
  }
  return {};
}

std::vector<std::string> Like::GetReferencedColumns() const {
  if (auto col = std::dynamic_pointer_cast<Column>(
          UnwrapProjection(LeftExpression()))) {
    return {col->ToString()};
  }
  return {};
}

std::vector<std::string> Not::GetReferencedColumns() const {
  if (auto pred = std::dynamic_pointer_cast<Predicate>(expr_)) {
    return pred->GetReferencedColumns();
  }
  return {};
}

arrow::Result<std::vector<std::shared_ptr<Expression>>>
Expression::ParseSimpleProjectionList(
    std::string_view proj_list_string,
    std::shared_ptr<arrow::Schema> table_schema,
    bool include_internal_columns) {
  std::vector<std::shared_ptr<Expression>> ret;

  if (proj_list_string.empty()) {
    return arrow::Status::Invalid(
        "Could not parse projection list: Empty projection list");
  }

  if (proj_list_string == "*") {
    return ret;
  }

  auto tokens = Tokenize(proj_list_string, ',');
  if (tokens.empty()) {
    return ret;
  }

  // Build name_set: exclude internal columns unless scan_internal
  std::set<std::string> name_set;
  for (const auto& field_name : table_schema->field_names()) {
    if (include_internal_columns || !vdb::IsInternalColumn(field_name)) {
      name_set.insert(field_name);
    }
  }

  auto trim = [](std::string_view str) -> std::string_view {
    const std::string_view whitespace = " \t\n\r\f\v";
    const auto start = str.find_first_not_of(whitespace);
    if (start == std::string_view::npos) {
      return {};  // String is all whitespace
    }
    const auto end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
  };

  for (const auto& token : tokens) {
    std::string col_name(trim(token));
    auto it = name_set.find(col_name);
    if (it == name_set.end()) {
      std::stringstream ss;
      ss << "Could not find column '" << token << "' from table schema";
      return arrow::Status::Invalid(ss.str());
    }

    auto field = table_schema->GetFieldByName(col_name);
    ret.push_back(std::make_shared<Projection>(
        std::make_shared<Column>(col_name, field->type())));
  }

  return ret;
}

static inline std::string AsciiUpper(std::string s) {
  for (char& c : s) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return s;
}

std::vector<ExpressionBuilder::Token> ExpressionBuilder::TokenizeFilter(
    std::string_view filter_string) {
  std::vector<Token> tokens;
  std::string current;
  bool in_quotes = false;
  size_t token_start = 0;

  auto push_nonquoted = [&]() {
    if (!current.empty()) {
      tokens.push_back(Token{current, AsciiUpper(current), token_start, false});
      current.clear();
    }
  };

  auto push_quoted = [&](size_t start_pos) {
    if (!current.empty()) {
      // keep raw and upper identical for quoted literals
      tokens.push_back(Token{current, current, start_pos, true});
      current.clear();
    }
  };

  for (size_t i = 0; i < filter_string.length(); ++i) {
    char c = filter_string[i];

    if (c == '\'') {
      if (!in_quotes) {
        // starting a quoted literal
        if (!current.empty()) push_nonquoted();
        in_quotes = true;
        token_start = i;
        current.push_back(c);
      } else {
        // closing a quoted literal
        current.push_back(c);
        in_quotes = false;
        push_quoted(token_start);
      }
    } else if (in_quotes) {
      current.push_back(c);
    } else if (std::isspace(static_cast<unsigned char>(c))) {
      push_nonquoted();
    } else if (c == '(' || c == ')' || c == ',') {
      push_nonquoted();
      std::string s(1, c);
      tokens.push_back(Token{s, s, i, false});
    } else if (c == '=' || c == '!' || c == '>' || c == '<') {
      push_nonquoted();
      std::string op(1, c);
      if (i + 1 < filter_string.length()) {
        char n = filter_string[i + 1];
        if ((c == '<' && n == '=') || (c == '>' && n == '=') ||
            (c == '!' && n == '=')) {
          op.push_back(n);
          ++i;
        } else if (c == '<' && n == '>') {  // handle '<>'
          op.push_back(n);
          ++i;
        }
      }
      tokens.push_back(Token{op, AsciiUpper(op), i, false});
    } else {
      if (current.empty()) token_start = i;
      current.push_back(c);
    }
  }
  if (in_quotes) {
    // unterminated string literal - still push as quoted to preserve pos
    push_quoted(token_start);
  } else {
    push_nonquoted();
  }

  // Post-process tokens to handle composite keywords
  std::vector<Token> processed;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i + 2 < tokens.size() && tokens[i].upper == "IS" &&
        tokens[i + 1].upper == "NOT" && tokens[i + 2].upper == "NULL") {
      std::string raw = "IS NOT NULL";
      processed.push_back(Token{raw, raw, tokens[i].pos, false});
      i += 2;
    } else if (i + 1 < tokens.size() && tokens[i].upper == "IS" &&
               tokens[i + 1].upper == "NULL") {
      std::string raw = "IS NULL";
      processed.push_back(Token{raw, raw, tokens[i].pos, false});
      ++i;
    } else if (i + 1 < tokens.size() && tokens[i].upper == "NOT" &&
               tokens[i + 1].upper == "IN") {
      std::string raw = "NOT IN";
      processed.push_back(Token{raw, raw, tokens[i].pos, false});
      ++i;
    } else if (i + 1 < tokens.size() && tokens[i].upper == "NOT" &&
               tokens[i + 1].upper == "LIKE") {
      std::string raw = "NOT LIKE";
      processed.push_back(Token{raw, raw, tokens[i].pos, false});
      ++i;
    } else if (!tokens[i].is_quoted &&
               (tokens[i].upper == "IN" || tokens[i].upper == "LIKE" ||
                tokens[i].upper == "AND" || tokens[i].upper == "OR" ||
                tokens[i].upper == "NOT")) {
      // normalize keyword tokens' raw text to uppercase for consistency
      std::string kw = tokens[i].upper;
      processed.push_back(Token{kw, kw, tokens[i].pos, false});
    } else {
      processed.push_back(tokens[i]);
    }
  }

  return processed;
}

arrow::Result<std::shared_ptr<Predicate>> ExpressionBuilder::ParseSimpleFilter(
    const std::vector<Token>& tokens, size_t& index) {
  if (index >= tokens.size()) {
    return arrow::Status::Invalid("Unexpected end of input");
  }

  const std::string column_name = tokens[index].raw;
  auto field = schema_->GetFieldByName(column_name);
  if (!field) {
    return arrow::Status::Invalid("Column not found: " + column_name);
  }
  auto column_type = field->type();
  const bool is_string_type_column = is_string(column_type->id());
  ++index;

  if (index >= tokens.size()) {
    return arrow::Status::Invalid("Unexpected end of input after column name");
  }

  const auto& op_tok = tokens[index];
  if (op_tok.raw == "IS NULL" || op_tok.raw == "IS NOT NULL") {
    const bool is_negated = (op_tok.raw == "IS NOT NULL");
    ++index;
    return std::make_shared<IsNull>(
        std::make_shared<Column>(column_name, field->type()), is_negated);
  } else if (op_tok.raw == "IN" || op_tok.raw == "NOT IN") {
    const bool is_negated = (op_tok.raw == "NOT IN");
    ++index;
    if (index >= tokens.size() || tokens[index].raw != "(") {
      std::stringstream ss;
      ss << "Expected opening parenthesis after IN. Near '" << op_tok.raw
         << "' at pos " << op_tok.pos;
      return arrow::Status::Invalid(ss.str());
    }
    ++index;  // skip '('

    std::vector<std::string> values;
    while (index < tokens.size() && tokens[index].raw != ")") {
      if (is_string_type_column) {
        if (!(tokens[index].is_quoted && tokens[index].raw.size() >= 2 &&
              tokens[index].raw.front() == '\'' &&
              tokens[index].raw.back() == '\'')) {
          std::stringstream ss;
          ss << "Column '" << column_name
             << "' is of string type; values in IN list must be string "
                "literals enclosed in single quotes. Near '"
             << tokens[index].raw << "' at pos " << tokens[index].pos;
          return arrow::Status::Invalid(ss.str());
        }
        const std::string& lit = tokens[index].raw;
        values.push_back(lit.substr(1, lit.size() - 2));
      } else {
        values.push_back(tokens[index].raw);
      }
      ++index;
      if (index < tokens.size() && tokens[index].raw == ",") {
        ++index;  // skip comma
      }
    }
    if (index >= tokens.size() || tokens[index].raw != ")") {
      return arrow::Status::Invalid(
          "Expected closing parenthesis after IN list");
    }
    ++index;  // skip ')'
    return std::make_shared<In>(
        std::make_shared<Column>(column_name,
                                 schema_->GetFieldByName(column_name)->type()),
        values, is_negated);
  } else if (op_tok.raw == "LIKE" || op_tok.raw == "NOT LIKE") {
    const bool is_negated = (op_tok.raw == "NOT LIKE");
    ++index;
    if (index >= tokens.size()) {
      return arrow::Status::Invalid("Unexpected end of input after LIKE");
    }

    if (!is_string_type_column) {
      std::stringstream ss;
      ss << "LIKE operator can only be used with string type columns for "
            "column '"
         << column_name << "'";
      return arrow::Status::Invalid(ss.str());
    }

    if (!(tokens[index].is_quoted && tokens[index].raw.size() >= 2 &&
          tokens[index].raw.front() == '\'' &&
          tokens[index].raw.back() == '\'')) {
      std::stringstream ss;
      ss << "LIKE pattern for column '" << column_name
         << "' must be a string literal enclosed in single quotes. Near '"
         << tokens[index].raw << "' at pos " << tokens[index].pos;
      return arrow::Status::Invalid(ss.str());
    }
    std::string pattern =
        tokens[index].raw.substr(1, tokens[index].raw.size() - 2);
    ++index;
    return std::make_shared<Like>(
        std::make_shared<Column>(column_name,
                                 schema_->GetFieldByName(column_name)->type()),
        pattern, is_negated);
  } else {
    ComparisonOperation op;
    if (op_tok.raw == "=") {
      op = ComparisonOperation::kEqual;
    } else if (op_tok.raw == "!=" || op_tok.raw == "<>") {
      op = ComparisonOperation::kNotEqual;
    } else if (op_tok.raw == ">") {
      op = ComparisonOperation::kGreaterThan;
    } else if (op_tok.raw == "<") {
      op = ComparisonOperation::kLessThan;
    } else if (op_tok.raw == ">=") {
      op = ComparisonOperation::kGreaterThanEqual;
    } else if (op_tok.raw == "<=") {
      op = ComparisonOperation::kLessThanEqual;
    } else {
      std::stringstream ss;
      ss << "Invalid operator: " << op_tok.raw << ". Near '" << op_tok.raw
         << "' at pos " << op_tok.pos
         << ". Expected one of: =, !=, <, <=, >, >=, LIKE, IN, IS NULL";
      return arrow::Status::Invalid(ss.str());
    }

    ++index;
    if (index >= tokens.size()) {
      std::stringstream ss;
      ss << "Unexpected end of input after operator '" << op_tok.raw
         << "' for column '" << column_name << "'";
      return arrow::Status::Invalid(ss.str());
    }
    std::string value = tokens[index].raw;
    if (is_string_type_column) {
      if (!(tokens[index].is_quoted && value.size() >= 2 &&
            value.front() == '\'' && value.back() == '\'')) {
        std::stringstream ss;
        ss << "Column '" << column_name
           << "' is of string type; expected a string literal enclosed in "
              "single quotes. Near '"
           << tokens[index].raw << "' at pos " << tokens[index].pos;
        return arrow::Status::Invalid(ss.str());
      }
      value = value.substr(1, value.size() - 2);
    }
    ++index;
    auto datatype = schema_->GetFieldByName(column_name)->type();
    return std::make_shared<Comparison>(
        op, std::make_shared<Column>(column_name, datatype),
        std::make_shared<Literal>(value, datatype));
  }
}

arrow::Result<std::shared_ptr<Predicate>> ExpressionBuilder::ParseExpression(
    const std::vector<Token>& tokens, size_t& index) {
  std::vector<std::shared_ptr<Predicate>> expressions;

  ARROW_ASSIGN_OR_RAISE(auto first_expression,
                        ParseAndExpression(tokens, index));
  expressions.push_back(std::move(first_expression));

  while (index < tokens.size() && tokens[index].upper == "OR") {
    ++index;  // skip OR
    ARROW_ASSIGN_OR_RAISE(auto next_expression,
                          ParseAndExpression(tokens, index));
    expressions.push_back(std::move(next_expression));
  }

  if (expressions.size() == 1) {
    return std::move(expressions[0]);
  }

  return std::make_shared<Or>(std::move(expressions));
}

arrow::Result<std::shared_ptr<Predicate>> ExpressionBuilder::ParseAndExpression(
    const std::vector<Token>& tokens, size_t& index) {
  std::vector<std::shared_ptr<Predicate>> expressions;

  ARROW_ASSIGN_OR_RAISE(auto first_expression,
                        ParsePrimaryExpression(tokens, index));
  expressions.push_back(std::move(first_expression));

  while (index < tokens.size() && tokens[index].upper == "AND") {
    ++index;  // skip AND
    ARROW_ASSIGN_OR_RAISE(auto next_expression,
                          ParsePrimaryExpression(tokens, index));
    expressions.push_back(std::move(next_expression));
  }

  if (expressions.size() == 1) {
    return std::move(expressions[0]);
  }

  return std::make_shared<And>(std::move(expressions));
}

arrow::Result<std::shared_ptr<Predicate>>
ExpressionBuilder::ParsePrimaryExpression(const std::vector<Token>& tokens,
                                          size_t& index) {
  if (index >= tokens.size()) {
    return arrow::Status::Invalid("Unexpected end of input");
  }

  if (tokens[index].raw == "(") {
    ++index;  // skip '('
    ARROW_ASSIGN_OR_RAISE(auto expression, ParseExpression(tokens, index));
    if (index >= tokens.size() || tokens[index].raw != ")") {
      return arrow::Status::Invalid("Mismatched parentheses");
    }
    ++index;  // skip ')'
    return expression;
  } else if (tokens[index].upper == "NOT") {
    ++index;  // skip NOT
    ARROW_ASSIGN_OR_RAISE(auto inner_expression,
                          ParsePrimaryExpression(tokens, index));
    return std::make_shared<Not>(std::move(inner_expression));
  } else {
    return ParseSimpleFilter(tokens, index);
  }
}

arrow::Result<std::shared_ptr<Predicate>> ExpressionBuilder::ParseFilter(
    std::string_view filter_string) {
  static bool is_initialized = false;
  if (!is_initialized) {
    ARROW_RETURN_NOT_OK(arrow::compute::Initialize());
    is_initialized = true;
  }

  auto tokens = TokenizeFilter(filter_string);
  if (tokens.empty()) {
    return nullptr;
  }

  size_t index = 0;
  ARROW_ASSIGN_OR_RAISE(auto result, ParseExpression(tokens, index));
  if (index != tokens.size()) {
    std::stringstream ss;
    ss << "Unexpected token '" << tokens[index].raw << "' at pos "
       << tokens[index].pos;
    return arrow::Status::Invalid(ss.str());
  }
  return result;
}

}  // namespace vdb::expression
