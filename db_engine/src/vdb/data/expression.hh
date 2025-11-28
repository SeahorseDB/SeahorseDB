#pragma once

#include <arrow/compute/api.h>
#include <memory>
#include <string>
#include "vdb/common/memory_allocator.hh"

namespace vdb {
class Segment;
}

namespace vdb::expression {

enum class ExpressionType { kNone, kPredicate, kProjection, kColumn, kLiteral };

enum class PredicateType { kAnd, kOr, kComparison, kIsNull, kIn, kLike, kNot };

enum class ComparisonOperation {
  kEqual,
  kNotEqual,
  kLessThan,
  kGreaterThan,
  kLessThanEqual,
  kGreaterThanEqual
};

enum class StringOperationType {
  kStartsWith,
  kEndsWith,
  kContains,
};

class Expression {
 public:
  Expression() = delete;

  Expression(ExpressionType type) : type_(type) {}

  Expression(const Expression& rhs) = default;

  Expression(Expression&& rhs) = default;

  Expression& operator=(const Expression& rhs) = default;

  Expression& operator=(Expression&& rhs) = default;

  virtual ~Expression() = default;

  ExpressionType GetType() const { return type_; }

  virtual arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const = 0;

  virtual std::string ToString() const = 0;

  static arrow::Result<std::vector<std::shared_ptr<Expression>>>
  ParseSimpleProjectionList(std::string_view proj_list_string,
                            std::shared_ptr<arrow::Schema> table_schema,
                            bool include_internal_columns = false);

 protected:
  ExpressionType type_;
};

class Column : public Expression {
 public:
  Column() = delete;

  Column(const std::string name)
      : Expression(ExpressionType::kColumn), name_(name) {}

  Column(const std::string name, std::shared_ptr<arrow::DataType> datatype)
      : Expression(ExpressionType::kColumn), name_(name), datatype_(datatype) {}

  Column(const Column& rhs) = default;

  Column(Column&& rhs) = default;

  Column& operator=(const Column& rhs) = default;

  Column& operator=(Column&& rhs) = default;

  virtual ~Column() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override { return name_; }

  std::shared_ptr<arrow::DataType> GetDataType() const { return datatype_; }

 protected:
  std::string name_;
  std::shared_ptr<arrow::DataType> datatype_;
};

class Projection : public Expression {
 public:
  Projection() = delete;

  Projection(std::shared_ptr<Expression> expr)
      : Expression(ExpressionType::kProjection), expr_(expr) {}

  Projection(const Projection& rhs) = default;

  Projection(Projection&& rhs) = default;

  Projection& operator=(const Projection& rhs) = default;

  virtual ~Projection() = default;

  Projection& operator=(Projection&& rhs) = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override { return expr_->ToString(); }

  // Accessor for wrapped expression (to allow unwrapping in generic helpers)
  std::shared_ptr<Expression> GetInnerExpression() const { return expr_; }

 protected:
  std::shared_ptr<Expression> expr_;
};

class Literal : public Expression {
 public:
  Literal() = delete;

  Literal(std::string scalar, std::shared_ptr<arrow::DataType> datatype)
      : Expression(ExpressionType::kLiteral),
        scalar_string_(scalar),
        datatype_(datatype) {
    auto result = arrow::Scalar::Parse(datatype, scalar);
    if (!result.ok()) {
      scalar_ = nullptr;
    } else {
      scalar_ = result.ValueUnsafe();
    }
  }

  explicit Literal(std::shared_ptr<arrow::Scalar> scalar)
      : Expression(ExpressionType::kLiteral),
        scalar_string_(scalar ? scalar->ToString() : ""),
        scalar_(scalar),
        datatype_(scalar ? scalar->type : nullptr) {}

  Literal(const Literal& rhs) = default;

  Literal(Literal&& rhs) = default;

  Literal& operator=(const Literal& rhs) = default;

  Literal& operator=(Literal&& rhs) = default;

  virtual ~Literal() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override {
    if (is_string(datatype_->id())) {
      return "'" + scalar_string_ + "'";
    } else {
      return scalar_string_;
    }
  }

  arrow::Result<std::shared_ptr<arrow::Scalar>> GetScalar() const {
    return scalar_;
  }

  std::shared_ptr<arrow::DataType> GetDataType() const { return datatype_; }

 protected:
  std::string scalar_string_;
  std::shared_ptr<arrow::Scalar> scalar_;
  std::shared_ptr<arrow::DataType> datatype_;
};

class Predicate : public Expression {
 public:
  Predicate() = delete;

  Predicate(PredicateType type)
      : Expression(ExpressionType::kPredicate), pred_type_(type) {}

  Predicate(const Predicate& rhs) = default;

  Predicate(Predicate&& rhs) = default;

  Predicate& operator=(const Predicate& rhs) = default;

  Predicate& operator=(Predicate&& rhs) = default;

  virtual ~Predicate() = default;

  PredicateType GetType() const { return pred_type_; }

  virtual arrow::Result<bool> EvaluateSegment(
      const std::unordered_map<std::string, std::string>& segment_id_map) = 0;

  virtual arrow::Result<bool> EvaluateRecord(
      std::shared_ptr<arrow::RecordBatch> rb, const size_t record_number) = 0;

  virtual arrow::Result<std::vector<std::shared_ptr<vdb::Segment>>>
  PruneSegments(
      const vdb::map<std::string_view, std::shared_ptr<vdb::Segment>>& segments,
      const std::shared_ptr<arrow::Schema>& schema);

  // Get column names referenced in this predicate (for lazy loading
  // optimization)
  virtual std::vector<std::string> GetReferencedColumns() const = 0;

 protected:
  PredicateType pred_type_;
};

// Base class for predicates that evaluate based on a left scalar value.
class ScalarPredicateBase : public Predicate {
 public:
  ScalarPredicateBase() = delete;

  explicit ScalarPredicateBase(PredicateType type) : Predicate(type) {}

  ScalarPredicateBase(const ScalarPredicateBase& rhs) = default;
  ScalarPredicateBase(ScalarPredicateBase&& rhs) = default;
  ScalarPredicateBase& operator=(const ScalarPredicateBase& rhs) = default;
  ScalarPredicateBase& operator=(ScalarPredicateBase&& rhs) = default;
  virtual ~ScalarPredicateBase() = default;

  arrow::Result<bool> EvaluateSegment(
      const std::unordered_map<std::string, std::string>& segment_id_map)
      override;

  arrow::Result<bool> EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                     const size_t record_number) override;

 protected:
  // Default fetchers implemented in base using LeftColumn()
  virtual arrow::Result<std::shared_ptr<arrow::Scalar>> FetchLeftScalarRecord(
      std::shared_ptr<arrow::RecordBatch> rb, size_t record_number) const;

  virtual arrow::Result<std::shared_ptr<arrow::Scalar>> FetchLeftScalarSegment(
      const std::unordered_map<std::string, std::string>& segment_id_map) const;

  // General left-hand side expression provider
  virtual std::shared_ptr<Expression> LeftExpression() const = 0;

  // Column provider for left-hand side (default: cast from LeftExpression)
  virtual std::shared_ptr<Column> LeftColumn() const {
    return std::dynamic_pointer_cast<Column>(LeftExpression());
  }

  // Evaluate using left scalar and class-specific right-hand side
  virtual arrow::Result<bool> Evaluate(
      const std::shared_ptr<arrow::Scalar>& left) const = 0;
};

class And : public Predicate {
 public:
  And() = delete;

  And(std::vector<std::shared_ptr<Predicate>> exprs)
      : Predicate(PredicateType::kAnd), exprs_(exprs) {}

  And(const And& rhs) = default;

  And(And&& rhs) = default;

  And& operator=(const And& rhs) = default;

  And& operator=(And&& rhs) = default;

  virtual ~And() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  arrow::Result<bool> EvaluateSegment(
      const std::unordered_map<std::string, std::string>& segment_id_map)
      override;

  arrow::Result<bool> EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                     const size_t record_number) override;

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  std::vector<std::shared_ptr<Predicate>> exprs_;
};

class Or : public Predicate {
 public:
  Or() = delete;

  Or(std::vector<std::shared_ptr<Predicate>> exprs)
      : Predicate(PredicateType::kOr), exprs_(exprs) {}

  Or(const Or& rhs) = default;

  Or(Or&& rhs) = default;

  Or& operator=(const Or& rhs) = default;

  Or& operator=(Or&& rhs) = default;

  virtual ~Or() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  arrow::Result<bool> EvaluateSegment(
      const std::unordered_map<std::string, std::string>& segment_id_map)
      override;

  arrow::Result<bool> EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                     const size_t record_number) override;

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  std::vector<std::shared_ptr<Predicate>> exprs_;
};

class Comparison : public ScalarPredicateBase {
 public:
  Comparison() = delete;

  Comparison(ComparisonOperation operation, std::shared_ptr<Expression> left,
             std::shared_ptr<Expression> right)
      : ScalarPredicateBase(PredicateType::kComparison),
        operation_(operation),
        left_(left),
        right_(right) {}

  Comparison(const Comparison& rhs) = default;

  Comparison(Comparison&& rhs) = default;

  Comparison& operator=(const Comparison& rhs) = default;

  Comparison& operator=(Comparison&& rhs) = default;

  virtual ~Comparison() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  ComparisonOperation GetOperation() const { return operation_; }

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  ComparisonOperation operation_;
  std::shared_ptr<Expression> left_;
  std::shared_ptr<Expression> right_;
  std::shared_ptr<Expression> LeftExpression() const override { return left_; }
  std::shared_ptr<Column> LeftColumn() const override;
  arrow::Result<bool> Evaluate(
      const std::shared_ptr<arrow::Scalar>& left) const override;

 private:
  arrow::Result<bool> EvaluateComparison(
      const std::shared_ptr<arrow::Scalar>& left_value,
      const std::shared_ptr<arrow::Scalar>& right_value,
      const ComparisonOperation operation) const;
};

class IsNull : public ScalarPredicateBase {
 public:
  IsNull(std::shared_ptr<Expression> expr)
      : ScalarPredicateBase(PredicateType::kIsNull),
        expr_(expr),
        is_negated_(false) {}

  IsNull(std::shared_ptr<Expression> expr, bool is_negated)
      : ScalarPredicateBase(PredicateType::kIsNull),
        expr_(expr),
        is_negated_(is_negated) {}

  IsNull(const IsNull& rhs) = default;

  IsNull(IsNull&& rhs) = default;

  IsNull& operator=(const IsNull& rhs) = default;

  IsNull& operator=(IsNull&& rhs) = default;

  virtual ~IsNull() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  std::shared_ptr<Expression> expr_;
  bool is_negated_;

 private:
  std::shared_ptr<Expression> LeftExpression() const override { return expr_; }
  std::shared_ptr<Column> LeftColumn() const override {
    return std::dynamic_pointer_cast<Column>(expr_);
  }
  arrow::Result<bool> Evaluate(
      const std::shared_ptr<arrow::Scalar>& left) const override;
};

class In : public ScalarPredicateBase {
 public:
  In() = delete;

  In(std::shared_ptr<Expression> expr, std::vector<std::string> values)
      : ScalarPredicateBase(PredicateType::kIn),
        expr_(expr),
        values_(values),
        is_negated_(false) {}

  In(std::shared_ptr<Expression> expr, std::vector<std::string> values,
     bool is_negated)
      : ScalarPredicateBase(PredicateType::kIn),
        expr_(expr),
        values_(values),
        is_negated_(is_negated) {}

  In(const In& rhs) = default;

  In(In&& rhs) = default;

  In& operator=(const In& rhs) = default;

  In& operator=(In&& rhs) = default;

  virtual ~In() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  std::shared_ptr<Expression> expr_;
  std::vector<std::string> values_;
  bool is_negated_;

 private:
  std::string BuildJsonArrayString() const;
  std::shared_ptr<Expression> LeftExpression() const override { return expr_; }
  std::shared_ptr<Column> LeftColumn() const override;
  arrow::Result<bool> Evaluate(
      const std::shared_ptr<arrow::Scalar>& left) const override;
};

class Like : public ScalarPredicateBase {
 public:
  Like() = delete;

  Like(std::shared_ptr<Expression> expr, std::string pattern)
      : ScalarPredicateBase(PredicateType::kLike),
        expr_(expr),
        pattern_(pattern),
        is_negated_(false) {}

  Like(std::shared_ptr<Expression> expr, std::string pattern, bool is_negated)
      : ScalarPredicateBase(PredicateType::kLike),
        expr_(expr),
        pattern_(pattern),
        is_negated_(is_negated) {}

  Like(const Like& rhs) = default;

  Like(Like&& rhs) = default;

  Like& operator=(const Like& rhs) = default;

  Like& operator=(Like&& rhs) = default;

  virtual ~Like() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  std::shared_ptr<Expression> expr_;
  std::string pattern_;
  bool is_negated_;
  std::shared_ptr<Expression> LeftExpression() const override { return expr_; }
  std::shared_ptr<Column> LeftColumn() const override;
  arrow::Result<bool> Evaluate(
      const std::shared_ptr<arrow::Scalar>& left) const override;
};

class Not : public Predicate {
 public:
  Not() = delete;

  Not(std::shared_ptr<Expression> expr)
      : Predicate(PredicateType::kNot), expr_(expr) {}

  Not(const Not& rhs) = default;

  Not(Not&& rhs) = default;

  Not& operator=(const Not& rhs) = default;

  Not& operator=(Not&& rhs) = default;

  virtual ~Not() = default;

  arrow::Result<arrow::compute::Expression> BuildArrowExpression()
      const override;

  std::string ToString() const override;

  arrow::Result<bool> EvaluateSegment(
      const std::unordered_map<std::string, std::string>& segment_id_map)
      override;

  arrow::Result<bool> EvaluateRecord(std::shared_ptr<arrow::RecordBatch> rb,
                                     const size_t record_number) override;

  std::vector<std::string> GetReferencedColumns() const override;

 protected:
  std::shared_ptr<Expression> expr_;
};

class ExpressionBuilder {
 public:
  ExpressionBuilder() = delete;

  ExpressionBuilder(std::shared_ptr<arrow::Schema> schema) : schema_(schema) {}

  ExpressionBuilder(const ExpressionBuilder& rhs) = default;

  ExpressionBuilder(ExpressionBuilder&& rhs) = default;

  ExpressionBuilder& operator=(const ExpressionBuilder& rhs) = default;

  ExpressionBuilder& operator=(ExpressionBuilder&& rhs) = default;

  virtual ~ExpressionBuilder() = default;

  arrow::Result<std::shared_ptr<Predicate>> ParseFilter(
      std::string_view filter_string);

 protected:
  struct Token {
    std::string raw;    // original lexeme
    std::string upper;  // uppercase form for keyword/operator comparisons
    size_t pos;         // start position in input string
    bool is_quoted;     // true if token is a quoted string literal
  };

  std::vector<Token> TokenizeFilter(std::string_view filter_string);

  arrow::Result<std::shared_ptr<Predicate>> ParseSimpleFilter(
      const std::vector<Token>& tokens, size_t& index);

  arrow::Result<std::shared_ptr<Predicate>> ParseExpression(
      const std::vector<Token>& tokens, size_t& index);

  arrow::Result<std::shared_ptr<Predicate>> ParseAndExpression(
      const std::vector<Token>& tokens, size_t& index);

  arrow::Result<std::shared_ptr<Predicate>> ParsePrimaryExpression(
      const std::vector<Token>& tokens, size_t& index);

  std::shared_ptr<arrow::Schema> schema_;
};
}  // namespace vdb::expression
