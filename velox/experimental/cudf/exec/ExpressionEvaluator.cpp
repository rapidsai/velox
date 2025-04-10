/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/VectorTypeUtils.h"

#include <cudf/datetime.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

namespace facebook::velox::cudf_velox {
namespace {
template <TypeKind kind>
cudf::ast::literal make_scalar_and_literal(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    size_t at_index = 0) {
  using T = typename facebook::velox::KindToFlatVector<kind>::WrapperType;
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const auto& type = vector->type();

  if constexpr (cudf::is_fixed_width<T>()) {
    auto constVector = vector->as<facebook::velox::SimpleVector<T>>();
    VELOX_CHECK_NOT_NULL(constVector, "ConstantVector is null");
    T value = constVector->valueAt(at_index);
    if (type->isShortDecimal()) {
      VELOX_FAIL("Short decimal not supported");
      /* TODO: enable after rewriting using binary ops
      using CudfDecimalType = cudf::numeric::decimal64;
      using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
      auto scalar = std::make_unique<cudfScalarType>(value,
                    type->scale(),
                     true,
                     stream,
                     mr);
      scalars.emplace_back(std::move(scalar));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
      */
    } else if (type->isLongDecimal()) {
      VELOX_FAIL("Long decimal not supported");
      /* TODO: enable after rewriting using binary ops
      using CudfDecimalType = cudf::numeric::decimal128;
      using cudfScalarType = cudf::fixed_point_scalar<CudfDecimalType>;
      auto scalar = std::make_unique<cudfScalarType>(value,
                    type->scale(),
                     true,
                     stream,
                     mr);
      scalars.emplace_back(std::move(scalar));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
      */
    } else if (type->isIntervalYearMonth()) {
      // no support for interval year month in cudf
      VELOX_FAIL("Interval year month not supported");
    } else if (type->isIntervalDayTime()) {
      using CudfDurationType = cudf::duration_ms;
      if constexpr (std::is_same_v<T, CudfDurationType::rep>) {
        using cudfScalarType = cudf::duration_scalar<CudfDurationType>;
        auto scalar = std::make_unique<cudfScalarType>(value, true, stream, mr);
        scalars.emplace_back(std::move(scalar));
        return cudf::ast::literal{
            *static_cast<cudfScalarType*>(scalars.back().get())};
      }
    } else if (type->isDate()) {
      using CudfDateType = cudf::timestamp_D;
      if constexpr (std::is_same_v<T, CudfDateType::rep>) {
        using cudfScalarType = cudf::timestamp_scalar<CudfDateType>;
        auto scalar = std::make_unique<cudfScalarType>(value, true, stream, mr);
        scalars.emplace_back(std::move(scalar));
        return cudf::ast::literal{
            *static_cast<cudfScalarType*>(scalars.back().get())};
      }
    } else {
      // Create a numeric scalar of type T, store it in the scalars vector,
      // and use its reference in the literal expression.
      using cudfScalarType = cudf::numeric_scalar<T>;
      scalars.emplace_back(
          std::make_unique<cudfScalarType>(value, true, stream, mr));
      return cudf::ast::literal{
          *static_cast<cudfScalarType*>(scalars.back().get())};
    }
    VELOX_FAIL("Unsupported base type for literal");
  } else if (kind == TypeKind::VARCHAR) {
    auto constVector = vector->as<facebook::velox::SimpleVector<StringView>>();
    auto value = constVector->valueAt(at_index);
    std::string_view stringValue = static_cast<std::string_view>(value);
    scalars.emplace_back(
        std::make_unique<cudf::string_scalar>(stringValue, true, stream, mr));
    return cudf::ast::literal{
        *static_cast<cudf::string_scalar*>(scalars.back().get())};
  } else {
    // TODO for non-numeric types too.
    VELOX_NYI(
        "Non-numeric types not yet implemented for kind " +
        mapTypeKindToName(kind));
  }
}

cudf::ast::literal createLiteral(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    size_t at_index = 0) {
  const auto kind = vector->typeKind();
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      make_scalar_and_literal, kind, std::move(vector), scalars, at_index);
}

// Helper function to extract literals from array elements based on type
void extractArrayLiterals(
    const ArrayVector* arrayVector,
    std::vector<cudf::ast::literal>& literals,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    vector_size_t offset,
    vector_size_t size) {
  auto elements = arrayVector->elements();

  for (auto i = offset; i < offset + size; ++i) {
    if (elements->isNullAt(i)) {
      // Skip null values for IN expressions
      continue;
    } else {
      literals.emplace_back(createLiteral(elements, scalars, i));
    }
  }
}

// Function to create literals from an array vector
std::vector<cudf::ast::literal> createLiteralsFromArray(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  std::vector<cudf::ast::literal> literals;

  // Check if it's a constant vector containing an array
  if (vector->isConstantEncoding()) {
    auto constantVector = vector->asUnchecked<ConstantVector<ComplexType>>();
    if (constantVector->isNullAt(0)) {
      // Return empty vector for null array
      return literals;
    }

    auto valueVector = constantVector->valueVector();
    if (valueVector->encoding() == VectorEncoding::Simple::ARRAY) {
      auto arrayVector = valueVector->as<ArrayVector>();
      auto index = constantVector->index();
      auto size = arrayVector->sizeAt(index);
      if (size == 0) {
        // Return empty vector for empty array
        return literals;
      }

      auto offset = arrayVector->offsetAt(index);
      auto elements = arrayVector->elements();

      // Handle different element types
      if (elements->isScalar()) {
        literals.reserve(size);
        extractArrayLiterals(arrayVector, literals, scalars, offset, size);
      } else if (elements->typeKind() == TypeKind::ARRAY) {
        // Nested arrays not supported in IN expressions
        VELOX_FAIL("Nested arrays not supported in IN expressions");
      } else {
        VELOX_FAIL(
            "Unsupported element type in array: {}",
            elements->type()->toString());
      }
    } else {
      VELOX_FAIL("Expected ARRAY encoding");
    }
  } else {
    VELOX_FAIL("Expected constant vector for IN list");
  }

  return literals;
}
} // namespace

using op = cudf::ast::ast_operator;
const std::map<std::string, op> binary_ops = {
    {"plus", op::ADD},
    {"minus", op::SUB},
    {"multiply", op::MUL},
    {"divide", op::DIV},
    {"eq", op::EQUAL},
    {"neq", op::NOT_EQUAL},
    {"lt", op::LESS},
    {"gt", op::GREATER},
    {"lte", op::LESS_EQUAL},
    {"gte", op::GREATER_EQUAL},
    {"and", op::NULL_LOGICAL_AND},
    {"or", op::NULL_LOGICAL_OR}};

const std::map<std::string, op> unary_ops = {{"not", op::NOT}};

const std::unordered_set<std::string> supported_ops = {
    "literal",
    "between",
    "in",
    "cast",
    "switch",
    "year",
    "length",
    "substr",
    "like"};

namespace detail {

bool can_be_evaluated(const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto& name = expr->name();
  if (supported_ops.count(name) || binary_ops.count(name) ||
      unary_ops.count(name)) {
    return std::all_of(
        expr->inputs().begin(), expr->inputs().end(), can_be_evaluated);
  }
  return std::dynamic_pointer_cast<velox::exec::FieldReference>(expr) !=
      nullptr;
}

} // namespace detail

struct AstContext {
  cudf::ast::tree& tree;
  std::vector<std::unique_ptr<cudf::scalar>>& scalars;
  const std::vector<RowTypePtr> inputRowSchema;
  const std::vector<std::reference_wrapper<std::vector<PrecomputeInstruction>>>
      precompute_instructions;
  cudf::ast::expression const& push_expr_to_tree(
      const std::shared_ptr<velox::exec::Expr>& expr);
  cudf::ast::expression const& add_precompute_instruction(
      std::string const& name,
      std::string const& instruction);
  cudf::ast::expression const& multiple_inputs_to_pair_wise(
      const std::shared_ptr<velox::exec::Expr>& expr);
  static bool can_be_evaluated(const std::shared_ptr<velox::exec::Expr>& expr);
};

// Create tree from Expr
// and collect precompute instructions for non-ast operations
cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<PrecomputeInstruction>& precompute_instructions) {
  AstContext context{
      tree, scalars, {inputRowSchema}, {precompute_instructions}};
  return context.push_expr_to_tree(expr);
}

cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& leftRowSchema,
    const RowTypePtr& rightRowSchema,
    std::vector<PrecomputeInstruction>& left_precompute_instructions,
    std::vector<PrecomputeInstruction>& right_precompute_instructions) {
  AstContext context{
      tree,
      scalars,
      {leftRowSchema, rightRowSchema},
      {left_precompute_instructions, right_precompute_instructions}};
  return context.push_expr_to_tree(expr);
}

cudf::ast::expression const& AstContext::add_precompute_instruction(
    std::string const& name,
    std::string const& instruction) {
  for (size_t side_idx = 0; side_idx < inputRowSchema.size(); ++side_idx) {
    if (inputRowSchema[side_idx].get()->containsChild(name)) {
      auto column_index = inputRowSchema[side_idx].get()->getChildIdx(name);
      auto new_column_index = inputRowSchema[side_idx].get()->size() +
          precompute_instructions[side_idx].get().size();
      // This custom op should be added to input columns.
      precompute_instructions[side_idx].get().emplace_back(
          column_index, instruction, new_column_index);
      auto side = static_cast<cudf::ast::table_reference>(side_idx);
      return tree.push(cudf::ast::column_reference(new_column_index, side));
    }
  }
  VELOX_FAIL("Field not found, " + name);
}

/// Handles logical AND/OR expressions with multiple inputs by converting them
/// into a chain of binary operations. For example, "a AND b AND c" becomes
/// "(a AND b) AND c".
///
/// @param expr The expression containing multiple inputs for AND/OR operation
/// @return A reference to the resulting AST expression
cudf::ast::expression const& AstContext::multiple_inputs_to_pair_wise(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using operation = cudf::ast::operation;

  const auto& name = expr->name();
  auto len = expr->inputs().size();
  // Create a simple chain of operations
  auto result = &push_expr_to_tree(expr->inputs()[0]);

  // Chain the rest of the inputs sequentially
  for (size_t i = 1; i < len; i++) {
    auto const& next_input = push_expr_to_tree(expr->inputs()[i]);
    result = &tree.push(operation{binary_ops.at(name), *result, next_input});
  }
  return *result;
}

/// Pushes an expression into the AST tree and returns a reference to the
/// resulting expression.
///
/// @param expr The expression to push into the AST tree
/// @return A reference to the resulting AST expression
cudf::ast::expression const& AstContext::push_expr_to_tree(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using op = cudf::ast::ast_operator;
  using operation = cudf::ast::operation;
  using velox::exec::ConstantExpr;
  using velox::exec::FieldReference;

  auto& name = expr->name();
  auto len = expr->inputs().size();

  if (name == "literal") {
    auto c = dynamic_cast<ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    VELOX_CHECK(value->isConstantEncoding());
    // convert to cudf scalar
    return tree.push(createLiteral(value, scalars));
  } else if (binary_ops.find(name) != binary_ops.end()) {
    if (len > 2 and (name == "and" or name == "or")) {
      return multiple_inputs_to_pair_wise(expr);
    }
    VELOX_CHECK_EQ(len, 2);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    auto const& op2 = push_expr_to_tree(expr->inputs()[1]);
    return tree.push(operation{binary_ops.at(name), op1, op2});
  } else if (unary_ops.find(name) != unary_ops.end()) {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    return tree.push(operation{unary_ops.at(name), op1});
  } else if (name == "between") {
    VELOX_CHECK_EQ(len, 3);
    auto const& value = push_expr_to_tree(expr->inputs()[0]);
    auto const& lower = push_expr_to_tree(expr->inputs()[1]);
    auto const& upper = push_expr_to_tree(expr->inputs()[2]);
    // construct between(op2, op3) using >= and <=
    auto const& ge_lower =
        tree.push(operation{op::GREATER_EQUAL, value, lower});
    auto const& le_upper = tree.push(operation{op::LESS_EQUAL, value, upper});
    return tree.push(operation{op::NULL_LOGICAL_AND, ge_lower, le_upper});
  } else if (name == "in") {
    // number of inputs is variable. >=2
    VELOX_CHECK_EQ(len, 2);
    // actually len is 2, second input is ARRAY
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    auto c = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    VELOX_CHECK_NOT_NULL(value, "ConstantExpr value is null");

    // Use the new createLiteralsFromArray function to get literals
    auto literals = createLiteralsFromArray(value, scalars);

    // Create equality expressions for each literal and OR them together
    std::vector<const cudf::ast::expression*> expr_vec;
    for (auto& literal : literals) {
      auto const& opi = tree.push(std::move(literal));
      auto const& logical_node = tree.push(operation{op::EQUAL, op1, opi});
      expr_vec.push_back(&logical_node);
    }

    // Handle empty IN list case
    if (expr_vec.empty()) {
      // FAIL
      VELOX_FAIL("Empty IN list");
      // Return FALSE for empty IN list
      // auto falseValue = std::make_shared<ConstantVector<bool>>(
      //     value->pool(), 1, false, TypeKind::BOOLEAN, false);
      // return tree.push(createLiteral(falseValue, scalars));
    }

    // OR all logical nodes
    auto* result = expr_vec[0];
    for (size_t i = 1; i < expr_vec.size(); i++) {
      auto const& tree_node =
          tree.push(operation{op::NULL_LOGICAL_OR, *result, *expr_vec[i]});
      result = &tree_node;
    }
    return *result;
  } else if (name == "cast") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
    if (expr->type()->kind() == TypeKind::INTEGER) {
      // No int32 cast in cudf ast
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::BIGINT) {
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::DOUBLE) {
      return tree.push(operation{op::CAST_TO_FLOAT64, op1});
    } else {
      VELOX_FAIL("Unsupported type for cast operation");
    }
  } else if (name == "switch") {
    VELOX_CHECK_EQ(len, 3);
    // check if input[1], input[2] are literals 1 and 0.
    // then simplify as typecast bool to int
    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    if (c1 and c1->toString() == "1:BIGINT" and c2 and
        c2->toString() == "0:BIGINT") {
      auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
      return tree.push(operation{op::CAST_TO_INT64, op1});
    } else if (c2 and c2->toString() == "0:DOUBLE") {
      auto const& op1 = push_expr_to_tree(expr->inputs()[0]);
      auto const& op1d = tree.push(operation{op::CAST_TO_FLOAT64, op1});
      auto const& op2 = push_expr_to_tree(expr->inputs()[1]);
      return tree.push(operation{op::MUL, op1d, op2});
    } else {
      VELOX_NYI("Unsupported switch complex operation " + expr->toString());
    }
  } else if (name == "year") {
    VELOX_CHECK_EQ(len, 1);

    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");

    auto const& col_ref = add_precompute_instruction(fieldExpr->name(), "year");

    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else if (name == "length") {
    VELOX_CHECK_EQ(len, 1);

    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");

    auto const& col_ref =
        add_precompute_instruction(fieldExpr->name(), "length");

    return tree.push(operation{op::CAST_TO_INT64, col_ref});
  } else if (name == "substr") {
    // Extract the start and length parameters from the substr function call
    // and create a precomputed column with the substring operation.
    // This will be handled during AST evaluation with special column reference.
    VELOX_CHECK_EQ(len, 3);
    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");

    auto c1 = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    auto c2 = dynamic_cast<ConstantExpr*>(expr->inputs()[2].get());
    std::string substr_expr =
        "substr " + c1->value()->toString(0) + " " + c2->value()->toString(0);

    return add_precompute_instruction(fieldExpr->name(), substr_expr);
  } else if (name == "like") {
    VELOX_CHECK_EQ(len, 2);

    auto fieldExpr =
        std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(fieldExpr, "Expression is not a field");
    auto literalExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(literalExpr, "Expression is not a literal");

    createLiteral(literalExpr->value(), scalars);

    std::string like_expr = "like " + std::to_string(scalars.size() - 1);

    return add_precompute_instruction(fieldExpr->name(), like_expr);
  } else if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    // Refer to the appropriate side
    for (size_t side_idx = 0; side_idx < inputRowSchema.size(); ++side_idx) {
      auto& schema = inputRowSchema[side_idx];
      if (schema.get()->containsChild(name)) {
        auto column_index = schema.get()->getChildIdx(name);
        auto side = static_cast<cudf::ast::table_reference>(side_idx);
        return tree.push(cudf::ast::column_reference(column_index, side));
      }
    }
    VELOX_FAIL("Field not found, " + name);
  } else {
    std::cerr << "Unsupported expression: " << expr->toString() << std::endl;
    VELOX_FAIL("Unsupported expression: " + name);
  }
}

void addPrecomputedColumns(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    const std::vector<PrecomputeInstruction>& precompute_instructions,
    const std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    rmm::cuda_stream_view stream) {
  for (const auto& instruction : precompute_instructions) {
    auto [dependent_column_index, ins_name, new_column_index] = instruction;
    if (ins_name == "year") {
      auto new_column = cudf::datetime::extract_datetime_component(
          input_table_columns[dependent_column_index]->view(),
          cudf::datetime::datetime_component::YEAR,
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name == "length") {
      auto new_column = cudf::strings::count_characters(
          input_table_columns[dependent_column_index]->view(),
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name.rfind("substr", 0) == 0) {
      std::istringstream iss(ins_name.substr(6));
      int begin_value, length;
      iss >> begin_value >> length;
      auto begin_scalar = cudf::numeric_scalar<cudf::size_type>(
          begin_value - 1,
          true,
          stream,
          cudf::get_current_device_resource_ref());
      auto end_scalar = cudf::numeric_scalar<cudf::size_type>(
          begin_value - 1 + length,
          true,
          stream,
          cudf::get_current_device_resource_ref());
      auto step_scalar = cudf::numeric_scalar<cudf::size_type>(
          1, true, stream, cudf::get_current_device_resource_ref());
      auto new_column = cudf::strings::slice_strings(
          input_table_columns[dependent_column_index]->view(),
          begin_scalar,
          end_scalar,
          step_scalar,
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else if (ins_name.rfind("like", 0) == 0) {
      auto scalar_index = std::stoi(ins_name.substr(4));
      auto new_column = cudf::strings::like(
          input_table_columns[dependent_column_index]->view(),
          *static_cast<cudf::string_scalar*>(scalars[scalar_index].get()),
          cudf::string_scalar(
              "", true, stream, cudf::get_current_device_resource_ref()),
          stream,
          cudf::get_current_device_resource_ref());
      input_table_columns.emplace_back(std::move(new_column));
    } else {
      VELOX_FAIL("Unsupported precompute operation " + ins_name);
    }
  }
}

ExpressionEvaluator::ExpressionEvaluator(
    const std::vector<std::shared_ptr<velox::exec::Expr>>& exprs,
    const RowTypePtr& inputRowSchema) {
  exprAst_.reserve(exprs.size());
  for (const auto& expr : exprs) {
    cudf::ast::tree tree;
    create_ast_tree(
        expr, tree, scalars_, inputRowSchema, precompute_instructions_);
    exprAst_.emplace_back(std::move(tree));
  }
}

void ExpressionEvaluator::close() {
  exprAst_.clear();
  scalars_.clear();
  precompute_instructions_.clear();
}

std::vector<std::unique_ptr<cudf::column>> ExpressionEvaluator::compute(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto num_columns = input_table_columns.size();
  addPrecomputedColumns(
      input_table_columns, precompute_instructions_, scalars_, stream);
  auto ast_input_table =
      std::make_unique<cudf::table>(std::move(input_table_columns));
  auto ast_input_table_view = ast_input_table->view();
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto& tree : exprAst_) {
    if (auto col_ref_ptr =
            dynamic_cast<cudf::ast::column_reference const*>(&tree.back())) {
      auto col = std::make_unique<cudf::column>(
          ast_input_table_view.column(col_ref_ptr->get_column_index()),
          stream,
          mr);
      columns.emplace_back(std::move(col));
    } else {
      auto col =
          cudf::compute_column(ast_input_table_view, tree.back(), stream, mr);
      columns.emplace_back(std::move(col));
    }
  }
  input_table_columns = ast_input_table->release();
  input_table_columns.resize(num_columns);
  return columns;
}

bool ExpressionEvaluator::can_be_evaluated(
    const std::vector<std::shared_ptr<velox::exec::Expr>>& exprs) {
  return std::all_of(exprs.begin(), exprs.end(), detail::can_be_evaluated);
}
} // namespace facebook::velox::cudf_velox
