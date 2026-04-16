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
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/DateArithmeticFunctions.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/type/Time.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

namespace {
// Parse interval string in format "[-]DAYS HH:MM:SS.mmm" to milliseconds.
// This is the inverse of IntervalDayTimeType::valueToString().
// Examples: "5 00:00:00.000", "1 03:48:20.100", "-2 12:30:00.500"
int64_t parseIntervalDayTimeString(const std::string& str) {
  int64_t sign = 1;
  size_t pos = 0;

  // Check for negative sign
  if (!str.empty() && str[0] == '-') {
    sign = -1;
    pos = 1;
  }

  // Find the space separating days from time
  auto spacePos = str.find(' ', pos);
  VELOX_USER_CHECK(
      spacePos != std::string::npos,
      "Invalid interval format, expected 'DAYS HH:MM:SS.mmm': {}",
      str);

  // Parse days
  int64_t days = std::stoll(str.substr(pos, spacePos - pos));

  // Parse time part HH:MM:SS.mmm using int for sscanf compatibility
  int hours = 0, minutes = 0, seconds = 0, millis = 0;
  int parsedFields = sscanf(
      str.c_str() + spacePos + 1,
      "%d:%d:%d.%d",
      &hours,
      &minutes,
      &seconds,
      &millis);

  VELOX_USER_CHECK(
      parsedFields >= 3,
      "Invalid interval time format, expected 'HH:MM:SS[.mmm]': {}",
      str);

  return sign *
      (days * kMillisInDay + hours * kMillisInHour + minutes * kMillisInMinute +
       seconds * kMillisInSecond + millis);
}
} // namespace

DateAddFunction::DateAddFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "date_add function expects exactly 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->type()->isDate(),
      "First argument to date_add must be a date");
  VELOX_CHECK_NULL(
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]));
  value_ = makeScalarFromConstantExpr(
      expr->inputs()[1], cudf::type_id::DURATION_DAYS);
}

ColumnOrView DateAddFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto inputCol = asView(inputColumns[0]);
  return cudf::binary_operation(
      inputCol,
      *value_,
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
      stream,
      mr);
}

DatePlusIntervalFunction::DatePlusIntervalFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(),
      2,
      "plus(date, interval) expects exactly 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->type()->isDate(),
      "First argument to plus must be a date");
  VELOX_CHECK(
      expr->inputs()[1]->type()->isIntervalDayTime(),
      "Second argument to plus must be an interval day to second");

  // Check if the interval argument is a constant.
  // It could be either:
  // 1. A direct ConstantExpr with interval type (constant-folded)
  // 2. A cast(ConstantExpr<VARCHAR>) to interval - we parse the string ourselves
  //    since Velox doesn't support cast from VARCHAR to interval
  auto intervalExpr = expr->inputs()[1];
  int64_t intervalMillis = 0;
  bool foundConstant = false;

  // Case 1: Direct ConstantExpr with interval type
  auto constExpr =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(intervalExpr);
  if (constExpr && constExpr->type()->isIntervalDayTime()) {
    auto constValue = constExpr->value();
    intervalMillis =
        constValue->as<velox::ConstantVector<int64_t>>()->valueAt(0);
    foundConstant = true;
  }

  // Case 2: cast(ConstantExpr<VARCHAR>) to interval
  if (!foundConstant &&
      (intervalExpr->name() == "cast" || intervalExpr->name() == "try_cast") &&
      intervalExpr->type()->isIntervalDayTime() &&
      !intervalExpr->inputs().empty()) {
    auto innerConstExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
        intervalExpr->inputs()[0]);
    if (innerConstExpr && innerConstExpr->type()->isVarchar()) {
      // Extract the VARCHAR value and parse it
      auto constValue = innerConstExpr->value();
      VELOX_CHECK_NOT_NULL(constValue, "ConstantExpr value is null");
      VELOX_CHECK(
          !constValue->isNullAt(0),
          "Cannot cast null VARCHAR to interval");
      auto varcharValue =
          constValue->asUnchecked<velox::ConstantVector<velox::StringView>>()
              ->valueAt(0);
      intervalMillis = parseIntervalDayTimeString(varcharValue.str());
      foundConstant = true;
    }
  }

  if (foundConstant) {
    isConstantInterval_ = true;

    // Validate that the interval represents whole days.
    VELOX_USER_CHECK_EQ(
        intervalMillis % kMillisInDay,
        0,
        "Cannot add hours, minutes, seconds or milliseconds to a date");

    // Convert milliseconds to days and create the scalar.
    auto days = static_cast<int32_t>(intervalMillis / kMillisInDay);
    auto stream = cudf::get_default_stream(cudf::allow_default_stream);
    auto mr = get_temp_mr();
    value_ = std::make_unique<cudf::duration_scalar<cudf::duration_D>>(
        days, true, stream, mr);
    stream.synchronize();
  }
  // else: non-constant interval column - will be handled in eval()
}

ColumnOrView DatePlusIntervalFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto dateCol = asView(inputColumns[0]);

  if (isConstantInterval_) {
    // Use the pre-computed scalar for constant intervals.
    return cudf::binary_operation(
        dateCol,
        *value_,
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  }

  // Handle non-constant interval column.
  auto intervalCol = asView(inputColumns[1]);

  // Validate that all intervals are whole days.
  auto divisor = cudf::numeric_scalar<int64_t>(kMillisInDay, true, stream, mr);
  auto remainder = cudf::binary_operation(
      intervalCol,
      divisor,
      cudf::binary_operator::MOD,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, mr);
  auto isWholeDays = cudf::binary_operation(
      remainder->view(),
      zero,
      cudf::binary_operator::EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto allWholeDays = cudf::reduce(
      isWholeDays->view(),
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto* result = static_cast<cudf::scalar_type_t<bool>*>(allWholeDays.get());
  VELOX_USER_CHECK(
      result->is_valid(stream) && result->value(stream),
      "Cannot add hours, minutes, seconds or milliseconds to a date");

  // Divide millis by kMillisInDay to get days.
  auto daysInt = cudf::binary_operation(
      intervalCol,
      divisor,
      cudf::binary_operator::DIV,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);

  // Cast days to duration_days and add to date.
  auto daysDuration = cudf::cast(
      daysInt->view(),
      cudf::data_type(cudf::type_id::DURATION_DAYS),
      stream,
      mr);
  return cudf::binary_operation(
      dateCol,
      daysDuration->view(),
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox
