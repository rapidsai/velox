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

#include "velox/functions/prestosql/aggregates/ArbitraryAggregate.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/aggregates/SimpleNumericAggregate.h"
#include "velox/functions/lib/aggregates/SingleValueAccumulator.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::aggregate::prestosql {

namespace {

// Arbitrary aggregate returns any arbitrary non-NULL value.
// We always keep the first (non-NULL) element seen.
template <typename T>
class ArbitraryAggregate : public SimpleNumericAggregate<T, T, T> {
  using BaseAggregate = SimpleNumericAggregate<T, T, T>;

 public:
  explicit ArbitraryAggregate(TypePtr resultType) : BaseAggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(T);
  }

  int32_t accumulatorAlignmentSize() const override {
    return 1;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::doExtractValues(groups, numGroups, result, [&](char* group) {
      return *BaseAggregate::Aggregate::template value<T>(group);
    });
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    DecodedVector decoded(*args[0], rows);

    if (decoded.isConstantMapping()) {
      auto begin = rows.begin();
      if (decoded.isNullAt(begin)) {
        return;
      }
      auto value = decoded.valueAt<T>(begin);
      rows.applyToSelected([&](vector_size_t i) {
        if (exec::Aggregate::isNull(groups[i])) {
          updateValue(groups[i], value);
        }
      });
    } else if (decoded.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decoded.isNullAt(i) && exec::Aggregate::isNull(groups[i])) {
          updateValue(groups[i], decoded.valueAt<T>(i));
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        if (exec::Aggregate::isNull(groups[i])) {
          updateValue(groups[i], decoded.valueAt<T>(i));
        }
      });
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    if (!exec::Aggregate::isNull(group)) {
      return;
    }
    DecodedVector decoded(*args[0], rows);
    auto begin = rows.begin();

    if (decoded.isConstantMapping()) {
      if (decoded.isNullAt(begin)) {
        return;
      }
      updateValue(group, decoded.valueAt<T>(begin));
    } else if (!decoded.mayHaveNulls()) {
      updateValue(group, decoded.valueAt<T>(begin));
    } else {
      // Find the first non-null value.
      rows.testSelected([&](vector_size_t i) {
        if (!decoded.isNullAt(i)) {
          updateValue(group, decoded.valueAt<T>(i));
          return false; // Stop
        }
        return true; // Continue
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
  }

 private:
  inline void updateValue(char* group, T value) {
    exec::Aggregate::clearNull(group);
    *exec::Aggregate::value<T>(group) = value;
  }
};

/// Override 'accumulatorAlignmentSize' for UnscaledLongDecimal values as it
/// uses int128_t type. Some CPUs don't support misaligned access to int128_t
/// type.
template <>
inline int32_t ArbitraryAggregate<int128_t>::accumulatorAlignmentSize() const {
  return static_cast<int32_t>(sizeof(int128_t));
}

// Arbitrary for non-numeric types. We always keep the first (non-NULL) element
// seen. Arbitrary (x) will produce partial and final aggregations of type x.
class NonNumericArbitrary : public exec::Aggregate {
 public:
  explicit NonNumericArbitrary(const TypePtr& resultType)
      : exec::Aggregate(resultType) {}

  // We use singleValueAccumulator to save the results for each group. This
  // struct will allow us to save variable-width value.
  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SingleValueAccumulator);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    (*result)->resize(numGroups);

    auto* rawNulls = exec::Aggregate::getRawNulls(result->get());

    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = value<SingleValueAccumulator>(group);
      if (!accumulator->hasValue()) {
        (*result)->setNull(i, true);
      } else {
        exec::Aggregate::clearNull(rawNulls, i);
        accumulator->read(*result, i);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    DecodedVector decoded(*args[0], rows, true);
    if (decoded.isConstantMapping() && decoded.isNullAt(rows.begin())) {
      // nothing to do; all values are nulls
      return;
    }

    const auto* indices = decoded.indices();
    const auto* baseVector = decoded.base();
    rows.applyToSelected([&](vector_size_t i) {
      if (decoded.isNullAt(i)) {
        return;
      }
      auto* accumulator = value<SingleValueAccumulator>(groups[i]);
      if (!accumulator->hasValue()) {
        accumulator->write(baseVector, indices[i], allocator_);
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addRawInput(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*unused*/) override {
    auto* accumulator = value<SingleValueAccumulator>(group);
    if (accumulator->hasValue()) {
      return;
    }

    DecodedVector decoded(*args[0], rows, true);
    if (decoded.isConstantMapping() && decoded.isNullAt(rows.begin())) {
      // nothing to do; all values are nulls
      return;
    }

    const auto* indices = decoded.indices();
    const auto* baseVector = decoded.base();
    // Find the first non-null value.
    rows.testSelected([&](vector_size_t i) {
      if (!decoded.isNullAt(i)) {
        accumulator->write(baseVector, indices[i], allocator_);
        return false; // Stop
      }
      return true; // Continue
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    addSingleGroupRawInput(group, rows, args, mayPushdown);
  }

 protected:
  // Initialize each group, we will not use the null flags because
  // SingleValueAccumulator has its own flag.
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto i : indices) {
      new (groups[i] + offset_) SingleValueAccumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        value<SingleValueAccumulator>(group)->destroy(allocator_);
      }
    }
  }
};

} // namespace

void registerArbitraryAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("T")
          .intermediateType("T")
          .argumentType("T")
          .build()};

  std::vector<std::string> names = {prefix + kArbitrary, prefix + kAnyValue};
  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [name = names.front()](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& /*resultType*/,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(argTypes.size(), 1, "{} takes only one argument", name);
        auto inputType = argTypes[0];
        switch (inputType->kind()) {
          case TypeKind::BOOLEAN:
            return std::make_unique<ArbitraryAggregate<bool>>(inputType);
          case TypeKind::TINYINT:
            return std::make_unique<ArbitraryAggregate<int8_t>>(inputType);
          case TypeKind::SMALLINT:
            return std::make_unique<ArbitraryAggregate<int16_t>>(inputType);
          case TypeKind::INTEGER:
            return std::make_unique<ArbitraryAggregate<int32_t>>(inputType);
          case TypeKind::BIGINT:
            return std::make_unique<ArbitraryAggregate<int64_t>>(inputType);
          case TypeKind::HUGEINT:
            if (inputType->isLongDecimal()) {
              return std::make_unique<ArbitraryAggregate<int128_t>>(inputType);
            }
            VELOX_NYI();
          case TypeKind::REAL:
            return std::make_unique<ArbitraryAggregate<float>>(inputType);
          case TypeKind::DOUBLE:
            return std::make_unique<ArbitraryAggregate<double>>(inputType);
          case TypeKind::TIMESTAMP:
            return std::make_unique<ArbitraryAggregate<Timestamp>>(inputType);
          case TypeKind::VARBINARY:
            [[fallthrough]];
          case TypeKind::VARCHAR:
            [[fallthrough]];
          case TypeKind::ARRAY:
            [[fallthrough]];
          case TypeKind::MAP:
            [[fallthrough]];
          case TypeKind::ROW:
            [[fallthrough]];
          case TypeKind::UNKNOWN:
            return std::make_unique<NonNumericArbitrary>(inputType);
          default:
            VELOX_FAIL(
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
