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
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <fmt/format.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace {

class OrderByTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
    rng_.seed(123);

    rowType_ = ROW(
        {{"c0", INTEGER()},
         {"c1", INTEGER()},
         {"c2", VARCHAR()},
         {"c3", VARCHAR()}});
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key) {
    core::PlanNodeId orderById;
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);
    auto plan = PlanBuilder()
                    .values(input)
                    .orderBy({fmt::format("{} ASC NULLS LAST", key)}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    runTest(
        plan,
        orderById,
        fmt::format("SELECT * FROM tmp ORDER BY {} NULLS LAST", key),
        {keyIndex});

    plan = PlanBuilder()
               .values(input)
               .orderBy({fmt::format("{} DESC NULLS FIRST", key)}, false)
               .planNode();
    runTest(
        plan,
        orderById,
        fmt::format("SELECT * FROM tmp ORDER BY {} DESC NULLS FIRST", key),
        {keyIndex});
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key,
      const std::string& filter) {
    core::PlanNodeId orderById;
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);
    auto plan = PlanBuilder()
                    .values(input)
                    .filter(filter)
                    .orderBy({fmt::format("{} ASC NULLS LAST", key)}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    runTest(
        plan,
        orderById,
        fmt::format(
            "SELECT * FROM tmp WHERE {} ORDER BY {} NULLS LAST", filter, key),
        {keyIndex});

    plan = PlanBuilder()
               .values(input)
               .filter(filter)
               .orderBy({fmt::format("{} DESC NULLS FIRST", key)}, false)
               .capturePlanNodeId(orderById)
               .planNode();
    runTest(
        plan,
        orderById,
        fmt::format(
            "SELECT * FROM tmp WHERE {} ORDER BY {} DESC NULLS FIRST",
            filter,
            key),
        {keyIndex});
  }

  void testTwoKeys(
      const std::vector<RowVectorPtr>& input,
      const std::string& key1,
      const std::string& key2) {
    auto& rowType = input[0]->type()->asRow();
    auto keyIndices = {rowType.getChildIdx(key1), rowType.getChildIdx(key2)};

    std::vector<core::SortOrder> sortOrders = {
        core::kAscNullsLast, core::kDescNullsFirst};
    std::vector<std::string> sortOrderSqls = {"NULLS LAST", "DESC NULLS FIRST"};

    for (int i = 0; i < sortOrders.size(); i++) {
      for (int j = 0; j < sortOrders.size(); j++) {
        core::PlanNodeId orderById;
        auto plan = PlanBuilder()
                        .values(input)
                        .orderBy(
                            {fmt::format("{} {}", key1, sortOrderSqls[i]),
                             fmt::format("{} {}", key2, sortOrderSqls[j])},
                            false)
                        .capturePlanNodeId(orderById)
                        .planNode();
        runTest(
            plan,
            orderById,
            fmt::format(
                "SELECT * FROM tmp ORDER BY {} {}, {} {}",
                key1,
                sortOrderSqls[i],
                key2,
                sortOrderSqls[j]),
            keyIndices);
      }
    }
  }

  void runTest(
      core::PlanNodePtr planNode,
      const core::PlanNodeId& orderById,
      const std::string& duckDbSql,
      const std::vector<uint32_t>& sortingKeys) {
    {
      SCOPED_TRACE("run without spilling");
      assertQueryOrdered(planNode, duckDbSql, sortingKeys);
    }
  }

  std::vector<RowVectorPtr> makeVectors(
      const RowTypePtr& rowType,
      int32_t numVectors,
      int32_t rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          facebook::velox::test::BatchMaker::createBatch(
              rowType, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  folly::Random::DefaultGenerator rng_;
  RowTypePtr rowType_;
};

TEST_F(OrderByTest, selectiveFilter) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 3; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  // c0 values are unique across batches
  testSingleKey(vectors, "c0", "c0 % 333 = 0");

  // c1 values are unique only within a batch
  testSingleKey(vectors, "c1", "c1 % 333 = 0");
}

TEST_F(OrderByTest, singleKey) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");

  // parser doesn't support "is not null" expression, hence, using c0 % 2 >= 0
  testSingleKey(vectors, "c0", "c0 % 2 >= 0");

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy({"c0 DESC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan, orderById, "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST", {0});

  plan = PlanBuilder()
             .values(vectors)
             .orderBy({"c0 ASC NULLS FIRST"}, false)
             .capturePlanNodeId(orderById)
             .planNode();
  runTest(plan, orderById, "SELECT * FROM tmp ORDER BY c0 NULLS FIRST", {0});
}

TEST_F(OrderByTest, multipleKeys) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    // c0: half of rows are null, a quarter is 0 and remaining quarter is 1
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [](vector_size_t row) { return row % 4; }, nullEvery(2, 1));
    auto c1 = makeFlatVector<int32_t>(
        batchSize, [](vector_size_t row) { return row; }, nullEvery(7));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testTwoKeys(vectors, "c0", "c1");

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy({"c0 ASC NULLS FIRST", "c1 ASC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan,
      orderById,
      "SELECT * FROM tmp ORDER BY c0 NULLS FIRST, c1 NULLS LAST",
      {0, 1});

  plan = PlanBuilder()
             .values(vectors)
             .orderBy({"c0 DESC NULLS LAST", "c1 DESC NULLS FIRST"}, false)
             .capturePlanNodeId(orderById)
             .planNode();
  runTest(
      plan,
      orderById,
      "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST, c1 DESC NULLS FIRST",
      {0, 1});
}

TEST_F(OrderByTest, multiBatchResult) {
  vector_size_t batchSize = 5000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 10; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c1, c1, c1, c1}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");
}

TEST_F(OrderByTest, varfields) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    auto c2 = makeFlatVector<StringView>(
        batchSize,
        [](vector_size_t row) {
          return StringView::makeInline(std::to_string(row));
        },
        nullEvery(17));
    // TODO: Add support for array/map in createDuckDbTable and verify
    // that we can sort by array/map as well.
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c2");
}

#if 0
// flattening for scalar types unsupported in arrow!
TEST_F(OrderByTest, unknown) {
  vector_size_t size = 1'000;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(size, [](auto row) { return row % 7; }),
      BaseVector::createNullConstant(UNKNOWN(), size, pool()),
  });

  // Exclude "UNKNOWN" column as DuckDB doesn't understand UNKNOWN type
  createDuckDbTable(
      {makeRowVector({vector->childAt(0)}),
       makeRowVector({vector->childAt(0)})});

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values({vector, vector})
                  .orderBy({"c0 DESC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan,
      orderById,
      "SELECT *, null FROM tmp ORDER BY c0 DESC NULLS LAST",
      {0});
}

/// Verifies output batch rows of OrderBy
TEST_F(OrderByTest, outputBatchRows) {
  struct {
    int numRowsPerBatch;
    int preferredOutBatchBytes;
    int maxOutBatchRows;
    int expectedOutputVectors;

    // TODO: add output size check with spilling enabled
    std::string debugString() const {
      return fmt::format(
          "numRowsPerBatch:{}, preferredOutBatchBytes:{}, maxOutBatchRows:{}, expectedOutputVectors:{}",
          numRowsPerBatch,
          preferredOutBatchBytes,
          maxOutBatchRows,
          expectedOutputVectors);
    }
  } testSettings[] = {
      {1024, 1, 100, 1024},
      // estimated size per row is ~2092, set preferredOutBatchBytes to 20920,
      // so each batch has 10 rows, so it would return 100 batches
      {1000, 20920, 100, 100},
      // same as above, but maxOutBatchRows is 1, so it would return 1000
      // batches
      {1000, 20920, 1, 1000}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const vector_size_t batchSize = testData.numRowsPerBatch;
    std::vector<RowVectorPtr> rowVectors;
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(11));
    std::vector<VectorPtr> vectors;
    vectors.push_back(c0);
    for (int i = 0; i < 256; ++i) {
      vectors.push_back(c1);
    }
    rowVectors.push_back(makeRowVector(vectors));
    createDuckDbTable(rowVectors);

    core::PlanNodeId orderById;
    auto plan = PlanBuilder()
                    .values(rowVectors)
                    .orderBy({fmt::format("{} ASC NULLS LAST", "c0")}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    queryCtx->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes,
          std::to_string(testData.preferredOutBatchBytes)},
         {core::QueryConfig::kMaxOutputBatchRows,
          std::to_string(testData.maxOutBatchRows)}});
    CursorParameters params;
    params.planNode = plan;
    params.queryCtx = queryCtx;
    auto task = assertQueryOrdered(
        params, "SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST", {0});
    EXPECT_EQ(
        testData.expectedOutputVectors,
        toPlanStats(task->taskStats()).at(orderById).outputVectors);
  }
}
#endif

TEST_F(OrderByTest, allTypesWithIntegerKey) {
  vector_size_t batchSize = 500;
  std::vector<RowVectorPtr> vectors;
  
  // Define the types we'll use
  auto keyType = INTEGER();
  auto boolType = BOOLEAN();
  auto tinyintType = TINYINT();
  auto smallintType = SMALLINT();
  auto bigintType = BIGINT();
  auto realType = REAL();
  auto doubleType = DOUBLE();
  auto hugeIntType = HUGEINT(); // Note: not supported by ArrowSchema
  auto varcharType = VARCHAR();
  auto varbinaryType = VARBINARY(); // Unsupported
  auto timestampType = TIMESTAMP();
  auto arrayType = ARRAY(INTEGER());
  auto rowType = ROW({{"nested1", BOOLEAN()}, {"nested2", INTEGER()}});
  auto shortDecimalType = DECIMAL(10, 2); // Unsupported due to a bug in ArrowSchema in Velox
  auto longDecimalType = DECIMAL(38, 10);
  auto dateType = DATE();
  auto intervalDayTimeType = INTERVAL_DAY_TIME();
  auto intervalYearMonthType = INTERVAL_YEAR_MONTH(); // Unsupported
  // MAP, VARBINARY, UKNOWN, FUNCTION, OPAQUE // Unsupported
  
  for (int32_t i = 0; i < 3; ++i) {
    std::vector<VectorPtr> children;
    std::vector<std::string> names;
    
    // Integer key column for ordering
    auto c0 = makeFlatVector<int32_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(7),
        keyType);
    children.push_back(c0);
    names.push_back("c0");

    // BOOLEAN
    auto c1 = makeFlatVector<bool>(
        batchSize, 
        [](vector_size_t row) { return row % 2 == 0; }, 
        nullEvery(5),
        boolType);
    children.push_back(c1);
    names.push_back("c1");

    // TINYINT
    auto c2 = makeFlatVector<int8_t>(
        batchSize, 
        [](vector_size_t row) { return row % 127; }, 
        nullEvery(9),
        tinyintType);
    children.push_back(c2);
    names.push_back("c2");

    // SMALLINT
    auto c3 = makeFlatVector<int16_t>(
        batchSize, 
        [](vector_size_t row) { return row % 32767; }, 
        nullEvery(11),
        smallintType);
    children.push_back(c3);
    names.push_back("c3");
    
    // BIGINT
    auto c4 = makeFlatVector<int64_t>(
        batchSize, 
        [](vector_size_t row) { return row * 10000; }, 
        nullEvery(13),
        bigintType);
    children.push_back(c4);
    names.push_back("c4");

    // HUGEINT Note:(not supported by ArrowSchema)
    // auto c5 = makeFlatVector<int128_t>(
    //     batchSize, 
    //     [](vector_size_t row) { return row * 1000000000000000000ull; }, 
    //     nullEvery(17),
    //     hugeIntType);
    // children.push_back(c5);

    // REAL (float)
    auto c6 = makeFlatVector<float>(
        batchSize, 
        [](vector_size_t row) { return row * 0.1f; }, 
        nullEvery(15),
        realType);
    children.push_back(c6);
    names.push_back("c6");

    // DOUBLE
    auto c7 = makeFlatVector<double>(
        batchSize, 
        [](vector_size_t row) { return row * 0.01; }, 
        nullEvery(17),
        doubleType);
    children.push_back(c7);
    names.push_back("c7");
    
    // VARCHAR
    auto c8 = makeFlatVector<StringView>(
        batchSize,
        [](vector_size_t row) {
          return StringView::makeInline("str_" + std::to_string(row));
        },
        nullEvery(19),
        varcharType);
    children.push_back(c8);
    names.push_back("c8");

    // VARBINARY
    // auto c9 = makeFlatVector<StringView>(
    //     batchSize,
    //     [](vector_size_t row) {
    //       return StringView::makeInline("bin_" + std::to_string(row));
    //     },
    //     nullEvery(21),
    //     varbinaryType);
    // children.push_back(c9);
    // names.push_back("c9");
    
    // TIMESTAMP
    auto c10 = makeFlatVector<Timestamp>(
        batchSize,
        [](vector_size_t row) {
          // seconds, nanoseconds
          return Timestamp(1600000000 + row, row * 1000);
        },
        nullEvery(23),
        timestampType);
    children.push_back(c10);
    names.push_back("c10");

    // ARRAY(INTEGER())
    auto c11 = makeArrayVector<int32_t>(
        batchSize,
        [](vector_size_t row) { return row % 5 + 1; }, // array sizes
        [](vector_size_t idx) { return idx * 2; },      // array contents
        nullEvery(13));
    children.push_back(c11);
    names.push_back("c11");

    // ROW/STRUCT type
    auto nestedBool = makeFlatVector<bool>(
        batchSize,
        [](vector_size_t row) { return row % 3 == 0; },
        nullEvery(11),
        BOOLEAN());
    auto nestedInt = makeFlatVector<int32_t>(
        batchSize,
        [](vector_size_t row) { return row * 5; },
        nullEvery(13),
        INTEGER());
    auto c12 = makeRowVector(
        {"nested1", "nested2"},
        {nestedBool, nestedInt});
    children.push_back(c12);
    names.push_back("c12");

    // DECIMAL(10, 2) - short decimal
    // auto c13 = makeFlatVector<int64_t>(
    //     batchSize,
    //     [](vector_size_t row) { return row * 123; }, // Will be interpreted as row*1.23
    //     nullEvery(27),
    //     shortDecimalType);
    // children.push_back(c13);
    // names.push_back("c13");

    // DECIMAL(38, 10) - long decimal
    auto c14 = makeFlatVector<int128_t>(
        batchSize,
        [](vector_size_t row) { 
          return static_cast<int128_t>(row) * 1234567890; 
        },
        nullEvery(29),
        longDecimalType);
    children.push_back(c14);
    names.push_back("c14");
    
    // DATE
    auto c15 = makeFlatVector<int32_t>(
        batchSize,
        [](vector_size_t row) { 
          // Days since epoch, starting from 2020-01-01 (18262) plus row offset
          return 18262 + row % 1000; 
        },
        nullEvery(31),
        dateType);
    children.push_back(c15);
    names.push_back("c15");

    // INTERVAL_DAY_TIME - stored as int64_t
    auto c16 = makeFlatVector<int64_t>(
        batchSize,
        [](vector_size_t row) { 
          // Interval of days and milliseconds: row days and row*100 milliseconds
          return row * 86400000 + row * 100; 
        },
        nullEvery(33),
        intervalDayTimeType);
    children.push_back(c16);
    names.push_back("c16");
    
    // INTERVAL_YEAR_MONTH - stored as int32_t
    // auto c17 = makeFlatVector<int32_t>(
    //     batchSize,
    //     [](vector_size_t row) { 
    //       // Interval of months: row months
    //       return row % 120; // 0-120 months (0-10 years)
    //     },
    //     nullEvery(35),
    //     intervalYearMonthType);
    // children.push_back(c17);
    // names.push_back("c17");
    
    // Dictionary vector
    auto flatVector = makeFlatVector<double>(batchSize, [](vector_size_t row) { return row; });
    auto indices = makeIndices(batchSize, [](vector_size_t i) { return i % 100; });
    auto nulls = makeNulls(batchSize, [](vector_size_t row) { return row % 3 == 0; });
    
    auto c18 = BaseVector::wrapInDictionary(nulls, indices, batchSize, flatVector);
    // For comparison, create a dictionary without nulls
    // auto dictWithoutNulls = BaseVector::wrapInDictionary(nullptr, indices, batchSize, flatVector);
    children.push_back(c18);
    names.push_back("c18");

    vectors.push_back(makeRowVector(names, children));
  }
  
  createDuckDbTable(vectors);
  
  // Test ordering by the integer key
  testSingleKey(vectors, "c0");
  
  // Test with a filter
  testSingleKey(vectors, "c0", "c0 % 100 = 0");
  
  // Test descending order
  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                .values(vectors)
                .orderBy({"c0 DESC NULLS LAST"}, false)
                .capturePlanNodeId(orderById)
                .planNode();
  runTest(
      plan, orderById, "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST", {0});
      
  // Test with secondary key
  testTwoKeys(vectors, "c0", "c1");
  
  // Test ordering by boolean column
  testSingleKey(vectors, "c1");
  
  // Test ordering by tinyint column
  testSingleKey(vectors, "c2");
  
  // Test ordering by smallint column
  testSingleKey(vectors, "c3");
  
  // Test ordering by bigint column
  testSingleKey(vectors, "c4");
  
  // Test ordering by date column
  testSingleKey(vectors, "c15");
  
  // Test ordering by interval column
  testSingleKey(vectors, "c16");
  
  // Test ordering by dictionary column
  testSingleKey(vectors, "c18");
  
  // Test multiple ordering directions
  core::PlanNodeId multiOrderById;
  auto multiOrderPlan = PlanBuilder()
                .values(vectors)
                .orderBy({"c0 ASC NULLS FIRST", "c1 DESC NULLS LAST"}, false)
                .capturePlanNodeId(multiOrderById)
                .planNode();
  runTest(
      multiOrderPlan, 
      multiOrderById, 
      "SELECT * FROM tmp ORDER BY c0 ASC NULLS FIRST, c1 DESC NULLS LAST", 
      {0, 1});
      
  // Test with complex filter
  testSingleKey(vectors, "c0", "c1 = true AND c2 < 100");
  
  // Test with three keys
  core::PlanNodeId threeKeysOrderById;
  auto threeKeysPlan = PlanBuilder()
                .values(vectors)
                .orderBy({"c0 ASC", "c1 DESC", "c2 ASC"}, false)
                .capturePlanNodeId(threeKeysOrderById)
                .planNode();
  runTest(
      threeKeysPlan, 
      threeKeysOrderById, 
      "SELECT * FROM tmp ORDER BY c0 ASC, c1 DESC, c2 ASC", 
      {0, 1, 2});
}
} // namespace
