#include "triton/Dialect/TritonMetal/IR/Dialect.h"
#include "triton/Dialect/TritonMetal/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace {

using namespace mlir;
using namespace mlir::triton::metal;

class TritonMetalTransformsTest : public ::testing::Test {
public:
  TritonMetalTransformsTest() {
    context.loadDialect<TritonMetalDialect>();
    context.allowUnregisteredDialects();
    builder = std::make_unique<OpBuilder>(&context);
  }

protected:
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

TEST_F(TritonMetalTransformsTest, BasicTransformTest) {
  // This test verifies that we can create a pass manager and register Metal passes
  PassManager pm(&context);
  // Basic verification that registering a pass doesn't crash
  ASSERT_NO_THROW({
    pm.addPass(createM3MemoryOptimizationPass());
  });
}

// Add more specific Metal transform tests here

} // namespace

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 