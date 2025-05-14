#include "triton/Dialect/TritonMetal/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace {

using namespace mlir;
using namespace mlir::triton::metal;

class TritonMetalDialectTest : public ::testing::Test {
public:
  TritonMetalDialectTest() {
    context.loadDialect<TritonMetalDialect>();
    context.allowUnregisteredDialects();
    builder = std::make_unique<OpBuilder>(&context);
  }

protected:
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

TEST_F(TritonMetalDialectTest, CanLoadDialect) {
  // Simple test to verify the Metal dialect can be loaded
  ASSERT_TRUE(context.isDialectRegistered<TritonMetalDialect>());
}

// Add more specific Metal dialect IR tests here

} // namespace

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 