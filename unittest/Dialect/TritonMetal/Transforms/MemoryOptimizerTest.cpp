#include "triton/Dialect/TritonMetal/IR/Dialect.h"
#include "triton/Dialect/TritonMetal/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace {

using namespace mlir;
using namespace mlir::triton::metal;

class TritonMetalMemoryOptimizerTest : public ::testing::Test {
public:
  TritonMetalMemoryOptimizerTest() {
    context.loadDialect<TritonMetalDialect>();
    context.allowUnregisteredDialects();
    builder = std::make_unique<OpBuilder>(&context);
  }
  
  // Helper to create a simple module with memory operations
  OwningOpRef<ModuleOp> createTestModule() {
    OwningOpRef<ModuleOp> module = ModuleOp::create(builder->getUnknownLoc());
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(module->getBody());
    
    // Create a function with memory operations
    FunctionType fnType = builder->getFunctionType({}, {});
    auto func = builder->create<FuncOp>(builder->getUnknownLoc(), "test_func", fnType);
    
    auto entry = builder->createBlock(&func.getRegion());
    OpBuilder::InsertionGuard bodyGuard(*builder);
    builder->setInsertionPointToStart(entry);
    
    // Create return
    builder->create<ReturnOp>(builder->getUnknownLoc());
    
    return module;
  }

protected:
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
};

TEST_F(TritonMetalMemoryOptimizerTest, BasicMemoryOptimizationTest) {
  // Create a test module
  auto module = createTestModule();
  
  // Create and run the memory optimization pass
  PassManager pm(&context);
  pm.addPass(createM3MemoryOptimizationPass());
  
  // Test that the pass runs without errors
  ASSERT_TRUE(succeeded(pm.run(module.get())));
}

// Add more specific memory optimization tests here

} // namespace

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 