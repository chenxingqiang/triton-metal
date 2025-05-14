#include "triton/Dialect/TritonMetal/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace {

using namespace mlir;
using namespace mlir::triton::metal;

class TritonMetalMLXIntegrationTest : public ::testing::Test {
public:
  TritonMetalMLXIntegrationTest() {
    context.loadDialect<TritonMetalDialect>();
    context.allowUnregisteredDialects();
    builder = std::make_unique<OpBuilder>(&context);
  }
  
  // Helper to create a simple module for testing MLX integration
  OwningOpRef<ModuleOp> createTestModule() {
    OwningOpRef<ModuleOp> module = ModuleOp::create(builder->getUnknownLoc());
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(module->getBody());
    
    // Create a function
    FunctionType fnType = builder->getFunctionType({}, {});
    auto func = builder->create<FuncOp>(builder->getUnknownLoc(), "test_mlx_integration", fnType);
    
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

TEST_F(TritonMetalMLXIntegrationTest, BasicMLXIntegrationTest) {
  // Create a test module
  auto module = createTestModule();
  
  // Apply the MLX lowering pass
  PassManager pm(&context);
  pm.addPass(createTritonToMLXPass());
  
  // Test that the pass runs without errors
  ASSERT_TRUE(succeeded(pm.run(module.get())));
}

// Add more MLX integration tests here

} // namespace

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 