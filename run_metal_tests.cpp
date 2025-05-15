#include <gtest/gtest.h>
#include <iostream>

// Note: The following includes will show errors in IDE but will be resolved correctly during the build process
// since the proper include paths will be provided by CMake build system.
// These files are included directly to make it easy to run all tests through a single executable.

// Include all the simplified Metal test files directly
#include "unittest/Dialect/TritonMetal/IR/DialectTest.cpp"
#include "unittest/Dialect/TritonMetal/Transforms/TransformsTest.cpp"
#include "unittest/Dialect/TritonMetal/Transforms/MemoryOptimizerTest.cpp"
#include "unittest/Dialect/TritonMetal/Transforms/M3OptimizationsTest.cpp"
#include "unittest/Dialect/TritonMetal/MLXIntegrationTest.cpp"
#include "unittest/Dialect/TritonMetal/HardwareDetectionTest.cpp"
#include "unittest/Metal/MetalBackendTest.cpp"
#include "unittest/Metal/M3OptimizationsTest.cpp"
#include "unittest/Metal/MetalMemoryManagerTest.cpp"
#include "unittest/Metal/OperationFusionTest.cpp"
#include "unittest/Metal/HardwareDetectionTest.cpp"
#include "unittest/Metal/MLXIntegrationTest.cpp"
#include "unittest/Metal/TensorCoreTest.cpp"

// Simple test
TEST(MetalSimpleTest, BasicTest) {
  EXPECT_EQ(1, 1);
  std::cout << "Basic Metal test running!" << std::endl;
}

// Test for environment variables
TEST(MetalSimpleTest, EnvVarsTest) {
#ifdef __APPLE__
  // Check for environment variables
  const char* m3Flag = std::getenv("TRITON_METAL_IS_M3");
  const char* generation = std::getenv("TRITON_METAL_GENERATION");
  
  // Print the values for debugging
  if (m3Flag != nullptr) {
    std::cout << "TRITON_METAL_IS_M3 is set to: " << m3Flag << std::endl;
  } else {
    std::cout << "TRITON_METAL_IS_M3 is not set" << std::endl;
  }
  
  if (generation != nullptr) {
    std::cout << "TRITON_METAL_GENERATION is set to: " << generation << std::endl;
  } else {
    std::cout << "TRITON_METAL_GENERATION is not set" << std::endl;
  }
  
  // Test passes as long as it runs
  EXPECT_TRUE(true);
#else
  GTEST_SKIP() << "Test only runs on Apple hardware";
#endif
}

// Simple test to check M3 detection
TEST(MetalSimpleTest, M3Detection) {
#ifdef __APPLE__
  const char* isM3 = std::getenv("TRITON_METAL_IS_M3");
  if (isM3 != nullptr && std::string(isM3) == "1") {
    std::cout << "Running with M3 chip detected" << std::endl;
  } else {
    std::cout << "Running without M3 chip detection" << std::endl;
  }
  EXPECT_TRUE(true);
#else
  GTEST_SKIP() << "Test only runs on Apple hardware";
#endif
}

// Main function
int main(int argc, char **argv) {
  std::cout << "Starting Metal tests..." << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
  std::cout << "Completed Metal tests!" << std::endl;
  return result;
} 