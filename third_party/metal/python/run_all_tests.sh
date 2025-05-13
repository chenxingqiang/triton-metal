#!/bin/bash
# Run all tests for the special_ops module

# Set up environment variables if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  Testing Special Math Functions    ${NC}"
echo -e "${BLUE}=====================================${NC}"

# Make test results directory
mkdir -p test_results

# Run the main test suite
echo -e "\n${BLUE}Running main test suite...${NC}"
python run_tests.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Main test suite PASSED!${NC}"
else
    echo -e "${RED}Main test suite FAILED!${NC}"
    exit 1
fi

# Run edge case tests
echo -e "\n${BLUE}Running edge case tests...${NC}"
python test_special_ops_edge_cases.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Edge case tests PASSED!${NC}"
else
    echo -e "${RED}Edge case tests FAILED!${NC}"
    exit 1
fi

# Check if we have any test results to summarize
if [ -d "test_results" ]; then
    echo -e "\n${BLUE}Test results and performance data saved to test_results/ directory${NC}"
    ls -la test_results/
fi

echo -e "\n${GREEN}All tests completed successfully!${NC}"
echo -e "${BLUE}=====================================${NC}"

exit 0 