"""
Test script to verify reduction memory optimization with COALESCED layout
"""

import os
import sys

# Add parent directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the metal memory manager
from metal_memory_manager import get_metal_memory_manager, MemoryLayout

# Create a sample reduction operation
reduce_op = {
    "type": "tt.reduce",
    "id": "reduce1",
    "input_shapes": [[1024, 1024]],
    "args": {"axis": 1}
}

# Get the memory manager
memory_manager = get_metal_memory_manager()

# Optimize the reduction operation
print("Optimizing reduction operation...")
optimized_op = memory_manager._optimize_reduction_memory(reduce_op)

# Check if the memory layout was set to COALESCED
if "execution_parameters" in optimized_op:
    memory_layout_value = optimized_op["execution_parameters"].get("memory_layout")
    print(f"Memory layout value set to: {memory_layout_value}")
    
    # Check if it matches COALESCED
    if memory_layout_value == MemoryLayout.COALESCED.value:
        print("SUCCESS: Memory layout is correctly set to COALESCED!")
        print(f"MemoryLayout.COALESCED.value = {MemoryLayout.COALESCED.value}")
    else:
        print(f"ERROR: Memory layout is not COALESCED. Expected {MemoryLayout.COALESCED.value}, got {memory_layout_value}")
else:
    print("ERROR: No execution_parameters found in optimized operation")

# Print all execution parameters for inspection
print("\nAll optimization parameters:")
if "execution_parameters" in optimized_op:
    for key, value in optimized_op["execution_parameters"].items():
        print(f"  {key}: {value}")
else:
    print("  No execution parameters found") 