#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Reduction using barrier synchronization
kernel void reduce_sum_kernel(
    device const float* src,
    device float* dst,
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]]) {
    
    // Load data into shared memory
    shared_mem[tid] = src[bid * block_size + tid];
    
    // Wait for all threads to load their data
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for(uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        // Wait for all threads to complete the current reduction step
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        dst[bid] = shared_mem[0];
    }
}

// Atomic operations example
kernel void atomic_operations_kernel(
    device const float* src,
    device atomic_float* sum,
    device atomic_int* max_val,
    device atomic_int* min_val,
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]) {
    
    // Atomic add
    atomic_fetch_add_explicit(sum, src[gid], memory_order_relaxed);
    
    // Atomic max
    int val = as_type<int>(src[gid]);
    atomic_fetch_max_explicit(max_val, val, memory_order_relaxed);
    
    // Atomic min
    atomic_fetch_min_explicit(min_val, val, memory_order_relaxed);
}

// SIMD group synchronization example
kernel void simd_group_kernel(
    device const float* src,
    device float* dst,
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
    
    // Each SIMD group loads its data
    shared_mem[tid] = src[tid];
    
    // SIMD group barrier
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each SIMD group performs a reduction
    float sum = shared_mem[tid];
    for (uint i = 0; i < 5; i++) {
        sum += simd_shuffle_down(sum, 1 << i);
    }
    
    if (simd_lane_id == 0) {
        shared_mem[simd_id] = sum;
    }
    
    // Wait for all SIMD groups to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction and output
    if (tid == 0) {
        float final_sum = 0.0f;
        for (uint i = 0; i < 32; i++) {
            final_sum += shared_mem[i];
        }
        dst[0] = final_sum;
    }
} 