#!/usr/bin/env python3
"""
Metal后端的简单示例
"""

import os
import sys
import argparse

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Metal后端测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    
    # 检查环境
    print("检查环境...")
    
    # 检查操作系统
    import platform
    print(f"操作系统: {platform.system()} {platform.release()}")
    if platform.system() != "Darwin":
        print("错误: 此测试需要在macOS上运行")
        return 1
        
    # 检查处理器架构
    print(f"处理器架构: {platform.processor()}")
    if platform.processor() != "arm":
        print("警告: 此测试设计用于Apple Silicon芯片，在其他处理器上可能无法正常工作")
        
    # 检查MLX
    try:
        import mlx.core as mx
        print(f"MLX版本: {getattr(mx, '__version__', '未知')}")
        print(f"Metal可用: {mx.metal.is_available()}")
    except ImportError:
        print("错误: 无法导入MLX，请确保已安装: pip install mlx")
        return 1
    
    # 添加必要的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metal_dir = os.path.dirname(current_dir)
    sys.path.insert(0, metal_dir)
    
    # 模拟Triton的backends模块
    import types
    
    class MockBaseBackend:
        pass

    class MockDriverBase:
        pass

    class MockGPUTarget:
        def __init__(self, backend, arch, warp_size):
            self.backend = backend
            self.arch = arch
            self.warp_size = warp_size
            
    # 创建mock模块
    sys.modules['triton'] = types.ModuleType('triton')
    sys.modules['triton.backends'] = types.ModuleType('triton.backends')
    sys.modules['triton.backends.compiler'] = types.ModuleType('triton.backends.compiler')
    sys.modules['triton.backends.driver'] = types.ModuleType('triton.backends.driver')
    sys.modules['triton.backends.compiler'].BaseBackend = MockBaseBackend
    sys.modules['triton.backends.compiler'].GPUTarget = MockGPUTarget
    sys.modules['triton.backends.driver'].DriverBase = MockDriverBase
        
    # 检查Metal后端
    try:
        from backend.driver import MetalDriver
        driver = MetalDriver()
        print("Metal驱动已成功初始化")
        
        target = driver.get_current_target()
        print(f"Metal目标: backend={target.backend}, arch={target.arch}, warp_size={target.warp_size}")
        
    except Exception as e:
        print(f"错误: Metal后端测试失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    print("Metal后端测试成功!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 