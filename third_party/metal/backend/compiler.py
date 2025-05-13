from triton.backends.compiler import BaseBackend, GPUTarget
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from types import ModuleType
import functools
import tempfile
import os
import pathlib
import sys
import json
import subprocess
import hashlib

@dataclass
class MetalOptions:
    """Metal compilation options"""
    arch: str = "apple-silicon"
    num_warps: int = 4
    num_ctas: int = 1
    enable_fp_fusion: bool = True
    max_shared_memory: int = 65536  # 64KB
    opt_level: int = 3  # Optimization level
    enable_interleaving: bool = True  # Enable instruction interleaving
    vectorize: bool = True  # Enable vectorization
    mlx_shard_size: int = 128  # Default shard size for MLX ops
    debug_info: bool = False  # Include debug info
    
class MetalBackend(BaseBackend):
    """Triton Metal backend implementation"""
    
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'metal'
        
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"
        self._mlx = None
        self._converter = None
        self._driver = None
        
    @property
    def mlx(self):
        """Lazy load MLX"""
        if self._mlx is None:
            try:
                import mlx.core as mx
                self._mlx = mx
            except ImportError:
                raise ImportError("MLX is required for Metal backend. Install it with 'pip install mlx'")
        return self._mlx
        
    @property
    def converter(self):
        """Get Triton to MLX converter"""
        if self._converter is None:
            import sys
            # Add metal package to path if needed
            metal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if metal_path not in sys.path:
                sys.path.append(metal_path)
                
            try:
                from python.triton_to_metal_converter import TritonToMLXConverter
                self._converter = TritonToMLXConverter()
            except ImportError:
                raise ImportError("TritonToMLXConverter not found. Make sure the metal package is installed properly.")
        return self._converter
    
    @property
    def driver(self):
        """Get Metal driver instance"""
        if self._driver is None:
            from .driver import MetalDriver
            self._driver = MetalDriver()
        return self._driver
        
    def hash(self) -> str:
        """Get unique backend identifier"""
        try:
            import mlx.core as mx
            version = mx.__version__
            return f'mlx-{version}-metal'
        except ImportError:
            return 'mlx-unknown-metal'
        
    def parse_options(self, options: dict) -> MetalOptions:
        """Parse compilation options"""
        args = {'arch': self.target.arch if hasattr(self.target, 'arch') else 'apple-silicon'}
        args.update({k: options[k] for k in MetalOptions.__dataclass_fields__.keys() 
                    if k in options and options[k] is not None})
        return MetalOptions(**args)
        
    def add_stages(self, stages, options):
        """Define compilation stages"""
        options = self.parse_options(options)
        
        # Define the compilation pipeline stages
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["mlxir"] = lambda src, metadata: self.make_mlxir(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)
        
    def make_ttir(self, src, metadata, options: MetalOptions):
        """Optimize Triton IR to canonical form"""
        # Apply target-specific optimizations to the IR
        from triton.compiler.compiler import optimize_ir_for_backend
        optimized_src = optimize_ir_for_backend(src, self.target, options)
        
        # Store options in metadata for later stages
        if metadata is not None:
            metadata["opt_level"] = options.opt_level
            metadata["arch"] = options.arch
            
        return optimized_src
        
    def make_ttgir(self, src, metadata, options: MetalOptions):
        """Convert TTIR to TTGIR"""
        # This would typically involve layout planning and optimization
        # For now, we'll make minimal changes and rely on MLX for optimizations
        
        # Include thread/grid dimensions in metadata
        if metadata is not None:
            metadata["num_warps"] = options.num_warps
            metadata["num_ctas"] = options.num_ctas
            metadata["max_shared_memory"] = options.max_shared_memory
            
        return src
        
    def make_mlxir(self, src, metadata, options: MetalOptions):
        """Convert TTGIR to MLX computation graph representation"""
        # Convert Triton IR to MLX computation graph
        try:
            # Import Metal IR transformations for advanced optimizations
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
            try:
                import metal_ir_transforms
                has_ir_transforms = True
            except ImportError:
                has_ir_transforms = False
                print("Warning: metal_ir_transforms module not found. Using basic conversion without Metal-specific optimizations.")
            
            # First, parse the IR to a format that can be transformed
            parsed_ir = self._parse_ir_for_transform(src)
            
            # Apply Metal-specific IR transformations if available
            if has_ir_transforms and parsed_ir:
                # Add compilation options to metadata for the transformations
                transform_metadata = metadata.copy() if metadata else {}
                transform_metadata.update({
                    "num_warps": options.num_warps,
                    "vectorize": options.vectorize,
                    "shard_size": options.mlx_shard_size,
                    "arch": options.arch,
                    "opt_level": options.opt_level,
                    "enable_fp_fusion": options.enable_fp_fusion,
                    "max_shared_memory": options.max_shared_memory,
                    "enable_interleaving": options.enable_interleaving
                })
                
                # Apply transformations
                transformed_ir, transform_summary = metal_ir_transforms.transform_ir(parsed_ir, transform_metadata)
                
                # Store transformation summary in metadata
                if metadata is not None:
                    metadata["transform_summary"] = transform_summary
                
                # Convert transformed IR back to the format expected by the converter
                src = self._serialize_ir_after_transform(transformed_ir)
            
            # Convert to MLX IR using the standard converter
            mlx_ir = self.converter.convert_to_mlx(
                src,
                num_warps=options.num_warps,
                vectorize=options.vectorize,
                shard_size=options.mlx_shard_size
            )
            
            # Store MLX metadata
            if metadata is not None:
                metadata["mlx_version"] = self.mlx.__version__
                metadata["has_custom_ops"] = self.converter.has_custom_ops
                
            return mlx_ir
        except Exception as e:
            import traceback
            error_msg = f"MLX conversion failed: {str(e)}\n{traceback.format_exc()}"
            if options.debug_info:
                print(error_msg)
            raise RuntimeError(error_msg)
    
    def _parse_ir_for_transform(self, src):
        """Parse IR to a format suitable for transformation
        
        Args:
            src: Source IR
            
        Returns:
            Parsed IR operations or None if parsing fails
        """
        try:
            # For now, we'll assume 'src' can be passed to the transformer directly
            # In a real implementation, this would parse the IR format to the 
            # expected list of operation dictionaries
            
            # This is a placeholder implementation that could be expanded based on 
            # the actual IR format received from the previous compilation stage
            if isinstance(src, str):
                # Try to parse as JSON if it's a string
                try:
                    import json
                    return json.loads(src)
                except json.JSONDecodeError:
                    # Not JSON, might be another format
                    pass
            
            # If we can't parse the IR, return None to skip transformations
            return None
        except Exception as e:
            print(f"Warning: Failed to parse IR for transformation: {e}")
            return None
    
    def _serialize_ir_after_transform(self, transformed_ir):
        """Serialize transformed IR back to the format expected by the converter
        
        Args:
            transformed_ir: Transformed IR operations
            
        Returns:
            Serialized IR
        """
        try:
            # For now, we'll assume the converter expects a JSON string
            # In a real implementation, this would convert the transformed IR
            # to the format expected by the converter
            
            # This is a placeholder implementation
            import json
            return json.dumps(transformed_ir)
        except Exception as e:
            print(f"Warning: Failed to serialize transformed IR: {e}")
            # Return the original transformed_ir as a fallback
            return transformed_ir
        
    def make_metallib(self, src, metadata, options: MetalOptions):
        """Generate Metal library from MLX computation graph"""
        try:
            # MLX handles the Metal library generation
            # We'll create a serialized representation of the computation
            
            # Get a temporary directory for Metal library generation
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = pathlib.Path(tmp_dir)
                
                # Use metadata to generate a unique name
                kernel_name = metadata.get("name", "triton_kernel")
                unique_id = hashlib.md5(src.encode('utf-8')).hexdigest()[:8]
                lib_path = tmp_dir / f"{kernel_name}_{unique_id}.metallib"
                
                # Store IR for debugging if needed
                if options.debug_info:
                    ir_path = tmp_dir / f"{kernel_name}_{unique_id}.mlxir"
                    with open(ir_path, "w") as f:
                        f.write(src)
                
                # Compile to Metal via MLX's compilation functions
                # This is simplified - we would need to integrate with MLX's actual compilation
                serialized_graph = self.converter.mlx_ir_to_binary(src)
                
                # In the actual implementation, we would call MLX's Metal compiler
                # For now, we'll simulate the resulting binary
                
                # Save any compilation metadata
                if metadata is not None:
                    metadata["metal_lib_path"] = str(lib_path)
                    metadata["metal_kernel_name"] = kernel_name
                
                return serialized_graph
        except Exception as e:
            import traceback
            error_msg = f"Metal library generation failed: {str(e)}\n{traceback.format_exc()}"
            if options.debug_info:
                print(error_msg)
            raise RuntimeError(error_msg)
        
    def get_module_map(self) -> Dict[str, ModuleType]:
        """Return module mapping"""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            return {
                "mlx": mx,
                "mlx.core": mx,
                "mlx.nn": nn
            }
        except ImportError:
            return {}
        
    def load_dialects(self, ctx):
        """Load dialects"""
        # Load Metal-specific dialects when we have them
        pass
        
    def get_runtime_library(self):
        """Return runtime library with Metal-specific functions"""
        lib_dir = os.path.join(os.path.dirname(__file__), "lib")
        if os.path.exists(lib_dir):
            return lib_dir
        return None
        
    def get_device_properties(self):
        """Get Metal device properties"""
        return self.driver.metal_info 