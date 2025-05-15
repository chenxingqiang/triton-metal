def check_metal_hardware() -> bool:
    """Check Metal hardware availability"""
    # Try to import PyMetal or MLX to check Metal support
    try:
        import mlx.core as mx
        
        # Check if running on Apple Silicon using MLX
        # MLX only runs on Metal, so if we can create an array and execute an operation,
        # it means Metal hardware is available
        try:
            # Create a small array and perform a simple operation
            a = mx.array([1, 2, 3])
            b = a + 1
            # Force execution
            b.item()
            print("✅ Metal hardware is available")
            return True
        except Exception as e:
            print(f"❌ Metal hardware execution failed: {e}")
            return False
    except ImportError:
        print("⚠️ Could not check Metal hardware (MLX not installed)")
        return False
    except Exception as e:
        print(f"❌ Error checking Metal hardware: {e}")
        return False 