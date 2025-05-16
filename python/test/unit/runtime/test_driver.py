import sys
from concurrent.futures import ThreadPoolExecutor
import torch

import triton_metal
import triton_metal.language as tl


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton_metal.runtime.driver"])
    reload(sys.modules["triton_metal.runtime"])
    mod = sys.modules[triton_metal.runtime.driver.__module__]
    assert isinstance(triton_metal.runtime.driver.active, getattr(mod, "LazyProxy"))
    assert triton_metal.runtime.driver.active._obj is None
    utils = triton_metal.runtime.driver.active.utils  # noqa: F841
    assert issubclass(triton_metal.runtime.driver.active._obj.__class__, getattr(triton_metal.backends.driver, "DriverBase"))


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton_metal.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton_metal.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()
