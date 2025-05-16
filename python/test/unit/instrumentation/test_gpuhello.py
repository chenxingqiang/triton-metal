import torch

import pytest
import os

import triton_metal
import triton_metal.language as tl

test_stdout = 'Hello From First Instruction of GPU Kernel: kernel1\ttest_gpuhello.py:17:4\n\
Hello From First Instruction of GPU Kernel: kernel2\ttest_gpuhello.py:23:4\n\
Hello From First Instruction of GPU Kernel: kernel3\ttest_gpuhello.py:29:4\n'


@pytest.mark.parametrize(None, [None])
@triton_metal.jit
def kernel1(BLOCK_SIZE: tl.constexpr):
    return


@pytest.mark.parametrize(None, [None])
@triton_metal.jit
def kernel2(BLOCK_SIZE: tl.constexpr):
    return


@pytest.mark.parametrize(None, [None])
@triton_metal.jit
def kernel3(BLOCK_SIZE: tl.constexpr):
    return


def func(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton_metal.cdiv(n_elements, meta['BLOCK_SIZE']), )
    kernel1[grid](BLOCK_SIZE=1024)
    kernel2[grid](BLOCK_SIZE=1024)
    kernel3[grid](BLOCK_SIZE=1024)


def test_op(capfd, device: str):
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    func(x, y)
    stdout, stderr = capfd.readouterr()
    if 'LLVM_PASS_PLUGIN_PATH' in os.environ:
        assert repr(stderr) == repr(test_stdout)
