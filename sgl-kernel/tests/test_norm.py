# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_norm.py

import pytest
import sgl_kernel
import torch


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def gemma_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


def bias_residual_rms_norm_ref(hidden_states, residual, weight, bias, eps):
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    residual = residual.to(torch.float32)
    
    # 1. Add residual and bias
    residual = residual + hidden_states
    if bias is not None:
        residual = residual + bias
    
    # 2. Store residual to hidden_states
    hidden_states = residual.clone()
    
    # 3. RMS Norm
    variance = residual.pow(2).mean(dim=-1, keepdim=True)
    residual = residual * torch.rsqrt(variance + eps)
    
    # 4. Apply weight and store to hidden_states
    hidden_states = (residual * weight.float()).to(orig_dtype)
    residual = residual.to(orig_dtype)
    
    return hidden_states, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [128, 256, 1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False])
def test_bias_residual_rmsnorm(batch_size, hidden_size, dtype, has_bias):
    eps = 1e-6
    
    hidden_states = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    bias = torch.randn(hidden_size, dtype=dtype, device="cuda") if has_bias else None
    
    # Reference implementation
    hidden_states_ref = hidden_states.clone()
    residual_ref = residual.clone()
    hidden_states_native, residual_native = bias_residual_rms_norm_ref(
        hidden_states_ref, residual_ref, weight, bias, eps
    )
    
    # CUDA implementation
    hidden_states_cuda = hidden_states.clone()
    residual_cuda = residual.clone()
    sgl_kernel.bias_residual_rms_norm(residual_cuda, hidden_states_cuda, weight, bias, eps)
    
    # Compare results
    torch.testing.assert_close(hidden_states_cuda, hidden_states_native, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual_cuda, residual_native, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
