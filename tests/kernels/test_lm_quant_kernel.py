"""GPU tests for the LM (q2_lm / q3_lm) Triton store kernel.

The integer schemes (q4_0/q6_0) can assert kernel-vs-oracle bit equality
end-to-end because their scales come from ``amax`` -- an order-independent
reduction. The LM scale is a block RMS: a fp32 SUM, whose rounding depends
on reduction order, so the torch oracle and the Triton kernel can land one
fp8 step apart on rare blocks. The invariant asserted here is therefore:

  * wherever the fp8 scale BYTES agree, the payload bytes must agree exactly
    (code assignment + packing are deterministic given the scale);
  * scale disagreements must be rare (< 2% of blocks);
  * the kernel is deterministic run-to-run (same reduction tree every call).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.quant import Q2_LM, Q3_LM


def _kurtotic_kv(shape=(4, 8, 128), mag=3.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g) * mag
    x[..., : shape[-1] // 16] *= 5.0
    return x.to(torch.bfloat16)


def _store_and_load_invariant(spec):
    """Shared body: store via Triton, compare to oracle under the
    scale-equality invariant, then dequant both and bound the divergence."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    torch.manual_seed(2026_0830)
    k = _kurtotic_kv(shape=(4, 8, 128), mag=3.0).cuda()
    v = _kurtotic_kv(shape=(4, 8, 128), mag=3.0).cuda()
    num_tokens, num_heads, head_dim = k.shape

    slots = num_tokens
    d_physical = spec.physical_head_dim(head_dim)
    n_scales = head_dim // spec.block_size
    kc = torch.zeros(slots, num_heads, d_physical, dtype=torch.uint8, device="cuda")
    vc = torch.zeros(slots, num_heads, d_physical, dtype=torch.uint8, device="cuda")
    ks = torch.zeros(slots, num_heads, n_scales, dtype=spec.scale_dtype, device="cuda")
    vs = torch.zeros(slots, num_heads, n_scales, dtype=spec.scale_dtype, device="cuda")
    indices = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    store_kv_quant(kc, ks, vc, vs, indices, k, v, spec)

    kp, kso = spec.quantize(k)
    vp, vso = spec.quantize(v)

    # Per-block invariant: equal fp8 scale bytes => equal payload bytes.
    pb = spec.payload_bytes_per_block
    nb = d_physical // pb
    for cache, oracle, scale_c, scale_o in (
        (kc, kp, ks, kso),
        (vc, vp, vs, vso),
    ):
        scale_match = (
            scale_c.view(torch.uint8) == scale_o.view(torch.uint8).cuda()
        )  # [slots, heads, n_scales]
        payload_match = cache.view(slots, num_heads, nb, pb) == oracle.view(
            slots, num_heads, nb, pb
        ).cuda()
        # A payload byte may only differ where its block's scale differs.
        bad = payload_match & ~scale_match.unsqueeze(-1)
        assert not bad.any(), (
            f"{spec.name}: payload diverges from oracle in blocks with identical scales"
        )
        mismatch_rate = 1.0 - scale_match.float().mean().item()
        assert mismatch_rate < 0.02, (
            f"{spec.name}: fp8 scale mismatch rate {mismatch_rate:.4f} > 2%"
        )

    # End-to-end divergence guard: dequantize kernel output vs oracle output.
    k_hat = spec.dequantize(kc, ks.float())
    k_ref = spec.dequantize(kp.cuda(), kso.float().cuda())
    diff = (k_hat - k_ref).abs().max().item()
    assert diff < 0.25, f"{spec.name}: store+dequant vs oracle max-abs diff {diff}"


def test_q2_lm_store_kernel_scale_invariant():
    _store_and_load_invariant(Q2_LM)


def test_q3_lm_store_kernel_scale_invariant():
    _store_and_load_invariant(Q3_LM)


def test_q2_lm_store_kernel_deterministic():
    """Same inputs -> same bytes, run to run (the reduction tree is fixed)."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from freetoken.kernel.triton.kv_quant import store_kv_quant

    k = _kurtotic_kv(shape=(4, 8, 128), mag=3.0).cuda()
    v = _kurtotic_kv(shape=(4, 8, 128), mag=3.0).cuda()
    num_tokens, num_heads, head_dim = k.shape
    d_physical = Q2_LM.physical_head_dim(head_dim)
    n_scales = head_dim // Q2_LM.block_size
    indices = torch.arange(num_tokens, device="cuda", dtype=torch.int32)

    runs = []
    for _ in range(2):
        kc = torch.zeros(num_tokens, num_heads, d_physical, dtype=torch.uint8, device="cuda")
        vc = torch.zeros_like(kc)
        ks = torch.zeros(num_tokens, num_heads, n_scales, dtype=Q2_LM.scale_dtype, device="cuda")
        vs = torch.zeros_like(ks)
        store_kv_quant(kc, ks, vc, vs, indices, k, v, Q2_LM)
        runs.append((kc.clone(), ks.clone()))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])


def test_q2_lm_end_to_end_attention_diff():
    """Attention-output divergence guard (pure torch, mirrors the q4/q6
    tests in test_attention_subbyte.py). q2_lm element noise is ~3.5x
    q4_0's, amplified by softmax; a broken unpack path produces ~1.0+."""
    torch.manual_seed(0)
    K = (torch.randn(64, 8, 128) * 3.0).bfloat16()
    V = (torch.randn(64, 8, 128) * 3.0).bfloat16()
    Q = torch.randn(64, 8, 128).to(torch.bfloat16)

    ref = torch.softmax(Q @ K.transpose(-1, -2) / (128 ** 0.5), dim=-1) @ V

    kp, ks = Q2_LM.quantize(K)
    vp, vs = Q2_LM.quantize(V)
    Kq = Q2_LM.dequantize(kp, ks).bfloat16()
    Vq = Q2_LM.dequantize(vp, vs).bfloat16()
    got = torch.softmax(Q @ Kq.transpose(-1, -2) / (128 ** 0.5), dim=-1) @ Vq

    diff = (ref - got).abs().mean().item() / ref.abs().mean().item()
    assert diff < 0.65, f"end-to-end q2_lm attention diff {diff:.4f} > 0.65"


def test_q3_lm_end_to_end_attention_diff():
    torch.manual_seed(0)
    K = (torch.randn(64, 8, 128) * 3.0).bfloat16()
    V = (torch.randn(64, 8, 128) * 3.0).bfloat16()
    Q = torch.randn(64, 8, 128).to(torch.bfloat16)

    ref = torch.softmax(Q @ K.transpose(-1, -2) / (128 ** 0.5), dim=-1) @ V

    kp, ks = Q3_LM.quantize(K)
    vp, vs = Q3_LM.quantize(V)
    Kq = Q3_LM.dequantize(kp, ks).bfloat16()
    Vq = Q3_LM.dequantize(vp, vs).bfloat16()
    got = torch.softmax(Q @ Kq.transpose(-1, -2) / (128 ** 0.5), dim=-1) @ Vq

    diff = (ref - got).abs().mean().item() / ref.abs().mean().item()
    assert diff < 0.40, f"end-to-end q3_lm attention diff {diff:.4f} > 0.40"
