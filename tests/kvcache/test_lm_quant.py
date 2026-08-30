"""Unit tests for the LM (Lloyd-Max) KV cache quantization specs (q2_lm, q3_lm).

Pure-Python + PyTorch tests. No GPU, no Triton. Same contract as
``test_subbyte_quant.py``: the quantize/dequantize methods here are the
**oracle** the Triton store kernel (``kv_quant.py``) and the ``_load_kv``
path (``attention.py``) must match.

LM layouts differ from the integer schemes in one structural way: the
per-block scale is the block **RMS** (the codebooks are trained on
RMS-normalized data), stored as fp8 E4M3. A fp32 sum is reduction-order
dependent, so kernel-vs-oracle bit-exactness is asserted at the
scale-equality level in ``tests/kernels/test_lm_quant_kernel.py`` instead
of end-to-end.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.quant import (
    LAYOUT_LM2,
    LAYOUT_LM3,
    NONE,
    Q2_LM,
    Q3_LM,
    Q4_0,
    _lm_levels,
    _lm_thresholds,
    resolve_kv_quant,
)


# ---- helpers ----

def _kurtotic_kv(shape=(4, 8, 128), mag=3.0, seed=0):
    """Real-data-shaped K/V (see test_subbyte_quant.py for the rationale)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g) * mag
    x[..., : shape[-1] // 16] *= 5.0
    return x.to(torch.bfloat16)


def _rel_l2(rec: torch.Tensor, x: torch.Tensor) -> float:
    return ((rec - x.float()).norm() / x.float().norm()).item()


# ---- spec constants ----

def test_q2_lm_spec_constants():
    """q2_lm: 2-bit codes, 8 payload bytes per 32-value block, fp8 scale."""
    assert Q2_LM.layout == LAYOUT_LM2
    assert Q2_LM.bits == 2
    assert Q2_LM.payload_bytes_per_block == 8
    assert Q2_LM.block_size == 32
    assert Q2_LM.scale_dtype is torch.float8_e4m3fn


def test_q3_lm_spec_constants():
    """q3_lm: 3-bit codes, 64-value blocks, 24 payload bytes, fp8 scale."""
    assert Q3_LM.layout == LAYOUT_LM3
    assert Q3_LM.bits == 3
    assert Q3_LM.payload_bytes_per_block == 24
    assert Q3_LM.block_size == 64
    assert Q3_LM.scale_dtype is torch.float8_e4m3fn


def test_lm_codebooks_sorted_and_symmetric():
    """The codebook IS the code index: ascending order is load-bearing
    (threshold assignment and the kernel's level ladders both assume it).
    Symmetry around zero keeps RMS-normalized blocks centered."""
    for layout in (LAYOUT_LM2, LAYOUT_LM3):
        levels = _lm_levels(layout)
        assert levels == tuple(sorted(levels))
        assert levels == tuple(sorted(levels, reverse=True))[::-1] or all(
            abs(a + b) < 1e-6 for a, b in zip(levels, levels[::-1])
        )
        assert len(levels) == 2 ** (2 if layout == LAYOUT_LM2 else 3)


def test_lm_thresholds_are_midpoints():
    """Thresholds sit at the exact midpoints of adjacent levels, so a value
    equidistant between two levels lands on the LOWER code (matching
    torch.argmin's first-occurrence tie-break)."""
    levels = _lm_levels(LAYOUT_LM2)
    thresholds = _lm_thresholds(LAYOUT_LM2)
    assert len(thresholds) == len(levels) - 1
    for t, (a, b) in zip(thresholds, zip(levels, levels[1:])):
        assert t == (a + b) / 2


# ---- density ----

def test_q2_lm_bytes_per_element():
    """q2_lm must be EXACTLY half of q4_0: (8 payload + 1 fp8 scale) / 32."""
    assert Q2_LM.bytes_per_element(torch.bfloat16) == pytest.approx(0.28125)
    assert Q4_0.bytes_per_element(torch.bfloat16) == pytest.approx(0.5625)
    assert Q2_LM.bytes_per_element(torch.bfloat16) == pytest.approx(
        Q4_0.bytes_per_element(torch.bfloat16) / 2
    )


def test_q3_lm_bytes_per_element():
    assert Q3_LM.bytes_per_element(torch.bfloat16) == pytest.approx(0.390625)


def test_q2_lm_physical_head_dim():
    assert Q2_LM.physical_head_dim(128) == 32
    assert Q2_LM.physical_head_dim(256) == 64


def test_q3_lm_physical_head_dim():
    assert Q3_LM.physical_head_dim(128) == 48
    assert Q3_LM.physical_head_dim(256) == 96


def test_q2_lm_scale_shape():
    """physical 32 -> logical 128 -> 128 / 32 = 4 scales along the last dim."""
    assert Q2_LM.scale_shape((3, 4, 32)) == (3, 4, 4)


def test_q3_lm_scale_shape():
    """64-value blocks: physical 96 -> logical 256 -> 256 / 64 = 4 scales."""
    assert Q3_LM.scale_shape((3, 4, 96)) == (3, 4, 4)


# ---- round-trip (oracle) ----

def test_q2_lm_roundtrip_oracle_kurtotic():
    """q2_lm floor on kurtotic K/V: measured 0.445-0.453 rel L2 across
    seeds (~3.7x q4_0's 0.12 on this fixture). This fixture puts the
    outliers INSIDE each 32-element block (worst case for a per-token
    RMS scale); the planned per-channel K path is what removes that
    penalty. Broken layouts give ~1.0+."""
    x = _kurtotic_kv(shape=(4, 8, 128), mag=3.0)
    payload, scales = Q2_LM.quantize(x)
    rec = Q2_LM.dequantize(payload, scales)
    err = _rel_l2(rec, x)
    assert err < 0.50, f"q2_lm rel_err {err:.4f} exceeded 0.50 floor"


def test_q3_lm_roundtrip_oracle_kurtotic():
    """q3_lm floor on the same fixture: measured 0.350-0.362. On this
    within-block-outlier distribution q3 and q2 converge (clipped
    outliers vs coarser inner level trade off); the Gaussian gap
    (0.18 vs 0.33) only shows up on tame blocks."""
    x = _kurtotic_kv(shape=(4, 8, 128), mag=3.0)
    payload, scales = Q3_LM.quantize(x)
    rec = Q3_LM.dequantize(payload, scales)
    err = _rel_l2(rec, x)
    assert err < 0.40, f"q3_lm rel_err {err:.4f} exceeded 0.40 floor"


def test_q2_lm_payload_shape():
    x = torch.randn(2, 4, 256, dtype=torch.bfloat16)
    payload, scales = Q2_LM.quantize(x)
    assert payload.shape == (2, 4, 64)   # 256 * 2 / 8
    assert scales.shape == (2, 4, 8)     # 256 / 32


def test_q3_lm_payload_shape():
    x = torch.randn(2, 4, 256, dtype=torch.bfloat16)
    payload, scales = Q3_LM.quantize(x)
    assert payload.shape == (2, 4, 96)   # 256 * 3 / 8
    assert scales.shape == (2, 4, 4)     # 256 / 64


# ---- byte layout (hand-verified, mirrors the kernel's read order) ----

def test_q2_lm_zero_block_layout():
    """q=0 -> code = #{thresholds < 0} = 1 (the -0.4516 level). Every byte
    packs four 2-bit fields of 01 = 0b01010101 = 0x55."""
    x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
    payload, _ = Q2_LM.quantize(x)
    assert (payload[0, 0] == 0x55).all()
    rec = Q2_LM.dequantize(payload, Q2_LM.quantize(x)[1])
    assert torch.allclose(rec.float(), torch.full((1, 1, 32), -0.4516), atol=1e-3)


def test_q2_lm_single_spike_layout():
    """31 values ~0, one large value: rms ~ V/sqrt(32), so the spike maps to
    q ~ sqrt(32) = 5.66 -> code 3, the zeros to code 1. Byte 0 holds
    v0*1 + v1*4 + v2*16 + v3*64 = 3 + 1*4 + 1*16 + 1*64 = 0x57."""
    x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
    x[0, 0, 0] = 10.0
    payload, _ = Q2_LM.quantize(x)
    assert payload[0, 0, 0].item() == 0x57
    assert all(payload[0, 0, j].item() == 0x55 for j in range(1, 8))


def test_q3_lm_zero_block_layout():
    """q=0 -> code 3 (the -0.2450 level): lo2 = 0b11 fills the low plane
    (0xFF), hi1 = 0 leaves the high plane zeroed."""
    x = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    payload, _ = Q3_LM.quantize(x)
    assert (payload[0, 0, :16] == 0xFF).all()
    assert (payload[0, 0, 16:] == 0).all()
    rec = Q3_LM.dequantize(payload, Q3_LM.quantize(x)[1])
    assert torch.allclose(rec.float(), torch.full((1, 1, 64), -0.2450), atol=1e-3)


def test_q3_lm_single_spike_high_plane():
    """The spike maps to code 7 (lo2=3, hi1=1); the zeros to code 3
    (lo2=3, hi1=0). So the low plane is all 0xFF and the high plane has
    exactly one bit set at byte 0, position 0."""
    x = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    x[0, 0, 0] = 10.0
    payload, _ = Q3_LM.quantize(x)
    assert (payload[0, 0, :16] == 0xFF).all()
    assert payload[0, 0, 16].item() == 0x01
    assert all(payload[0, 0, j].item() == 0 for j in range(17, 24))


def test_q2_lm_constant_block_lands_between_levels():
    """A constant block has q = v/rms = 1.0, which sits between the inner
    (0.4516) and outer (1.5096) levels. This is INHERENT to RMS-normalized
    codebook quantization, not a bug -- document it so nobody 'fixes' it.
    The reconstruction error for this degenerate case is ~0.34 of the value."""
    x = torch.full((1, 1, 32), 3.0, dtype=torch.bfloat16)
    payload, scales = Q2_LM.quantize(x)
    rec = Q2_LM.dequantize(payload, scales)
    err = _rel_l2(rec, x)
    assert 0.2 < err < 0.6


# ---- resolve ----

def test_resolve_q2_lm():
    assert resolve_kv_quant("q2_lm") is Q2_LM


def test_resolve_q3_lm():
    assert resolve_kv_quant("q3_lm") is Q3_LM


def test_resolve_unknown_still_raises():
    with pytest.raises(ValueError, match="unknown --kv-cache-dtype"):
        resolve_kv_quant("q2_0")  # GGUF-style name must NOT resolve


# ---- CPU/CUDA parity ----

def test_lm_cpu_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    for spec in (Q2_LM, Q3_LM):
        x_cpu = _kurtotic_kv(shape=(2, 4, 128), mag=3.0)
        x_cuda = x_cpu.cuda()
        p_cpu, s_cpu = spec.quantize(x_cpu)
        p_cuda, s_cuda = spec.quantize(x_cuda)
        assert torch.equal(p_cpu, p_cuda.cpu())
        assert torch.equal(s_cpu, s_cuda.cpu())


# ---- LM is not an integer scheme ----

def test_lm_is_not_integer():
    """The store kernel keys rounding on IS_INT; LM assigns nearest codebook
    entries and must NOT round."""
    assert not Q2_LM.is_integer
    assert not Q3_LM.is_integer


def test_none_spec_untouched():
    assert resolve_kv_quant("auto") is NONE
    assert not NONE.enabled
