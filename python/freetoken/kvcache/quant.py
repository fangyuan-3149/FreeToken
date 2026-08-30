"""KV-cache quantization schemes.

Two layout families live here:

  * 8-bit: one int8 (or fp8) value per element + one fp16 scale per :data:`BLOCK` elements
    along ``head_dim``. The 8-bit value uses the full byte; the scale is amortized into a
    fixed 1.0625 bytes/element cost. `Q8_0` and `FP8_E4M3` are the two variants.

  * Sub-byte: each element takes only ``BITS`` of a byte, packed into a per-block payload
    of ``payload_bytes_per_block`` bytes. One fp16 scale per :data:`BLOCK` elements
    (same as 8-bit) is stored alongside. `Q4_0` (4 bits/element, 16 bytes/32 elements)
    and `Q6_0` (6 bits/element, 24 bytes/32 elements -- 16 low + 8 high planes) are the
    GGUF-style variants. ``bytes_per_element`` is ``payload_bytes_per_block / BLOCK +
    scale_bytes / BLOCK`` = 0.5625 and 0.8125 respectively.

  * LM codebook (``lm2`` / ``lm3``): sub-byte code INDICES into a frozen Lloyd-Max
    codebook, with one fp8 E4M3 scale per block. The scale is the block **RMS**, not
    amax/MAX_MAG -- the codebooks are trained on RMS-normalized data, and RMS's low
    block-to-block variance is what keeps the shared codebook shape from being
    stretched. `Q2_LM` (2-bit codes, 8 bytes/32 values + 1 fp8 scale = 0.28125
    B/elem, half of q4_0) and `Q3_LM` (3-bit codes, 64-value blocks, 24 bytes + 1
    fp8 scale = 0.390625 B/elem) sit on the accuracy-per-byte convex hull below
    q4_0. See ``_LM2_LEVELS`` / ``_LM3_LEVELS`` for the frozen codebooks.

Storage layout (last axis = head_dim) changes per scheme:

  * bf16 / fp8 / int8: ``head_dim`` slots, each one byte/element dtype
  * q4_0: ``head_dim // 2`` slots of uint8 (two 4-bit values per byte, low nibble = even
    element, high nibble = odd element)
  * q6_0: ``head_dim * 3 // 4`` slots of uint8 (24 bytes pack 32 6-bit values, split as a
    16-byte low plane holding the low 4 bits of every value plus an 8-byte high plane
    holding the top 2 bits of every value, byte g in the high plane serving values 4g..4g+3
    at bit positions 0, 2, 4, 6)
  * q2_lm: ``head_dim // 4`` slots of uint8 (8 bytes pack 32 2-bit code indices, byte g
    serving values 4g..4g+3 at bit positions 0, 2, 4, 6)
  * q3_lm: ``head_dim * 3 // 8`` slots of uint8 (24 bytes pack 64 3-bit code indices,
    split as a 16-byte low plane of 2-bit fields -- byte g serving values 4g..4g+3 at bit
    positions 0, 2, 4, 6 -- plus an 8-byte high plane holding the top bit of every value,
    byte g serving values 8g..8g+7 at bit positions 0..7)

The KV pool allocates the buffer in uint8 (so the element size is 1 regardless of scheme)
with the scheme's packed last-dim. The attention kernel is told the logical ``head_dim``
AND the physical last-dim (``D_PHYSICAL``) and unpacks inside the load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch

# Elements per scale, along head_dim. Matches GGUF Q8_0/Q4_0/Q6_0's block.
BLOCK = 32
# One fp16 scale per block.
SCALE_DTYPE = torch.float16
# All quantized schemes use uint8 storage at the byte level (sub-byte just packs
# multiple values per byte). Pools allocate with this dtype.
STORAGE_BYTE_DTYPE = torch.uint8

# Layout identifiers. Add a new one here to plug in a new scheme.
LAYOUT_Q8 = "q8"        # 1 byte per element (Q8_0 / FP8_E4M3)
LAYOUT_Q2 = "q2"        # 8 bytes pack 32 2-bit values (4 per byte at bits 0/2/4/6)
LAYOUT_Q4 = "q4"        # 16 bytes pack 32 4-bit values
LAYOUT_NVFP4 = "nvfp4"  # 8 bytes pack 16 4-bit E2M1 float values + 1 fp8 E4M3 block scale
LAYOUT_Q6 = "q6"        # 24 bytes pack 32 6-bit values (16 lo + 8 hi)
LAYOUT_LM2 = "lm2"      # 8 bytes pack 32 2-bit Lloyd-Max code indices + 1 fp8 E4M3 scale
LAYOUT_LM3 = "lm3"      # 24 bytes pack 64 3-bit Lloyd-Max code indices + 1 fp8 E4M3 scale

# Lloyd-Max codebooks for the ``lm2`` / ``lm3`` layouts, trained by k-means
# (Lloyd iteration) on 4M standard-Gaussian samples normalized by their
# per-32-element-block RMS -- the same normalization the store kernel applies
# at write time. Symmetrized around zero; sorted ascending so the code index
# is also the sort order. For Gaussian-shaped blocks these are the MSE-optimal
# 4/8-level quantizers (they reproduce the closed-form Lloyd-Max values
# +-0.4528/+-1.5104 and +-0.2450/+-0.7561/+-1.3441/+-2.1508 to 4 decimals).
# On channel-outlier K data the outer levels migrate toward the tail when the
# codebook is retrained per-channel -- see the LM quant survey notes; the
# frozen Gaussian values here are the no-calibration default.
_LM2_LEVELS = (-1.5096, -0.4516, 0.4516, 1.5096)
_LM3_LEVELS = (
    -2.1508, -1.3441, -0.7561, -0.2450,
    0.2450, 0.7561, 1.3441, 2.1508,
)


def _lm_levels(layout: str) -> Tuple[float, ...]:
    """Ascending codebook for an LM layout. The code index IS the position."""
    if layout == LAYOUT_LM2:
        return _LM2_LEVELS
    if layout == LAYOUT_LM3:
        return _LM3_LEVELS
    raise ValueError(f"not an LM layout: {layout!r}")


def _lm_thresholds(layout: str) -> Tuple[float, ...]:
    """Decision thresholds between adjacent levels: (c_i + c_{i+1}) / 2.

    A value maps to code = #{thresholds < value}, so an exact tie between two
    levels lands on the lower code -- matching torch.argmin's first-occurrence
    tie-break. Stored as fp32 (both the torch oracle and the Triton kernel
    compare against the same fp32 constants, keeping them bit-identical)."""
    levels = _lm_levels(layout)
    return tuple((a + b) / 2 for a, b in zip(levels, levels[1:]))


@dataclass(frozen=True)
class KVQuantSpec:
    """How a KV pool stores its K/V elements.

    ``name`` is the ``--kv-cache-dtype`` value. ``storage_dtype`` is the underlying
    byte dtype (always ``uint8`` once we go sub-byte; 8-bit schemes may use int8/float8
    and let the user-side interpretation matter for the actual bits). ``layout`` is one
    of :data:`LAYOUT_Q8`, :data:`LAYOUT_Q2`, :data:`LAYOUT_Q4`, :data:`LAYOUT_NVFP4`,
    :data:`LAYOUT_Q6`. ``max_magnitude`` is the largest absolute value a quantized
    element can take (127 / 1 / 7 / 6 / 31 for q8_0 / q2_0 / q4_0 / nvfp4 / q6_0).

    For sub-byte schemes, ``bits`` is the number of bits per element (2 / 4 / 4 / 6
    for q2_0 / q4_0 / nvfp4 / q6_0) and ``payload_bytes_per_block`` is the number of
    payload bytes that hold one ``block_size``-element block. For 8-bit schemes these
    are set automatically from ``max_magnitude``.

    ``block_size`` defaults to ``BLOCK`` (32, used by q2_0 / q4_0 / q6_0); nvfp4 uses
    16-element blocks. ``scale_dtype`` defaults to fp16; nvfp4 uses fp8 E4M3 to
    halve the per-block scale storage.
    """

    name: str
    storage_dtype: torch.dtype | None
    max_magnitude: float
    layout: str = LAYOUT_Q8
    bits: int = 8
    payload_bytes_per_block: int = BLOCK
    block_size: int = BLOCK
    scale_dtype: torch.dtype = SCALE_DTYPE

    @property
    def enabled(self) -> bool:
        return self.storage_dtype is not None

    @property
    def is_integer(self) -> bool:
        """Integer schemes round; float ones divide or table-lookup. Q2/Q4/Q6
        are integer; NVFP4 is float (E2M1); LM layouts assign the nearest
        codebook entry (no rounding)."""
        if self.layout in (LAYOUT_NVFP4, LAYOUT_LM2, LAYOUT_LM3):
            return False
        if self.layout != LAYOUT_Q8:
            return True
        return self.storage_dtype == torch.int8

    def bytes_per_element(self, compute_dtype: torch.dtype) -> float:
        """Storage bytes per K/V element, scales amortized over the block.

        Unquantized: the compute dtype's itemsize. 8-bit quantized: 1 byte + 2/32 for
        the fp16 scale = 1.0625. Sub-byte: ``payload_bytes_per_block / block_size +
        scale.itemsize / block_size``.
        """
        if not self.enabled:
            return float(compute_dtype.itemsize)
        return self.payload_bytes_per_block / self.block_size + self.scale_dtype.itemsize / self.block_size

    def physical_head_dim(self, head_dim: int) -> int:
        """Number of bytes in the buffer's last (head_dim) axis under this scheme.

        Equal to ``head_dim`` for 8-bit (each element = 1 byte). Smaller for sub-byte:
        ``head_dim * bits / 8``. Rounds down; the caller is responsible for ensuring
        ``head_dim * bits`` is a multiple of 8.
        """
        if self.layout == LAYOUT_Q8:
            return head_dim
        return head_dim * self.bits // 8

    def scale_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Scale-tensor shape for a KV buffer shape: last dim divided by the block.

        For sub-byte schemes, the buffer's last dim is the *packed* byte count, so the
        scale shape is derived from the *logical* head dim -- which the caller passes
        via the buffer shape's last dim. We rely on the buffer's last dim being
        ``physical_head_dim(logical_head_dim)``; recover logical by multiplying by
        ``8 // bits``.
        """
        if shape[-1] % self.payload_bytes_per_block:
            raise ValueError(
                f"physical last-dim {shape[-1]} is not a multiple of the NVFP4 block payload"
                if self.layout == LAYOUT_NVFP4
                else f"physical head_dim {shape[-1]} is not a multiple of the KV quant block {BLOCK}"
            )
        if self.layout == LAYOUT_Q8:
            return (*shape[:-1], shape[-1] // BLOCK)
        # Sub-byte: logical = physical * 8 / bits; scale extent = logical / block_size.
        logical = shape[-1] * 8 // self.bits
        return (*shape[:-1], logical // self.block_size)

    # ---- reference implementations (correctness oracle for the Triton kernels) ----

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """``x[..., D]`` (float) -> ``(payload[...], scales[..., D // BLOCK])``.

        For 8-bit schemes, payload has the same last-dim as input. For sub-byte
        schemes, payload's last-dim is ``physical_head_dim(D)`` (packed).
        """
        assert self.enabled, "quantize() on an unquantized spec"
        if self.layout == LAYOUT_Q8:
            return self._quantize_8bit(x)
        if self.layout == LAYOUT_NVFP4:
            return self._quantize_nvfp4(x)
        if self.layout == LAYOUT_LM2:
            return self._quantize_lm(x, bits=2)
        if self.layout == LAYOUT_LM3:
            return self._quantize_lm(x, bits=3)
        if self.layout == LAYOUT_Q4:
            return self._quantize_subbyte(x, bits=4)
        if self.layout == LAYOUT_Q6:
            return self._quantize_subbyte(x, bits=6)
        raise ValueError(f"unknown layout {self.layout!r}")

    def dequantize(self, *payload_scales: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`quantize`, in float32. Accepts ``(payload, scales)`` for
        8-bit and ``(payload, scales)`` for sub-byte (single packed payload)."""
        assert self.enabled, "dequantize() on an unquantized spec"
        if self.layout == LAYOUT_Q8:
            payload, scales = payload_scales
            return self._dequantize_8bit(payload, scales)
        if self.layout == LAYOUT_NVFP4:
            payload, scales = payload_scales
            return self._dequantize_nvfp4(payload, scales)
        if self.layout == LAYOUT_LM2:
            payload, scales = payload_scales
            return self._dequantize_lm(payload, scales, bits=2)
        if self.layout == LAYOUT_LM3:
            payload, scales = payload_scales
            return self._dequantize_lm(payload, scales, bits=3)
        if self.layout == LAYOUT_Q4:
            payload, scales = payload_scales
            return self._dequantize_subbyte(payload, scales, bits=4)
        if self.layout == LAYOUT_Q6:
            payload, scales = payload_scales
            return self._dequantize_subbyte(payload, scales, bits=6)
        raise ValueError(f"unknown layout {self.layout!r}")

    # ---- 8-bit ----

    def _quantize_8bit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        blocks = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
        amax = blocks.abs().amax(dim=-1)
        scales = torch.where(amax > 0, amax / self.max_magnitude, torch.ones_like(amax))
        scales = scales.to(SCALE_DTYPE)
        q = blocks / scales.float().unsqueeze(-1)
        if self.is_integer:
            q = torch.where(q >= 0, (q + 0.5).floor(), (q - 0.5).ceil())
            q = q.clamp_(-self.max_magnitude, self.max_magnitude)
        return (q.flatten(-2).to(self.storage_dtype), scales)

    def _dequantize_8bit(self, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        blocks = q.float().unflatten(-1, (q.shape[-1] // BLOCK, BLOCK))
        return (blocks * scales.float().unsqueeze(-1)).flatten(-2)

    # ---- NVFP4: E2M1 4-bit float, 16-value blocks, fp8 E4M3 block scale ----

    # E2M1 4-bit float code table. codes 0..7 are non-negative; 8..15 are the same
    # magnitudes with the sign bit set. Per NVIDIA OCP NVFP4 spec.
    _E2M1_VALUES = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    def _quantize_nvfp4(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-block NVFP4 quantization: 16 values -> 8 payload bytes (4-bit
        E2M1 codes, 2 per byte same nibble layout as Q4) + 1 fp8 E4M3 scale.
        Symmetric around 0; max representable magnitude is 6, so scale = amax / 6
        and values in (-6, +6) are coded exactly, beyond that clip."""
        assert x.shape[-1] % self.block_size == 0, (
            f"head_dim {x.shape[-1]} is not a multiple of the NVFP4 block {self.block_size}"
        )
        nb = x.shape[-1] // self.block_size
        tail_shape = list(x.shape[:-1]) + [nb, self.block_size]
        blocks = x.float().reshape(tail_shape)
        amax = blocks.abs().amax(dim=-1)
        scales = torch.where(amax > 0, amax / self.max_magnitude, torch.ones_like(amax))
        # Round scale to fp8 E4M3 storage precision before using it, so the value
        # the attention kernel reads back is scaled by the identical number.
        scales_fp8 = scales.to(self.scale_dtype).float()
        qf = blocks / scales_fp8.unsqueeze(-1)
        # Find the nearest E2M1 code per value (16 codes; argmin |x - table|).
        e2m1 = torch.tensor(self._E2M1_VALUES, dtype=qf.dtype, device=qf.device)
        diffs = (qf.unsqueeze(-1) - e2m1).abs()
        codes = diffs.argmin(dim=-1).to(torch.int32)  # [..., NB, 16] in [0, 15]
        # Pack two 4-bit codes per byte (low/even, high/odd -- same layout as q4_0).
        lo = codes[..., 0::2]  # [..., NB, 8]
        hi = codes[..., 1::2]
        payload = (lo | (hi << 4)).to(torch.uint8).reshape(*x.shape[:-1], nb * 8)
        return (payload, scales.to(self.scale_dtype))

    def _dequantize_nvfp4(self, payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_quantize_nvfp4`. Returns float32 with logical head_dim."""
        # payload shape: [..., nb * payload_bytes_per_block] = [..., nb * 8] for NVFP4
        nb = payload.shape[-1] // self.payload_bytes_per_block
        # Per-block layout: 8 bytes, 2 nibbles per byte, 1 value per nibble.
        bytes_view = payload.unflatten(-1, (nb, self.payload_bytes_per_block))
        lo = (bytes_view & 0xF).to(torch.int64)
        hi = ((bytes_view >> 4) & 0xF).to(torch.int64)
        # Interleave: lo[0], hi[0], lo[1], hi[1], ...
        stacked = torch.stack([lo, hi], dim=-1)  # [..., NB, 8, 2]
        stacked = stacked.flatten(-2)  # [..., NB, 16]
        # Look up the E2M1 value table. 16 entries indexed by unsigned 4-bit code.
        e2m1 = torch.tensor(self._E2M1_VALUES, dtype=torch.float32, device=payload.device)
        vals = e2m1[stacked]  # [..., NB, 16]
        return (vals * scales.float().unsqueeze(-1)).flatten(-2)

    # ---- LM codebook (lm2 / lm3): 2/3-bit Lloyd-Max indices + fp8 E4M3 RMS scale ----

    def _quantize_lm(self, x: torch.Tensor, *, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-block Lloyd-Max codebook quantization.

        The scale is the block's RMS (NOT amax/MAX_MAG like the integer
        schemes): the codebook is trained on RMS-normalized data, so RMS is
        the matching per-block statistic, and its low variance across blocks
        is what keeps the shared codebook shape from being stretched
        block-to-block. amax fluctuates ~20% on 32-sample Gaussian blocks;
        RMS only ~12%, and the error difference is large (0.60 vs 0.33 rel).

        Layout (lm2): 32 values -> 8 payload bytes. Byte j holds code[4j..4j+3]
        as 2-bit fields at bit positions 0, 2, 4, 6 (same field layout as the
        q6 high plane). Layout (lm3): 64 values -> 24 payload bytes = 16-byte
        low plane (low 2 bits, same field layout) + 8-byte high plane (top
        bit of each value, one bit per position 0..7).

        Code assignment: code = #{thresholds < value} with thresholds at the
        exact midpoints of adjacent levels -- ties land on the lower code,
        matching torch.argmin's first-occurrence tie-break.
        """
        bs = self.block_size
        assert x.shape[-1] % bs == 0, f"head_dim {x.shape[-1]} is not a multiple of {bs}"
        blocks = x.float().unflatten(-1, (x.shape[-1] // bs, bs))  # [..., NB, bs]
        rms = blocks.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # An all-zero block stores any code under any scale; 1.0 keeps fp8 finite.
        rms = torch.where(rms > 0, rms, torch.ones_like(rms))
        # Round to the stored fp8 precision before dividing, so the value used
        # here and the value the attention kernel reads back are identical.
        rms = rms.to(self.scale_dtype).float()
        q = blocks / rms  # rms already keepdim: [..., NB, 1]
        thresholds = torch.tensor(
            _lm_thresholds(self.layout), dtype=torch.float32, device=x.device
        )
        codes = torch.zeros(q.shape, dtype=torch.int32, device=x.device)
        for t in thresholds:
            codes += (q > t).to(torch.int32)  # [..., NB, bs] in [0, 2**bits - 1]

        if bits == 2:
            groups = codes.unflatten(-1, (bs // 4, 4))  # [..., NB, 8, 4]
            masks = torch.tensor([1, 4, 16, 64], dtype=torch.int32, device=x.device)
            packed = (groups * masks).sum(dim=-1).to(torch.uint8)  # [..., NB, 8]
            # squeeze the keepdim scale axis so the returned scales match the
            # pool's scale-buffer shape [..., NB] (see scale_shape()).
            return (packed.flatten(-2), rms.squeeze(-1).to(self.scale_dtype))

        if bits == 3:
            lo2 = codes & 0x3  # low 2 bits -> 16-byte field plane
            hi1 = (codes >> 2) & 0x1  # top bit -> 8-byte plane
            lo_groups = lo2.unflatten(-1, (bs // 4, 4))  # [..., NB, 16, 4]
            lo_masks = torch.tensor([1, 4, 16, 64], dtype=torch.int32, device=x.device)
            packed_lo = (lo_groups * lo_masks).sum(dim=-1).to(torch.uint8)  # [..., NB, 16]
            hi_groups = hi1.unflatten(-1, (bs // 8, 8))  # [..., NB, 8, 8]
            hi_masks = torch.tensor(
                [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int32, device=x.device
            )
            packed_hi = (hi_groups * hi_masks).sum(dim=-1).to(torch.uint8)  # [..., NB, 8]
            payload = torch.cat([packed_lo, packed_hi], dim=-1).flatten(-2)  # [..., NB * 24]
            return (payload, rms.squeeze(-1).to(self.scale_dtype))

        raise ValueError(f"unsupported LM bits: {bits}")

    def _dequantize_lm(
        self, payload: torch.Tensor, scales: torch.Tensor, *, bits: int
    ) -> torch.Tensor:
        """Inverse of :meth:`_quantize_lm`. Returns float32 with logical head_dim."""
        bs = self.block_size
        payload_bytes = self.payload_bytes_per_block
        nb = payload.shape[-1] // payload_bytes
        levels = torch.tensor(_lm_levels(self.layout), dtype=torch.float32, device=payload.device)

        if bits == 2:
            bytes_view = payload.unflatten(-1, (nb, 8))  # [..., NB, 8]
            shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=payload.device)
            codes = ((bytes_view.unsqueeze(-1) >> shifts) & 0x3)  # [..., NB, 8, 4]
            codes = codes.flatten(-2)  # [..., NB, 32]
        elif bits == 3:
            block_view = payload.unflatten(-1, (nb, 24))  # [..., NB, 24]
            lo_bytes = block_view[..., :16]  # [..., NB, 16]
            hi_bytes = block_view[..., 16:]  # [..., NB, 8]
            lo_shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=payload.device)
            lo2 = ((lo_bytes.unsqueeze(-1) >> lo_shifts) & 0x3).flatten(-2)  # [..., NB, 64]
            hi_shifts = torch.arange(8, dtype=torch.int32, device=payload.device)
            hi1 = ((hi_bytes.unsqueeze(-1) >> hi_shifts) & 0x1).flatten(-2)  # [..., NB, 64]
            codes = lo2 | (hi1 << 2)  # [..., NB, 64]
        else:
            raise ValueError(f"unsupported LM bits: {bits}")

        vals = levels[codes.long()]  # [..., NB, bs]
        return (vals * scales.float().unsqueeze(-1)).flatten(-2)

    # ---- sub-byte (Q4 / Q6) ----

    def _quantize_subbyte(self, x: torch.Tensor, *, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric per-block quantization to a sub-byte packed uint8 buffer.

        Packing (Q4): 32 values -> 16 bytes. Byte j holds val[2j] in low nibble,
        val[2j+1] in high nibble. Sign-extended on read: low/high nibble becomes a
        value in [-(2^(bits-1)), 2^(bits-1)-1] = [-8, 7] for 4-bit, [-32, 31] for 6-bit.

        Packing (Q6): 32 values -> 24 bytes = 16-byte low plane + 8-byte high plane.
        Low plane is the same nibble layout as Q4 but holds the LOW 4 bits of each
        6-bit value (val[2j] & 0xF | (val[2j+1] & 0xF) << 4). High plane byte g holds
        the top 2 bits of val[4g..4g+3] at bit positions 0, 2, 4, 6.
        """
        assert x.shape[-1] % BLOCK == 0, (
            f"head_dim {x.shape[-1]} is not a multiple of {BLOCK}"
        )
        max_mag = self.max_magnitude
        blocks = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))  # [..., NB, 32]
        amax = blocks.abs().amax(dim=-1)
        scales = torch.where(amax > 0, amax / max_mag, torch.ones_like(amax))
        scales = scales.to(SCALE_DTYPE)
        # Quantize in float, round half away from zero, clamp. The 4-bit signed
        # range is [-8, 7] (16 levels, stored as unsigned 0..15 -- 8 maps to
        # -8 in the XOR-sub sign extension), and 6-bit signed is [-32, 31]
        # (64 levels). For Q4, MAX_MAG=8 and the writer must clamp to
        # MAX_MAG-1 = 7 so the dequant sees a real signed value; for Q6,
        # MAX_MAG=31 already aligns the storage with the 6-bit signed range
        # so we clamp at MAX_MAG.
        if bits == 4:
            upper = max_mag - 1
        else:
            upper = max_mag
        qf = blocks / scales.float().unsqueeze(-1)
        qf = torch.where(qf >= 0, (qf + 0.5).floor(), (qf - 0.5).ceil())
        qf = qf.clamp_(-max_mag, upper).to(torch.int32)  # [..., NB, 32] int32

        # Sign-extend via int32 shift (advanced indexing promotes to int64, which
        # would NOT round-trip -- see freetoken-kv-subbyte-quant memory).
        if bits == 4:
            # 4-bit signed: only the low 4 bits are stored; values -8..7.
            mask4 = torch.tensor(0xF, dtype=torch.int32, device=qf.device)
            lo = qf & mask4  # [..., NB, 32] each value in [0, 15]
            # Reshape: pair values (even, odd) -> 1 byte: low nibble = even,
            # high nibble = odd. The low plane has 16 bytes per block.
            lo_pairs = lo.unflatten(-1, (BLOCK // 2, 2))  # [..., NB, 16, 2]
            packed_lo = (lo_pairs[..., 0] | (lo_pairs[..., 1] << 4)).to(torch.uint8)
            payload = packed_lo.flatten(-2)  # [..., NB * 16]
            return (payload, scales)

        if bits == 6:
            # 6-bit: low 4 bits go to lo plane (16 bytes), top 2 bits go to hi plane
            # (8 bytes). Sign extension recovers -32..31 on read.
            # Layout per block: 16 lo bytes followed by 8 hi bytes, 24 bytes total,
            # so the two planes live adjacently inside each block (cache-friendly
            # when a single block is read). The dequant side slices the same way.
            mask4 = torch.tensor(0xF, dtype=torch.int32, device=qf.device)
            mask2 = torch.tensor(0x3, dtype=torch.int32, device=qf.device)
            lo4 = qf & mask4  # low 4 bits of each value
            hi2 = (qf >> 4) & mask2  # top 2 bits of each value

            # Low plane: same nibble layout as Q4. 16 bytes per block hold 32 values'
            # low 4 bits.
            lo_pairs = lo4.unflatten(-1, (BLOCK // 2, 2))
            packed_lo = (lo_pairs[..., 0] | (lo_pairs[..., 1] << 4)).to(torch.uint8)
            # [..., NB, 16]

            # High plane: 32 values' 2-bit tops go into 8 bytes. Each byte holds 4
            # values at bit positions 0, 2, 4, 6. Each 2-bit value lives at bit 2*v
            # within the byte, so the per-position mask is 1 << (2*v) = 1, 4, 16, 64.
            hi_groups = hi2.unflatten(-1, (BLOCK // 4, 4))  # [..., NB, 8, 4]
            masks = torch.tensor([1, 4, 16, 64], dtype=torch.int32, device=qf.device)
            packed_hi = (hi_groups * masks).sum(dim=-1).to(torch.uint8)  # [..., NB, 8]

            # Per-block concat: 16 lo + 8 hi = 24 bytes per block, in [.., NB, 24].
            payload = torch.cat([packed_lo, packed_hi], dim=-1).flatten(-2)  # [..., NB * 24]
            return (payload, scales)

        raise ValueError(f"unsupported sub-byte bits: {bits}")

    def _dequantize_subbyte(
        self, payload: torch.Tensor, scales: torch.Tensor, *, bits: int
    ) -> torch.Tensor:
        """Inverse of :meth:`_quantize_subbyte`. Returns float32 with logical head_dim."""
        logical_per_block = BLOCK
        # payload shape: [..., NB * payload_bytes_per_block]
        payload_bytes = self.payload_bytes_per_block
        nb = payload.shape[-1] // payload_bytes
        if bits == 4:
            # 16 bytes -> 32 nibbles (2 nibbles per byte: low, high).
            bytes_view = payload.unflatten(-1, (nb, 16))  # [..., NB, 16]
            # low nibble = byte & 0xF, high nibble = (byte >> 4) & 0xF
            lo = (bytes_view & 0xF).to(torch.int32)
            hi = ((bytes_view >> 4) & 0xF).to(torch.int32)
            # Interleave: per 16 bytes -> 32 values in order lo[0], hi[0], lo[1], hi[1], ...
            stacked = torch.stack([lo, hi], dim=-1)  # [..., NB, 16, 2]
            vals = stacked.flatten(-2)  # [..., NB, 32] int32 in [0, 15]
            # Sign-extend 4-bit: (v << 28) >> 28 == (v - (v & 8) * 2) but the canonical
            # arithmetic shift is clearer.
            vals = (vals << (32 - 4)) >> (32 - 4)  # [-8, 7]
            return (vals.to(torch.float32) * scales.float().unsqueeze(-1)).flatten(-2)

        if bits == 6:
            # Per-block layout (must match _quantize_subbyte): each of NB blocks is
            # 24 bytes = 16 lo + 8 hi. Unflatten the whole thing once.
            block_view = payload.unflatten(-1, (nb, 24))  # [..., NB, 24]
            lo_bytes = block_view[..., :16]  # [..., NB, 16]
            hi_bytes = block_view[..., 16:]  # [..., NB, 8]

            lo = (lo_bytes & 0xF).to(torch.int32)
            hi_lo = ((lo_bytes >> 4) & 0xF).to(torch.int32)  # high nibble of lo plane
            # Recover full 4-bit lo and 2-bit hi per value
            lo4 = torch.stack([lo, hi_lo], dim=-1).flatten(-2)  # [..., NB, 32]

            # High plane: 8 bytes, each with 4 2-bit values at bits 0, 2, 4, 6
            # Extract: value_in_group g = (byte >> (2*g)) & 0x3
            hi_view = hi_bytes.to(torch.int32)  # [..., NB, 8]
            shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=payload.device)
            hi_groups = ((hi_view.unsqueeze(-1) >> shifts) & 0x3)  # [..., NB, 8, 4]
            hi2 = hi_groups.flatten(-2)  # [..., NB, 32]

            # Combine: 6-bit value = lo4 | (hi2 << 4)
            vals6 = (lo4 | (hi2 << 4))  # [..., NB, 32] int32 in [0, 63]
            # Sign-extend 6-bit
            vals = (vals6 << (32 - 6)) >> (32 - 6)  # [-32, 31]
            return (vals.to(torch.float32) * scales.float().unsqueeze(-1)).flatten(-2)

        raise ValueError(f"unsupported sub-byte bits: {bits}")


# 8-bit schemes.
Q8_0 = KVQuantSpec(name="q8_0", storage_dtype=torch.int8, max_magnitude=127.0)
FP8_E4M3 = KVQuantSpec(
    name="fp8_e4m3", storage_dtype=torch.float8_e4m3fn, max_magnitude=448.0
)

# Sub-byte schemes. Q4_0: 4-bit signed, 16 bytes/32 values = 0.5 byte/element.
# The range is [-8, 7] (16 levels). max_magnitude = 8 -- the symmetric limit
# the quantizer targets; values at the -8 boundary are exact, the +7 boundary
# is exact. GGUF's Q4_0 spec uses 8 (this is the canonical setting; max=7 wastes
# one quantization step on an unreachable value AND empirically costs ~20% rel_err
# on K/V-shaped data because the distribution tail biases scale upward, leaving
# the +7 boundary the more frequent side).
#
# Note: switching max_magnitude changes the binary layout, so any saved KV
# caches from the old spec need to be invalidated. Currently this only
# matters for fresh Q4 deployments; existing Q8 services are unaffected.
Q4_0 = KVQuantSpec(
    name="q4_0",
    storage_dtype=STORAGE_BYTE_DTYPE,
    max_magnitude=8.0,
    layout=LAYOUT_Q4,
    bits=4,
    payload_bytes_per_block=16,
)
# Q6_0: 6-bit signed, 24 bytes/32 values = 0.75 byte/element.
Q6_0 = KVQuantSpec(
    name="q6_0",
    storage_dtype=STORAGE_BYTE_DTYPE,
    max_magnitude=31.0,
    layout=LAYOUT_Q6,
    bits=6,
    payload_bytes_per_block=24,
)

# LM (Lloyd-Max) schemes: trained non-uniform codebooks instead of uniform
# integer grids. lm2: 2-bit codes, 8 payload bytes / 32 values + 1 fp8 E4M3
# RMS scale = 0.28125 B/elem -- HALF of q4_0's density -- at ~3.5x q4_0's
# kernel rel_err (0.334 vs ~0.09 on Gaussian). lm3: 3-bit codes, 64-value
# blocks, 24 payload bytes + 1 fp8 scale = 0.390625 B/elem at ~0.18 rel_err,
# which sits ON the accuracy-per-byte convex hull between q4_0 and q2_lm
# (linear interpolation at that byte density would predict ~0.24).
# The scale is the block RMS, not amax/MAX_MAG: the codebooks are trained on
# RMS-normalized data, and RMS's low block-to-block variance is what keeps
# the shared codebook shape from being stretched (amax-driven scales measured
# 0.60 rel_err vs 0.33 for RMS on the same data).
Q2_LM = KVQuantSpec(
    name="q2_lm",
    storage_dtype=STORAGE_BYTE_DTYPE,
    max_magnitude=1.5096,
    layout=LAYOUT_LM2,
    bits=2,
    payload_bytes_per_block=8,
    scale_dtype=torch.float8_e4m3fn,
)
Q3_LM = KVQuantSpec(
    name="q3_lm",
    storage_dtype=STORAGE_BYTE_DTYPE,
    max_magnitude=2.1508,
    layout=LAYOUT_LM3,
    bits=3,
    payload_bytes_per_block=24,
    block_size=64,
    scale_dtype=torch.float8_e4m3fn,
)

# NVFP4: NVIDIA FP4 (E2M1) per-block, 16 values per block with one
# fp8 E4M3 block scale. (8 payload + 1 scale) / 16 = 0.5625 B/elem --
# the same density as q4_0, but the E2M1 floating-point code (1 sign
# + 2 exponent + 1 mantissa) is log-spaced (0, 0.5, 1, 1.5, 2, 3, 4, 6)
# rather than uniform-integer, which is materially more accurate on
# kurtotic K/V distributions. max_magnitude=6 is the largest representable
# magnitude (the 0b0111 code); we don't clamp during quantize because
# the float code handles overflow naturally. Reuses the existing MoE
# expert-weight NVFP4 quantize/dequantize machinery (kernel/triton/nvfp4_linear.py),
# so porting cost is mostly the attention path.
NVFP4 = KVQuantSpec(
    name="nvfp4",
    storage_dtype=STORAGE_BYTE_DTYPE,
    max_magnitude=6.0,
    layout=LAYOUT_NVFP4,
    bits=4,
    payload_bytes_per_block=8,
    block_size=16,
    scale_dtype=torch.float8_e4m3fn,
)

NONE = KVQuantSpec(name="auto", storage_dtype=None, max_magnitude=0.0)

_BY_NAME = {spec.name: spec for spec in (NONE, Q8_0, FP8_E4M3, NVFP4, Q4_0, Q6_0, Q2_LM, Q3_LM)}
KV_CACHE_DTYPES = tuple(_BY_NAME)


def resolve_kv_quant(name: str | None) -> KVQuantSpec:
    """``--kv-cache-dtype`` value -> spec. ``None``/``"auto"`` means unquantized."""
    if name is None:
        return NONE
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown --kv-cache-dtype {name!r}; choose from {', '.join(KV_CACHE_DTYPES)}"
        ) from None


__all__ = [
    "BLOCK",
    "SCALE_DTYPE",
    "STORAGE_BYTE_DTYPE",
    "LAYOUT_Q8",
    "LAYOUT_Q4",
    "LAYOUT_Q6",
    "LAYOUT_LM2",
    "LAYOUT_LM3",
    "KVQuantSpec",
    "KV_CACHE_DTYPES",
    "Q8_0",
    "FP8_E4M3",
    "Q4_0",
    "Q6_0",
    "Q2_LM",
    "Q3_LM",
    "NONE",
    "resolve_kv_quant",
]
