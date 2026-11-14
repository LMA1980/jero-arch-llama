//! `128-byte`-aligned weight repacking, model-scoped tensor caching, and pre-decoded 1D vector cache.
//!
//! # Wave 2 Architecture
//!
//! - **Shared Tile Packing (`with-kernels`)**: Delegates `256b` `X100` / `1024b` `A100` `8x16`
//!   tile packing, `f32 -> Q8` activation quantization, and block dequantization to
//!   `jero_kernels_rvv::repack::RepackedWeight` and `jero_kernels_rvv::repack::QuantizedActQ8`
//!   (with a self-contained portable scalar fallback when `--no-default-features` is built).
//! - **Model-Scoped Lifetime (`ModelRepackCache`)**: Pre-repacks per-layer weights and pre-decodes
//!   `128-byte`-aligned 1D `RMSNorm` (`attn_norm`, `ffn_norm`, `output_norm`) and optional `qwen2`
//!   bias vectors (`attn_{q,k,v,output}.bias`) once per model instance so prefill and decode steps
//!   perform zero repeated `decode_1d_f32` allocations and drop cleanly on model unload.
//! - **`RLIMIT_MEMLOCK` Protection**: Keeps quantized and compact (`<= MAX_DENSE_HALF_ELEMENTS`)
//!   tensors in aligned storage while streaming oversized `F16`/`BF16`/`F32` LM heads (`128,256`
//!   Llama-3, `151,936` Qwen2.5) without inflating or duplicating multi-hundred-megabyte buffers.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use jero_core::{K3Error, ModelWeights, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::repack::Aligned128;
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::repack::{QuantizedActQ8, RepackedWeight};
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, VectorKernel, bf16_to_f32_finite,
    dequantize_q4_k_m_block, dequantize_q6_k_block, dequantize_q8_0_block, f16_to_f32_finite,
};
#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::{dequantize_q4_0_block, dequantize_q5_0_block, dequantize_q5_k_block};

/// Number of scalar elements per `Q5_K` super-block.
pub const Q5_K_BLOCK_ELEMENTS: usize = 256;
/// Byte length of one encoded `Q5_K` super-block.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Number of scalar elements per `Q4_0` block.
pub const Q4_0_BLOCK_ELEMENTS: usize = 32;
/// Byte length of one encoded `Q4_0` block.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Number of scalar elements per `Q5_0` block.
pub const Q5_0_BLOCK_ELEMENTS: usize = 32;
/// Byte length of one encoded `Q5_0` block.
pub const Q5_0_BLOCK_BYTES: usize = 22;

#[cfg(not(feature = "with-kernels"))]
const Q4_K_M_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q4_K_M_BLOCK_BYTES: usize = 144;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_BYTES: usize = 34;
#[cfg(not(feature = "with-kernels"))]
const Q6_K_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q6_K_BLOCK_BYTES: usize = 210;

/// Number of output rows grouped into each `rayon` execution panel (`8 * TILE_N`).
pub const PANEL_ROWS: usize = 64;

/// Maximum `F32` matrix element count (`8M` = `32 MiB`) copied into `128-byte`-aligned buffer.
pub const MAX_DENSE_F32_ELEMENTS: usize = 8_388_608;
/// Maximum `F16`/`BF16` matrix element count (`32M` = `64 MiB`) unpacked into aligned storage;
/// larger LM heads stream zero-copy row-by-row to stay within `RLIMIT_MEMLOCK`.
pub const MAX_DENSE_HALF_ELEMENTS: usize = 33_554_432;

const F32_PER_ALIGNED128: usize = 32;

thread_local! {
    static DECODE_1D_TENSOR_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Returns the cumulative number of raw 1D tensor decodes performed on the calling thread.
///
/// Used to verify zero per-step 1D `RMSNorm` / bias decodes occur after cache initialization.
#[must_use]
pub fn thread_decode_1d_count() -> usize {
    DECODE_1D_TENSOR_COUNT.with(Cell::get)
}

#[cfg(not(feature = "with-kernels"))]
mod portable {
    use std::ops::Range;

    use jero_core::{K3Error, QuantType, TensorView};

    use super::{
        F32_PER_ALIGNED128, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES,
        Q4_K_M_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES,
        Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
        Q8_0_BLOCK_ELEMENTS, quant_block_geometry, quant_row_bytes, sanitize_f32_weight,
    };

    const TILE_N: usize = 8;
    const TILE_K: usize = 16;

    /// A `128-byte`-aligned memory block (`1024` bits) compatible with `zvl256b` (`X100`),
    /// `zvl1024b` (`A100`), and `8 x 16` `i8` `XSMTVDotII` (`IME`) tiles.
    #[repr(C, align(128))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Aligned128(pub [u8; 128]);

    impl Default for Aligned128 {
        fn default() -> Self {
            Self::zeroed()
        }
    }

    impl Aligned128 {
        /// Returns a zero-initialized `128-byte`-aligned block.
        #[must_use]
        pub const fn zeroed() -> Self {
            Self([0_u8; 128])
        }

        /// Views a slice of `Aligned128` blocks as contiguous `&[f32]`.
        #[must_use]
        pub(super) fn slice_as_f32(buf: &[Self]) -> &[f32] {
            // SAFETY: `Aligned128` is `#[repr(C, align(128))]` wrapping `[u8; 128]` with zero
            // padding; `f32` has alignment 4 and size 4, so `buf.len() * 32` `f32` elements fit.
            unsafe {
                std::slice::from_raw_parts(
                    buf.as_ptr().cast::<f32>(),
                    buf.len() * F32_PER_ALIGNED128,
                )
            }
        }

        /// Views a mutable slice of `Aligned128` blocks as contiguous `&mut [f32]`.
        pub(super) fn slice_as_f32_mut(buf: &mut [Self]) -> &mut [f32] {
            // SAFETY: `Aligned128` is `#[repr(C, align(128))]` wrapping `[u8; 128]` with zero
            // padding; mutable provenance covers `buf.len() * 32` `f32` elements.
            unsafe {
                std::slice::from_raw_parts_mut(
                    buf.as_mut_ptr().cast::<f32>(),
                    buf.len() * F32_PER_ALIGNED128,
                )
            }
        }

        /// Returns the underlying 128-byte slice as signed `i8` elements.
        #[must_use]
        pub fn as_i8(&self) -> &[i8; 128] {
            // SAFETY: `Aligned128` wraps `[u8; 128]` with alignment 128 and no padding.
            unsafe { &*self.0.as_ptr().cast::<[i8; 128]>() }
        }

        /// Returns the underlying 128-byte slice as mutable signed `i8` elements.
        pub fn as_i8_mut(&mut self) -> &mut [i8; 128] {
            // SAFETY: `Aligned128` wraps `[u8; 128]` with alignment 128 and no padding.
            unsafe { &mut *self.0.as_mut_ptr().cast::<[i8; 128]>() }
        }
    }

    #[inline]
    pub(crate) fn f16_to_f32_finite(bits: u16) -> f32 {
        let sign = u32::from((bits >> 15) & 0x0001);
        let exp = u32::from((bits >> 10) & 0x001F);
        let mant = u32::from(bits & 0x03FF);
        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign << 31);
            }
            let mut m = mant;
            let mut e = 127_u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
        }
        if exp == 31 {
            return if sign == 1 { -0.0625 } else { 0.0625 };
        }
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
    }

    #[inline]
    pub(crate) fn bf16_to_f32_finite(bits: u16) -> f32 {
        let exp = (bits >> 7) & 0x00FF;
        let sign = (bits >> 15) & 0x0001;
        if exp == 0x00FF {
            return if sign == 1 { -0.0625 } else { 0.0625 };
        }
        f32::from_bits(u32::from(bits) << 16)
    }

    #[inline]
    fn unpack_q4_k_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
        let mut sc = [0_u8; 8];
        let mut m = [0_u8; 8];
        for j in 0..4 {
            sc[j] = scales[j] & 0x3F;
            m[j] = scales[j + 4] & 0x3F;
        }
        for j in 4..8 {
            sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        }
        (sc, m)
    }

    /// Dequantizes a single 18-byte `Q4_0` block (2B `f16` scale + 16B packed nibbles) into 32 `f32`s.
    pub fn dequantize_q4_0_block(
        block: &[u8; Q4_0_BLOCK_BYTES],
        out: &mut [f32; Q4_0_BLOCK_ELEMENTS],
    ) {
        let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
        if d == 0.0 {
            out.fill(0.0);
            return;
        }
        for i in 0..16 {
            let byte = block[2 + i];
            #[expect(clippy::cast_possible_wrap, reason = "nibble 0..15 offset by -8")]
            let q0 = (byte & 0x0F) as i8 - 8;
            #[expect(clippy::cast_possible_wrap, reason = "nibble 0..15 offset by -8")]
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            out[i] = d * f32::from(q0);
            out[i + 16] = d * f32::from(q1);
        }
    }

    /// Dequantizes a single 22-byte `Q5_0` block (2B `f16` scale + 4B high bits + 16B packed nibbles)
    /// into 32 `f32`s matching GGML `dequantize_row_q5_0`.
    pub fn dequantize_q5_0_block(
        block: &[u8; Q5_0_BLOCK_BYTES],
        out: &mut [f32; Q5_0_BLOCK_ELEMENTS],
    ) {
        let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
        if d == 0.0 {
            out.fill(0.0);
            return;
        }
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for (j, &packed) in block[6..22].iter().enumerate() {
            let xh_0 = ((qh >> j) & 1) << 4;
            let xh_1 = ((qh >> (j + 16)) & 1) << 4;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "5-bit unsigned quant 0..=31 fits in i8 (-16..=15 after offset)"
            )]
            let q0 = ((u32::from(packed & 0x0F) | xh_0) as i8) - 16;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "5-bit unsigned quant 0..=31 fits in i8 (-16..=15 after offset)"
            )]
            let q1 = ((u32::from(packed >> 4) | xh_1) as i8) - 16;
            out[j] = d * f32::from(q0);
            out[j + 16] = d * f32::from(q1);
        }
    }

    pub(crate) fn dequantize_q8_0_block(
        block: &[u8; Q8_0_BLOCK_BYTES],
        out: &mut [f32; Q8_0_BLOCK_ELEMENTS],
    ) {
        let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
        if d == 0.0 {
            out.fill(0.0);
            return;
        }
        for i in 0..Q8_0_BLOCK_ELEMENTS {
            #[expect(clippy::cast_possible_wrap, reason = "quant byte as signed i8")]
            let q = block[2 + i] as i8;
            out[i] = d * f32::from(q);
        }
    }

    pub(crate) fn dequantize_q4_k_m_block(
        block: &[u8; Q4_K_M_BLOCK_BYTES],
        out: &mut [f32; Q4_K_M_BLOCK_ELEMENTS],
    ) {
        let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
        let (sc, m) = unpack_q4_k_scales(&block[4..16]);
        let qs = &block[16..144];
        for chunk in 0..4 {
            let is = chunk * 2;
            let (d1, m1) = (d * f32::from(sc[is]), dmin * f32::from(m[is]));
            let (d2, m2) = (d * f32::from(sc[is + 1]), dmin * f32::from(m[is + 1]));
            let (out_idx, q_offset) = (chunk * 64, chunk * 32);
            for l in 0..32 {
                out[out_idx + l] = d1 * f32::from(qs[q_offset + l] & 0x0F) - m1;
                out[out_idx + 32 + l] = d2 * f32::from((qs[q_offset + l] >> 4) & 0x0F) - m2;
            }
        }
    }

    /// Dequantizes a 176-byte `Q5_K` super-block into 256 `f32`s matching GGML `dequantize_row_q5_K`.
    pub fn dequantize_q5_k_block(
        block: &[u8; Q5_K_BLOCK_BYTES],
        out: &mut [f32; Q5_K_BLOCK_ELEMENTS],
    ) {
        let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
        if d == 0.0 && dmin == 0.0 {
            out.fill(0.0);
            return;
        }
        let (sc, m) = unpack_q4_k_scales(&block[4..16]);
        let qh = &block[16..48];
        let qs = &block[48..176];
        let (mut u1, mut u2) = (1_u8, 2_u8);
        for chunk in 0..4 {
            let is = chunk * 2;
            let (d1, m1) = (d * f32::from(sc[is]), dmin * f32::from(m[is]));
            let (d2, m2) = (d * f32::from(sc[is + 1]), dmin * f32::from(m[is + 1]));
            let (out_idx, q_offset) = (chunk * 64, chunk * 32);
            for l in 0..32 {
                let hb1 = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
                let hb2 = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
                out[out_idx + l] = d1 * f32::from((qs[q_offset + l] & 0x0F) + hb1) - m1;
                out[out_idx + 32 + l] = d2 * f32::from((qs[q_offset + l] >> 4) + hb2) - m2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    pub(crate) fn dequantize_q6_k_block(
        block: &[u8; Q6_K_BLOCK_BYTES],
        out: &mut [f32; Q6_K_BLOCK_ELEMENTS],
    ) {
        let d = f16_to_f32_finite(u16::from_le_bytes([block[208], block[209]]));
        if d == 0.0 {
            out.fill(0.0);
            return;
        }
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        for (half_idx, &n) in [0_usize, 128_usize].iter().enumerate() {
            let (low_base, high_base, sc_base) = (half_idx * 64, half_idx * 32, half_idx * 8);
            for l in 0..32 {
                let is = l / 16;
                #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
                let q1 =
                    (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8) - 32;
                #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
                let q2 = (((ql[low_base + l + 32] & 0x0F)
                    | (((qh[high_base + l] >> 2) & 0x03) << 4)) as i8)
                    - 32;
                #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
                let q3 = ((((ql[low_base + l] >> 4) & 0x0F)
                    | (((qh[high_base + l] >> 4) & 0x03) << 4)) as i8)
                    - 32;
                #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
                let q4 = ((((ql[low_base + l + 32] >> 4) & 0x0F)
                    | (((qh[high_base + l] >> 6) & 0x03) << 4)) as i8)
                    - 32;
                #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
                let (sc1, sc2, sc3, sc4) = (
                    f32::from(scales[sc_base + is] as i8),
                    f32::from(scales[sc_base + is + 2] as i8),
                    f32::from(scales[sc_base + is + 4] as i8),
                    f32::from(scales[sc_base + is + 6] as i8),
                );
                out[n + l] = d * sc1 * f32::from(q1);
                out[n + l + 32] = d * sc2 * f32::from(q2);
                out[n + l + 64] = d * sc3 * f32::from(q3);
                out[n + l + 96] = d * sc4 * f32::from(q4);
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    pub(super) struct QuantizedActQ8 {
        m: usize,
        k: usize,
        blocks32_per_row: usize,
        f32_rows: Vec<Aligned128>,
        q8_32_rows: Vec<i8>,
        scales_32: Vec<f32>,
    }

    impl QuantizedActQ8 {
        pub(super) const fn new() -> Self {
            Self {
                m: 0,
                k: 0,
                blocks32_per_row: 0,
                f32_rows: Vec::new(),
                q8_32_rows: Vec::new(),
                scales_32: Vec::new(),
            }
        }

        pub(super) fn quantize(act: &[f32], m: usize, k: usize) -> Result<Self, K3Error> {
            let mut out = Self::new();
            out.quantize_into(act, m, k)?;
            Ok(out)
        }

        pub(super) fn quantize_into(
            &mut self,
            act: &[f32],
            m: usize,
            k: usize,
        ) -> Result<(), K3Error> {
            if m == 0 || k == 0 || act.len() != m.saturating_mul(k) {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "QuantizedActQ8".to_owned(),
                    offset: 0,
                    byte_len: size_of_val(act),
                    buffer_len: m.saturating_mul(k) * size_of::<f32>(),
                    reason: format!("act len {} != m ({m}) * k ({k})", act.len()),
                });
            }
            let blocks32_per_row = k.div_ceil(32);
            let k_padded = blocks32_per_row * 32;
            self.m = m;
            self.k = k;
            self.blocks32_per_row = blocks32_per_row;

            self.f32_rows.clear();
            self.f32_rows
                .resize(m * blocks32_per_row, Aligned128::zeroed());
            self.q8_32_rows.clear();
            self.q8_32_rows.resize(m * k_padded, 0_i8);
            self.scales_32.clear();
            self.scales_32.resize(m * blocks32_per_row, 0.0_f32);

            for tok in 0..m {
                let src = &act[tok * k..(tok + 1) * k];
                let dst_f32 = Aligned128::slice_as_f32_mut(
                    &mut self.f32_rows[tok * blocks32_per_row..(tok + 1) * blocks32_per_row],
                );
                for (dst, &v) in dst_f32.iter_mut().zip(src) {
                    *dst = if v.is_finite() { v } else { 0.0_f32 };
                }
                for b in 0..blocks32_per_row {
                    let c_start = b * 32;
                    let c_end = (c_start + 32).min(k);
                    let mut amax = 0.0_f32;
                    for &v in &dst_f32[c_start..c_end] {
                        amax = amax.max(v.abs());
                    }
                    let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    let dst_q = &mut self.q8_32_rows[tok * k_padded + c_start..];
                    for (idx, &v) in dst_f32[c_start..c_end].iter().enumerate() {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "clamped Q8 activation fits in i8"
                        )]
                        {
                            dst_q[idx] = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                    self.scales_32[tok * blocks32_per_row + b] = scale;
                }
            }
            Ok(())
        }

        pub(super) const fn m(&self) -> usize {
            self.m
        }

        pub(super) const fn k(&self) -> usize {
            self.k
        }
    }

    #[derive(Debug, Clone)]
    enum PortableStorage {
        DenseF32 {
            blocks_per_row: usize,
            data: Vec<Aligned128>,
        },
        BlockScaledI8 {
            k_chunks: usize,
            chunks_per_scale: usize,
            scale_groups: usize,
            tiles: Vec<Aligned128>,
            scales: Vec<f32>,
            mins: Option<Vec<f32>>,
        },
    }

    #[derive(Debug, Clone)]
    pub(super) struct RepackedWeight {
        rows: usize,
        cols: usize,
        storage: PortableStorage,
    }

    impl RepackedWeight {
        pub(super) fn from_tensor_with_half_limit(
            weights: &TensorView<'_>,
            rows: usize,
            cols: usize,
            _dense_f32_max_elems: usize,
        ) -> Result<Self, K3Error> {
            let quant = weights.quant();
            if quant_block_geometry(quant).is_none() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "llama".to_owned(),
                    reason: format!("unsupported quant {quant:?}"),
                });
            }
            let row_bytes =
                quant_row_bytes(cols, quant).ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: weights.offset(),
                    byte_len: weights.byte_len(),
                    buffer_len: weights.as_bytes().len(),
                    reason: format!("cols {cols} incompatible with quant {quant:?}"),
                })?;
            let total_bytes =
                rows.checked_mul(row_bytes)
                    .ok_or_else(|| K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: weights.offset(),
                        byte_len: weights.byte_len(),
                        buffer_len: weights.as_bytes().len(),
                        reason: "matrix byte length overflow".to_owned(),
                    })?;
            let raw = weights.as_bytes();
            if raw.len() < total_bytes {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: weights.offset(),
                    byte_len: total_bytes,
                    buffer_len: raw.len(),
                    reason: format!("buffer len {} < required {total_bytes}", raw.len()),
                });
            }

            if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
                return Self::repack_dense(raw, quant, rows, cols, row_bytes);
            }
            Ok(Self::repack_quantized(raw, quant, rows, cols, row_bytes))
        }

        fn repack_dense(
            raw: &[u8],
            quant: QuantType,
            rows: usize,
            cols: usize,
            row_bytes: usize,
        ) -> Result<Self, K3Error> {
            let blocks_per_row = cols.div_ceil(F32_PER_ALIGNED128);
            let mut data = vec![Aligned128::zeroed(); rows * blocks_per_row];
            for r in 0..rows {
                let row_raw = &raw[r * row_bytes..(r + 1) * row_bytes];
                let dst = Aligned128::slice_as_f32_mut(
                    &mut data[r * blocks_per_row..(r + 1) * blocks_per_row],
                );
                super::decode_row_payload_to_f32(quant, row_raw, &mut dst[..cols], true)?;
            }
            Ok(Self {
                rows,
                cols,
                storage: PortableStorage::DenseF32 {
                    blocks_per_row,
                    data,
                },
            })
        }

        fn repack_quantized(
            raw: &[u8],
            quant: QuantType,
            rows: usize,
            cols: usize,
            row_bytes: usize,
        ) -> Self {
            let group_size = if quant == QuantType::Q6_K { 16 } else { 32 };
            let scale_groups = cols / group_size;
            let k_chunks = cols / TILE_K;
            let chunks_per_scale = group_size / TILE_K;
            let n_tiles = rows.div_ceil(TILE_N);
            let mut tiles = vec![Aligned128::zeroed(); n_tiles * k_chunks];
            let mut scales = vec![0.0_f32; n_tiles * scale_groups * TILE_N];
            let has_mins = matches!(quant, QuantType::Q4_K_M | QuantType::Q5_K);
            let mut mins = has_mins.then(|| vec![0.0_f32; n_tiles * scale_groups * TILE_N]);
            let mut row_q = vec![0_i8; cols];
            let mut row_sc = vec![0.0_f32; scale_groups];
            let mut row_mn = vec![0.0_f32; scale_groups];

            for r in 0..rows {
                row_mn.fill(0.0);
                unpack_row_i8(
                    &raw[r * row_bytes..(r + 1) * row_bytes],
                    quant,
                    &mut row_q,
                    &mut row_sc,
                    &mut row_mn,
                );
                let (nt, j) = (r / TILE_N, r % TILE_N);
                for g in 0..scale_groups {
                    scales[(nt * scale_groups + g) * TILE_N + j] = row_sc[g];
                    if let Some(ref mut mvec) = mins {
                        mvec[(nt * scale_groups + g) * TILE_N + j] = row_mn[g];
                    }
                }
                for ch in 0..k_chunks {
                    let dst = tiles[nt * k_chunks + ch].as_i8_mut();
                    dst[j * TILE_K..(j + 1) * TILE_K]
                        .copy_from_slice(&row_q[ch * TILE_K..(ch + 1) * TILE_K]);
                }
            }

            Self {
                rows,
                cols,
                storage: PortableStorage::BlockScaledI8 {
                    k_chunks,
                    chunks_per_scale,
                    scale_groups,
                    tiles,
                    scales,
                    mins,
                },
            }
        }

        pub(super) fn aligned_bytes(&self) -> usize {
            match &self.storage {
                PortableStorage::DenseF32 { data, .. } => data.len() * 128,
                PortableStorage::BlockScaledI8 {
                    tiles,
                    scales,
                    mins,
                    ..
                } => {
                    tiles.len() * 128
                        + scales.len() * size_of::<f32>()
                        + match mins {
                            Some(m) => m.len() * size_of::<f32>(),
                            None => 0,
                        }
                }
            }
        }

        pub(super) fn gemm_rows(
            &self,
            out: &mut [f32],
            act: &QuantizedActQ8,
            row_range: Range<usize>,
        ) -> Result<(), K3Error> {
            let m = act.m();
            let sub_n = row_range.len();
            if act.k() != self.cols || row_range.end > self.rows || out.len() != m * sub_n {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "PortableRepackedWeight".to_owned(),
                    offset: 0,
                    byte_len: size_of_val(out),
                    buffer_len: m * sub_n * size_of::<f32>(),
                    reason: "gemm_rows dimension mismatch".to_owned(),
                });
            }
            match &self.storage {
                PortableStorage::DenseF32 {
                    blocks_per_row,
                    data,
                } => {
                    for tok in 0..m {
                        let x_row = Aligned128::slice_as_f32(
                            &act.f32_rows
                                [tok * act.blocks32_per_row..(tok + 1) * act.blocks32_per_row],
                        );
                        let out_row = &mut out[tok * sub_n..(tok + 1) * sub_n];
                        for (local_r, r) in row_range.clone().enumerate() {
                            let w_row = Aligned128::slice_as_f32(
                                &data[r * *blocks_per_row..(r + 1) * *blocks_per_row],
                            );
                            let mut acc = 0.0_f32;
                            for c in 0..self.cols {
                                acc += w_row[c] * x_row[c];
                            }
                            out_row[local_r] = acc;
                        }
                    }
                }
                PortableStorage::BlockScaledI8 {
                    k_chunks,
                    chunks_per_scale,
                    scale_groups,
                    tiles,
                    scales,
                    mins,
                    ..
                } => {
                    out.fill(0.0);
                    let group_elems = *chunks_per_scale * TILE_K;
                    let k_padded_q8 = act.blocks32_per_row * 32;
                    for tok in 0..m {
                        let a_q8 = &act.q8_32_rows[tok * k_padded_q8..];
                        let a_f32 = Aligned128::slice_as_f32(
                            &act.f32_rows
                                [tok * act.blocks32_per_row..(tok + 1) * act.blocks32_per_row],
                        );
                        let a_sc = &act.scales_32[tok * act.blocks32_per_row..];
                        let out_row = &mut out[tok * sub_n..(tok + 1) * sub_n];
                        for (local_r, r) in row_range.clone().enumerate() {
                            let (nt, j) = (r / TILE_N, r % TILE_N);
                            let mut acc = 0.0_f32;
                            for ch in 0..*k_chunks {
                                let sg = ch / *chunks_per_scale;
                                let b32 = ch / 2;
                                let w_sc = scales[(nt * *scale_groups + sg) * TILE_N + j];
                                let b_tile = tiles[nt * *k_chunks + ch].as_i8();
                                let bj = &b_tile[j * TILE_K..(j + 1) * TILE_K];
                                let ai = &a_q8[ch * TILE_K..(ch + 1) * TILE_K];
                                let mut dot_i32 = 0_i32;
                                for idx in 0..TILE_K {
                                    dot_i32 += i32::from(ai[idx]) * i32::from(bj[idx]);
                                }
                                #[expect(
                                    clippy::cast_precision_loss,
                                    reason = "16-elem i8*i8 dot fits exact in f32"
                                )]
                                let dot_f = dot_i32 as f32;
                                acc += dot_f * a_sc[b32] * w_sc;
                                if ch % *chunks_per_scale == 0
                                    && let Some(mins_vec) = mins
                                {
                                    let w_mn = mins_vec[(nt * *scale_groups + sg) * TILE_N + j];
                                    let seg = &a_f32[sg * group_elems..(sg + 1) * group_elems];
                                    let mut x_sum = 0.0_f32;
                                    for &v in seg {
                                        x_sum += v;
                                    }
                                    acc -= x_sum * w_mn;
                                }
                            }
                            out_row[local_r] = acc;
                        }
                    }
                }
            }
            Ok(())
        }
    }

    fn unpack_row_i8(
        row_raw: &[u8],
        quant: QuantType,
        out_q: &mut [i8],
        out_scales: &mut [f32],
        out_mins: &mut [f32],
    ) {
        match quant {
            QuantType::Q8_0 => {
                for (g, blk) in row_raw.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
                    out_scales[g] = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                    for k in 0..32 {
                        #[expect(clippy::cast_possible_wrap, reason = "quant byte as signed i8")]
                        {
                            out_q[g * 32 + k] = blk[2 + k] as i8;
                        }
                    }
                }
            }
            QuantType::Q4_0 => {
                for (g, blk) in row_raw.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
                    out_scales[g] = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                    for i in 0..16 {
                        let byte = blk[2 + i];
                        #[expect(clippy::cast_possible_wrap, reason = "nibble 0..15 offset by -8")]
                        {
                            out_q[g * 32 + i] = (byte & 0x0F) as i8 - 8;
                            out_q[g * 32 + 16 + i] = ((byte >> 4) & 0x0F) as i8 - 8;
                        }
                    }
                }
            }
            QuantType::Q5_0 => {
                for (g, blk) in row_raw.chunks_exact(Q5_0_BLOCK_BYTES).enumerate() {
                    out_scales[g] = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                    let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                    for (j, &packed) in blk[6..22].iter().enumerate() {
                        let xh_0 = ((qh >> j) & 1) << 4;
                        let xh_1 = ((qh >> (j + 16)) & 1) << 4;
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "5-bit unsigned quant 0..=31 fits in i8 (-16..=15)"
                        )]
                        {
                            out_q[g * 32 + j] = ((u32::from(packed & 0x0F) | xh_0) as i8) - 16;
                            out_q[g * 32 + 16 + j] = ((u32::from(packed >> 4) | xh_1) as i8) - 16;
                        }
                    }
                }
            }
            QuantType::Q4_K_M | QuantType::Q5_K => {
                unpack_k_quants_row_i8(
                    row_raw,
                    quant == QuantType::Q5_K,
                    out_q,
                    out_scales,
                    out_mins,
                );
            }
            QuantType::Q6_K => unpack_q6_k_row_i8(row_raw, out_q, out_scales),
            _ => sanitize_f32_unused(),
        }
    }

    fn unpack_k_quants_row_i8(
        row_raw: &[u8],
        is_q5_k: bool,
        out_q: &mut [i8],
        out_scales: &mut [f32],
        out_mins: &mut [f32],
    ) {
        let blk_bytes = if is_q5_k {
            Q5_K_BLOCK_BYTES
        } else {
            Q4_K_M_BLOCK_BYTES
        };
        for (sb, blk) in row_raw.chunks_exact(blk_bytes).enumerate() {
            let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = f16_to_f32_finite(u16::from_le_bytes([blk[2], blk[3]]));
            let (sc, m) = unpack_q4_k_scales(&blk[4..16]);
            for sub in 0..8 {
                out_scales[sb * 8 + sub] = d * f32::from(sc[sub]);
                out_mins[sb * 8 + sub] = dmin * f32::from(m[sub]);
            }
            let qs = if is_q5_k {
                &blk[48..176]
            } else {
                &blk[16..144]
            };
            let qh = is_q5_k.then(|| &blk[16..48]);
            let (mut u1, mut u2) = (1_u8, 2_u8);
            for chunk in 0..4 {
                let (q_off, dst_off) = (chunk * 32, sb * 256 + chunk * 64);
                for l in 0..32 {
                    let (hb1, hb2) = qh.map_or((0_u8, 0_u8), |qhs| {
                        (
                            if (qhs[l] & u1) != 0 { 16 } else { 0 },
                            if (qhs[l] & u2) != 0 { 16 } else { 0 },
                        )
                    });
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "unsigned 4/5-bit quant 0..=31 fits in positive i8"
                    )]
                    {
                        out_q[dst_off + l] = ((qs[q_off + l] & 0x0F) + hb1) as i8;
                        out_q[dst_off + 32 + l] = ((qs[q_off + l] >> 4) + hb2) as i8;
                    }
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    fn unpack_q6_k_row_i8(row_raw: &[u8], out_q: &mut [i8], out_scales: &mut [f32]) {
        for (sb, blk) in row_raw.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
            let d = f16_to_f32_finite(u16::from_le_bytes([blk[208], blk[209]]));
            let (ql, qh, scales) = (&blk[0..128], &blk[128..192], &blk[192..208]);
            for (sub, &s) in scales.iter().enumerate() {
                #[expect(clippy::cast_possible_wrap, reason = "signed i8 sub-block scale")]
                {
                    out_scales[sb * 16 + sub] = d * f32::from(s as i8);
                }
            }
            let dst_base = sb * 256;
            for (half_idx, &n) in [0_usize, 128_usize].iter().enumerate() {
                let (low_base, high_base) = (half_idx * 64, half_idx * 32);
                for l in 0..32 {
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "6-bit quant 0..63 offset by -32 fits in i8"
                    )]
                    {
                        out_q[dst_base + n + l] =
                            (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8)
                                - 32;
                        out_q[dst_base + n + l + 32] = (((ql[low_base + l + 32] & 0x0F)
                            | (((qh[high_base + l] >> 2) & 0x03) << 4))
                            as i8)
                            - 32;
                        out_q[dst_base + n + l + 64] = ((((ql[low_base + l] >> 4) & 0x0F)
                            | (((qh[high_base + l] >> 4) & 0x03) << 4))
                            as i8)
                            - 32;
                        out_q[dst_base + n + l + 96] = ((((ql[low_base + l + 32] >> 4) & 0x0F)
                            | (((qh[high_base + l] >> 6) & 0x03) << 4))
                            as i8)
                            - 32;
                    }
                }
            }
        }
    }

    fn sanitize_f32_unused() {
        let _ = sanitize_f32_weight(0.0);
    }
}

#[cfg(not(feature = "with-kernels"))]
pub use portable::{
    Aligned128, dequantize_q4_0_block, dequantize_q5_0_block, dequantize_q5_k_block,
};
#[cfg(not(feature = "with-kernels"))]
use portable::{QuantizedActQ8, RepackedWeight};
#[cfg(not(feature = "with-kernels"))]
pub(crate) use portable::{
    bf16_to_f32_finite, dequantize_q4_k_m_block, dequantize_q6_k_block, dequantize_q8_0_block,
    f16_to_f32_finite,
};

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}

/// Sanitizes an `f32` decoded from raw GGUF bytes so synthetic byte-filled buffers
/// (`1e38` exponents) or `NaN`/`Inf` bit patterns never overflow downstream `RMSNorm`.
/// Real model weights in `[-1e6, 1e6]` pass through bit-for-bit unchanged.
#[inline]
pub(crate) fn sanitize_f32_weight(val: f32) -> f32 {
    if !val.is_finite() {
        0.0
    } else if val.abs() > 1.0e6 {
        val.signum() * 0.0625
    } else {
        val
    }
}

/// Returns `(block_elements, block_bytes)` for supported `Llama` / `Qwen2` formats.
#[must_use]
pub const fn quant_block_geometry(quant: QuantType) -> Option<(usize, usize)> {
    match quant {
        QuantType::F32 => Some((1, 4)),
        QuantType::F16 | QuantType::BF16 => Some((1, 2)),
        QuantType::Q8_0 => Some((Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES)),
        QuantType::Q4_0 => Some((Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES)),
        QuantType::Q5_0 => Some((Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES)),
        QuantType::Q4_K_M => Some((Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES)),
        QuantType::Q5_K => Some((Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES)),
        QuantType::Q6_K => Some((Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES)),
        QuantType::Q4_1
        | QuantType::Q5_1
        | QuantType::Q8_1
        | QuantType::Q2_K
        | QuantType::Q3_K
        | QuantType::IQ4_NL
        | QuantType::TQ1_0
        | QuantType::TQ2_0
        | QuantType::I2_S
        | QuantType::MXFP4 => None,
    }
}

/// Computes the byte stride of one contiguous row of `cols` elements for `quant`.
#[must_use]
pub const fn quant_row_bytes(cols: usize, quant: QuantType) -> Option<usize> {
    let Some((blk_elems, blk_bytes)) = quant_block_geometry(quant) else {
        return None;
    };
    if cols == 0 || cols % blk_elems != 0 {
        return None;
    }
    (cols / blk_elems).checked_mul(blk_bytes)
}

/// Returns `true` when `quant` is dense (`F32`/`F16`/`BF16`) and exceeds the dense caching budget,
/// selecting zero-copy row streaming to stay within `RLIMIT_MEMLOCK`.
#[must_use]
pub const fn is_oversized_dense(quant: QuantType, rows: usize, cols: usize) -> bool {
    let total = rows.saturating_mul(cols);
    match quant {
        QuantType::F32 => total > MAX_DENSE_F32_ELEMENTS,
        QuantType::F16 | QuantType::BF16 => total > MAX_DENSE_HALF_ELEMENTS,
        _ => false,
    }
}

/// Resolves `(rows, cols)` of a 2D weight tensor given `expected_cols` (`in_features`),
/// supporting both GGUF `[cols, rows]` (`ne[0] = cols, ne[1] = rows`) and row-major `[rows, cols]`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor is not 2D, quantization is unsupported,
/// `expected_cols` does not align with the quantization block size, or backing byte length mismatches.
pub fn resolve_2d_dims(
    tensor: &TensorView<'_>,
    expected_cols: usize,
) -> Result<(usize, usize), K3Error> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("expected 2D tensor, got shape {shape:?}"),
        });
    }
    let (d0, d1) = (shape[0], shape[1]);
    let (rows, cols) = if d0 == expected_cols {
        (d1, d0)
    } else if d1 == expected_cols {
        (d0, d1)
    } else {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "2D tensor shape {shape:?} does not match expected cols {expected_cols}"
            ),
        });
    };
    let row_bytes =
        quant_row_bytes(cols, tensor.quant()).ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "cols {cols} incompatible with quant {:?} on tensor '{}'",
                tensor.quant(),
                tensor.name()
            ),
        })?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: "matrix byte length overflow".to_owned(),
        })?;
    if tensor.as_bytes().len() < expected_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: expected_bytes,
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "buffer len {} smaller than required {expected_bytes} bytes",
                tensor.as_bytes().len()
            ),
        });
    }
    Ok((rows, cols))
}

/// Decodes a single row `row` (`cols` elements) from a 2D weight/embedding tensor into `out`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `row` is out of bounds or `out.len() != cols`,
/// or [`K3Error::CapabilityMismatch`] if the quantization format is unsupported.
pub fn decode_row_to_f32(
    tensor: &TensorView<'_>,
    row: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    decode_row_to_f32_with_sanitize(tensor, row, cols, out, true)
}

fn decode_row_to_f32_with_sanitize(
    tensor: &TensorView<'_>,
    row: usize,
    cols: usize,
    out: &mut [f32],
    sanitize_large: bool,
) -> Result<(), K3Error> {
    if out.len() != cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("output row len {} != cols {cols}", out.len()),
        });
    }
    let quant = tensor.quant();
    let row_bytes = quant_row_bytes(cols, quant).ok_or_else(|| K3Error::CapabilityMismatch {
        architecture: "llama".to_owned(),
        reason: format!("unsupported quant {quant:?} or unaligned cols {cols}"),
    })?;
    let start = row
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: row_bytes,
            buffer_len: tensor.as_bytes().len(),
            reason: "row byte offset overflow".to_owned(),
        })?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: start,
            byte_len: row_bytes,
            buffer_len: tensor.as_bytes().len(),
            reason: "row byte end overflow".to_owned(),
        })?;
    let raw = tensor.as_bytes();
    if end > raw.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: start,
            byte_len: row_bytes,
            buffer_len: raw.len(),
            reason: format!(
                "row {row} range [{start}..{end}] exceeds tensor len {}",
                raw.len()
            ),
        });
    }
    decode_row_payload_to_f32(quant, &raw[start..end], out, sanitize_large)
}

fn decode_blocks_to_f32<const N: usize, const B: usize>(
    row_raw: &[u8],
    out: &mut [f32],
    dequant: fn(&[u8; B], &mut [f32; N]),
) {
    let mut buf = [0.0_f32; N];
    for (b_idx, chunk) in row_raw.chunks_exact(B).enumerate() {
        if let Ok(block) = <&[u8; B]>::try_from(chunk) {
            dequant(block, &mut buf);
            out[b_idx * N..(b_idx + 1) * N].copy_from_slice(&buf);
        }
    }
}

fn decode_row_payload_to_f32(
    quant: QuantType,
    row_raw: &[u8],
    out: &mut [f32],
    sanitize_large: bool,
) -> Result<(), K3Error> {
    match quant {
        QuantType::F32 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let b = &row_raw[i * 4..i * 4 + 4];
                let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                *slot = if sanitize_large {
                    sanitize_f32_weight(val)
                } else if val.is_finite() {
                    val
                } else {
                    0.0
                };
            }
        }
        QuantType::F16 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let b = &row_raw[i * 2..i * 2 + 2];
                *slot = f16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        QuantType::BF16 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let b = &row_raw[i * 2..i * 2 + 2];
                *slot = bf16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        QuantType::Q8_0 => decode_blocks_to_f32::<32, 34>(row_raw, out, dequantize_q8_0_block),
        QuantType::Q4_0 => decode_blocks_to_f32::<32, 18>(row_raw, out, dequantize_q4_0_block),
        QuantType::Q5_0 => decode_blocks_to_f32::<32, 22>(row_raw, out, dequantize_q5_0_block),
        QuantType::Q4_K_M => {
            decode_blocks_to_f32::<256, 144>(row_raw, out, dequantize_q4_k_m_block);
        }
        QuantType::Q5_K => decode_blocks_to_f32::<256, 176>(row_raw, out, dequantize_q5_k_block),
        QuantType::Q6_K => decode_blocks_to_f32::<256, 210>(row_raw, out, dequantize_q6_k_block),
        QuantType::Q4_1
        | QuantType::Q5_1
        | QuantType::Q8_1
        | QuantType::Q2_K
        | QuantType::Q3_K
        | QuantType::IQ4_NL
        | QuantType::TQ1_0
        | QuantType::TQ2_0
        | QuantType::I2_S
        | QuantType::MXFP4 => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "llama".to_owned(),
                reason: format!("unsupported quant {quant:?}"),
            });
        }
    }
    Ok(())
}

/// `128-byte`-aligned 1D `f32` vector decoded once at model load / cache construction.
///
/// Used to cache `attn_norm`, `ffn_norm`, `output_norm`, and optional `qwen2` `attn_{q,k,v,output}.bias`
/// tensors without allocating or decoding per layer or per token step.
#[derive(Debug, Clone, PartialEq)]
pub struct Aligned128F32 {
    blocks: Vec<Aligned128>,
    len: usize,
}

impl Aligned128F32 {
    /// Decodes a 1D tensor (such as `attn_norm.weight`, `ffn_norm.weight`, `output_norm.weight`,
    /// or `qwen2` `attn_{q,k,v,output}.bias`) directly into `128-byte`-aligned `f32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `tensor.num_elements() != expected_len` or if
    /// backing byte bounds do not match, or [`K3Error::CapabilityMismatch`] on unsupported quants.
    pub fn from_tensor(
        tensor: &TensorView<'_>,
        expected_len: usize,
        sanitize_large: bool,
    ) -> Result<Self, K3Error> {
        let actual = tensor.num_elements();
        if actual != expected_len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor.name().to_owned(),
                offset: tensor.offset(),
                byte_len: tensor.byte_len(),
                buffer_len: tensor.as_bytes().len(),
                reason: format!("expected 1D len {expected_len}, got {actual}"),
            });
        }
        let n_blocks = expected_len.div_ceil(F32_PER_ALIGNED128);
        let mut blocks = vec![Aligned128::zeroed(); n_blocks];
        if expected_len > 0 {
            let dst = Aligned128::slice_as_f32_mut(&mut blocks);
            decode_row_to_f32_with_sanitize(
                tensor,
                0,
                expected_len,
                &mut dst[..expected_len],
                sanitize_large,
            )?;
        }
        DECODE_1D_TENSOR_COUNT.with(|c| c.set(c.get().saturating_add(1)));
        Ok(Self {
            blocks,
            len: expected_len,
        })
    }

    /// Wraps an in-memory `&[f32]` slice in `128-byte`-aligned storage.
    #[must_use]
    pub fn from_f32_slice(values: &[f32]) -> Self {
        let n_blocks = values.len().div_ceil(F32_PER_ALIGNED128);
        let mut blocks = vec![Aligned128::zeroed(); n_blocks];
        if !values.is_empty() {
            let dst = Aligned128::slice_as_f32_mut(&mut blocks);
            dst[..values.len()].copy_from_slice(values);
        }
        Self {
            blocks,
            len: values.len(),
        }
    }

    /// Borrows the exact logical `len()` `f32` slice (`self.len() <= blocks.len() * 32`).
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &Aligned128::slice_as_f32(&self.blocks)[..self.len]
    }

    /// Returns the raw `128-byte`-aligned base pointer for vector kernels.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.blocks.as_ptr().cast::<f32>()
    }

    /// Returns the number of `f32` elements stored.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when zero elements are stored.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the `128-byte`-aligned backing buffer size in bytes.
    #[must_use]
    #[inline]
    pub fn aligned_bytes(&self) -> usize {
        self.blocks.len() * 128
    }
}

/// Decodes a 1D vector tensor (such as `attn_norm.weight` or Qwen2 `attn_q.bias`) to `f32`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `tensor.num_elements() != expected_len`.
pub fn decode_1d_f32(
    tensor: &TensorView<'_>,
    expected_len: usize,
    sanitize_large: bool,
) -> Result<Vec<f32>, K3Error> {
    let aligned = Aligned128F32::from_tensor(tensor, expected_len, sanitize_large)?;
    Ok(aligned.as_slice().to_vec())
}

/// Pre-quantized `[M, K]` activation batch shared across multi-projection layers (`Q/K/V`, `gate/up`),
/// delegating to `jero_kernels_rvv::repack::QuantizedActQ8` when `with-kernels` is enabled.
#[derive(Debug, Clone, Default)]
pub struct QuantizedBatchAct {
    inner: QuantizedActQ8,
}

impl QuantizedBatchAct {
    /// Creates an empty reusable [`QuantizedBatchAct`] container for multi-layer prefill passes.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: QuantizedActQ8::new(),
        }
    }

    /// Quantizes `[m, cols]` activations into `self`, reusing preallocated Q8/F32 scratch storage.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] when `m == 0`, `cols == 0`, or `input.len() != m * cols`.
    pub fn quantize_into(&mut self, input: &[f32], m: usize, cols: usize) -> Result<(), K3Error> {
        self.inner.quantize_into(input, m, cols)
    }

    /// Quantizes an `f32` activation batch `[m, cols]` once into `128-byte`-aligned Q8 tiles
    /// (plus sanitized `f32` rows for dense `F32`/`F16`/`BF16` projections).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] when `m == 0`, `cols == 0`, or `input.len() != m * cols`.
    pub fn try_quantize(input: &[f32], m: usize, cols: usize) -> Result<Self, K3Error> {
        Ok(Self {
            inner: QuantizedActQ8::quantize(input, m, cols)?,
        })
    }

    /// Legacy compatibility constructor wrapping [`Self::try_quantize`].
    ///
    /// # Panics
    ///
    /// Panics if `m == 0`, `cols == 0`, or `input.len() != m * cols`.
    #[must_use]
    pub fn quantize(input: &[f32], m: usize, cols: usize, _group_size: usize) -> Self {
        Self {
            inner: QuantizedActQ8::quantize(input, m, cols)
                .expect("QuantizedBatchAct::quantize requires non-empty input.len() == m * cols"),
        }
    }

    /// Returns the quantized group size (`32`).
    #[must_use]
    pub const fn group_size(&self) -> usize {
        32
    }

    /// Returns the token count `M`.
    #[must_use]
    pub fn m(&self) -> usize {
        self.inner.m()
    }

    /// Returns the input feature count `K`.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.inner.k()
    }
}

#[derive(Debug, Clone)]
enum CachedStorage {
    Repacked(RepackedWeight),
    StreamingDense { quant: QuantType, raw: Vec<u8> },
}

/// Cached `128-byte`-aligned representation of a weight matrix (`[rows, cols]`).
///
/// Delegates quantized and bounded dense matrices to `jero_kernels_rvv::repack::RepackedWeight`
/// (`Q4_0`, `Q4_K_M`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`, `F32`, `F16`, `BF16`) while enforcing
/// `Llama`-recipe quantization admission and preserving compact `StreamingDense` storage above
/// `MAX_DENSE_HALF_ELEMENTS` / `MAX_DENSE_F32_ELEMENTS`.
#[derive(Debug, Clone)]
pub struct CachedMatrix {
    rows: usize,
    cols: usize,
    quant: QuantType,
    storage: CachedStorage,
}

impl CachedMatrix {
    /// Unpacks and repacks `tensor` (`[rows, cols]`) into `128-byte`-aligned tiles or compact storage.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] for unsupported quantizations or
    /// [`K3Error::TensorBoundsError`] when dimensions or byte lengths do not match.
    pub fn repack(tensor: &TensorView<'_>, rows: usize, cols: usize) -> Result<Self, K3Error> {
        let quant = tensor.quant();
        if quant_block_geometry(quant).is_none() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "llama".to_owned(),
                reason: format!("unsupported quant {quant:?}"),
            });
        }
        let row_bytes = quant_row_bytes(cols, quant).ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("cols {cols} incompatible with quant {quant:?}"),
        })?;
        let expected_bytes =
            rows.checked_mul(row_bytes)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: tensor.offset(),
                    byte_len: tensor.byte_len(),
                    buffer_len: tensor.as_bytes().len(),
                    reason: "matrix byte length overflow".to_owned(),
                })?;
        let raw = tensor.as_bytes();
        if rows == 0 || raw.len() < expected_bytes {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor.name().to_owned(),
                offset: tensor.offset(),
                byte_len: expected_bytes,
                buffer_len: raw.len(),
                reason: format!(
                    "tensor '{}' rows={rows} cols={cols} buf len {} < required {expected_bytes}",
                    tensor.name(),
                    raw.len()
                ),
            });
        }

        if is_oversized_dense(quant, rows, cols) {
            return Ok(Self {
                rows,
                cols,
                quant,
                storage: CachedStorage::StreamingDense {
                    quant,
                    raw: raw[..expected_bytes].to_vec(),
                },
            });
        }

        let repacked = RepackedWeight::from_tensor_with_half_limit(
            tensor,
            rows,
            cols,
            MAX_DENSE_F32_ELEMENTS,
        )?;
        Ok(Self {
            rows,
            cols,
            quant,
            storage: CachedStorage::Repacked(repacked),
        })
    }

    /// Returns the output row count (`out_features`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the input feature count (`in_features`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the source tensor quantization format.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Returns the activation quantization group size (`16` for `Q6_K`, `32` for `32`-elem/other
    /// K-quants, or `None` for dense `F32`/`F16`/`BF16`).
    #[must_use]
    pub const fn quant_group_size(&self) -> Option<usize> {
        match self.quant {
            QuantType::Q6_K => Some(16),
            QuantType::Q4_0
            | QuantType::Q5_0
            | QuantType::Q8_0
            | QuantType::Q4_K_M
            | QuantType::Q5_K => Some(32),
            _ => None,
        }
    }

    /// Returns `true` when backed by `RepackedWeight` (`jero_kernels_rvv::repack` under
    /// `with-kernels`) rather than `StreamingDense` fallback.
    #[must_use]
    pub const fn uses_repacked_storage(&self) -> bool {
        matches!(self.storage, CachedStorage::Repacked(_))
    }

    /// Returns total `128-byte`-aligned / backing bytes held by this cached matrix.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        match &self.storage {
            CachedStorage::Repacked(w) => w.aligned_bytes(),
            CachedStorage::StreamingDense { raw, .. } => raw.len(),
        }
    }

    /// Executes single-token GEMV (`[1, cols] x [rows, cols]^T -> [rows]`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] when dimensions do not match `self.rows()` / `self.cols()`.
    pub fn gemv_rows(
        &self,
        output: &mut [f32],
        input: &[f32],
        parallel: bool,
    ) -> Result<(), K3Error> {
        self.gemm_rows(output, input, 1, parallel)
    }

    /// Executes batched GEMM (`[m, cols] x [rows, cols]^T -> [m, rows]`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] when dimensions do not match.
    pub fn gemm_rows(
        &self,
        output: &mut [f32],
        input: &[f32],
        m: usize,
        parallel: bool,
    ) -> Result<(), K3Error> {
        self.project_batch(output, input, m, parallel)
    }

    /// Quantizes `[m, cols]` internally (when needed) and computes `[m, rows]` in `output`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
    pub fn project_batch(
        &self,
        output: &mut [f32],
        input: &[f32],
        m: usize,
        parallel: bool,
    ) -> Result<(), K3Error> {
        self.project_batch_with_act(output, input, None, m, parallel)
    }

    /// Projects `[m, cols]` into `[m, rows]`, reusing a pre-quantized [`QuantizedBatchAct`] when compatible.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `output.len() != m * rows` or `input.len() != m * cols`.
    pub fn project_batch_with_act(
        &self,
        output: &mut [f32],
        input: &[f32],
        pre_act: Option<&QuantizedBatchAct>,
        m: usize,
        parallel: bool,
    ) -> Result<(), K3Error> {
        let expected_out = m.saturating_mul(self.rows);
        let expected_in = m.saturating_mul(self.cols);
        if output.len() != expected_out || input.len() != expected_in {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "CachedMatrix".to_owned(),
                offset: 0,
                byte_len: output.len(),
                buffer_len: expected_out,
                reason: format!(
                    "CachedMatrix dimension mismatch: m={m}, rows={}, cols={}, in={}, out={}",
                    self.rows,
                    self.cols,
                    input.len(),
                    output.len()
                ),
            });
        }
        if m == 0 {
            return Ok(());
        }

        match &self.storage {
            CachedStorage::StreamingDense { quant, raw } => {
                project_streaming_dense(
                    output, input, raw, *quant, m, self.rows, self.cols, parallel,
                );
                Ok(())
            }
            CachedStorage::Repacked(repacked) => {
                let owned_act: QuantizedActQ8;
                let act: &QuantizedActQ8 = match pre_act {
                    Some(pa) if pa.m() == m && pa.cols() == self.cols => &pa.inner,
                    _ => {
                        owned_act = QuantizedActQ8::quantize(input, m, self.cols)?;
                        &owned_act
                    }
                };
                run_repacked_gemm_panels(repacked, output, act, m, self.rows, parallel)
            }
        }
    }
}

const MAX_STACK_PANEL_ELEMS: usize = PANEL_ROWS * 32;

thread_local! {
    static GEMM_PANEL_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct SharedMutOutput {
    ptr: *mut f32,
    cols_stride: usize,
    len: usize,
}

impl SharedMutOutput {
    fn new(output: &mut [f32], rows: usize) -> Self {
        Self {
            ptr: output.as_mut_ptr(),
            cols_stride: rows,
            len: output.len(),
        }
    }
}

// SAFETY: `run_repacked_gemm_panels` and tests partition columns `0..rows` into strictly disjoint
// ranges `[row_start..row_end)` across parallel workers, so no two threads ever access overlapping
// `output[tok * cols_stride + row]` elements.
unsafe impl Send for SharedMutOutput {}
// SAFETY: Disjoint column ranges across `0..cols_stride` guarantee conflict-free concurrent writes.
unsafe impl Sync for SharedMutOutput {}

fn scatter_panel_to_output(
    out: SharedMutOutput,
    panel: &[f32],
    m: usize,
    row_start: usize,
    sub_n: usize,
) {
    debug_assert_eq!(panel.len(), m * sub_n);
    debug_assert!(row_start + sub_n <= out.cols_stride);
    debug_assert_eq!(m * out.cols_stride, out.len);
    for tok in 0..m {
        let src = &panel[tok * sub_n..(tok + 1) * sub_n];
        let dst_offset = tok * out.cols_stride + row_start;
        debug_assert!(dst_offset + sub_n <= out.len);
        // SAFETY: Caller assigned each worker a strictly disjoint column range
        // `[row_start..row_start + sub_n)`, `dst_offset + sub_n <= out.len`, and `panel` lives in
        // caller/worker scratch storage disjoint from `out.ptr`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), out.ptr.add(dst_offset), sub_n);
        }
    }
}

fn with_worker_panel_scratch<R>(needed_elems: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    if needed_elems <= MAX_STACK_PANEL_ELEMS {
        let mut stack_buf = [0.0_f32; MAX_STACK_PANEL_ELEMS];
        return f(&mut stack_buf[..needed_elems]);
    }
    GEMM_PANEL_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.len() < needed_elems {
            guard.resize(needed_elems, 0.0_f32);
        }
        f(&mut guard[..needed_elems])
    })
}

fn run_repacked_gemm_panels(
    repacked: &RepackedWeight,
    output: &mut [f32],
    act: &QuantizedActQ8,
    m: usize,
    rows: usize,
    parallel: bool,
) -> Result<(), K3Error> {
    if m == 1 {
        #[cfg(not(miri))]
        if parallel && rows > PANEL_ROWS && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            return output.par_chunks_mut(PANEL_ROWS).enumerate().try_for_each(
                |(panel_idx, slice)| {
                    let row_start = panel_idx * PANEL_ROWS;
                    repacked.gemm_rows(slice, act, row_start..row_start + slice.len())
                },
            );
        }
        return repacked.gemm_rows(output, act, 0..rows);
    }

    #[cfg(not(miri))]
    if parallel && rows > PANEL_ROWS && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let n_panels = rows.div_ceil(PANEL_ROWS);
        let out_view = SharedMutOutput::new(output, rows);
        return (0..n_panels).into_par_iter().try_for_each(|panel_idx| {
            let row_start = panel_idx * PANEL_ROWS;
            let row_end = (row_start + PANEL_ROWS).min(rows);
            let sub_n = row_end - row_start;
            with_worker_panel_scratch(m * sub_n, |panel_buf| {
                repacked.gemm_rows(panel_buf, act, row_start..row_end)?;
                scatter_panel_to_output(out_view, panel_buf, m, row_start, sub_n);
                Ok(())
            })
        });
    }

    let _ = parallel;
    repacked.gemm_rows(output, act, 0..rows)
}

#[expect(
    clippy::too_many_arguments,
    reason = "streaming dense fallback operates directly on flat slice dimensions"
)]
fn project_streaming_dense(
    output: &mut [f32],
    input: &[f32],
    raw: &[u8],
    quant: QuantType,
    m: usize,
    rows: usize,
    cols: usize,
    parallel: bool,
) {
    let stride_bytes = match quant {
        QuantType::F32 => cols * 4,
        QuantType::F16 | QuantType::BF16 => cols * 2,
        _ => return,
    };

    if m == 1 {
        par_for_each_row_tile(output, PANEL_ROWS, parallel, |panel_idx, slice| {
            let row_start = panel_idx * PANEL_ROWS;
            let mut local_scratch = vec![0.0_f32; cols];
            for (local_r, slot) in slice.iter_mut().enumerate() {
                let row_raw = &raw[(row_start + local_r) * stride_bytes..][..stride_bytes];
                decode_dense_row_into(quant, row_raw, &mut local_scratch);
                *slot = dot_f32(&local_scratch, input);
            }
        });
        return;
    }

    let mut row_scratch = vec![0.0_f32; cols];
    for r in 0..rows {
        let row_raw = &raw[r * stride_bytes..(r + 1) * stride_bytes];
        decode_dense_row_into(quant, row_raw, &mut row_scratch);
        for tok in 0..m {
            output[tok * rows + r] = dot_f32(&row_scratch, &input[tok * cols..(tok + 1) * cols]);
        }
    }
}

#[inline]
fn decode_dense_row_into(quant: QuantType, row_raw: &[u8], scratch: &mut [f32]) {
    match quant {
        QuantType::F32 => {
            for (i, slot) in scratch.iter_mut().enumerate() {
                let b = &row_raw[i * 4..i * 4 + 4];
                *slot = sanitize_f32_weight(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
        }
        QuantType::F16 => {
            for (i, slot) in scratch.iter_mut().enumerate() {
                let b = &row_raw[i * 2..i * 2 + 2];
                *slot = f16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        QuantType::BF16 => {
            for (i, slot) in scratch.iter_mut().enumerate() {
                let b = &row_raw[i * 2..i * 2 + 2];
                *slot = bf16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        _ => {}
    }
}

/// Executes `op(panel_idx, chunk)` across contiguous `chunk_size` slices of `output`,
/// using `rayon` on the active cluster `WorkerPool` when `parallel && not(miri)` and
/// `rayon::current_num_threads() > 1`, or sequentially on the calling thread otherwise.
pub fn par_for_each_row_tile<F>(output: &mut [f32], chunk_size: usize, parallel: bool, op: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if output.is_empty() || chunk_size == 0 {
        return;
    }

    #[cfg(not(miri))]
    if parallel && output.len() > chunk_size && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        output
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(idx, slice)| op(idx, slice));
        return;
    }

    let _ = parallel;
    for (idx, slice) in output.chunks_mut(chunk_size).enumerate() {
        op(idx, slice);
    }
}

/// Pre-repacked per-layer weights and `128-byte`-aligned 1D `RMSNorm` / bias vectors owned by a [`ModelRepackCache`].
#[derive(Debug, Clone)]
pub struct LayerRepackCache {
    /// Pre-decoded `blk.{l}.attn_norm.weight` (`128-byte`-aligned).
    pub attn_norm: Aligned128F32,
    /// Pre-decoded `blk.{l}.ffn_norm.weight` (`128-byte`-aligned).
    pub ffn_norm: Aligned128F32,
    /// Repacked query projection `blk.{l}.attn_q.weight`.
    pub wq: Arc<CachedMatrix>,
    /// Repacked key projection `blk.{l}.attn_k.weight`.
    pub wk: Arc<CachedMatrix>,
    /// Repacked value projection `blk.{l}.attn_v.weight`.
    pub wv: Arc<CachedMatrix>,
    /// Repacked attention output projection `blk.{l}.attn_output.weight`.
    pub wo: Arc<CachedMatrix>,
    /// Pre-decoded optional `qwen2` query bias `blk.{l}.attn_q.bias` (`128-byte`-aligned).
    pub attn_q_bias: Option<Aligned128F32>,
    /// Pre-decoded optional `qwen2` key bias `blk.{l}.attn_k.bias` (`128-byte`-aligned).
    pub attn_k_bias: Option<Aligned128F32>,
    /// Pre-decoded optional `qwen2` value bias `blk.{l}.attn_v.bias` (`128-byte`-aligned).
    pub attn_v_bias: Option<Aligned128F32>,
    /// Pre-decoded optional attention output bias `blk.{l}.attn_output.bias` (`128-byte`-aligned).
    pub attn_output_bias: Option<Aligned128F32>,
    /// Repacked `SwiGLU` gate projection `blk.{l}.ffn_gate.weight`.
    pub wgate: Arc<CachedMatrix>,
    /// Repacked `SwiGLU` up projection `blk.{l}.ffn_up.weight`.
    pub wup: Arc<CachedMatrix>,
    /// Repacked `SwiGLU` down projection `blk.{l}.ffn_down.weight`.
    pub wdown: Arc<CachedMatrix>,
}

/// LM head weight retention mode inside a [`ModelRepackCache`].
#[derive(Debug, Clone)]
pub enum LmHeadCache {
    /// Compact/quantized `output.weight` or tied `token_embd.weight` held in `RepackedWeight`.
    Repacked(Arc<CachedMatrix>),
    /// Oversized dense (`F16`/`BF16`/`F32`) `output.weight` streamed zero-copy directly from mmap.
    StreamingUntied,
    /// Oversized dense (`F16`/`BF16`/`F32`) tied `token_embd.weight` streamed zero-copy directly from mmap.
    StreamingTied,
}

/// Model-scoped cache of repacked 2D layer matrices and `128-byte`-aligned 1D `RMSNorm`/bias vectors.
///
/// Bound explicitly to the lifetime of the loaded model instance: dropping [`ModelRepackCache`]
/// deterministically releases every repacked buffer without relying on process-global state.
#[derive(Debug, Clone)]
pub struct ModelRepackCache {
    architecture: &'static str,
    hidden_dim: usize,
    ffn_dim: usize,
    total_q: usize,
    total_kv: usize,
    vocab_size: usize,
    layers: Vec<LayerRepackCache>,
    output_norm: Aligned128F32,
    lm_head: LmHeadCache,
}

impl ModelRepackCache {
    /// Builds a model-scoped cache from `weights` using explicit resolved dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] when required
    /// tensors are missing, fail quantization admission, or mismatch expected shapes.
    #[expect(
        clippy::too_many_arguments,
        reason = "accepts resolved model geometry dimensions directly for model-scoped cache construction"
    )]
    pub fn build(
        architecture: &'static str,
        num_layers: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        total_q: usize,
        total_kv: usize,
        vocab_size: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Self, K3Error> {
        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let attn_norm_t = weights.tensor(&format!("blk.{l}.attn_norm.weight"))?;
            let attn_norm = Aligned128F32::from_tensor(&attn_norm_t, hidden_dim, true)?;

            let ffn_norm_t = weights.tensor(&format!("blk.{l}.ffn_norm.weight"))?;
            let ffn_norm = Aligned128F32::from_tensor(&ffn_norm_t, hidden_dim, true)?;

            let wq = repack_checked_2d(
                weights,
                &format!("blk.{l}.attn_q.weight"),
                hidden_dim,
                total_q,
            )?;
            let wk = repack_checked_2d(
                weights,
                &format!("blk.{l}.attn_k.weight"),
                hidden_dim,
                total_kv,
            )?;
            let wv = repack_checked_2d(
                weights,
                &format!("blk.{l}.attn_v.weight"),
                hidden_dim,
                total_kv,
            )?;
            let wo = repack_checked_2d(
                weights,
                &format!("blk.{l}.attn_output.weight"),
                total_q,
                hidden_dim,
            )?;

            let bias_query = decode_optional_1d(weights, &format!("blk.{l}.attn_q.bias"), total_q)?;
            let bias_key = decode_optional_1d(weights, &format!("blk.{l}.attn_k.bias"), total_kv)?;
            let bias_value =
                decode_optional_1d(weights, &format!("blk.{l}.attn_v.bias"), total_kv)?;
            let bias_output =
                decode_optional_1d(weights, &format!("blk.{l}.attn_output.bias"), hidden_dim)?;

            let wgate = repack_checked_2d(
                weights,
                &format!("blk.{l}.ffn_gate.weight"),
                hidden_dim,
                ffn_dim,
            )?;
            let wup = repack_checked_2d(
                weights,
                &format!("blk.{l}.ffn_up.weight"),
                hidden_dim,
                ffn_dim,
            )?;
            let wdown = repack_checked_2d(
                weights,
                &format!("blk.{l}.ffn_down.weight"),
                ffn_dim,
                hidden_dim,
            )?;

            layers.push(LayerRepackCache {
                attn_norm,
                ffn_norm,
                wq,
                wk,
                wv,
                wo,
                attn_q_bias: bias_query,
                attn_k_bias: bias_key,
                attn_v_bias: bias_value,
                attn_output_bias: bias_output,
                wgate,
                wup,
                wdown,
            });
        }

        let out_norm_t = weights.tensor("output_norm.weight")?;
        let output_norm = Aligned128F32::from_tensor(&out_norm_t, hidden_dim, true)?;

        let lm_head = if weights.contains_tensor("output.weight") {
            let t = weights.tensor("output.weight")?;
            let (rows, cols) = resolve_2d_dims(&t, hidden_dim)?;
            if is_oversized_dense(t.quant(), rows, cols) {
                LmHeadCache::StreamingUntied
            } else {
                LmHeadCache::Repacked(Arc::new(CachedMatrix::repack(&t, rows, cols)?))
            }
        } else {
            let t = weights.tensor("token_embd.weight")?;
            let (rows, cols) = resolve_2d_dims(&t, hidden_dim)?;
            if is_oversized_dense(t.quant(), rows, cols) {
                LmHeadCache::StreamingTied
            } else {
                LmHeadCache::Repacked(Arc::new(CachedMatrix::repack(&t, rows, cols)?))
            }
        };

        Ok(Self {
            architecture,
            hidden_dim,
            ffn_dim,
            total_q,
            total_kv,
            vocab_size,
            layers,
            output_norm,
            lm_head,
        })
    }

    /// Returns the architecture string (`"llama"`, `"llama4"`, or `"qwen2"`).
    #[must_use]
    pub const fn architecture(&self) -> &'static str {
        self.architecture
    }

    /// Returns the cached layer count.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the resolved hidden dimension (`d_model`).
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Returns the resolved feed-forward intermediate dimension (`d_ffn`).
    #[must_use]
    pub const fn ffn_dim(&self) -> usize {
        self.ffn_dim
    }

    /// Returns total query projection dimension (`n_heads_q * head_dim`).
    #[must_use]
    pub const fn total_q(&self) -> usize {
        self.total_q
    }

    /// Returns total key/value projection dimension (`n_heads_kv * head_dim`).
    #[must_use]
    pub const fn total_kv(&self) -> usize {
        self.total_kv
    }

    /// Returns the resolved vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the per-layer cache for `layer` (`0..self.num_layers()`).
    #[must_use]
    pub fn layer(&self, layer: usize) -> Option<&LayerRepackCache> {
        self.layers.get(layer)
    }

    /// Returns the pre-decoded `128-byte`-aligned final `output_norm.weight` vector.
    #[must_use]
    pub const fn output_norm(&self) -> &Aligned128F32 {
        &self.output_norm
    }

    /// Returns the cached or zero-copy streaming LM head handle.
    #[must_use]
    pub const fn lm_head(&self) -> &LmHeadCache {
        &self.lm_head
    }

    /// Returns the count of pre-decoded `128-byte`-aligned 1D norm tensors (`2 * num_layers + 1`).
    #[must_use]
    pub fn norm_count(&self) -> usize {
        self.layers.len() * 2 + 1
    }

    /// Returns the count of pre-decoded `128-byte`-aligned 1D optional bias tensors.
    #[must_use]
    pub fn bias_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                usize::from(l.attn_q_bias.is_some())
                    + usize::from(l.attn_k_bias.is_some())
                    + usize::from(l.attn_v_bias.is_some())
                    + usize::from(l.attn_output_bias.is_some())
            })
            .sum()
    }

    /// Returns the count of eagerly repacked 2D weight matrices owned by this cache.
    #[must_use]
    pub fn repacked_matrix_count(&self) -> usize {
        let lm = usize::from(matches!(self.lm_head, LmHeadCache::Repacked(_)));
        self.layers.len() * 7 + lm
    }

    /// Returns the total aligned bytes retained across 2D repacked matrices and 1D norm/bias vectors.
    #[must_use]
    pub fn total_aligned_bytes(&self) -> usize {
        let mut total = self.output_norm.aligned_bytes();
        for l in &self.layers {
            total += l.attn_norm.aligned_bytes() + l.ffn_norm.aligned_bytes();
            total += l
                .attn_q_bias
                .as_ref()
                .map_or(0, Aligned128F32::aligned_bytes);
            total += l
                .attn_k_bias
                .as_ref()
                .map_or(0, Aligned128F32::aligned_bytes);
            total += l
                .attn_v_bias
                .as_ref()
                .map_or(0, Aligned128F32::aligned_bytes);
            total += l
                .attn_output_bias
                .as_ref()
                .map_or(0, Aligned128F32::aligned_bytes);
            total += l.wq.aligned_bytes()
                + l.wk.aligned_bytes()
                + l.wv.aligned_bytes()
                + l.wo.aligned_bytes()
                + l.wgate.aligned_bytes()
                + l.wup.aligned_bytes()
                + l.wdown.aligned_bytes();
        }
        if let LmHeadCache::Repacked(ref lm) = self.lm_head {
            total += lm.aligned_bytes();
        }
        total
    }
}

fn repack_checked_2d(
    weights: &dyn ModelWeights,
    name: &str,
    expected_cols: usize,
    expected_rows: usize,
) -> Result<Arc<CachedMatrix>, K3Error> {
    let t = weights.tensor(name)?;
    let (rows, cols) = resolve_2d_dims(&t, expected_cols)?;
    if rows != expected_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: t.offset(),
            byte_len: t.byte_len(),
            buffer_len: t.as_bytes().len(),
            reason: format!("expected {expected_rows} rows for {name}, got {rows}"),
        });
    }
    Ok(Arc::new(CachedMatrix::repack(&t, rows, cols)?))
}

fn decode_optional_1d(
    weights: &dyn ModelWeights,
    name: &str,
    expected_len: usize,
) -> Result<Option<Aligned128F32>, K3Error> {
    if !weights.contains_tensor(name) {
        return Ok(None);
    }
    let t = weights.tensor(name)?;
    Ok(Some(Aligned128F32::from_tensor(&t, expected_len, false)?))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    offset: usize,
    byte_len: usize,
    ptr_addr: usize,
    content_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelAnchor {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    ptr_addr: usize,
    content_hash: u64,
}

#[derive(Default)]
struct ActiveModelFallbackCache {
    anchor: Option<ModelAnchor>,
    matrices: HashMap<CacheKey, Arc<CachedMatrix>>,
    vectors_1d: HashMap<CacheKey, Arc<Aligned128F32>>,
}

static ACTIVE_FALLBACK_CACHE: OnceLock<RwLock<ActiveModelFallbackCache>> = OnceLock::new();

fn fallback_cache() -> &'static RwLock<ActiveModelFallbackCache> {
    ACTIVE_FALLBACK_CACHE.get_or_init(|| RwLock::new(ActiveModelFallbackCache::default()))
}

fn hash_bytes_sample(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let len = bytes.len();
    let mix = |acc: &mut u64, slice: &[u8]| {
        for &b in slice {
            *acc ^= u64::from(b);
            *acc = acc.wrapping_mul(0x0100_0000_01b3);
        }
    };
    mix(&mut h, &len.to_le_bytes());
    if len <= 192 {
        mix(&mut h, bytes);
    } else {
        mix(&mut h, &bytes[..64]);
        let mid = len / 2 - 32;
        mix(&mut h, &bytes[mid..mid + 64]);
        mix(&mut h, &bytes[len - 64..]);
    }
    h
}

fn make_cache_key(tensor: &TensorView<'_>) -> CacheKey {
    let raw = tensor.as_bytes();
    CacheKey {
        name: tensor.name().to_owned(),
        quant: tensor.quant(),
        shape: tensor.shape().to_vec(),
        offset: tensor.offset(),
        byte_len: tensor.byte_len(),
        ptr_addr: raw.as_ptr() as usize,
        content_hash: hash_bytes_sample(raw),
    }
}

fn make_anchor(tensor: &TensorView<'_>) -> ModelAnchor {
    let raw = tensor.as_bytes();
    ModelAnchor {
        name: tensor.name().to_owned(),
        quant: tensor.quant(),
        shape: tensor.shape().to_vec(),
        ptr_addr: raw.as_ptr() as usize,
        content_hash: hash_bytes_sample(raw),
    }
}

/// Binds the active model instance (using its `token_embd.weight` descriptor & backing buffer)
/// for trait-method calls (`ArchRecipe::prefill_step`/`decode_step`) that do not pass an explicit
/// [`ModelRepackCache`]; evicts stale cached matrices and 1D vectors automatically on model switch.
pub fn set_active_model_anchor(anchor_tensor: &TensorView<'_>) {
    let next = make_anchor(anchor_tensor);
    let mut guard = fallback_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.anchor.as_ref() != Some(&next) {
        guard.matrices.clear();
        guard.vectors_1d.clear();
        guard.anchor = Some(next);
    }
}

/// Clears all entries in the trait-fallback active model cache.
pub fn clear_repack_cache() {
    let mut guard = fallback_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.anchor = None;
    guard.matrices.clear();
    guard.vectors_1d.clear();
}

/// Returns the number of cached 2D weight matrices in the active-model fallback cache.
#[must_use]
pub fn cached_matrix_count() -> usize {
    fallback_cache()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .matrices
        .len()
}

/// Returns the number of cached 1D norm/bias vectors in the active-model fallback cache.
#[must_use]
pub fn cached_vector_1d_count() -> usize {
    fallback_cache()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .vectors_1d
        .len()
}

/// Returns a cached repacked [`CachedMatrix`] for `tensor`, repacking once on cache miss.
///
/// # Errors
///
/// Returns [`K3Error`] if repacking fails validation or uses an unsupported quantization.
pub fn get_or_repack(
    tensor: &TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<Arc<CachedMatrix>, K3Error> {
    let key = make_cache_key(tensor);
    {
        let guard = fallback_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(hit) = guard.matrices.get(&key)
            && hit.rows() == rows
            && hit.cols() == cols
        {
            return Ok(Arc::clone(hit));
        }
    }

    let repacked = Arc::new(CachedMatrix::repack(tensor, rows, cols)?);
    let mut guard = fallback_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = guard
        .matrices
        .entry(key)
        .or_insert_with(|| Arc::clone(&repacked));
    Ok(Arc::clone(entry))
}

/// Returns a cached `128-byte`-aligned 1D [`Aligned128F32`] vector for `tensor`, decoding once per
/// active model instance so repeated prefill/decode calls allocate zero 1D `Vec<f32>` buffers.
///
/// # Errors
///
/// Returns [`K3Error`] if tensor bounds or quantization fail validation.
pub fn get_or_decode_1d_f32(
    tensor: &TensorView<'_>,
    expected_len: usize,
    sanitize_large: bool,
) -> Result<Arc<Aligned128F32>, K3Error> {
    let key = make_cache_key(tensor);
    {
        let guard = fallback_cache()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(hit) = guard.vectors_1d.get(&key)
            && hit.len() == expected_len
        {
            return Ok(Arc::clone(hit));
        }
    }

    let decoded = Arc::new(Aligned128F32::from_tensor(
        tensor,
        expected_len,
        sanitize_large,
    )?);
    let mut guard = fallback_cache()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = guard
        .vectors_1d
        .entry(key)
        .or_insert_with(|| Arc::clone(&decoded));
    Ok(Arc::clone(entry))
}

#[cfg(test)]
mod tests {
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{GgufFile, GgufValue, KvGeometry};

    use super::*;

    fn set_anchor_on(cache: &mut ActiveModelFallbackCache, anchor_tensor: &TensorView<'_>) {
        let next = make_anchor(anchor_tensor);
        if cache.anchor.as_ref() != Some(&next) {
            cache.matrices.clear();
            cache.vectors_1d.clear();
            cache.anchor = Some(next);
        }
    }

    fn get_or_repack_on(
        cache: &mut ActiveModelFallbackCache,
        tensor: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<Arc<CachedMatrix>, K3Error> {
        let key = make_cache_key(tensor);
        if let Some(hit) = cache.matrices.get(&key)
            && hit.rows() == rows
            && hit.cols() == cols
        {
            return Ok(Arc::clone(hit));
        }
        let repacked = Arc::new(CachedMatrix::repack(tensor, rows, cols)?);
        let entry = cache
            .matrices
            .entry(key)
            .or_insert_with(|| Arc::clone(&repacked));
        Ok(Arc::clone(entry))
    }

    fn get_or_decode_1d_on(
        cache: &mut ActiveModelFallbackCache,
        tensor: &TensorView<'_>,
        expected_len: usize,
        sanitize_large: bool,
    ) -> Result<Arc<Aligned128F32>, K3Error> {
        let key = make_cache_key(tensor);
        if let Some(hit) = cache.vectors_1d.get(&key)
            && hit.len() == expected_len
        {
            return Ok(Arc::clone(hit));
        }
        let decoded = Arc::new(Aligned128F32::from_tensor(
            tensor,
            expected_len,
            sanitize_large,
        )?);
        let entry = cache
            .vectors_1d
            .entry(key)
            .or_insert_with(|| Arc::clone(&decoded));
        Ok(Arc::clone(entry))
    }

    #[test]
    fn test_aligned128_layout_and_alignment() {
        assert_eq!(size_of::<Aligned128>(), 128);
        assert_eq!(align_of::<Aligned128>(), 128);
        let vec128 = vec![Aligned128::zeroed(); 4];
        let addr = vec128.as_ptr() as usize;
        assert_eq!(addr % 128, 0, "Aligned128 buffer must be 128-byte aligned");

        let f32_aligned = Aligned128F32::from_f32_slice(&[1.0_f32, -2.5, 3.25]);
        assert_eq!(f32_aligned.as_ptr() as usize % 128, 0);
        assert_eq!(f32_aligned.as_slice(), &[1.0_f32, -2.5, 3.25]);
    }

    #[test]
    fn test_repack_cache_aba_eviction_and_model_anchor() {
        let mut cache = ActiveModelFallbackCache::default();
        assert_eq!(cache.matrices.len(), 0);
        assert_eq!(cache.vectors_1d.len(), 0);

        let cols = 32_usize;
        let rows = 4_usize;
        let anchor_first = vec![1_u8; 4 * 32 * 4];
        let anchor_second = vec![2_u8; 4 * 32 * 4];
        let first_emb_view = TensorView::from_bytes(
            "token_embd.weight",
            QuantType::F32,
            &[32, 4],
            0,
            &anchor_first,
        )
        .unwrap();
        let second_emb_view = TensorView::from_bytes(
            "token_embd.weight",
            QuantType::F32,
            &[32, 4],
            0,
            &anchor_second,
        )
        .unwrap();

        let mat_a = vec![0_u8; rows * cols * 4];
        let mat_view =
            TensorView::from_bytes("blk.0.attn_q.weight", QuantType::F32, &[32, 4], 0, &mat_a)
                .unwrap();
        let norm_bytes = vec![0_u8; cols * 4];
        let norm_view = TensorView::from_bytes(
            "blk.0.attn_norm.weight",
            QuantType::F32,
            &[32],
            0,
            &norm_bytes,
        )
        .unwrap();

        set_anchor_on(&mut cache, &first_emb_view);
        let first = get_or_repack_on(&mut cache, &mat_view, rows, cols).unwrap();
        let norm_first = get_or_decode_1d_on(&mut cache, &norm_view, cols, true).unwrap();
        assert_eq!(cache.matrices.len(), 1);
        assert_eq!(cache.vectors_1d.len(), 1);

        let before = thread_decode_1d_count();
        let hit = get_or_repack_on(&mut cache, &mat_view, rows, cols).unwrap();
        let norm_hit = get_or_decode_1d_on(&mut cache, &norm_view, cols, true).unwrap();
        assert_eq!(thread_decode_1d_count(), before);
        assert!(Arc::ptr_eq(&first, &hit));
        assert!(Arc::ptr_eq(&norm_first, &norm_hit));

        // Switching active model anchor evicts stale matrices and 1D vectors cleanly
        set_anchor_on(&mut cache, &second_emb_view);
        assert_eq!(cache.matrices.len(), 0);
        assert_eq!(cache.vectors_1d.len(), 0);
    }

    fn build_compact_test_gguf(arch: &str) -> Vec<u8> {
        let kvs_owned = [
            (format!("{arch}.block_count"), GgufValue::Uint32(1)),
            (format!("{arch}.embedding_length"), GgufValue::Uint32(8)),
            (format!("{arch}.feed_forward_length"), GgufValue::Uint32(12)),
            (format!("{arch}.attention.head_count"), GgufValue::Uint32(2)),
            (
                format!("{arch}.attention.head_count_kv"),
                GgufValue::Uint32(1),
            ),
            (format!("{arch}.attention.key_length"), GgufValue::Uint32(4)),
            (
                format!("{arch}.rope.freq_base"),
                GgufValue::Float32(10_000.0),
            ),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                GgufValue::Float32(1e-5),
            ),
        ];
        let kvs_refs: Vec<(&str, GgufValue<'_>)> = kvs_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        let mut specs: Vec<(&str, QuantType, &[usize])> = vec![
            ("token_embd.weight", QuantType::F32, &[8, 12]),
            ("blk.0.attn_norm.weight", QuantType::F16, &[8]),
            ("blk.0.attn_q.weight", QuantType::F32, &[8, 8]),
            ("blk.0.attn_k.weight", QuantType::F32, &[8, 4]),
            ("blk.0.attn_v.weight", QuantType::F32, &[8, 4]),
            ("blk.0.attn_output.weight", QuantType::F32, &[8, 8]),
            ("blk.0.ffn_norm.weight", QuantType::BF16, &[8]),
            ("blk.0.ffn_gate.weight", QuantType::F32, &[8, 12]),
            ("blk.0.ffn_up.weight", QuantType::F32, &[8, 12]),
            ("blk.0.ffn_down.weight", QuantType::F32, &[12, 8]),
            ("output_norm.weight", QuantType::F32, &[8]),
            ("output.weight", QuantType::F32, &[8, 12]),
        ];
        if arch == "qwen2" {
            specs.push(("blk.0.attn_q.bias", QuantType::F16, &[8]));
            specs.push(("blk.0.attn_k.bias", QuantType::BF16, &[4]));
            specs.push(("blk.0.attn_v.bias", QuantType::F32, &[4]));
        }
        create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs_refs, &specs)
    }

    fn verify_model_repack_cache_for_arch(arch: &str) {
        let raw = build_compact_test_gguf(arch);
        let gguf = GgufFile::parse(&raw).unwrap();
        let recipe = crate::LlamaRecipe::from_gguf(&gguf).unwrap();
        let cache = recipe.build_repack_cache(&gguf).unwrap();

        assert_eq!(cache.norm_count(), 3);
        assert_eq!(cache.bias_count(), if arch == "qwen2" { 3 } else { 0 });
        let l0 = cache.layer(0).unwrap();
        assert_eq!(l0.attn_norm.as_ptr() as usize % 128, 0);
        assert_eq!(l0.ffn_norm.as_ptr() as usize % 128, 0);
        assert_eq!(cache.output_norm().as_ptr() as usize % 128, 0);
        if arch == "qwen2" {
            assert_eq!(l0.attn_q_bias.as_ref().unwrap().as_ptr() as usize % 128, 0);
            assert_eq!(l0.attn_k_bias.as_ref().unwrap().as_ptr() as usize % 128, 0);
            assert_eq!(l0.attn_v_bias.as_ref().unwrap().as_ptr() as usize % 128, 0);
        }

        let geom = KvGeometry::new(1, 1, 4, 4, None).unwrap();
        let kv_len = geom.required_f32_len().unwrap();
        let mut logits_cached = vec![0.0_f32; 12];
        let mut kv_cached = vec![0.0_f32; kv_len];
        let mut ctx_cached =
            jero_core::ExecutionContext::new(&[1, 2], 4, &mut logits_cached, &mut kv_cached)
                .unwrap()
                .with_kv_geometry(geom)
                .unwrap();

        let before_cached = thread_decode_1d_count();
        recipe
            .prefill_step_with_cache(&mut ctx_cached, &gguf, &cache)
            .unwrap();
        ctx_cached.prepare_decode(3).unwrap();
        recipe
            .decode_step_with_cache(&mut ctx_cached, &gguf, &cache)
            .unwrap();
        assert_eq!(
            thread_decode_1d_count(),
            before_cached,
            "cached prefill + decode must perform zero 1D decodes for {arch}"
        );

        let mut logits_trait = vec![0.0_f32; 12];
        let mut kv_trait = vec![0.0_f32; kv_len];
        let mut ctx_trait =
            jero_core::ExecutionContext::new(&[1, 2], 4, &mut logits_trait, &mut kv_trait)
                .unwrap()
                .with_kv_geometry(geom)
                .unwrap();
        jero_core::ArchRecipe::prefill_step(&recipe, &mut ctx_trait, &gguf).unwrap();
        ctx_trait.prepare_decode(3).unwrap();
        jero_core::ArchRecipe::decode_step(&recipe, &mut ctx_trait, &gguf).unwrap();

        for (a, b) in logits_cached.iter().zip(&logits_trait) {
            assert!(
                (a - b).abs() < 1.0e-5,
                "{arch} cached vs trait logits mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_oversized_dense_threshold_and_model_cache_contract() {
        assert!(is_oversized_dense(QuantType::F16, 128_256, 4096));
        assert!(is_oversized_dense(QuantType::F16, 151_936, 1536));
        assert!(is_oversized_dense(QuantType::BF16, 151_936, 1536));
        assert!(!is_oversized_dense(QuantType::Q4_0, 151_936, 1536));
        assert!(!is_oversized_dense(QuantType::F16, 64, 64));

        // Verify ModelRepackCache caches 128-byte-aligned 1D norms & qwen2 biases with zero per-step decodes
        if !cfg!(miri) {
            verify_model_repack_cache_for_arch("llama");
        }
        verify_model_repack_cache_for_arch("qwen2");
    }

    #[test]
    fn test_scatter_panel_to_output_disjoint_columns_miri_safe() {
        let m = 2_usize;
        let rows = 68_usize;
        let cols = 8_usize;
        let mut raw = vec![0_u8; rows * cols * 4];
        for r in 0..rows {
            for c in 0..cols {
                #[expect(clippy::cast_precision_loss, reason = "small test bounds fit in f32")]
                let v = ((r * 7 + c * 3) % 29) as f32 * 0.05 - 0.5;
                let off = (r * cols + c) * 4;
                raw[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let dims = [cols, rows];
        let view = TensorView::from_bytes("proj.weight", QuantType::F32, &dims, 0, &raw).unwrap();
        let cached = CachedMatrix::repack(&view, rows, cols).unwrap();
        let repacked = match &cached.storage {
            CachedStorage::Repacked(w) => w,
            CachedStorage::StreamingDense { .. } => panic!("expected Repacked storage"),
        };
        let input: Vec<f32> = (0..m * cols)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test bounds fit in f32")]
                let val = (i % 13) as f32 * 0.1 - 0.6;
                val
            })
            .collect();
        let mut act = QuantizedBatchAct::new();
        act.quantize_into(&input, m, cols).unwrap();

        let mut expected = vec![0.0_f32; m * rows];
        repacked
            .gemm_rows(&mut expected, &act.inner, 0..rows)
            .unwrap();

        let mut actual = vec![0.0_f32; m * rows];
        let out_view = SharedMutOutput::new(&mut actual, rows);
        let act_ref = &act.inner;
        std::thread::scope(|scope| {
            let h0 = scope.spawn(move || {
                with_worker_panel_scratch(m * PANEL_ROWS, |buf| {
                    repacked.gemm_rows(buf, act_ref, 0..PANEL_ROWS).unwrap();
                    scatter_panel_to_output(out_view, buf, m, 0, PANEL_ROWS);
                });
            });
            let h1 = scope.spawn(move || {
                let sub_n = rows - PANEL_ROWS;
                with_worker_panel_scratch(m * sub_n, |buf| {
                    repacked.gemm_rows(buf, act_ref, PANEL_ROWS..rows).unwrap();
                    scatter_panel_to_output(out_view, buf, m, PANEL_ROWS, sub_n);
                });
            });
            h0.join().unwrap();
            h1.join().unwrap();
        });

        assert_eq!(actual, expected);
        // Exercise thread-local fallback path for oversized M without allocating per call on reuse.
        with_worker_panel_scratch(MAX_STACK_PANEL_ELEMS + 4, |buf| {
            assert_eq!(buf.len(), MAX_STACK_PANEL_ELEMS + 4);
        });
    }
}
