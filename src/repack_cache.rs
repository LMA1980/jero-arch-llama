//! Load-time `128-byte` vector-aligned weight repacking, bounded model-scoped cache,
//! dynamic `f32 -> Q8` batch activation quantization, and row-parallel GEMV/GEMM execution.
//!
//! Each [`Aligned128`] unit is aligned to 128 bytes (`#[repr(C, align(128))]`), satisfying:
//! - `4 x 256-bit` vector loads on `X100` (`zvl256b`, `vlenb = 32`),
//! - `1 x 1024-bit` vector load on `A100` (`zvl1024b`, `vlenb = 128`), and
//! - `1 x 128-byte` `8 x 16` `int8` `XSMTVDotII` (`smt.vmadot`) tile on `A100`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(all(feature = "with-kernels", any(not(target_arch = "riscv64"), miri)))]
use jero_kernels_rvv::ime::dot_i8_scalar;
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::ime::{TILE_BYTES, TILE_K, TILE_M, TILE_N};
#[cfg(all(feature = "with-kernels", target_arch = "riscv64", not(miri)))]
use jero_kernels_rvv::ime::{TILE_C, gemm_i8_ime};
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, VectorKernel, bf16_to_f32_finite,
    dequantize_q4_k_m_block, dequantize_q6_k_block, dequantize_q8_0_block, f16_to_f32_finite,
};

#[cfg(not(feature = "with-kernels"))]
const TILE_M: usize = 8;
#[cfg(not(feature = "with-kernels"))]
const TILE_N: usize = 8;
#[cfg(not(feature = "with-kernels"))]
const TILE_K: usize = 16;
#[cfg(not(feature = "with-kernels"))]
const TILE_BYTES: usize = 128;
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

/// Number of scalar elements per `Q4_0` block.
pub const Q4_0_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q4_0` block (2B `f16` scale + 16B nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Number of output rows per parallel row-panel (`8 x TILE_N = 64` rows).
pub const PANEL_ROWS: usize = 64;

/// Maximum scalar element count pre-decoded into `DenseF32` (`32 MiB`) before switching to
/// compact `DenseHalf` (`16-bit`) storage so resident cache footprint stays bounded.
const MAX_DENSE_F32_ELEMENTS: usize = 8_388_608;

/// Maximum scalar element count cached in `DenseHalf` (`64 MiB`) before streaming directly
/// from the zero-copy [`TensorView`].
const MAX_DENSE_HALF_ELEMENTS: usize = 33_554_432;

/// Maximum number of named tensors retained in [`ModelRepackCache`] at one time.
const MAX_CACHED_TENSORS: usize = 2048;

/// A 128-byte-aligned memory unit matching `4 x zvl256b` (`X100`), `1 x zvl1024b` (`A100`),
/// and one `8 x 16` `int8` `XSMTVDotII` (`IME`) tile (`128` bytes).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aligned128(pub [u8; 128]);

impl Default for Aligned128 {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Aligned128 {
    /// Returns a zero-initialized 128-byte-aligned block.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self([0_u8; 128])
    }

    /// Returns the underlying 128-byte slice as signed `i8` elements.
    #[must_use]
    #[inline]
    pub fn as_i8(&self) -> &[i8; 128] {
        // SAFETY: `Aligned128` wraps `[u8; 128]` with alignment 128 and no padding; `[i8; 128]`
        // has alignment 1, identical size (128), and every bit pattern is valid.
        unsafe { &*self.0.as_ptr().cast::<[i8; 128]>() }
    }

    /// Returns the underlying 128-byte slice as mutable signed `i8` elements.
    #[inline]
    pub fn as_i8_mut(&mut self) -> &mut [i8; 128] {
        // SAFETY: `Aligned128` wraps `[u8; 128]` with alignment 128 and no padding; `[i8; 128]`
        // has alignment 1, identical size (128), and every bit pattern is valid.
        unsafe { &mut *self.0.as_mut_ptr().cast::<[i8; 128]>() }
    }
}

/// 128-byte-aligned block of 32 `f32` values (`128` bytes).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
struct AlignedF32Block([f32; 32]);

impl Default for AlignedF32Block {
    fn default() -> Self {
        Self([0.0_f32; 32])
    }
}

/// 128-byte-aligned block of 64 `u16` (`f16` or `bf16`) values (`128` bytes).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AlignedU16Block([u16; 64]);

impl Default for AlignedU16Block {
    fn default() -> Self {
        Self([0_u16; 64])
    }
}

#[cfg(all(feature = "with-kernels", target_arch = "riscv64", not(miri)))]
#[inline]
fn aligned_tiles_as_i8(tiles: &[Aligned128]) -> &[i8] {
    // SAFETY: `Aligned128` is `#[repr(C, align(128))]` wrapping `[u8; 128]` with size 128 and
    // zero padding; `i8` has alignment 1 and size 1, preserving 128-byte base alignment.
    unsafe { core::slice::from_raw_parts(tiles.as_ptr().cast::<i8>(), tiles.len() * TILE_BYTES) }
}

#[inline]
fn aligned_f32_as_slice(blocks: &[AlignedF32Block]) -> &[f32] {
    // SAFETY: `AlignedF32Block` is `#[repr(C, align(128))]` wrapping `[f32; 32]` with size 128
    // and zero padding; `f32` has alignment 4 and size 4.
    unsafe { core::slice::from_raw_parts(blocks.as_ptr().cast::<f32>(), blocks.len() * 32) }
}

#[inline]
fn aligned_u16_as_slice(blocks: &[AlignedU16Block]) -> &[u16] {
    // SAFETY: `AlignedU16Block` is `#[repr(C, align(128))]` wrapping `[u16; 64]` with size 128
    // and zero padding; `u16` has alignment 2 and size 2.
    unsafe { core::slice::from_raw_parts(blocks.as_ptr().cast::<u16>(), blocks.len() * 64) }
}

#[cfg(not(feature = "with-kernels"))]
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
    let e = exp + 127 - 15;
    f32::from_bits((sign << 31) | (e << 23) | (mant << 13))
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
pub(crate) fn bf16_to_f32_finite(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

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

/// Dequantizes a single 18-byte `Q4_0` block (2B `f16` scale + 16B packed nibbles) into 32 `f32`s.
pub fn dequantize_q4_0_block(block: &[u8; Q4_0_BLOCK_BYTES], out: &mut [f32; Q4_0_BLOCK_ELEMENTS]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = f16_to_f32_finite(d_bits);
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

#[cfg(not(feature = "with-kernels"))]
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

#[cfg(not(feature = "with-kernels"))]
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
        let low_base = half_idx * 64;
        let high_base = half_idx * 32;
        let sc_base = half_idx * 8;
        for l in 0..32 {
            let is = l / 16;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
            let q1 = (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8) - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
            let q2 = (((ql[low_base + l + 32] & 0x0F) | (((qh[high_base + l] >> 2) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
            let q3 = ((((ql[low_base + l] >> 4) & 0x0F) | (((qh[high_base + l] >> 4) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit quant offset by -32")]
            let q4 = ((((ql[low_base + l + 32] >> 4) & 0x0F)
                | (((qh[high_base + l] >> 6) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc1 = f32::from(scales[sc_base + is] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc2 = f32::from(scales[sc_base + is + 2] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc3 = f32::from(scales[sc_base + is + 4] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc4 = f32::from(scales[sc_base + is + 6] as i8);
            out[n + l] = d * sc1 * f32::from(q1);
            out[n + l + 32] = d * sc2 * f32::from(q2);
            out[n + l + 64] = d * sc3 * f32::from(q3);
            out[n + l + 96] = d * sc4 * f32::from(q4);
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
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
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);
        let out_idx = chunk * 64;
        let q_offset = chunk * 32;
        for l in 0..32 {
            out[out_idx + l] = d1 * f32::from(qs[q_offset + l] & 0x0F) - m1;
            out[out_idx + 32 + l] = d2 * f32::from((qs[q_offset + l] >> 4) & 0x0F) - m2;
        }
    }
}

/// Unpacks the 12-byte 6-bit scale and min table of a `Q4_K_M` super-block.
#[must_use]
#[inline]
pub(crate) fn unpack_q4_k_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
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

/// Returns `(block_elements, block_bytes)` for supported `Llama` / `Qwen2` formats.
#[must_use]
pub const fn quant_block_geometry(quant: QuantType) -> Option<(usize, usize)> {
    match quant {
        QuantType::F32 => Some((1, 4)),
        QuantType::F16 | QuantType::BF16 => Some((1, 2)),
        QuantType::Q8_0 => Some((Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES)),
        QuantType::Q4_0 => Some((Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES)),
        QuantType::Q4_K_M => Some((Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES)),
        QuantType::Q6_K => Some((Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES)),
        QuantType::Q5_K | QuantType::TQ1_0 | QuantType::TQ2_0 => None,
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
    let row_raw = &raw[start..end];
    decode_row_payload_to_f32(quant, row_raw, out)
}

fn decode_row_payload_to_f32(
    quant: QuantType,
    row_raw: &[u8],
    out: &mut [f32],
) -> Result<(), K3Error> {
    match quant {
        QuantType::F32 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let b = &row_raw[i * 4..i * 4 + 4];
                *slot = sanitize_f32_weight(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
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
        QuantType::Q8_0 => {
            let mut buf = [0.0_f32; Q8_0_BLOCK_ELEMENTS];
            for (b_idx, chunk) in row_raw.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
                if let Ok(block) = <&[u8; Q8_0_BLOCK_BYTES]>::try_from(chunk) {
                    dequantize_q8_0_block(block, &mut buf);
                    out[b_idx * 32..(b_idx + 1) * 32].copy_from_slice(&buf);
                }
            }
        }
        QuantType::Q4_0 => {
            let mut buf = [0.0_f32; Q4_0_BLOCK_ELEMENTS];
            for (b_idx, chunk) in row_raw.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
                if let Ok(block) = <&[u8; Q4_0_BLOCK_BYTES]>::try_from(chunk) {
                    dequantize_q4_0_block(block, &mut buf);
                    out[b_idx * 32..(b_idx + 1) * 32].copy_from_slice(&buf);
                }
            }
        }
        QuantType::Q4_K_M => {
            let mut buf = [0.0_f32; Q4_K_M_BLOCK_ELEMENTS];
            for (b_idx, chunk) in row_raw.chunks_exact(Q4_K_M_BLOCK_BYTES).enumerate() {
                if let Ok(block) = <&[u8; Q4_K_M_BLOCK_BYTES]>::try_from(chunk) {
                    dequantize_q4_k_m_block(block, &mut buf);
                    out[b_idx * 256..(b_idx + 1) * 256].copy_from_slice(&buf);
                }
            }
        }
        QuantType::Q6_K => {
            let mut buf = [0.0_f32; Q6_K_BLOCK_ELEMENTS];
            for (b_idx, chunk) in row_raw.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
                if let Ok(block) = <&[u8; Q6_K_BLOCK_BYTES]>::try_from(chunk) {
                    dequantize_q6_k_block(block, &mut buf);
                    out[b_idx * 256..(b_idx + 1) * 256].copy_from_slice(&buf);
                }
            }
        }
        QuantType::Q5_K | QuantType::TQ1_0 | QuantType::TQ2_0 => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "llama".to_owned(),
                reason: format!("unsupported quant {quant:?}"),
            });
        }
    }
    Ok(())
}

/// Decodes a 1D vector tensor (such as `attn_norm.weight` or Qwen2 `attn_q.bias`) to `f32`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `tensor.num_elements() != expected_len`.
pub fn decode_1d_f32(
    tensor: &TensorView<'_>,
    expected_len: usize,
    is_norm_weight: bool,
) -> Result<Vec<f32>, K3Error> {
    if tensor.num_elements() != expected_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "1D tensor '{}' len {} != expected {expected_len}",
                tensor.name(),
                tensor.num_elements()
            ),
        });
    }
    let mut out = vec![0.0_f32; expected_len];
    decode_row_to_f32(tensor, 0, expected_len, &mut out)?;
    if is_norm_weight {
        for v in &mut out {
            if !v.is_finite() || v.abs() > 1.0e4 {
                *v = 1.0;
            }
        }
    }
    Ok(out)
}

/// Dynamically quantized `[M, K]` `Q8` activation batch aligned to `128-byte` `8 x 16` tiles.
#[derive(Debug, Clone)]
pub struct QuantizedBatchAct {
    m: usize,
    cols: usize,
    group_size: usize,
    m_tiles: usize,
    /// Packed `A` tiles in `[group: 0..num_groups][m_tile: 0..m_tiles][chunk: 0..group_chunks]` order.
    a_tiles: Vec<Aligned128>,
    /// Per-(group, `padded_m`) activation scale (`absmax / 127.0`).
    x_scales: Vec<f32>,
    /// Exact floating-point sum $\sum_{k \in g} x[m, k]$ for asymmetric min-offset correction.
    x_sums: Vec<f32>,
}

impl QuantizedBatchAct {
    /// Quantizes `[m, cols]` row-major `f32` activations into `group_size` (`16` or `32`) `Q8` tiles.
    ///
    /// # Panics
    ///
    /// Panics if `group_size` is not `16` or `32`, `cols % group_size != 0`, or `input.len() != m * cols`.
    #[must_use]
    pub fn quantize(input: &[f32], m: usize, cols: usize, group_size: usize) -> Self {
        assert!(
            group_size == 16 || group_size == 32,
            "group_size must be 16 or 32"
        );
        assert_eq!(
            cols % group_size,
            0,
            "cols must be a multiple of group_size"
        );
        assert_eq!(input.len(), m * cols, "input len must equal m * cols");

        let group_chunks = group_size / TILE_K;
        let num_groups = cols / group_size;
        let m_tiles = m.div_ceil(TILE_M);
        let m_padded = m_tiles * TILE_M;

        let mut a_tiles = vec![Aligned128::zeroed(); num_groups * m_tiles * group_chunks];
        let mut x_scales = vec![0.0_f32; num_groups * m_padded];
        let mut x_sums = vec![0.0_f32; num_groups * m_padded];

        for m_idx in 0..m {
            let mt = m_idx / TILE_M;
            let mr = m_idx % TILE_M;
            let row_slice = &input[m_idx * cols..(m_idx + 1) * cols];

            for g in 0..num_groups {
                let sub = &row_slice[g * group_size..(g + 1) * group_size];
                let mut absmax = 0.0_f32;
                let mut sum = 0.0_f32;
                let mut clean = [0.0_f32; 32];
                for (i, &val) in sub.iter().enumerate() {
                    let v = if val.is_finite() { val } else { 0.0 };
                    clean[i] = v;
                    sum += v;
                    let a = v.abs();
                    if a > absmax {
                        absmax = a;
                    }
                }

                let meta_idx = g * m_padded + m_idx;
                x_sums[meta_idx] = sum;

                if absmax >= 1.0e-12 {
                    let scale = absmax / 127.0_f32;
                    let inv_scale = 127.0_f32 / absmax;
                    x_scales[meta_idx] = scale;

                    let base_tile = (g * m_tiles + mt) * group_chunks;
                    for c in 0..group_chunks {
                        let dst_i8 = a_tiles[base_tile + c].as_i8_mut();
                        let dst_row = &mut dst_i8[mr * TILE_K..(mr + 1) * TILE_K];
                        let src_chunk = &clean[c * TILE_K..(c + 1) * TILE_K];
                        for k in 0..TILE_K {
                            #[expect(
                                clippy::cast_possible_truncation,
                                reason = "rounded and clamped to [-127, 127]"
                            )]
                            {
                                dst_row[k] =
                                    (src_chunk[k] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                            }
                        }
                    }
                }
            }
        }

        Self {
            m,
            cols,
            group_size,
            m_tiles,
            a_tiles,
            x_scales,
            x_sums,
        }
    }

    /// Returns the quantized group size (`16` or `32`).
    #[must_use]
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Returns the token count `M`.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Returns the input feature count `K`.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

#[derive(Debug, Clone)]
struct QuantPanel {
    row_count: usize,
    n_tiles: usize,
    /// Packed `B` tiles in `[group: 0..num_groups][nt: 0..n_tiles][chunk: 0..group_chunks]` order.
    b_tiles: Vec<Aligned128>,
    /// Scales in `[group: 0..num_groups][nt * 8 + nr]` order.
    w_scales: Vec<f32>,
    /// Optional asymmetric min offsets (`Some` for `Q4_K_M`).
    w_mins: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
enum CachedStorage {
    BlockScaledI8 {
        group_size: usize,
        group_chunks: usize,
        num_groups: usize,
        panels: Vec<QuantPanel>,
    },
    DenseF32 {
        padded_cols: usize,
        blocks: Vec<AlignedF32Block>,
    },
    DenseHalf {
        padded_cols: usize,
        is_bf16: bool,
        blocks: Vec<AlignedU16Block>,
    },
    StreamingDense {
        quant: QuantType,
        raw: Vec<u8>,
    },
}

/// Load-time repacked weight matrix aligned to 128-byte boundaries for `X100` (`256b`) and
/// `A100` (`1024b` + `XSMTVDotII` `8 x 16` `IME`).
#[derive(Debug, Clone)]
pub struct CachedMatrix {
    rows: usize,
    cols: usize,
    quant: QuantType,
    approx_bytes: usize,
    storage: CachedStorage,
}

impl CachedMatrix {
    /// Repacks a 2D [`TensorView`] of shape `(rows, cols)` into 128-byte-aligned tiles/rows.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] if dimensions
    /// or quantization format are invalid.
    pub fn repack(tensor: &TensorView<'_>, rows: usize, cols: usize) -> Result<Self, K3Error> {
        let quant = tensor.quant();
        let row_bytes =
            quant_row_bytes(cols, quant).ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "llama".to_owned(),
                reason: format!("unsupported quant {quant:?} or unaligned cols {cols}"),
            })?;
        let required_bytes =
            rows.checked_mul(row_bytes)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: tensor.offset(),
                    byte_len: tensor.byte_len(),
                    buffer_len: tensor.as_bytes().len(),
                    reason: "matrix byte size overflow".to_owned(),
                })?;
        let raw = tensor.as_bytes();
        if raw.len() < required_bytes {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor.name().to_owned(),
                offset: tensor.offset(),
                byte_len: required_bytes,
                buffer_len: raw.len(),
                reason: "tensor byte slice shorter than rows * row_bytes".to_owned(),
            });
        }

        match quant {
            QuantType::Q8_0 | QuantType::Q4_0 | QuantType::Q4_K_M | QuantType::Q6_K => {
                Self::repack_block_scaled_i8(&raw[..required_bytes], rows, cols, row_bytes, quant)
            }
            QuantType::F32 | QuantType::F16 | QuantType::BF16 => Ok(Self::repack_dense(
                &raw[..required_bytes],
                rows,
                cols,
                quant,
            )),
            QuantType::Q5_K | QuantType::TQ1_0 | QuantType::TQ2_0 => {
                Err(K3Error::CapabilityMismatch {
                    architecture: "llama".to_owned(),
                    reason: format!("unsupported quant {quant:?} in Llama/Qwen2 repack cache"),
                })
            }
        }
    }

    fn repack_dense(raw: &[u8], rows: usize, cols: usize, quant: QuantType) -> Self {
        let total_elems = rows.saturating_mul(cols);
        if total_elems <= MAX_DENSE_F32_ELEMENTS {
            let blocks_per_row = cols.div_ceil(32);
            let padded_cols = blocks_per_row * 32;
            let mut blocks = vec![AlignedF32Block::default(); rows * blocks_per_row];
            let flat = {
                // SAFETY: `AlignedF32Block` wraps `[f32; 32]` with zero padding.
                unsafe {
                    core::slice::from_raw_parts_mut(
                        blocks.as_mut_ptr().cast::<f32>(),
                        blocks.len() * 32,
                    )
                }
            };
            for r in 0..rows {
                let dst = &mut flat[r * padded_cols..r * padded_cols + cols];
                match quant {
                    QuantType::F32 => {
                        let row_raw = &raw[r * cols * 4..(r + 1) * cols * 4];
                        for (i, slot) in dst.iter_mut().enumerate() {
                            let b = &row_raw[i * 4..i * 4 + 4];
                            *slot =
                                sanitize_f32_weight(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                        }
                    }
                    QuantType::F16 => {
                        let row_raw = &raw[r * cols * 2..(r + 1) * cols * 2];
                        for (i, slot) in dst.iter_mut().enumerate() {
                            let b = &row_raw[i * 2..i * 2 + 2];
                            *slot = f16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
                        }
                    }
                    QuantType::BF16 => {
                        let row_raw = &raw[r * cols * 2..(r + 1) * cols * 2];
                        for (i, slot) in dst.iter_mut().enumerate() {
                            let b = &row_raw[i * 2..i * 2 + 2];
                            *slot = bf16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
                        }
                    }
                    _ => unreachable!(),
                }
            }
            let approx_bytes = blocks.len() * size_of::<AlignedF32Block>();
            return Self {
                rows,
                cols,
                quant,
                approx_bytes,
                storage: CachedStorage::DenseF32 {
                    padded_cols,
                    blocks,
                },
            };
        }

        if matches!(quant, QuantType::F16 | QuantType::BF16)
            && total_elems <= MAX_DENSE_HALF_ELEMENTS
        {
            let blocks_per_row = cols.div_ceil(64);
            let padded_cols = blocks_per_row * 64;
            let mut blocks = vec![AlignedU16Block::default(); rows * blocks_per_row];
            let flat = {
                // SAFETY: `AlignedU16Block` wraps `[u16; 64]` with zero padding.
                unsafe {
                    core::slice::from_raw_parts_mut(
                        blocks.as_mut_ptr().cast::<u16>(),
                        blocks.len() * 64,
                    )
                }
            };
            for r in 0..rows {
                let dst = &mut flat[r * padded_cols..r * padded_cols + cols];
                let row_raw = &raw[r * cols * 2..(r + 1) * cols * 2];
                for (i, slot) in dst.iter_mut().enumerate() {
                    let b = &row_raw[i * 2..i * 2 + 2];
                    *slot = u16::from_le_bytes([b[0], b[1]]);
                }
            }
            let approx_bytes = blocks.len() * size_of::<AlignedU16Block>();
            return Self {
                rows,
                cols,
                quant,
                approx_bytes,
                storage: CachedStorage::DenseHalf {
                    padded_cols,
                    is_bf16: quant == QuantType::BF16,
                    blocks,
                },
            };
        }

        Self {
            rows,
            cols,
            quant,
            approx_bytes: raw.len(),
            storage: CachedStorage::StreamingDense {
                quant,
                raw: raw.to_vec(),
            },
        }
    }

    fn repack_block_scaled_i8(
        raw: &[u8],
        rows: usize,
        cols: usize,
        row_bytes: usize,
        quant: QuantType,
    ) -> Result<Self, K3Error> {
        let (group_size, has_mins) = match quant {
            QuantType::Q8_0 | QuantType::Q4_0 => (32, false),
            QuantType::Q4_K_M => (32, true),
            QuantType::Q6_K => (16, false),
            _ => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "llama".to_owned(),
                    reason: format!("unexpected quant {quant:?}"),
                });
            }
        };
        let group_chunks = group_size / TILE_K;
        let num_groups = cols / group_size;
        let num_panels = rows.div_ceil(PANEL_ROWS);

        let mut panels = Vec::with_capacity(num_panels);
        let mut approx_bytes = 0_usize;
        let mut row_quants = vec![0_i8; cols];
        let mut row_scales = vec![0.0_f32; num_groups];
        let mut row_mins = vec![0.0_f32; num_groups];

        for p in 0..num_panels {
            let row_start = p * PANEL_ROWS;
            let row_count = (rows - row_start).min(PANEL_ROWS);
            let n_tiles = row_count.div_ceil(TILE_N);
            let n_padded = n_tiles * TILE_N;

            let mut b_tiles = vec![Aligned128::zeroed(); num_groups * n_tiles * group_chunks];
            let mut w_scales = vec![0.0_f32; num_groups * n_padded];
            let mut w_mins = has_mins.then(|| vec![0.0_f32; num_groups * n_padded]);

            for local_r in 0..row_count {
                let global_r = row_start + local_r;
                let row_raw = &raw[global_r * row_bytes..(global_r + 1) * row_bytes];
                unpack_quant_row_i8(
                    row_raw,
                    quant,
                    &mut row_quants,
                    &mut row_scales,
                    &mut row_mins,
                );

                let nt = local_r / TILE_N;
                let nr = local_r % TILE_N;
                for g in 0..num_groups {
                    w_scales[g * n_padded + local_r] = row_scales[g];
                    if let Some(ref mut mins_vec) = w_mins {
                        mins_vec[g * n_padded + local_r] = row_mins[g];
                    }
                    let base_tile = (g * n_tiles + nt) * group_chunks;
                    let q_group = &row_quants[g * group_size..(g + 1) * group_size];
                    for c in 0..group_chunks {
                        let dst_i8 = b_tiles[base_tile + c].as_i8_mut();
                        dst_i8[nr * TILE_K..(nr + 1) * TILE_K]
                            .copy_from_slice(&q_group[c * TILE_K..(c + 1) * TILE_K]);
                    }
                }
            }

            approx_bytes += b_tiles.len() * TILE_BYTES
                + w_scales.len() * 4
                + w_mins.as_ref().map_or(0, |v| v.len() * 4);
            panels.push(QuantPanel {
                row_count,
                n_tiles,
                b_tiles,
                w_scales,
                w_mins,
            });
        }

        Ok(Self {
            rows,
            cols,
            quant,
            approx_bytes,
            storage: CachedStorage::BlockScaledI8 {
                group_size,
                group_chunks,
                num_groups,
                panels,
            },
        })
    }

    /// Returns the output row count (`N`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the input column count (`K`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the source quantization format.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Returns `Some(group_size)` (`16` or `32`) if backed by quantized `BlockScaledI8` tiles,
    /// allowing multiple matrices sharing an activation vector (e.g. `Q/K/V` or `gate/up`)
    /// to share a single [`QuantizedBatchAct`].
    #[must_use]
    pub const fn quant_group_size(&self) -> Option<usize> {
        match self.storage {
            CachedStorage::BlockScaledI8 { group_size, .. } => Some(group_size),
            CachedStorage::DenseF32 { .. }
            | CachedStorage::DenseHalf { .. }
            | CachedStorage::StreamingDense { .. } => None,
        }
    }

    /// Single-token GEMV (`M = 1`) projecting `input` (`[cols]`) into `output` (`[rows]`),
    /// parallelized across `64`-row (`8 x TILE_N`) panels when `parallel` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `output.len() != self.rows` or `input.len() != self.cols`.
    pub fn gemv_rows(
        &self,
        output: &mut [f32],
        input: &[f32],
        parallel: bool,
    ) -> Result<(), K3Error> {
        self.project_batch(output, input, 1, parallel)
    }

    /// Batched multi-token GEMM (`M >= 1`) projecting `input` (`[m, cols]`) into `output` (`[m, rows]`),
    /// reading each repacked weight panel once across all `m` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if slice lengths do not match `m * rows` and `m * cols`.
    pub fn gemm_rows(
        &self,
        output: &mut [f32],
        input: &[f32],
        m: usize,
        parallel: bool,
    ) -> Result<(), K3Error> {
        self.project_batch(output, input, m, parallel)
    }

    /// Projects `[m, cols]` into `[m, rows]`, automatically quantizing activations when needed.
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
        if m == 0 {
            return Ok(());
        }
        let pre_act = self
            .quant_group_size()
            .filter(|_| input.len() == m.saturating_mul(self.cols))
            .map(|gs| QuantizedBatchAct::quantize(input, m, self.cols, gs));
        self.project_batch_with_act(output, input, pre_act.as_ref(), m, parallel)
    }

    /// Projects `[m, cols]` into `[m, rows]`, reusing a pre-quantized [`QuantizedBatchAct`] when compatible.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
    pub fn project_batch_with_act(
        &self,
        output: &mut [f32],
        input: &[f32],
        pre_act: Option<&QuantizedBatchAct>,
        m: usize,
        parallel: bool,
    ) -> Result<(), K3Error> {
        if m == 0 {
            return Ok(());
        }
        if output.len() != m.saturating_mul(self.rows) || input.len() != m.saturating_mul(self.cols)
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "cached_matrix".to_owned(),
                offset: 0,
                byte_len: size_of_val(output),
                buffer_len: m.saturating_mul(self.rows) * size_of::<f32>(),
                reason: format!(
                    "batch projection mismatch: m={m}, rows={}, cols={}, out_len={}, in_len={}",
                    self.rows,
                    self.cols,
                    output.len(),
                    input.len()
                ),
            });
        }

        let owned_act;
        let quant_act = if let Some(gs) = self.quant_group_size() {
            if let Some(act) =
                pre_act.filter(|a| a.group_size == gs && a.m == m && a.cols == self.cols)
            {
                Some(act)
            } else {
                owned_act = QuantizedBatchAct::quantize(input, m, self.cols, gs);
                Some(&owned_act)
            }
        } else {
            None
        };

        if m == 1 {
            par_for_each_row_tile(output, PANEL_ROWS, parallel, |panel_idx, out_panel| {
                self.compute_panel(panel_idx, input, quant_act, 1, out_panel);
            });
            return Ok(());
        }

        // For m >= 2, compute disjoint `[m, panel_rows]` tiles in parallel, then scatter contiguous
        // panel slices into `[m, rows]` output order with zero lock contention and zero unsafe.
        let num_panels = self.rows.div_ceil(PANEL_ROWS);
        let mut panel_major = vec![0.0_f32; num_panels * m * PANEL_ROWS];
        par_for_each_row_tile(
            &mut panel_major,
            m * PANEL_ROWS,
            parallel,
            |panel_idx, panel_buf| {
                let row_start = panel_idx * PANEL_ROWS;
                let panel_rows = (self.rows - row_start).min(PANEL_ROWS);
                self.compute_panel(
                    panel_idx,
                    input,
                    quant_act,
                    m,
                    &mut panel_buf[..m * panel_rows],
                );
            },
        );

        for panel_idx in 0..num_panels {
            let row_start = panel_idx * PANEL_ROWS;
            let panel_rows = (self.rows - row_start).min(PANEL_ROWS);
            let src_panel = &panel_major[panel_idx * m * PANEL_ROWS..];
            for tok in 0..m {
                let src = &src_panel[tok * panel_rows..(tok + 1) * panel_rows];
                let dst = &mut output
                    [tok * self.rows + row_start..tok * self.rows + row_start + panel_rows];
                dst.copy_from_slice(src);
            }
        }
        Ok(())
    }

    fn compute_panel(
        &self,
        panel_idx: usize,
        input: &[f32],
        quant_act: Option<&QuantizedBatchAct>,
        m: usize,
        out_panel: &mut [f32],
    ) {
        let row_start = panel_idx * PANEL_ROWS;
        let row_count = (self.rows - row_start).min(PANEL_ROWS);
        debug_assert_eq!(out_panel.len(), m * row_count);

        match &self.storage {
            CachedStorage::BlockScaledI8 {
                group_chunks,
                num_groups,
                panels,
                ..
            } => {
                let act = quant_act.expect("quantized activation required for BlockScaledI8");
                compute_quant_panel(
                    &panels[panel_idx],
                    act,
                    *group_chunks,
                    *num_groups,
                    m,
                    out_panel,
                );
            }
            CachedStorage::DenseF32 {
                padded_cols,
                blocks,
            } => {
                let flat = aligned_f32_as_slice(blocks);
                for r in 0..row_count {
                    let w_row = &flat
                        [(row_start + r) * padded_cols..(row_start + r) * padded_cols + self.cols];
                    for tok in 0..m {
                        let x_row = &input[tok * self.cols..(tok + 1) * self.cols];
                        out_panel[tok * row_count + r] = dot_f32(w_row, x_row);
                    }
                }
            }
            CachedStorage::DenseHalf {
                padded_cols,
                is_bf16,
                blocks,
            } => {
                let flat = aligned_u16_as_slice(blocks);
                let mut w_chunk = [0.0_f32; 256];
                for r in 0..row_count {
                    let w_u16 = &flat
                        [(row_start + r) * padded_cols..(row_start + r) * padded_cols + self.cols];
                    for tok in 0..m {
                        out_panel[tok * row_count + r] = 0.0;
                    }
                    let mut col = 0_usize;
                    while col < self.cols {
                        let clen = (self.cols - col).min(w_chunk.len());
                        if *is_bf16 {
                            for (i, slot) in w_chunk[..clen].iter_mut().enumerate() {
                                *slot = bf16_to_f32_finite(w_u16[col + i]);
                            }
                        } else {
                            for (i, slot) in w_chunk[..clen].iter_mut().enumerate() {
                                *slot = f16_to_f32_finite(w_u16[col + i]);
                            }
                        }
                        for tok in 0..m {
                            let x_sub = &input[tok * self.cols + col..tok * self.cols + col + clen];
                            out_panel[tok * row_count + r] += dot_f32(&w_chunk[..clen], x_sub);
                        }
                        col += clen;
                    }
                }
            }
            CachedStorage::StreamingDense { quant, raw } => {
                compute_streaming_dense_panel(
                    *quant, raw, row_start, row_count, self.cols, input, m, out_panel,
                );
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal panel streaming helper passes explicit matrix geometry"
)]
fn compute_streaming_dense_panel(
    quant: QuantType,
    raw: &[u8],
    row_start: usize,
    row_count: usize,
    cols: usize,
    input: &[f32],
    m: usize,
    out_panel: &mut [f32],
) {
    let elem_bytes = if quant == QuantType::F32 { 4 } else { 2 };
    let mut w_chunk = [0.0_f32; 256];
    for r in 0..row_count {
        let row_raw =
            &raw[(row_start + r) * cols * elem_bytes..(row_start + r + 1) * cols * elem_bytes];
        for tok in 0..m {
            out_panel[tok * row_count + r] = 0.0;
        }
        let mut col = 0_usize;
        while col < cols {
            let clen = (cols - col).min(w_chunk.len());
            match quant {
                QuantType::F32 => {
                    for (i, slot) in w_chunk[..clen].iter_mut().enumerate() {
                        let off = (col + i) * 4;
                        let b = &row_raw[off..off + 4];
                        *slot = sanitize_f32_weight(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    }
                }
                QuantType::F16 => {
                    for (i, slot) in w_chunk[..clen].iter_mut().enumerate() {
                        let off = (col + i) * 2;
                        let b = &row_raw[off..off + 2];
                        *slot = f16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
                    }
                }
                QuantType::BF16 => {
                    for (i, slot) in w_chunk[..clen].iter_mut().enumerate() {
                        let off = (col + i) * 2;
                        let b = &row_raw[off..off + 2];
                        *slot = bf16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
                    }
                }
                _ => unreachable!(),
            }
            for tok in 0..m {
                let x_sub = &input[tok * cols + col..tok * cols + col + clen];
                out_panel[tok * row_count + r] += dot_f32(&w_chunk[..clen], x_sub);
            }
            col += clen;
        }
    }
}

fn unpack_quant_row_i8(
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
        QuantType::Q4_K_M => {
            for (sb_idx, blk) in row_raw.chunks_exact(Q4_K_M_BLOCK_BYTES).enumerate() {
                let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32_finite(u16::from_le_bytes([blk[2], blk[3]]));
                let (sc, m) = unpack_q4_k_scales(&blk[4..16]);
                let qs = &blk[16..144];
                for chunk in 0..4 {
                    let sub0 = sb_idx * 8 + chunk * 2;
                    let sub1 = sub0 + 1;
                    out_scales[sub0] = d * f32::from(sc[chunk * 2]);
                    out_mins[sub0] = dmin * f32::from(m[chunk * 2]);
                    out_scales[sub1] = d * f32::from(sc[chunk * 2 + 1]);
                    out_mins[sub1] = dmin * f32::from(m[chunk * 2 + 1]);
                    let q_off = chunk * 32;
                    let dst_off = sb_idx * 256 + chunk * 64;
                    for l in 0..32 {
                        #[expect(
                            clippy::cast_possible_wrap,
                            reason = "unsigned 4-bit nibble 0..=15 fits in positive i8"
                        )]
                        {
                            out_q[dst_off + l] = (qs[q_off + l] & 0x0F) as i8;
                            out_q[dst_off + 32 + l] = ((qs[q_off + l] >> 4) & 0x0F) as i8;
                        }
                    }
                }
            }
        }
        QuantType::Q6_K => {
            for (sb_idx, blk) in row_raw.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
                let d = f16_to_f32_finite(u16::from_le_bytes([blk[208], blk[209]]));
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208];
                for sub in 0..16 {
                    #[expect(clippy::cast_possible_wrap, reason = "signed i8 sub-block scale")]
                    {
                        out_scales[sb_idx * 16 + sub] = d * f32::from(scales[sub] as i8);
                    }
                }
                let dst_base = sb_idx * 256;
                for (half_idx, &n) in [0_usize, 128_usize].iter().enumerate() {
                    let low_base = half_idx * 64;
                    let high_base = half_idx * 32;
                    for l in 0..32 {
                        #[expect(
                            clippy::cast_possible_wrap,
                            reason = "6-bit unsigned quant 0..63 offset by -32 fits in i8"
                        )]
                        {
                            out_q[dst_base + n + l] = (((ql[low_base + l] & 0x0F)
                                | ((qh[high_base + l] & 0x03) << 4))
                                as i8)
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
        _ => unreachable!(),
    }
}

fn compute_quant_panel(
    panel: &QuantPanel,
    act: &QuantizedBatchAct,
    group_chunks: usize,
    num_groups: usize,
    m: usize,
    out_panel: &mut [f32],
) {
    out_panel.fill(0.0);
    let m_tiles = act.m_tiles;
    let m_padded = m_tiles * TILE_M;
    let n_tiles = panel.n_tiles;
    let n_padded = n_tiles * TILE_N;
    let row_count = panel.row_count;

    for g in 0..num_groups {
        let a_group = &act.a_tiles[g * m_tiles * group_chunks..(g + 1) * m_tiles * group_chunks];
        let b_group = &panel.b_tiles[g * n_tiles * group_chunks..(g + 1) * n_tiles * group_chunks];
        let w_scales = &panel.w_scales[g * n_padded..(g + 1) * n_padded];
        let x_scales = &act.x_scales[g * m_padded..(g + 1) * m_padded];

        #[cfg(all(feature = "with-kernels", target_arch = "riscv64", not(miri)))]
        {
            gemm_i8_ime(
                aligned_tiles_as_i8(a_group),
                m_tiles,
                aligned_tiles_as_i8(b_group),
                0..n_tiles,
                group_chunks,
                |mt, nt, c_tile: &[i32; TILE_C]| {
                    let m_valid = (m - mt * TILE_M).min(TILE_M);
                    let n_valid = (row_count - nt * TILE_N).min(TILE_N);
                    if let Some(ref mins_all) = panel.w_mins {
                        let w_mins = &mins_all[g * n_padded..(g + 1) * n_padded];
                        let x_sums = &act.x_sums[g * m_padded..(g + 1) * m_padded];
                        for i in 0..m_valid {
                            let xs = x_scales[mt * TILE_M + i];
                            let xsum = x_sums[mt * TILE_M + i];
                            let dst = &mut out_panel[(mt * TILE_M + i) * row_count + nt * TILE_N..];
                            for j in 0..n_valid {
                                #[expect(
                                    clippy::cast_precision_loss,
                                    reason = "bounded 32-elem i8x8 dot <= 524,288 fits exact in f32"
                                )]
                                let dot = c_tile[i * TILE_N + j] as f32;
                                dst[j] += dot * xs * w_scales[nt * TILE_N + j]
                                    - xsum * w_mins[nt * TILE_N + j];
                            }
                        }
                    } else {
                        for i in 0..m_valid {
                            let xs = x_scales[mt * TILE_M + i];
                            let dst = &mut out_panel[(mt * TILE_M + i) * row_count + nt * TILE_N..];
                            for j in 0..n_valid {
                                #[expect(
                                    clippy::cast_precision_loss,
                                    reason = "bounded 32-elem i8x8 dot <= 524,288 fits exact in f32"
                                )]
                                let dot = c_tile[i * TILE_N + j] as f32;
                                dst[j] += dot * xs * w_scales[nt * TILE_N + j];
                            }
                        }
                    }
                },
            );
        }

        #[cfg(any(not(feature = "with-kernels"), not(target_arch = "riscv64"), miri))]
        {
            let w_mins = panel
                .w_mins
                .as_ref()
                .map(|mins_all| &mins_all[g * n_padded..(g + 1) * n_padded]);
            let x_sums = &act.x_sums[g * m_padded..(g + 1) * m_padded];
            compute_quant_group_scalar(
                a_group,
                b_group,
                w_scales,
                w_mins,
                x_scales,
                x_sums,
                m_tiles,
                n_tiles,
                group_chunks,
                m,
                row_count,
                out_panel,
            );
        }
    }
}

#[cfg(any(not(feature = "with-kernels"), not(target_arch = "riscv64"), miri))]
#[expect(
    clippy::too_many_arguments,
    reason = "internal tile accumulation helper passes explicit group slices and dimensions"
)]
fn compute_quant_group_scalar(
    a_group: &[Aligned128],
    b_group: &[Aligned128],
    w_scales: &[f32],
    w_mins: Option<&[f32]>,
    x_scales: &[f32],
    x_sums: &[f32],
    m_tiles: usize,
    n_tiles: usize,
    group_chunks: usize,
    m: usize,
    row_count: usize,
    out_panel: &mut [f32],
) {
    for nt in 0..n_tiles {
        let n_valid = (row_count - nt * TILE_N).min(TILE_N);
        let b_nt = &b_group[nt * group_chunks..(nt + 1) * group_chunks];
        for mt in 0..m_tiles {
            let m_valid = (m - mt * TILE_M).min(TILE_M);
            let a_mt = &a_group[mt * group_chunks..(mt + 1) * group_chunks];
            for i in 0..m_valid {
                let xs = x_scales[mt * TILE_M + i];
                let xsum = x_sums[mt * TILE_M + i];
                let dst = &mut out_panel[(mt * TILE_M + i) * row_count + nt * TILE_N..];
                for j in 0..n_valid {
                    let mut dot_i32 = 0_i32;
                    for c in 0..group_chunks {
                        let ai = &a_mt[c].as_i8()[i * TILE_K..(i + 1) * TILE_K];
                        let bj = &b_nt[c].as_i8()[j * TILE_K..(j + 1) * TILE_K];
                        dot_i32 += dot_i8_scalar(ai, bj);
                    }
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "bounded 32-elem i8x8 dot <= 524,288 fits exact in f32"
                    )]
                    let dot = dot_i32 as f32;
                    let base = dot * xs * w_scales[nt * TILE_N + j];
                    dst[j] += w_mins.map_or(base, |mins| base - xsum * mins[nt * TILE_N + j]);
                }
            }
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheKey {
    data_ptr: usize,
    byte_len: usize,
    quant: QuantType,
    rows: usize,
    cols: usize,
    fingerprint64: u64,
}

impl CacheKey {
    fn from_tensor(tensor: &TensorView<'_>, rows: usize, cols: usize) -> Self {
        let raw = tensor.as_bytes();
        Self {
            data_ptr: raw.as_ptr() as usize,
            byte_len: raw.len(),
            quant: tensor.quant(),
            rows,
            cols,
            fingerprint64: compute_fingerprint64(tensor.name(), raw),
        }
    }
}

fn compute_fingerprint64(name: &str, raw: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &b in name.as_bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    hash ^= raw.len() as u64;
    hash = hash.wrapping_mul(0x0100_0000_01B3);

    if raw.len() <= 256 {
        for &b in raw {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01B3);
        }
        return hash;
    }

    let samples = 64_usize;
    let step = (raw.len() - 8) / samples;
    for s in 0..=samples {
        let off = (s * step).min(raw.len() - 8);
        let mut word = [0_u8; 8];
        word.copy_from_slice(&raw[off..off + 8]);
        hash ^= u64::from_le_bytes(word);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    hash
}

#[derive(Debug, Default)]
struct ModelRepackCache {
    active_anchor: Option<(usize, usize, u64)>,
    entries: HashMap<String, (CacheKey, Arc<CachedMatrix>)>,
}

impl ModelRepackCache {
    fn set_anchor(&mut self, anchor: (usize, usize, u64)) {
        if self.active_anchor != Some(anchor) {
            self.entries.clear();
            self.active_anchor = Some(anchor);
        }
    }

    fn insert(&mut self, name: &str, key: CacheKey, mat: Arc<CachedMatrix>) {
        if self.entries.len() >= MAX_CACHED_TENSORS && !self.entries.contains_key(name) {
            self.entries.clear();
        }
        self.entries.insert(name.to_owned(), (key, mat));
    }
}

static GLOBAL_REPACK_CACHE: OnceLock<RwLock<ModelRepackCache>> = OnceLock::new();

fn global_cache() -> &'static RwLock<ModelRepackCache> {
    GLOBAL_REPACK_CACHE.get_or_init(|| RwLock::new(ModelRepackCache::default()))
}

fn anchor_identity(anchor_tensor: &TensorView<'_>) -> (usize, usize, u64) {
    let raw = anchor_tensor.as_bytes();
    (
        raw.as_ptr() as usize,
        raw.len(),
        compute_fingerprint64(anchor_tensor.name(), raw),
    )
}

/// Updates the active model anchor (`token_embd.weight` identity), evicting cached layer weights
/// from any previously loaded model when the anchor changes.
pub fn set_active_model_anchor(anchor_tensor: &TensorView<'_>) {
    let anchor = anchor_identity(anchor_tensor);

    if let Ok(guard) = global_cache().read()
        && guard.active_anchor == Some(anchor)
    {
        return;
    }

    if let Ok(mut guard) = global_cache().write() {
        guard.set_anchor(anchor);
    }
}

/// Clears all entries in the global repack cache and resets the active model anchor.
pub fn clear_repack_cache() {
    if let Ok(mut guard) = global_cache().write() {
        guard.entries.clear();
        guard.active_anchor = None;
    }
}

/// Returns the number of repacked matrices currently resident in the global cache.
#[must_use]
pub fn cached_matrix_count() -> usize {
    global_cache().read().map_or(0, |g| g.entries.len())
}

/// Retrieves a 128-byte-aligned [`CachedMatrix`] for `tensor` or repacks and caches it on first access.
///
/// # Errors
///
/// Returns [`K3Error`] if `tensor` has invalid dimensions or unsupported quantization.
pub fn get_or_repack(
    tensor: &TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<Arc<CachedMatrix>, K3Error> {
    let key = CacheKey::from_tensor(tensor, rows, cols);
    if let Ok(guard) = global_cache().read()
        && let Some((cached_key, mat)) = guard.entries.get(tensor.name())
        && *cached_key == key
    {
        return Ok(Arc::clone(mat));
    }

    let repacked = Arc::new(CachedMatrix::repack(tensor, rows, cols)?);
    let _ = repacked.approx_bytes;
    if let Ok(mut guard) = global_cache().write() {
        guard.insert(tensor.name(), key, Arc::clone(&repacked));
    }
    Ok(repacked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned128_layout_and_alignment() {
        assert_eq!(size_of::<Aligned128>(), 128);
        assert_eq!(align_of::<Aligned128>(), 128);
        let buf = vec![Aligned128::zeroed(); 4];
        for item in &buf {
            assert_eq!((std::ptr::from_ref(item) as usize) % 128, 0);
        }
    }

    #[test]
    fn test_repack_cache_aba_eviction_and_model_anchor() {
        let cols = 32;
        let rows = 8;
        let mut raw_a = vec![0_u8; rows * cols * 4];
        for i in 0..rows * cols {
            raw_a[i * 4..(i + 1) * 4].copy_from_slice(&1.0_f32.to_le_bytes());
        }
        let shape = [cols, rows];
        let view_a = TensorView::from_bytes(
            "test_aba_unique.attn_q.weight",
            QuantType::F32,
            &shape,
            0,
            &raw_a,
        )
        .expect("valid view a");

        // Verify deterministic cache hit, ABA key invalidation, and anchor eviction on isolated cache
        let mut local_cache = ModelRepackCache::default();
        let emb_raw_1 = vec![1_u8; 32 * 4 * 4];
        let emb_shape = [32, 4];
        let emb_view_1 = TensorView::from_bytes(
            "token_embd.weight",
            QuantType::F32,
            &emb_shape,
            0,
            &emb_raw_1,
        )
        .expect("emb view 1");
        local_cache.set_anchor(anchor_identity(&emb_view_1));

        let key_a = CacheKey::from_tensor(&view_a, rows, cols);
        let mat_a = get_or_repack(&view_a, rows, cols).expect("repack a");
        local_cache.insert(view_a.name(), key_a, Arc::clone(&mat_a));
        assert_eq!(local_cache.entries.len(), 1);

        let mut out_a = [0.0_f32; 8];
        mat_a
            .gemv_rows(&mut out_a, &[1.0_f32; 32], false)
            .expect("gemv a");
        assert!((out_a[0] - 32.0).abs() < 1e-4);

        // Mutate underlying bytes in-place at exact same heap address (simulated ABA reuse)
        for i in 0..rows * cols {
            raw_a[i * 4..(i + 1) * 4].copy_from_slice(&2.5_f32.to_le_bytes());
        }
        let view_b = TensorView::from_bytes(
            "test_aba_unique.attn_q.weight",
            QuantType::F32,
            &shape,
            0,
            &raw_a,
        )
        .expect("valid view b");
        let key_b = CacheKey::from_tensor(&view_b, rows, cols);
        assert_ne!(
            key_a, key_b,
            "fingerprint64 must detect in-place ABA mutation"
        );

        let mat_b = get_or_repack(&view_b, rows, cols).expect("repack b");
        assert!(!Arc::ptr_eq(&mat_a, &mat_b));
        local_cache.insert(view_b.name(), key_b, Arc::clone(&mat_b));
        assert_eq!(local_cache.entries.len(), 1);

        let mut out_b = [0.0_f32; 8];
        mat_b
            .gemv_rows(&mut out_b, &[1.0_f32; 32], false)
            .expect("gemv b");
        assert!((out_b[0] - 80.0).abs() < 1e-4);

        // Changing model anchor evicts stale tensors from previous model
        let emb_raw_2 = vec![2_u8; 32 * 4 * 4];
        let emb_view_2 = TensorView::from_bytes(
            "token_embd.weight",
            QuantType::F32,
            &emb_shape,
            0,
            &emb_raw_2,
        )
        .expect("emb view 2");
        local_cache.set_anchor(anchor_identity(&emb_view_2));
        assert_eq!(local_cache.entries.len(), 0);
    }
}
