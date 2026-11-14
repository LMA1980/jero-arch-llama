//! Pre-computed per-layer tensor key tables, zero-allocation `M = 1` activation quantization,
//! and multi-row RVV 1.0 `GEMV` row-partitioned dispatch for autoregressive decode.

use std::mem::{align_of, size_of};
use std::sync::{Arc, RwLock};

use jero_core::{K3Error, ModelWeights, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::{
    BlockQ8_0, BlockQ8K, gemv_q4_0_q8_0, gemv_q4_k_q8_k, gemv_q5_0_q8_0, gemv_q5_k_q8_k,
    gemv_q6_k_q8_k, gemv_q8_0_q8_0, gemv_tq1_0_q8_k, gemv_tq2_0_q8_k, quantize_row_q8_0,
    quantize_row_q8_k,
};

#[cfg(not(feature = "with-kernels"))]
use crate::repack_cache::{bf16_to_f32_finite, f16_to_f32_finite};
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{bf16_to_f32_finite, f16_to_f32_finite};

use crate::dispatch::compute_row_dot;
use crate::repack_cache::{resolve_2d_dims, sanitize_f32_weight};

/// Pre-formatted GGUF tensor key strings for a single transformer layer (`blk.{layer}.*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTensorKeys {
    /// `blk.{layer}.attn_norm.weight`
    pub attn_norm: String,
    /// `blk.{layer}.attn_q.weight`
    pub attn_q: String,
    /// `blk.{layer}.attn_q.bias`
    pub attn_q_bias: String,
    /// `blk.{layer}.attn_k.weight`
    pub attn_k: String,
    /// `blk.{layer}.attn_k.bias`
    pub attn_k_bias: String,
    /// `blk.{layer}.attn_v.weight`
    pub attn_v: String,
    /// `blk.{layer}.attn_v.bias`
    pub attn_v_bias: String,
    /// `blk.{layer}.attn_output.weight`
    pub attn_output: String,
    /// `blk.{layer}.attn_output.bias`
    pub attn_output_bias: String,
    /// `blk.{layer}.ffn_norm.weight`
    pub ffn_norm: String,
    /// `blk.{layer}.ffn_gate.weight`
    pub ffn_gate: String,
    /// `blk.{layer}.ffn_up.weight`
    pub ffn_up: String,
    /// `blk.{layer}.ffn_down.weight`
    pub ffn_down: String,
}

impl LayerTensorKeys {
    /// Formats all 13 tensor key names once for layer index `layer`.
    #[must_use]
    pub fn for_layer(layer: usize) -> Self {
        Self {
            attn_norm: format!("blk.{layer}.attn_norm.weight"),
            attn_q: format!("blk.{layer}.attn_q.weight"),
            attn_q_bias: format!("blk.{layer}.attn_q.bias"),
            attn_k: format!("blk.{layer}.attn_k.weight"),
            attn_k_bias: format!("blk.{layer}.attn_k.bias"),
            attn_v: format!("blk.{layer}.attn_v.weight"),
            attn_v_bias: format!("blk.{layer}.attn_v.bias"),
            attn_output: format!("blk.{layer}.attn_output.weight"),
            attn_output_bias: format!("blk.{layer}.attn_output.bias"),
            ffn_norm: format!("blk.{layer}.ffn_norm.weight"),
            ffn_gate: format!("blk.{layer}.ffn_gate.weight"),
            ffn_up: format!("blk.{layer}.ffn_up.weight"),
            ffn_down: format!("blk.{layer}.ffn_down.weight"),
        }
    }
}

/// Model-scoped table of pre-formatted per-layer GGUF tensor keys (`blk.{0..N}.*`),
/// eliminating per-token `format!` heap allocations across prefill and decode steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerKeyTable {
    layers: Vec<LayerTensorKeys>,
}

impl LayerKeyTable {
    /// Pre-computes tensor key strings for `num_layers` layers.
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        let layers = (0..num_layers).map(LayerTensorKeys::for_layer).collect();
        Self { layers }
    }

    /// Number of pre-computed layers stored in this table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` when zero layers are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns pre-formatted tensor keys for `layer_idx`, if present.
    #[must_use]
    pub fn get(&self, layer_idx: usize) -> Option<&LayerTensorKeys> {
        self.layers.get(layer_idx)
    }

    /// Borrows all pre-computed layer key entries as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[LayerTensorKeys] {
        &self.layers
    }
}

static GLOBAL_LAYER_KEYS: RwLock<Option<Arc<LayerKeyTable>>> = RwLock::new(None);

/// Returns a shared pre-computed [`LayerKeyTable`] containing at least `min_layers` entries,
/// allocating only on initial creation or when a larger layer count is requested.
#[must_use]
pub fn ensure_global_layer_keys(min_layers: usize) -> Arc<LayerKeyTable> {
    if let Ok(guard) = GLOBAL_LAYER_KEYS.read()
        && let Some(table) = guard.as_ref()
        && table.len() >= min_layers
    {
        return Arc::clone(table);
    }

    if let Ok(mut guard) = GLOBAL_LAYER_KEYS.write() {
        if let Some(table) = guard.as_ref()
            && table.len() >= min_layers
        {
            return Arc::clone(table);
        }
        let target_len = min_layers.max(64);
        let built = Arc::new(LayerKeyTable::new(target_len));
        *guard = Some(Arc::clone(&built));
        return built;
    }

    Arc::new(LayerKeyTable::new(min_layers.max(1)))
}

/// Probes sequential `blk.{0..upper}.attn_norm.weight` tensors without per-probe `String`
/// allocations, using [`ensure_global_layer_keys`].
#[must_use]
pub fn probe_live_layer_count(weights: &dyn ModelWeights, upper: usize) -> usize {
    if upper == 0 {
        return 0;
    }
    let keys = ensure_global_layer_keys(upper);
    let mut probed = 0_usize;
    for (layer, entry) in keys.as_slice().iter().take(upper).enumerate() {
        let has_layer = weights.contains_tensor(&entry.attn_norm)
            || weights.contains_tensor(&entry.attn_q)
            || weights.contains_tensor(&entry.ffn_gate)
            || weights.contains_tensor(&entry.attn_output);
        if has_layer {
            probed = layer + 1;
        } else if probed > 0 {
            break;
        }
    }
    probed
}

#[cfg(not(feature = "with-kernels"))]
mod portable_q8 {
    use jero_core::{K3Error, QuantType, TensorView};

    use super::{bf16_to_f32_finite, f16_to_f32_finite};

    const Q8_0_BLOCK_ELEMENTS: usize = 32;
    const Q8_K_BLOCK_ELEMENTS: usize = 256;
    const Q4_0_BLOCK_BYTES: usize = 18;
    const Q5_0_BLOCK_BYTES: usize = 22;
    const Q8_0_BLOCK_BYTES: usize = 34;
    const Q4_K_M_BLOCK_BYTES: usize = 144;
    const Q5_K_BLOCK_BYTES: usize = 176;
    const Q6_K_BLOCK_BYTES: usize = 210;
    const TQ1_0_BLOCK_BYTES: usize = 54;
    const TQ2_0_BLOCK_BYTES: usize = 66;

    /// Symmetrically quantized 32-element activation block (`f32` scale + `32 x i8`).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct BlockQ8_0 {
        /// Positive block scale (`absmax / 127.0`), or `0.0` for all-zero blocks.
        pub d: f32,
        /// Symmetrically quantized signed 8-bit activation values in `[-127, 127]`.
        pub qs: [i8; Q8_0_BLOCK_ELEMENTS],
    }

    impl BlockQ8_0 {
        /// Zero-initialized constant block.
        pub const ZERO: Self = Self {
            d: 0.0,
            qs: [0; Q8_0_BLOCK_ELEMENTS],
        };
    }

    impl Default for BlockQ8_0 {
        fn default() -> Self {
            Self::ZERO
        }
    }

    /// Symmetrically quantized 256-element activation super-block (`f32` scale + `256 x i8` +
    /// `[i16; 16]` sub-block sums).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct BlockQ8K {
        /// Positive super-block scale (`absmax / 127.0`), or `0.0` for all-zero blocks.
        pub d: f32,
        /// Symmetrically quantized signed 8-bit activations across 256 elements in `[-127, 127]`.
        pub qs: [i8; Q8_K_BLOCK_ELEMENTS],
        /// Precomputed 16-element sub-block integer sums.
        pub bsums: [i16; 16],
    }

    impl BlockQ8K {
        /// Zero-initialized constant super-block.
        pub const ZERO: Self = Self {
            d: 0.0,
            qs: [0; Q8_K_BLOCK_ELEMENTS],
            bsums: [0; 16],
        };
    }

    impl Default for BlockQ8K {
        fn default() -> Self {
            Self::ZERO
        }
    }

    #[inline]
    #[expect(
        clippy::cast_precision_loss,
        reason = "integer dot accumulator fits 24-bit significand per GGML contract"
    )]
    const fn i32_to_f32(val: i32) -> f32 {
        val as f32
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped strictly to [-127.0, 127.0] before i8 cast"
    )]
    fn round_scaled_to_i8(val: f32, inv_scale: f32) -> i8 {
        if val.is_finite() {
            (val * inv_scale).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        }
    }

    fn validate_quant_len(
        name: &str,
        x_len: usize,
        out_len: usize,
        block_elems: usize,
    ) -> Result<usize, K3Error> {
        if x_len == 0 || x_len % block_elems != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: x_len * size_of::<f32>(),
                buffer_len: out_len,
                reason: format!(
                    "{name}: length {x_len} must be a positive multiple of {block_elems}"
                ),
            });
        }
        let n_blocks = x_len / block_elems;
        if out_len != n_blocks {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: x_len * size_of::<f32>(),
                buffer_len: out_len,
                reason: format!("{name}: output count {out_len} != {n_blocks}"),
            });
        }
        Ok(n_blocks)
    }

    /// Quantizes `x` into `BlockQ8_0` blocks in caller-provided scratch without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on empty/unaligned input or mismatched `out` length.
    pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) -> Result<(), K3Error> {
        let _n = validate_quant_len("quantize_row_q8_0", x.len(), out.len(), Q8_0_BLOCK_ELEMENTS)?;
        for (chunk, blk) in x.chunks_exact(Q8_0_BLOCK_ELEMENTS).zip(out.iter_mut()) {
            let amax = chunk
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0_f32, |acc, v| acc.max(v.abs()));
            if amax == 0.0 || !amax.is_finite() {
                *blk = BlockQ8_0::ZERO;
                continue;
            }
            blk.d = amax / 127.0_f32;
            let inv = 127.0_f32 / amax;
            for (dst, &src) in blk.qs.iter_mut().zip(chunk.iter()) {
                *dst = round_scaled_to_i8(src, inv);
            }
        }
        Ok(())
    }

    /// Quantizes `x` into `BlockQ8K` super-blocks in caller-provided scratch without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on empty/unaligned input or mismatched `out` length.
    pub fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) -> Result<(), K3Error> {
        let _n = validate_quant_len("quantize_row_q8_k", x.len(), out.len(), Q8_K_BLOCK_ELEMENTS)?;
        for (chunk, blk) in x.chunks_exact(Q8_K_BLOCK_ELEMENTS).zip(out.iter_mut()) {
            let amax = chunk
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0_f32, |acc, v| acc.max(v.abs()));
            if amax == 0.0 || !amax.is_finite() {
                *blk = BlockQ8K::ZERO;
                continue;
            }
            blk.d = amax / 127.0_f32;
            let inv = 127.0_f32 / amax;
            for (sub_idx, sub_chunk) in chunk.chunks_exact(16).enumerate() {
                let mut bsum = 0_i16;
                let base = sub_idx * 16;
                for (j, &val) in sub_chunk.iter().enumerate() {
                    let q = round_scaled_to_i8(val, inv);
                    blk.qs[base + j] = q;
                    bsum += i16::from(q);
                }
                blk.bsums[sub_idx] = bsum;
            }
        }
        Ok(())
    }

    fn validate_gemv_view<'a>(
        op: &str,
        quant: QuantType,
        out_len: usize,
        weights: &'a TensorView<'a>,
        cols: usize,
        blk_bytes: usize,
    ) -> Result<(usize, usize, &'a [u8]), K3Error> {
        if weights.quant() != quant {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!("{op}: expected {quant:?}, got {:?}", weights.quant()),
            });
        }
        let shape = weights.shape();
        let rows = if shape.len() == 2
            && ((shape[0] == cols && shape[1] == out_len)
                || (shape[1] == cols && shape[0] == out_len))
        {
            out_len
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!("{op}: shape {shape:?} does not match [{cols}, {out_len}]"),
            });
        };
        let blocks_per_row = cols / quant.block_elements();
        let row_bytes = blocks_per_row * blk_bytes;
        if weights.as_bytes().len() < rows * row_bytes {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!("{op}: truncated weight buffer"),
            });
        }
        Ok((rows, row_bytes, weights.as_bytes()))
    }

    #[inline]
    fn dot_q4_0_q8_0_blk(w_blk: &[u8], x_blk: &BlockQ8_0) -> f32 {
        let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        if dw == 0.0 || x_blk.d == 0.0 {
            return 0.0;
        }
        let mut sumi = 0_i32;
        for (i, &byte) in w_blk[2..18].iter().enumerate() {
            let q0 = i32::from(byte & 0x0F) - 8;
            let q1 = i32::from((byte >> 4) & 0x0F) - 8;
            sumi += q0 * i32::from(x_blk.qs[i]) + q1 * i32::from(x_blk.qs[i + 16]);
        }
        i32_to_f32(sumi) * (dw * x_blk.d)
    }

    #[inline]
    fn dot_q5_0_q8_0_blk(w_blk: &[u8], x_blk: &BlockQ8_0) -> f32 {
        let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        if dw == 0.0 || x_blk.d == 0.0 {
            return 0.0;
        }
        let qh = u32::from_le_bytes([w_blk[2], w_blk[3], w_blk[4], w_blk[5]]);
        let mut sumi = 0_i32;
        for (j, &byte) in w_blk[6..22].iter().enumerate() {
            let h0 = i32::try_from(((qh >> j) & 1) << 4).unwrap_or(0);
            let h1 = i32::try_from(((qh >> (j + 16)) & 1) << 4).unwrap_or(0);
            let q0 = (i32::from(byte & 0x0F) | h0) - 16;
            let q1 = (i32::from((byte >> 4) & 0x0F) | h1) - 16;
            sumi += q0 * i32::from(x_blk.qs[j]) + q1 * i32::from(x_blk.qs[j + 16]);
        }
        i32_to_f32(sumi) * (dw * x_blk.d)
    }

    #[inline]
    fn dot_q8_0_q8_0_blk(w_blk: &[u8], x_blk: &BlockQ8_0) -> f32 {
        let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        if dw == 0.0 || x_blk.d == 0.0 {
            return 0.0;
        }
        let mut sumi = 0_i32;
        for i in 0..32 {
            let wq = i32::from(i8::from_ne_bytes([w_blk[2 + i]]));
            sumi += wq * i32::from(x_blk.qs[i]);
        }
        i32_to_f32(sumi) * (dw * x_blk.d)
    }

    #[inline]
    fn unpack_q4_q5_k_scales_mins(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
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

    #[inline]
    fn dot_q4_k_q8_k_blk(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
        let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        let dmin = f16_to_f32_finite(u16::from_le_bytes([w_blk[2], w_blk[3]]));
        if (d == 0.0 && dmin == 0.0) || x_blk.d == 0.0 {
            return 0.0;
        }
        let (sc, mins) = unpack_q4_q5_k_scales_mins(&w_blk[4..16]);
        let qs = &w_blk[16..144];
        let mut sumi = 0_i32;
        let mut sum_mins = 0_i32;
        for chunk in 0..4 {
            let is = chunk * 2;
            let q_sub = &qs[chunk * 32..(chunk + 1) * 32];
            let x_lo = &x_blk.qs[chunk * 64..chunk * 64 + 32];
            let x_hi = &x_blk.qs[chunk * 64 + 32..chunk * 64 + 64];
            let mut s_lo = 0_i32;
            let mut s_hi = 0_i32;
            for l in 0..32 {
                let b = q_sub[l];
                s_lo += i32::from(b & 0x0F) * i32::from(x_lo[l]);
                s_hi += i32::from(b >> 4) * i32::from(x_hi[l]);
            }
            sumi += i32::from(sc[is]) * s_lo + i32::from(sc[is + 1]) * s_hi;
            let bsum_lo = i32::from(x_blk.bsums[chunk * 4]) + i32::from(x_blk.bsums[chunk * 4 + 1]);
            let bsum_hi =
                i32::from(x_blk.bsums[chunk * 4 + 2]) + i32::from(x_blk.bsums[chunk * 4 + 3]);
            sum_mins += i32::from(mins[is]) * bsum_lo + i32::from(mins[is + 1]) * bsum_hi;
        }
        (d * x_blk.d) * i32_to_f32(sumi) - (dmin * x_blk.d) * i32_to_f32(sum_mins)
    }

    #[inline]
    fn dot_q5_k_q8_k_blk(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
        let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        let dmin = f16_to_f32_finite(u16::from_le_bytes([w_blk[2], w_blk[3]]));
        if (d == 0.0 && dmin == 0.0) || x_blk.d == 0.0 {
            return 0.0;
        }
        let (sc, mins) = unpack_q4_q5_k_scales_mins(&w_blk[4..16]);
        let qh = &w_blk[16..48];
        let qs = &w_blk[48..176];
        let mut sumi = 0_i32;
        let mut sum_mins = 0_i32;
        for chunk in 0..4 {
            let is = chunk * 2;
            let u1 = 1_u8 << (chunk * 2);
            let u2 = 2_u8 << (chunk * 2);
            let q_sub = &qs[chunk * 32..(chunk + 1) * 32];
            let x_lo = &x_blk.qs[chunk * 64..chunk * 64 + 32];
            let x_hi = &x_blk.qs[chunk * 64 + 32..chunk * 64 + 64];
            let mut s_lo = 0_i32;
            let mut s_hi = 0_i32;
            for l in 0..32 {
                let b = q_sub[l];
                let hb1 = if (qh[l] & u1) != 0 { 16_i32 } else { 0_i32 };
                let hb2 = if (qh[l] & u2) != 0 { 16_i32 } else { 0_i32 };
                s_lo += (i32::from(b & 0x0F) | hb1) * i32::from(x_lo[l]);
                s_hi += (i32::from(b >> 4) | hb2) * i32::from(x_hi[l]);
            }
            sumi += i32::from(sc[is]) * s_lo + i32::from(sc[is + 1]) * s_hi;
            let bsum_lo = i32::from(x_blk.bsums[chunk * 4]) + i32::from(x_blk.bsums[chunk * 4 + 1]);
            let bsum_hi =
                i32::from(x_blk.bsums[chunk * 4 + 2]) + i32::from(x_blk.bsums[chunk * 4 + 3]);
            sum_mins += i32::from(mins[is]) * bsum_lo + i32::from(mins[is + 1]) * bsum_hi;
        }
        (d * x_blk.d) * i32_to_f32(sumi) - (dmin * x_blk.d) * i32_to_f32(sum_mins)
    }

    #[inline]
    fn dot_q6_k_q8_k_blk(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
        let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[208], w_blk[209]]));
        if d == 0.0 || x_blk.d == 0.0 {
            return 0.0;
        }
        let ql = &w_blk[0..128];
        let qh = &w_blk[128..192];
        let scales = &w_blk[192..208];
        let mut sumi = 0_i32;
        for half_idx in 0..2 {
            let n = half_idx * 128;
            let q_lo_off = half_idx * 64;
            let q_hi_off = half_idx * 32;
            let sc_base = half_idx * 8;
            for is in 0..2 {
                let l_base = is * 16;
                let sc1 = i32::from(i8::from_ne_bytes([scales[sc_base + is]]));
                let sc2 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 2]]));
                let sc3 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 4]]));
                let sc4 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 6]]));
                let (mut s1, mut s2, mut s3, mut s4) = (0_i32, 0_i32, 0_i32, 0_i32);
                for l in l_base..l_base + 16 {
                    let b_lo = ql[q_lo_off + l];
                    let b_hi = ql[q_lo_off + l + 32];
                    let hb = qh[q_hi_off + l];
                    let q1 = (i32::from(b_lo & 0x0F) | (i32::from(hb & 0x03) << 4)) - 32;
                    let q2 = (i32::from(b_hi & 0x0F) | (i32::from((hb >> 2) & 0x03) << 4)) - 32;
                    let q3 = (i32::from(b_lo >> 4) | (i32::from((hb >> 4) & 0x03) << 4)) - 32;
                    let q4 = (i32::from(b_hi >> 4) | (i32::from((hb >> 6) & 0x03) << 4)) - 32;
                    s1 += q1 * i32::from(x_blk.qs[n + l]);
                    s2 += q2 * i32::from(x_blk.qs[n + l + 32]);
                    s3 += q3 * i32::from(x_blk.qs[n + l + 64]);
                    s4 += q4 * i32::from(x_blk.qs[n + l + 96]);
                }
                sumi += sc1 * s1 + sc2 * s2 + sc3 * s3 + sc4 * s4;
            }
        }
        (d * x_blk.d) * i32_to_f32(sumi)
    }

    const TQ1_LUT: [[i8; 5]; 256] = {
        let mut table = [[0_i8; 5]; 256];
        let mut b = 0_usize;
        while b < 256 {
            let mut cur = b;
            let mut j = 0_usize;
            while j < 5 {
                let rem = match cur % 3 {
                    0 => 0_i8,
                    1 => 1_i8,
                    _ => 2_i8,
                };
                table[b][j] = rem - 1;
                cur /= 3;
                j += 1;
            }
            b += 1;
        }
        table
    };

    #[inline]
    fn dot_tq1_0_q8_k_blk(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
        let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        if d == 0.0 || x_blk.d == 0.0 {
            return 0.0;
        }
        let mut sumi = 0_i32;
        for (i, &byte) in w_blk[2..53].iter().enumerate() {
            let base = i * 5;
            let lut = TQ1_LUT[usize::from(byte)];
            for (j, &trit) in lut.iter().enumerate() {
                sumi += i32::from(trit) * i32::from(x_blk.qs[base + j]);
            }
        }
        sumi += i32::from(TQ1_LUT[usize::from(w_blk[53])][0]) * i32::from(x_blk.qs[255]);
        i32_to_f32(sumi) * (d * x_blk.d)
    }

    #[inline]
    fn dot_tq2_0_q8_k_blk(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
        let d = bf16_to_f32_finite(u16::from_le_bytes([w_blk[64], w_blk[65]]));
        if d == 0.0 || x_blk.d == 0.0 {
            return 0.0;
        }
        let mut sumi = 0_i32;
        for (i, &byte) in w_blk[..64].iter().enumerate() {
            let base = i * 4;
            for sub in 0..4 {
                let trit = i32::from((byte >> (sub * 2)) & 0x03) - 1;
                sumi += trit * i32::from(x_blk.qs[base + sub]);
            }
        }
        i32_to_f32(sumi) * (d * x_blk.d)
    }

    fn run_gemv<X, F>(
        out: &mut [f32],
        raw_bytes: &[u8],
        rows: usize,
        stride_bytes: usize,
        blk_bytes: usize,
        x_blocks: &[X],
        dot_blk: F,
    ) where
        F: Fn(&[u8], &X) -> f32,
    {
        for r in 0..rows {
            let row = &raw_bytes[r * stride_bytes..(r + 1) * stride_bytes];
            let mut acc = 0.0_f32;
            for (idx, xb) in x_blocks.iter().enumerate() {
                let off = idx * blk_bytes;
                acc += dot_blk(&row[off..off + blk_bytes], xb);
            }
            out[r] = acc;
        }
    }

    /// Computes scalar parity `M = 1` GEMV for `Q4_0` weights and pre-quantized [`BlockQ8_0`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_q4_0_q8_0(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8_0],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_0_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_q4_0_q8_0",
            QuantType::Q4_0,
            out.len(),
            weights,
            cols,
            Q4_0_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            Q4_0_BLOCK_BYTES,
            x_blocks,
            dot_q4_0_q8_0_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `Q5_0` weights and pre-quantized [`BlockQ8_0`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_q5_0_q8_0(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8_0],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_0_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_q5_0_q8_0",
            QuantType::Q5_0,
            out.len(),
            weights,
            cols,
            Q5_0_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            Q5_0_BLOCK_BYTES,
            x_blocks,
            dot_q5_0_q8_0_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `Q8_0` weights and pre-quantized [`BlockQ8_0`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_q8_0_q8_0(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8_0],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_0_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_q8_0_q8_0",
            QuantType::Q8_0,
            out.len(),
            weights,
            cols,
            Q8_0_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            Q8_0_BLOCK_BYTES,
            x_blocks,
            dot_q8_0_q8_0_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `Q4_K_M` weights and pre-quantized [`BlockQ8K`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_q4_k_q8_k(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8K],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_K_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_q4_k_q8_k",
            QuantType::Q4_K_M,
            out.len(),
            weights,
            cols,
            Q4_K_M_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            Q4_K_M_BLOCK_BYTES,
            x_blocks,
            dot_q4_k_q8_k_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `Q5_K` weights and pre-quantized [`BlockQ8K`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_q5_k_q8_k(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8K],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_K_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_q5_k_q8_k",
            QuantType::Q5_K,
            out.len(),
            weights,
            cols,
            Q5_K_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            Q5_K_BLOCK_BYTES,
            x_blocks,
            dot_q5_k_q8_k_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `Q6_K` weights and pre-quantized [`BlockQ8K`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_q6_k_q8_k(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8K],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_K_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_q6_k_q8_k",
            QuantType::Q6_K,
            out.len(),
            weights,
            cols,
            Q6_K_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            Q6_K_BLOCK_BYTES,
            x_blocks,
            dot_q6_k_q8_k_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `TQ1_0` weights and pre-quantized [`BlockQ8K`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_tq1_0_q8_k(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8K],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_K_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_tq1_0_q8_k",
            QuantType::TQ1_0,
            out.len(),
            weights,
            cols,
            TQ1_0_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            TQ1_0_BLOCK_BYTES,
            x_blocks,
            dot_tq1_0_q8_k_blk,
        );
        Ok(())
    }

    /// Computes scalar parity `M = 1` GEMV for `TQ2_0` weights and pre-quantized [`BlockQ8K`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on quant or dimension mismatch.
    pub fn gemv_tq2_0_q8_k(
        out: &mut [f32],
        weights: &TensorView<'_>,
        x_blocks: &[BlockQ8K],
    ) -> Result<(), K3Error> {
        let cols = x_blocks.len() * Q8_K_BLOCK_ELEMENTS;
        let (rows, row_bytes, raw) = validate_gemv_view(
            "gemv_tq2_0_q8_k",
            QuantType::TQ2_0,
            out.len(),
            weights,
            cols,
            TQ2_0_BLOCK_BYTES,
        )?;
        run_gemv(
            out,
            raw,
            rows,
            row_bytes,
            TQ2_0_BLOCK_BYTES,
            x_blocks,
            dot_tq2_0_q8_k_blk,
        );
        Ok(())
    }
}

#[cfg(not(feature = "with-kernels"))]
pub use portable_q8::{
    BlockQ8_0, BlockQ8K, gemv_q4_0_q8_0, gemv_q4_k_q8_k, gemv_q5_0_q8_0, gemv_q5_k_q8_k,
    gemv_q6_k_q8_k, gemv_q8_0_q8_0, gemv_tq1_0_q8_k, gemv_tq2_0_q8_k, quantize_row_q8_0,
    quantize_row_q8_k,
};

/// Converts a `128`-byte-aligned `ScratchWorkspace` byte slice into zeroed [`BlockQ8_0`] blocks.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] when `bytes` is unaligned or shorter than `count * 36`.
pub fn bytes_as_q8_0_blocks_mut(
    bytes: &mut [u8],
    count: usize,
) -> Result<&mut [BlockQ8_0], K3Error> {
    const _: () = assert!(align_of::<BlockQ8_0>() <= 128);
    const _: () = assert!(size_of::<BlockQ8_0>() == 36);
    const _: () = assert!(size_of::<BlockQ8_0>() % align_of::<BlockQ8_0>() == 0);
    if count == 0 {
        return Ok(&mut []);
    }
    let req_bytes = count.saturating_mul(size_of::<BlockQ8_0>());
    if bytes.as_ptr().align_offset(128) != 0 || bytes.len() < req_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "decode_q8_0_scratch".to_owned(),
            offset: 0,
            byte_len: req_bytes,
            buffer_len: bytes.len(),
            reason: "scratch slice for BlockQ8_0 must be 128B-aligned and fit count * 36 bytes"
                .to_owned(),
        });
    }
    let slice = &mut bytes[..req_bytes];
    slice.fill(0);
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "slice verified 128-byte aligned above, satisfying BlockQ8_0 alignment"
    )]
    let ptr = slice.as_mut_ptr().cast::<BlockQ8_0>();
    // SAFETY: `slice` is 128-byte aligned (>= align_of::<BlockQ8_0>()), covers `count * 36`
    // zero-initialized bytes (`BlockQ8_0` is #[repr(C)] with zero padding bytes and all-zero valid),
    // and is exclusively reborrowed from `bytes`.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
}

/// Converts a `128`-byte-aligned `ScratchWorkspace` byte slice into zeroed [`BlockQ8K`] blocks.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] when `bytes` is unaligned or shorter than `count * 292`.
pub fn bytes_as_q8_k_blocks_mut(
    bytes: &mut [u8],
    count: usize,
) -> Result<&mut [BlockQ8K], K3Error> {
    const _: () = assert!(align_of::<BlockQ8K>() <= 128);
    const _: () = assert!(size_of::<BlockQ8K>() == 292);
    const _: () = assert!(size_of::<BlockQ8K>() % align_of::<BlockQ8K>() == 0);
    if count == 0 {
        return Ok(&mut []);
    }
    let req_bytes = count.saturating_mul(size_of::<BlockQ8K>());
    if bytes.as_ptr().align_offset(128) != 0 || bytes.len() < req_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "decode_q8_k_scratch".to_owned(),
            offset: 0,
            byte_len: req_bytes,
            buffer_len: bytes.len(),
            reason: "scratch slice for BlockQ8K must be 128B-aligned and fit count * 292 bytes"
                .to_owned(),
        });
    }
    let slice = &mut bytes[..req_bytes];
    slice.fill(0);
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "slice verified 128-byte aligned above, satisfying BlockQ8K alignment"
    )]
    let ptr = slice.as_mut_ptr().cast::<BlockQ8K>();
    // SAFETY: `slice` is 128-byte aligned (>= align_of::<BlockQ8K>()), covers `count * 292`
    // zero-initialized bytes (`BlockQ8K` is #[repr(C)] with zero padding bytes and all-zero valid),
    // and is exclusively reborrowed from `bytes`.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
}

/// Caller-owned reusable quantization scratch (`BlockQ8_0` and `BlockQ8K`) backed by
/// `ScratchWorkspace`, caching one quantized representation per shared decode activation vector.
#[derive(Debug)]
pub struct DecodeQ8Scratch<'a> {
    q8_0: &'a mut [BlockQ8_0],
    q8_k: &'a mut [BlockQ8K],
    valid_q8_0_len: usize,
    valid_q8_k_len: usize,
}

impl<'a> DecodeQ8Scratch<'a> {
    /// Creates a decode Q8 scratch coordinator over pre-allocated `ScratchWorkspace` block slices.
    #[must_use]
    pub const fn new(q8_0: &'a mut [BlockQ8_0], q8_k: &'a mut [BlockQ8K]) -> Self {
        Self {
            q8_0,
            q8_k,
            valid_q8_0_len: 0,
            valid_q8_k_len: 0,
        }
    }

    /// Invalidates cached quantized activation blocks whenever the caller overwrites the input vector.
    pub fn invalidate(&mut self) {
        self.valid_q8_0_len = 0;
        self.valid_q8_k_len = 0;
    }

    fn ensure_q8_0(&mut self, x: &[f32]) -> Result<&[BlockQ8_0], K3Error> {
        let blocks = x.len() / 32;
        if blocks == 0 || x.len() % 32 != 0 || blocks > self.q8_0.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "decode_q8_0_scratch".to_owned(),
                offset: 0,
                byte_len: size_of_val(x),
                buffer_len: self.q8_0.len() * 32 * size_of::<f32>(),
                reason: format!(
                    "activation len {} incompatible with Q8_0 capacity {} blocks",
                    x.len(),
                    self.q8_0.len()
                ),
            });
        }
        if self.valid_q8_0_len != x.len() {
            quantize_row_q8_0(x, &mut self.q8_0[..blocks])?;
            self.valid_q8_0_len = x.len();
        }
        Ok(&self.q8_0[..blocks])
    }

    fn ensure_q8_k(&mut self, x: &[f32]) -> Result<&[BlockQ8K], K3Error> {
        let blocks = x.len() / 256;
        if blocks == 0 || x.len() % 256 != 0 || blocks > self.q8_k.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "decode_q8_k_scratch".to_owned(),
                offset: 0,
                byte_len: size_of_val(x),
                buffer_len: self.q8_k.len() * 256 * size_of::<f32>(),
                reason: format!(
                    "activation len {} incompatible with Q8K capacity {} blocks",
                    x.len(),
                    self.q8_k.len()
                ),
            });
        }
        if self.valid_q8_k_len != x.len() {
            quantize_row_q8_k(x, &mut self.q8_k[..blocks])?;
            self.valid_q8_k_len = x.len();
        }
        Ok(&self.q8_k[..blocks])
    }
}

#[derive(Clone, Copy)]
enum PreparedAct<'a> {
    Q8_0(&'a [BlockQ8_0]),
    Q8K(&'a [BlockQ8K]),
    DenseF32,
    GenericRowDot,
}

#[inline]
fn quant_row_stride_bytes(quant: QuantType, cols: usize) -> Result<usize, K3Error> {
    match quant {
        QuantType::F32 => return Ok(cols * 4),
        QuantType::F16 | QuantType::BF16 => return Ok(cols * 2),
        _ => {}
    }
    let blk_elems = quant.block_elements();
    if blk_elems == 0 || cols % blk_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gemv_decode_projection".to_owned(),
            offset: 0,
            byte_len: cols,
            buffer_len: blk_elems,
            reason: format!(
                "cols {cols} must be a multiple of block_elements {blk_elems} for {quant:?}"
            ),
        });
    }
    Ok((cols / blk_elems) * quant.block_bytes())
}

#[inline]
pub(crate) fn dot_dense_row_zero_alloc(quant: QuantType, row_bytes: &[u8], x_f32: &[f32]) -> f32 {
    const CHUNK: usize = 256;
    let cols = x_f32.len();
    let mut act_buf = [0.0_f32; CHUNK];
    match quant {
        QuantType::F32 => {
            let mut sum = 0.0_f32;
            let mut buf = [0.0_f32; CHUNK];
            let mut col = 0_usize;
            while col < cols {
                let n = (cols - col).min(CHUNK);
                for (i, dst) in buf[..n].iter_mut().enumerate() {
                    let off = (col + i) * 4;
                    let v = f32::from_le_bytes([
                        row_bytes[off],
                        row_bytes[off + 1],
                        row_bytes[off + 2],
                        row_bytes[off + 3],
                    ]);
                    *dst = sanitize_f32_weight(v);
                }
                let x_chunk = &x_f32[col..col + n];
                let term = if x_chunk.iter().all(|v| v.is_finite()) {
                    dot_f32_compat(&buf[..n], x_chunk)
                } else {
                    for (dst, &src) in act_buf[..n].iter_mut().zip(x_chunk) {
                        *dst = if src.is_finite() { src } else { 0.0_f32 };
                    }
                    dot_f32_compat(&buf[..n], &act_buf[..n])
                };
                if term.is_finite() {
                    sum += term;
                }
                col += n;
            }
            sum
        }
        QuantType::F16 | QuantType::BF16 => {
            let is_bf16 = quant == QuantType::BF16;
            let mut sum = 0.0_f32;
            let mut buf = [0.0_f32; CHUNK];
            let mut col = 0_usize;
            while col < cols {
                let n = (cols - col).min(CHUNK);
                for (i, dst) in buf[..n].iter_mut().enumerate() {
                    let off = (col + i) * 2;
                    let bits = u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]);
                    *dst = if is_bf16 {
                        bf16_to_f32_finite(bits)
                    } else {
                        f16_to_f32_finite(bits)
                    };
                }
                let x_chunk = &x_f32[col..col + n];
                let term = if x_chunk.iter().all(|v| v.is_finite()) {
                    dot_f32_compat(&buf[..n], x_chunk)
                } else {
                    for (dst, &src) in act_buf[..n].iter_mut().zip(x_chunk) {
                        *dst = if src.is_finite() { src } else { 0.0_f32 };
                    }
                    dot_f32_compat(&buf[..n], &act_buf[..n])
                };
                if term.is_finite() {
                    sum += term;
                }
                col += n;
            }
            sum
        }
        _ => 0.0,
    }
}

#[inline]
fn dot_f32_compat(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        use jero_kernels_rvv::VectorKernel as _;
        jero_kernels_rvv::BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[derive(Clone, Copy)]
struct GemvChunkCtx<'a> {
    weights: &'a TensorView<'a>,
    cols: usize,
    row_stride_bytes: usize,
    raw_bytes: &'a [u8],
    x_f32: &'a [f32],
    act: PreparedAct<'a>,
}

fn eval_gemv_chunk(
    output: &mut [f32],
    ctx: &GemvChunkCtx<'_>,
    row_start: usize,
) -> Result<(), K3Error> {
    let sub_rows = output.len();
    if sub_rows == 0 {
        return Ok(());
    }
    let byte_start = row_start * ctx.row_stride_bytes;
    let byte_end = (row_start + sub_rows) * ctx.row_stride_bytes;
    if byte_end > ctx.raw_bytes.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: ctx.weights.name().to_owned(),
            offset: byte_start,
            byte_len: sub_rows * ctx.row_stride_bytes,
            buffer_len: ctx.raw_bytes.len(),
            reason: "weight slice out of bounds during M=1 GEMV".to_owned(),
        });
    }
    let sub_bytes = &ctx.raw_bytes[byte_start..byte_end];
    match ctx.act {
        PreparedAct::Q8_0(x_q8) => {
            let sub_dims = [ctx.cols, sub_rows];
            let sub_view = TensorView::from_bytes(
                ctx.weights.name(),
                ctx.weights.quant(),
                &sub_dims,
                0,
                sub_bytes,
            )?;
            match ctx.weights.quant() {
                QuantType::Q4_0 => gemv_q4_0_q8_0(output, &sub_view, x_q8),
                QuantType::Q5_0 => gemv_q5_0_q8_0(output, &sub_view, x_q8),
                QuantType::Q8_0 => gemv_q8_0_q8_0(output, &sub_view, x_q8),
                _ => unreachable!("Q8_0 PreparedAct restricted to Q4_0/Q5_0/Q8_0"),
            }
        }
        PreparedAct::Q8K(x_q8k) => {
            let sub_dims = [ctx.cols, sub_rows];
            let sub_view = TensorView::from_bytes(
                ctx.weights.name(),
                ctx.weights.quant(),
                &sub_dims,
                0,
                sub_bytes,
            )?;
            match ctx.weights.quant() {
                QuantType::Q4_K_M => gemv_q4_k_q8_k(output, &sub_view, x_q8k),
                QuantType::Q5_K => gemv_q5_k_q8_k(output, &sub_view, x_q8k),
                QuantType::Q6_K => gemv_q6_k_q8_k(output, &sub_view, x_q8k),
                QuantType::TQ1_0 => gemv_tq1_0_q8_k(output, &sub_view, x_q8k),
                QuantType::TQ2_0 => gemv_tq2_0_q8_k(output, &sub_view, x_q8k),
                _ => unreachable!("Q8K PreparedAct restricted to K/ternary quants"),
            }
        }
        PreparedAct::DenseF32 => {
            let q = ctx.weights.quant();
            for (local_r, slot) in output.iter_mut().enumerate() {
                let row_slice = &sub_bytes
                    [local_r * ctx.row_stride_bytes..(local_r + 1) * ctx.row_stride_bytes];
                *slot = dot_dense_row_zero_alloc(q, row_slice, ctx.x_f32);
            }
            Ok(())
        }
        PreparedAct::GenericRowDot => {
            let q = ctx.weights.quant();
            for (local_r, slot) in output.iter_mut().enumerate() {
                *slot = compute_row_dot(row_start + local_r, ctx.cols, q, ctx.raw_bytes, ctx.x_f32);
            }
            Ok(())
        }
    }
}

fn recurse_gemv_4_aligned(
    output: &mut [f32],
    ctx: &GemvChunkCtx<'_>,
    row_start: usize,
    min_chunk_rows: usize,
    workers_rem: usize,
) -> Result<(), K3Error> {
    if cfg!(miri) || workers_rem <= 1 || output.len() <= min_chunk_rows || output.len() < 8 {
        return eval_gemv_chunk(output, ctx, row_start);
    }

    let mid = ((output.len() / 2) / 4) * 4;
    if mid == 0 {
        return eval_gemv_chunk(output, ctx, row_start);
    }

    let (out_lo, out_hi) = output.split_at_mut(mid);
    let left_w = (workers_rem / 2).max(1);
    let right_w = (workers_rem - left_w).max(1);
    let (res_lo, res_hi) = rayon::join(
        || recurse_gemv_4_aligned(out_lo, ctx, row_start, min_chunk_rows, left_w),
        || recurse_gemv_4_aligned(out_hi, ctx, row_start + mid, min_chunk_rows, right_w),
    );
    res_lo?;
    res_hi
}

/// Dispatches an `M = 1` linear projection (`output = weights * x`) across `num_workers` row
/// partitions aligned at multiples of `4`, routing quantized tensors to multi-row RVV 1.0 `GEMV`
/// kernels with zero heap allocations.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `weights.quant()` is unsupported, or
/// [`K3Error::TensorBoundsError`] when dimensions, block alignment, or scratch capacities
/// do not match `weights`.
pub fn gemv_decode_projection_with_workers(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_f32: &[f32],
    scratch: &mut DecodeQ8Scratch<'_>,
    num_workers: usize,
) -> Result<(), K3Error> {
    let quant = weights.quant();
    if !matches!(
        quant,
        QuantType::F32
            | QuantType::F16
            | QuantType::BF16
            | QuantType::Q4_0
            | QuantType::Q5_0
            | QuantType::Q8_0
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::TQ1_0
            | QuantType::TQ2_0
    ) {
        return Err(K3Error::CapabilityMismatch {
            architecture: "llama".to_owned(),
            reason: format!("unsupported quantization {quant:?} for M=1 GEMV projection"),
        });
    }

    let cols = x_f32.len();
    let (rows, _) = resolve_2d_dims(weights, cols)?;
    if output.len() != rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: rows * size_of::<f32>(),
            reason: format!("M=1 GEMV output len {} != weight rows {rows}", output.len()),
        });
    }
    if rows == 0 {
        return Ok(());
    }

    let (prepared, row_stride_bytes) = match quant {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 if cols % 32 == 0 => {
            let stride = quant_row_stride_bytes(quant, cols)?;
            let q8 = scratch.ensure_q8_0(x_f32)?;
            (PreparedAct::Q8_0(q8), stride)
        }
        QuantType::Q4_K_M
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::TQ1_0
        | QuantType::TQ2_0
            if cols % 256 == 0 =>
        {
            let stride = quant_row_stride_bytes(quant, cols)?;
            let q8k = scratch.ensure_q8_k(x_f32)?;
            (PreparedAct::Q8K(q8k), stride)
        }
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            let stride = quant_row_stride_bytes(quant, cols)?;
            (PreparedAct::DenseF32, stride)
        }
        QuantType::Q4_0
        | QuantType::Q5_0
        | QuantType::Q8_0
        | QuantType::Q4_K_M
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::TQ1_0
        | QuantType::TQ2_0 => (PreparedAct::GenericRowDot, 0),
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "llama".to_owned(),
                reason: format!("unsupported quantization {other:?} for M=1 GEMV projection"),
            });
        }
    };

    let raw_bytes = weights.as_bytes();
    if let PreparedAct::GenericRowDot = prepared {
        for (r, slot) in output.iter_mut().enumerate() {
            *slot = compute_row_dot(r, cols, quant, raw_bytes, x_f32);
        }
        return Ok(());
    }

    let workers = num_workers.max(1);
    let min_chunk_rows = (rows.div_ceil(workers)).max(16).next_multiple_of(4);
    let ctx = GemvChunkCtx {
        weights,
        cols,
        row_stride_bytes,
        raw_bytes,
        x_f32,
        act: prepared,
    };
    recurse_gemv_4_aligned(output, &ctx, 0, min_chunk_rows, workers)
}

/// Dispatches an `M = 1` linear projection across the active `rayon` worker pool.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] when weight dimensions or scratch capacity mismatch.
pub fn gemv_decode_projection(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_f32: &[f32],
    scratch: &mut DecodeQ8Scratch<'_>,
) -> Result<(), K3Error> {
    let workers = if cfg!(miri) {
        1
    } else {
        rayon::current_num_threads().max(1)
    };
    gemv_decode_projection_with_workers(output, weights, x_f32, scratch, workers)
}

std::thread_local! {
    static FALLBACK_Q8_WORKSPACE: std::cell::RefCell<Option<jero_core::ScratchWorkspace>> =
        const { std::cell::RefCell::new(None) };
}

/// Computes `M = 1` linear projection across `num_workers` row partitions using a reusable
/// thread-local `ScratchWorkspace`, avoiding heap allocations after initial cold sizing.
///
/// # Errors
///
/// Returns [`K3Error`] when workspace initialization fails or weight bounds mismatch.
pub fn gemv_decode_projection_with_thread_scratch(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_f32: &[f32],
    num_workers: usize,
) -> Result<(), K3Error> {
    let cols = x_f32.len();
    let small_blocks = if cols % 32 == 0 { cols / 32 } else { 0 };
    let super_blocks = if cols % 256 == 0 { cols / 256 } else { 0 };
    let small_bytes = small_blocks.saturating_mul(size_of::<BlockQ8_0>());
    let super_bytes = super_blocks.saturating_mul(size_of::<BlockQ8K>());
    let needed = small_bytes
        .saturating_add(128)
        .saturating_add(super_bytes)
        .saturating_add(128);

    FALLBACK_Q8_WORKSPACE.with(|cell| {
        let mut guard = cell.borrow_mut();
        let needs_alloc = guard.as_ref().is_none_or(|ws| ws.capacity_bytes() < needed);
        if needs_alloc {
            *guard = Some(jero_core::ScratchWorkspace::new(needed)?);
        }
        let Some(ws) = guard.as_mut() else {
            return Ok(());
        };
        ws.reset();
        let scope = ws.scope();
        let small_raw = scope.alloc_u8(small_bytes)?;
        let small_buf = bytes_as_q8_0_blocks_mut(small_raw, small_blocks)?;
        let super_raw = scope.alloc_u8(super_bytes)?;
        let super_buf = bytes_as_q8_k_blocks_mut(super_raw, super_blocks)?;
        let mut q8_scratch = DecodeQ8Scratch::new(small_buf, super_buf);
        gemv_decode_projection_with_workers(output, weights, x_f32, &mut q8_scratch, num_workers)
    })
}

#[cfg(test)]
mod tests {
    use jero_core::ScratchWorkspace;

    use super::*;

    #[test]
    fn test_layer_key_table_and_global_reuse_miri_safe() {
        let tbl = LayerKeyTable::new(2);
        assert_eq!(tbl.len(), 2);
        assert!(!tbl.is_empty());
        assert_eq!(tbl.get(0).unwrap().attn_norm, "blk.0.attn_norm.weight");
        assert_eq!(
            tbl.get(1).unwrap().attn_output_bias,
            "blk.1.attn_output.bias"
        );
        assert!(tbl.get(2).is_none());

        let g1 = ensure_global_layer_keys(4);
        let g2 = ensure_global_layer_keys(2);
        assert!(Arc::ptr_eq(&g1, &g2));
        assert!(g1.len() >= 4);
    }

    #[test]
    fn test_scratch_scope_q8_block_alignment_and_gemv_miri_safe() {
        let mut ws = ScratchWorkspace::new(4096).unwrap();
        let scope = ws.scope();
        let small_raw = scope.alloc_u8(size_of::<BlockQ8_0>()).unwrap();
        let q8_0 = bytes_as_q8_0_blocks_mut(small_raw, 1).unwrap();
        assert_eq!(q8_0.as_ptr() as usize % align_of::<BlockQ8_0>(), 0);

        let super_raw = scope.alloc_u8(size_of::<BlockQ8K>()).unwrap();
        let q8_k = bytes_as_q8_k_blocks_mut(super_raw, 1).unwrap();
        assert_eq!(q8_k.as_ptr() as usize % align_of::<BlockQ8K>(), 0);

        // Unaligned subslice fails closed
        let mut unaligned = [0_u8; 128];
        assert!(bytes_as_q8_0_blocks_mut(&mut unaligned[1..], 1).is_err());

        let mut x_32 = [0.0_f32; 32];
        for (i, slot) in x_32.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small bounded index")]
            {
                *slot = ((i % 9) as f32 - 4.0) * 0.25;
            }
        }
        let rows = 4_usize;
        let cols = 32_usize;
        let mut q8_raw = vec![0_u8; rows * 34];
        for r in 0..rows {
            let off = r * 34;
            q8_raw[off..off + 2].copy_from_slice(&0x3000_u16.to_le_bytes()); // 0.125
            for k in 0..32 {
                q8_raw[off + 2 + k] = u8::try_from((r + k) % 11).unwrap();
            }
        }
        let q8_shape = [cols, rows];
        let view = TensorView::from_bytes(
            "blk.0.attn_q.weight",
            QuantType::Q8_0,
            &q8_shape,
            0,
            &q8_raw,
        )
        .unwrap();

        let mut scratch = DecodeQ8Scratch::new(q8_0, q8_k);
        let mut out_1 = [0.0_f32; 4];
        let mut out_4 = [0.0_f32; 4];
        gemv_decode_projection_with_workers(&mut out_1, &view, &x_32, &mut scratch, 1).unwrap();
        scratch.invalidate();
        gemv_decode_projection_with_workers(&mut out_4, &view, &x_32, &mut scratch, 4).unwrap();
        assert_eq!(out_1, out_4);
        for (r, &got) in out_1.iter().enumerate() {
            let expected = compute_row_dot(r, cols, QuantType::Q8_0, &q8_raw, &x_32);
            assert!(
                (got - expected).abs() < 1.0e-2,
                "row {r}: {got} vs {expected}"
            );
        }

        // Unsupported quant fails closed
        let q2_raw = [0_u8; 84];
        let q2_view =
            TensorView::from_bytes("bad.weight", QuantType::Q2_K, &[256, 1], 0, &q2_raw).unwrap();
        let mut x_256 = [0.5_f32; 256];
        x_256[0] = f32::NAN;
        let mut bad_out = [0.0_f32; 1];
        let err = gemv_decode_projection_with_workers(
            &mut bad_out,
            &q2_view,
            &x_256[..256],
            &mut scratch,
            1,
        )
        .expect_err("Q2_K must fail closed");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));

        // Non-finite activation element is scrubbed in dot_dense_row_zero_alloc
        let row_f32 = [2.0_f32, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let dot = dot_dense_row_zero_alloc(QuantType::F32, &row_f32, &[f32::NAN, 4.0]);
        assert!((dot - 12.0).abs() < 1e-6);
    }
}
