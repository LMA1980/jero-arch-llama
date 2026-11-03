//! Large-vocabulary LM head projection with heterogeneous row-split parallel dispatch.
//!
//! Partitions output vocabulary rows across `X100` (RVV 1.0 `VLEN=256`) and `A100` (`VLEN=1024` / IME)
//! clusters on tile boundaries (`TILE_M = 8`), writing to non-overlapping contiguous slices
//! of the output logits buffer per ADR-029 and ADR-037.

use k3_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use kernels_rvv::{
    bf16_to_f32_finite, dequantize_q6_k_block, dequantize_q8_0_block, f16_to_f32_finite,
};

/// Hardware execution cluster lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// Out-of-order Cluster 0 (`X100`, cores 0-7, `VLEN=256`).
    X100,
    /// In-order AI Cluster 1 (`A100`, cores 8-15, `VLEN=1024`).
    A100,
}

impl Lane {
    /// Returns the canonical lowercase string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X100 => "x100",
            Self::A100 => "a100",
        }
    }
}

/// Execution layout and cluster placement configuration for model forward passes.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseLayout {
    /// Matrix multiplication cluster lanes and their respective fraction of output rows.
    pub matmul: Vec<(Lane, f64)>,
    /// Cluster lane dedicated to vector operations (normalization, `RoPE`, attention).
    pub vector: Lane,
}

impl PhaseLayout {
    /// Single cluster placement executing all operations on `lane`.
    #[must_use]
    pub fn single(lane: Lane) -> Self {
        Self {
            matmul: vec![(lane, 1.0)],
            vector: lane,
        }
    }

    /// Heterogeneous row-split placement with `x100_share` fraction assigned to `X100`
    /// and the remaining fraction assigned to `A100`.
    #[must_use]
    pub fn row_split(x100_share: f64) -> Self {
        let x100 = x100_share.clamp(0.0, 1.0);
        Self {
            matmul: vec![(Lane::X100, x100), (Lane::A100, 1.0 - x100)],
            vector: Lane::X100,
        }
    }
}

impl Default for PhaseLayout {
    fn default() -> Self {
        Self::row_split(0.5)
    }
}

#[cfg(not(feature = "with-kernels"))]
fn f16_to_f32_finite(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127u32 - 14;
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
fn bf16_to_f32_finite(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q8_0_block(block: &[u8; 34], out: &mut [f32; 32]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = f16_to_f32_finite(d_bits);
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for i in 0..32 {
        #[expect(clippy::cast_possible_wrap, reason = "quant byte as signed i8")]
        let q = block[2 + i] as i8;
        out[i] = d * f32::from(q);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q6_k_block(block: &[u8; 210], out: &mut [f32; 256]) {
    let d_bits = u16::from_le_bytes([block[208], block[209]]);
    let d = f16_to_f32_finite(d_bits);
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    for (half_idx, &n) in [0usize, 128usize].iter().enumerate() {
        let low_nibble_base = half_idx * 64;
        let high_bits_base = half_idx * 32;
        let sc_base = half_idx * 8;

        for l in 0..32 {
            let is = l / 16;
            #[expect(clippy::cast_possible_wrap, reason = "quant nibble offset by -32")]
            let q1 = (((ql[low_nibble_base + l] & 0x0F) | (((qh[high_bits_base + l]) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant nibble offset by -32")]
            let q2 = ((((ql[low_nibble_base + l] >> 4) & 0x0F)
                | (((qh[high_bits_base + l] >> 2) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "scale byte as signed i8")]
            let sc1 = scales[sc_base + is] as i8;
            out[n + l] = d * f32::from(sc1) * f32::from(q1);
            out[n + l + 32] = d * f32::from(sc1) * f32::from(q2);
        }
    }
}

fn dequantize_q4_0_block(block: &[u8; 18], out: &mut [f32; 32]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = f16_to_f32_finite(d_bits);
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for i in 0..16 {
        let byte = block[2 + i];
        #[expect(clippy::cast_possible_wrap, reason = "quant nibble offset by -8")]
        let x0 = (byte & 0x0F) as i8 - 8;
        #[expect(clippy::cast_possible_wrap, reason = "quant nibble offset by -8")]
        let x1 = ((byte >> 4) & 0x0F) as i8 - 8;
        out[i] = d * f32::from(x0);
        out[i + 16] = d * f32::from(x1);
    }
}

/// Computes the dot product of a single row of a quantized or dense weight matrix with `hidden`.
#[expect(
    clippy::too_many_lines,
    reason = "decodes multiple quantization formats (F32, F16, Q4_K_M, Q8_0, Q6_K)"
)]
fn compute_row_dot(
    row: usize,
    cols: usize,
    quant: QuantType,
    raw_bytes: &[u8],
    hidden: &[f32],
) -> f32 {
    match quant {
        QuantType::F32 => {
            let row_offset = row * cols * 4;
            let mut sum = 0.0_f32;
            for (i, &h) in hidden.iter().enumerate() {
                let off = row_offset + i * 4;
                if off + 4 <= raw_bytes.len() {
                    let b = &raw_bytes[off..off + 4];
                    let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    sum += val * h;
                }
            }
            sum
        }
        QuantType::F16 => {
            let row_offset = row * cols * 2;
            let mut sum = 0.0_f32;
            for (i, &h) in hidden.iter().enumerate() {
                let off = row_offset + i * 2;
                if off + 2 <= raw_bytes.len() {
                    let b = &raw_bytes[off..off + 2];
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    sum += f16_to_f32_finite(bits) * h;
                }
            }
            sum
        }
        QuantType::BF16 => {
            let row_offset = row * cols * 2;
            let mut sum = 0.0_f32;
            for (i, &h) in hidden.iter().enumerate() {
                let off = row_offset + i * 2;
                if off + 2 <= raw_bytes.len() {
                    let b = &raw_bytes[off..off + 2];
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    sum += bf16_to_f32_finite(bits) * h;
                }
            }
            sum
        }
        QuantType::Q8_0 => {
            let blocks_per_row = cols / 32;
            let bytes_per_row = blocks_per_row * 34;
            let row_offset = row * bytes_per_row;
            let mut sum = 0.0_f32;
            let mut dequant_buf = [0.0_f32; 32];
            for b in 0..blocks_per_row {
                let blk_off = row_offset + b * 34;
                if blk_off + 34 <= raw_bytes.len() {
                    let mut block = [0_u8; 34];
                    block.copy_from_slice(&raw_bytes[blk_off..blk_off + 34]);
                    dequantize_q8_0_block(&block, &mut dequant_buf);
                    let h_start = b * 32;
                    let h_chunk = &hidden[h_start..(h_start + 32).min(hidden.len())];
                    for (k, &h) in h_chunk.iter().enumerate() {
                        sum += dequant_buf[k] * h;
                    }
                }
            }
            sum
        }
        QuantType::Q6_K => {
            let blocks_per_row = cols / 256;
            let bytes_per_row = blocks_per_row * 210;
            let row_offset = row * bytes_per_row;
            let mut sum = 0.0_f32;
            let mut dequant_buf = [0.0_f32; 256];
            for b in 0..blocks_per_row {
                let blk_off = row_offset + b * 210;
                if blk_off + 210 <= raw_bytes.len() {
                    let mut block = [0_u8; 210];
                    block.copy_from_slice(&raw_bytes[blk_off..blk_off + 210]);
                    dequantize_q6_k_block(&block, &mut dequant_buf);
                    let h_start = b * 256;
                    let h_chunk = &hidden[h_start..(h_start + 256).min(hidden.len())];
                    for (k, &h) in h_chunk.iter().enumerate() {
                        sum += dequant_buf[k] * h;
                    }
                }
            }
            sum
        }
        QuantType::Q4_0 => {
            let blocks_per_row = cols / 32;
            let bytes_per_row = blocks_per_row * 18;
            let row_offset = row * bytes_per_row;
            let mut sum = 0.0_f32;
            let mut dequant_buf = [0.0_f32; 32];
            for b in 0..blocks_per_row {
                let blk_off = row_offset + b * 18;
                if blk_off + 18 <= raw_bytes.len() {
                    let mut block = [0_u8; 18];
                    block.copy_from_slice(&raw_bytes[blk_off..blk_off + 18]);
                    dequantize_q4_0_block(&block, &mut dequant_buf);
                    let h_start = b * 32;
                    let h_chunk = &hidden[h_start..(h_start + 32).min(hidden.len())];
                    for (k, &h) in h_chunk.iter().enumerate() {
                        sum += dequant_buf[k] * h;
                    }
                }
            }
            sum
        }
        QuantType::Q4_K_M => {
            let blocks_per_row = cols / 256;
            let bytes_per_row = blocks_per_row * 144;
            let row_offset = row * bytes_per_row;
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let blk_off = row_offset + b * 144;
                if blk_off + 144 <= raw_bytes.len() {
                    let block = &raw_bytes[blk_off..blk_off + 144];
                    let d_bits = u16::from_le_bytes([block[0], block[1]]);
                    let d = f16_to_f32_finite(d_bits);
                    let h_start = b * 256;
                    let h_chunk = &hidden[h_start..(h_start + 256).min(hidden.len())];
                    for (k, &h) in h_chunk.iter().enumerate() {
                        let q_idx = 16 + (k / 2);
                        #[expect(clippy::cast_possible_wrap, reason = "nibble as signed i8")]
                        let nibble = if k % 2 == 0 {
                            (block[q_idx] & 0x0F) as i8 - 8
                        } else {
                            ((block[q_idx] >> 4) & 0x0F) as i8 - 8
                        };
                        sum += d * f32::from(nibble) * h;
                    }
                }
            }
            sum
        }
        QuantType::TQ1_0 | QuantType::TQ2_0 => 0.0,
    }
}

/// Executes parallel heterogeneous row-split LM head projection across `X100` and `A100` clusters.
///
/// Designed for large-vocabulary models ($\ge 128\text{K}$ tokens, such as Llama-3's 128,256 tokens).
/// Partitions output rows on tile boundaries (`TILE_M = 8`) proportional to calibrated throughput shares.
/// Writes directly to non-overlapping partitions of output `logits: &mut [f32]` with zero contention.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `output_weight` is not 2D or dimensions do not match
/// `logits.len()` and `hidden.len()`.
pub fn project_lm_head_row_split(
    logits: &mut [f32],
    hidden: &[f32],
    output_weight: &TensorView<'_>,
    layout: &PhaseLayout,
) -> Result<(), K3Error> {
    if logits.is_empty() || hidden.is_empty() {
        return Ok(());
    }

    let shape = output_weight.shape();
    if shape.len() != 2 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: output_weight.name().to_owned(),
            offset: output_weight.offset(),
            byte_len: output_weight.byte_len(),
            buffer_len: output_weight.as_bytes().len(),
            reason: format!("expected 2D tensor for LM head projection, got shape {shape:?}"),
        });
    }

    let (rows, cols) = if logits.len() == shape[0] && hidden.len() == shape[1] {
        (shape[0], shape[1])
    } else if logits.len() == shape[1] && hidden.len() == shape[0] {
        (shape[1], shape[0])
    } else {
        return Err(K3Error::TensorBoundsError {
            tensor_name: output_weight.name().to_owned(),
            offset: output_weight.offset(),
            byte_len: output_weight.byte_len(),
            buffer_len: output_weight.as_bytes().len(),
            reason: format!(
                "shape {shape:?} does not match logits len {} and hidden len {}",
                logits.len(),
                hidden.len()
            ),
        });
    };

    let quant = output_weight.quant();
    let raw = output_weight.as_bytes();

    let is_large_vocab = rows >= 128_000;
    let has_multiple_lanes = layout.matmul.len() > 1;

    // Single-lane fallback for small vocabularies or single-lane layouts
    if (!is_large_vocab && !has_multiple_lanes) || layout.matmul.is_empty() {
        for (r, slot) in logits.iter_mut().enumerate() {
            *slot = compute_row_dot(r, cols, quant, raw, hidden);
        }
        return Ok(());
    }

    // Partition on TILE_M = 8 boundaries
    let x100_share = layout
        .matmul
        .iter()
        .find(|(lane, _)| *lane == Lane::X100)
        .map_or(0.5, |(_, share)| *share)
        .clamp(0.0, 1.0);

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "bounded positive rows and share conversion"
    )]
    let raw_x100_rows = ((rows as f64) * x100_share).round() as usize;
    // Align to 8-row tile boundary; last tile absorbs remainder if rows % 8 != 0 (Edge Case 15)
    let x100_rows = (raw_x100_rows / 8 * 8).min(rows);

    let (x100_out, a100_out) = logits.split_at_mut(x100_rows);

    std::thread::scope(|s| {
        s.spawn(|| {
            for (i, slot) in x100_out.iter_mut().enumerate() {
                *slot = compute_row_dot(i, cols, quant, raw, hidden);
            }
        });
        s.spawn(|| {
            for (i, slot) in a100_out.iter_mut().enumerate() {
                *slot = compute_row_dot(x100_rows + i, cols, quant, raw, hidden);
            }
        });
    });

    Ok(())
}
