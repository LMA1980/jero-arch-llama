//! Large-vocabulary LM head projection with cached repacked weights and row-parallel dispatch.
//!
//! Partitions output vocabulary rows across `TILE_M = 8`-aligned `64`-row panels on the active
//! cluster `WorkerPool` (`X100` `VLEN=256` or `A100` `VLEN=1024` / `IME`), writing directly to
//! non-overlapping contiguous slices of the output logits buffer per ADR-029 and ADR-037.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    bf16_to_f32_finite, dequantize_q4_k_m_block, dequantize_q5_k_block, dequantize_q6_k_block,
    dequantize_q8_0_block, f16_to_f32_finite,
};

#[cfg(not(feature = "with-kernels"))]
use crate::repack_cache::{
    bf16_to_f32_finite, dequantize_q4_k_m_block, dequantize_q5_k_block, dequantize_q6_k_block,
    dequantize_q8_0_block, f16_to_f32_finite,
};
use crate::repack_cache::{dequantize_q4_0_block, dequantize_q5_0_block, sanitize_f32_weight};

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

fn block_dequant_dot<const N: usize, const B: usize>(
    row: usize,
    cols: usize,
    raw_bytes: &[u8],
    hidden: &[f32],
    dequant: fn(&[u8; B], &mut [f32; N]),
) -> f32 {
    let blocks_per_row = cols / N;
    let bytes_per_row = blocks_per_row * B;
    let row_offset = row * bytes_per_row;
    let mut sum = 0.0_f32;
    let mut dequant_buf = [0.0_f32; N];
    for b in 0..blocks_per_row {
        let blk_off = row_offset + b * B;
        if let Some(slice) = raw_bytes.get(blk_off..blk_off + B)
            && let Ok(block) = <&[u8; B]>::try_from(slice)
        {
            dequant(block, &mut dequant_buf);
            let h_start = b * N;
            let h_chunk = &hidden[h_start..(h_start + N).min(hidden.len())];
            for (k, &h) in h_chunk.iter().enumerate() {
                sum += dequant_buf[k] * h;
            }
        }
    }
    sum
}

/// Computes the exact dequantized `f32` dot product of row `row` (`cols` elements) with `hidden`.
#[must_use]
pub fn compute_row_dot(
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
            for (i, &h) in hidden.iter().enumerate().take(cols) {
                let off = row_offset + i * 4;
                if off + 4 <= raw_bytes.len() {
                    let b = &raw_bytes[off..off + 4];
                    let val = sanitize_f32_weight(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    sum += val * h;
                }
            }
            sum
        }
        QuantType::F16 | QuantType::BF16 => {
            let row_offset = row * cols * 2;
            let mut sum = 0.0_f32;
            for (i, &h) in hidden.iter().enumerate().take(cols) {
                let off = row_offset + i * 2;
                if off + 2 <= raw_bytes.len() {
                    let b = &raw_bytes[off..off + 2];
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    let val = if quant == QuantType::BF16 {
                        bf16_to_f32_finite(bits)
                    } else {
                        f16_to_f32_finite(bits)
                    };
                    sum += val * h;
                }
            }
            sum
        }
        QuantType::Q8_0 => {
            block_dequant_dot::<32, 34>(row, cols, raw_bytes, hidden, dequantize_q8_0_block)
        }
        QuantType::Q4_0 => {
            block_dequant_dot::<32, 18>(row, cols, raw_bytes, hidden, dequantize_q4_0_block)
        }
        QuantType::Q5_0 => {
            block_dequant_dot::<32, 22>(row, cols, raw_bytes, hidden, dequantize_q5_0_block)
        }
        QuantType::Q4_K_M => {
            block_dequant_dot::<256, 144>(row, cols, raw_bytes, hidden, dequantize_q4_k_m_block)
        }
        QuantType::Q5_K => {
            block_dequant_dot::<256, 176>(row, cols, raw_bytes, hidden, dequantize_q5_k_block)
        }
        QuantType::Q6_K => {
            block_dequant_dot::<256, 210>(row, cols, raw_bytes, hidden, dequantize_q6_k_block)
        }
        QuantType::Q4_1
        | QuantType::Q5_1
        | QuantType::Q8_1
        | QuantType::Q2_K
        | QuantType::Q3_K
        | QuantType::IQ4_NL
        | QuantType::TQ1_0
        | QuantType::TQ2_0
        | QuantType::I2_S
        | QuantType::MXFP4 => 0.0,
    }
}

/// Executes row-parallel LM head projection using cached `128-byte`-aligned repacked weights
/// and the active cluster `WorkerPool` (`X100` or `A100`).
///
/// Designed for large-vocabulary models ($\ge 128\text{K}$ tokens, such as Llama-3's 128,256
/// and Qwen2.5's 151,936 tokens). Partitions output rows on `TILE_M = 8`-aligned `64`-row panels
/// writing directly to non-overlapping slices of `logits: &mut [f32]` with zero lock contention.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `output_weight` is not 2D or dimensions do not match
/// `logits.len()` and `hidden.len()`, or [`K3Error::CapabilityMismatch`] if `output_weight` uses
/// an unsupported quantization format.
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

    let parallel = !layout.matmul.is_empty();
    let quant = output_weight.quant();
    if crate::repack_cache::quant_block_geometry(quant).is_none() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "llama".to_owned(),
            reason: format!(
                "LM head tensor {} uses unsupported quantization {quant:?}",
                output_weight.name()
            ),
        });
    }
    if crate::repack_cache::is_oversized_dense(quant, rows, cols) {
        let raw = output_weight.as_bytes();
        let stride_bytes = match quant {
            QuantType::F32 => cols * 4,
            QuantType::F16 | QuantType::BF16 => cols * 2,
            _ => 0,
        };
        if stride_bytes > 0 {
            crate::repack_cache::par_for_each_row_tile(
                logits,
                crate::repack_cache::PANEL_ROWS,
                parallel,
                |panel_idx, slice| {
                    let row_start = panel_idx * crate::repack_cache::PANEL_ROWS;
                    for (local_r, slot) in slice.iter_mut().enumerate() {
                        let row_raw = &raw[(row_start + local_r) * stride_bytes..][..stride_bytes];
                        *slot =
                            crate::decode_gemv::dot_dense_row_zero_alloc(quant, row_raw, hidden);
                    }
                },
            );
            return Ok(());
        }
        crate::repack_cache::par_for_each_row_tile(
            logits,
            crate::repack_cache::PANEL_ROWS,
            parallel,
            |panel_idx, slice| {
                let row_start = panel_idx * crate::repack_cache::PANEL_ROWS;
                for (local_r, slot) in slice.iter_mut().enumerate() {
                    *slot = compute_row_dot(row_start + local_r, cols, quant, raw, hidden);
                }
            },
        );
        return Ok(());
    }

    let workers = if parallel && !cfg!(miri) {
        rayon::current_num_threads().max(1)
    } else {
        1
    };
    crate::decode_gemv::gemv_decode_projection_with_thread_scratch(
        logits,
        output_weight,
        hidden,
        workers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_row_dot_and_repacked_q4_k_m_6bit_scales_and_mins() {
        // Construct 2 rows of 256 elements in Q4_K_M with non-zero dmin and 6-bit high-bit scales/mins
        let rows = 2;
        let cols = 256;
        let mut raw = vec![0_u8; rows * 144];

        // Row 0: d = 0.5 (0x3800), dmin = 0.25 (0x3400)
        raw[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        raw[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
        // Exercise both j in 0..4 and j in 4..8 packed 6-bit scale/min fields
        raw[4..16].copy_from_slice(&[
            0x95, 0xA6, 0x57, 0x68, // low 6 bits = sc[0..4], top 2 bits -> sc[4..8]
            0x83, 0x44, 0xC5, 0x06, // low 6 bits = m[0..4], top 2 bits -> m[4..8]
            0x37, 0x48, 0x59, 0x6A, // low 4 bits -> sc[4..8], high 4 bits -> m[4..8]
        ]);
        for (i, b) in raw[16..144].iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "bounded byte pattern")]
            {
                *b = (((i * 7 + 3) & 0x0F) | (((i * 5 + 9) & 0x0F) << 4)) as u8;
            }
        }

        // Row 1: distinct scales and nibbles
        raw[144..146].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
        raw[146..148].copy_from_slice(&0x3800_u16.to_le_bytes()); // dmin = 0.5
        raw[148..160].copy_from_slice(&[3, 5, 7, 9, 2, 4, 6, 8, 0x21, 0x43, 0x65, 0x87]);
        for (i, b) in raw[160..288].iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "bounded byte pattern")]
            {
                *b = (((i * 11 + 1) & 0x0F) | (((i * 3 + 12) & 0x0F) << 4)) as u8;
            }
        }

        let hidden: Vec<f32> = (0..cols)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small index")]
                let f = (i as f32) * 0.01 - 1.2;
                f
            })
            .collect();

        let ref0 = compute_row_dot(0, cols, QuantType::Q4_K_M, &raw, &hidden);
        let ref1 = compute_row_dot(1, cols, QuantType::Q4_K_M, &raw, &hidden);

        let shape = [cols, rows];
        let view = TensorView::from_bytes("output.weight", QuantType::Q4_K_M, &shape, 0, &raw)
            .expect("valid Q4_K_M view");
        let mut logits = [0.0_f32; 2];
        project_lm_head_row_split(&mut logits, &hidden, &view, &PhaseLayout::default())
            .expect("repacked Q4_K_M projection succeeds");

        let rel_err0 = (logits[0] - ref0).abs() / ref0.abs().max(1.0);
        let rel_err1 = (logits[1] - ref1).abs() / ref1.abs().max(1.0);
        assert!(
            rel_err0 < 5e-3,
            "row 0 mismatch: got {}, ref {ref0} (rel_err {rel_err0})",
            logits[0]
        );
        assert!(
            rel_err1 < 5e-3,
            "row 1 mismatch: got {}, ref {ref1} (rel_err {rel_err1})",
            logits[1]
        );
    }
}
