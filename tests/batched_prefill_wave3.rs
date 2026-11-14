//! Wave 3 verification suite: direct output-slice batched GEMM (`M = 1, 8, 32`) without
//! `panel_major` staging allocations, batched causal GQA (`gqa_causal_attention_prefill_f32`),
//! and non-zero `start_position` chunked `u_batch` prefill with `compute_logits` gating.

use jero_arch_llama::{
    CachedMatrix, LlamaRecipe, PANEL_ROWS, QuantizedBatchAct, compute_row_dot,
    thread_decode_1d_count,
};
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, KvGeometry, QuantType, TensorView,
};

fn mod_byte(val: usize, modulus: usize) -> u8 {
    u8::try_from(val % modulus).unwrap_or(0)
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0);
    let mant = bits & 0x007F_FFFF;
    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        return sign | 0x7C00;
    }
    let h_exp = exp - 127 + 15;
    if h_exp >= 31 {
        return sign | 0x7BFF;
    }
    if h_exp <= 0 {
        return sign;
    }
    let exp_bits = u16::try_from(h_exp).unwrap_or(0) << 10;
    let mant_bits = u16::try_from(mant >> 13).unwrap_or(0);
    sign | exp_bits | mant_bits
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    u16::try_from(val.to_bits() >> 16).unwrap_or(0)
}

fn fill_dense_f32(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    let mut out = vec![0_u8; rows * cols * 4];
    for idx in 0..rows * cols {
        #[expect(
            clippy::cast_precision_loss,
            reason = "bounded test modulo fits in f32"
        )]
        let val = (((idx * 19 + seed) % 97) as f32 - 48.0) * 0.02;
        out[idx * 4..idx * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    out
}

fn fill_dense_half(rows: usize, cols: usize, quant: QuantType, seed: usize) -> Vec<u8> {
    let mut out = vec![0_u8; rows * cols * 2];
    let (mul, modulus, offset) = match quant {
        QuantType::F16 => (17, 67, 33.0),
        QuantType::BF16 => (23, 73, 36.0),
        _ => unreachable!(),
    };
    for idx in 0..rows * cols {
        #[expect(
            clippy::cast_precision_loss,
            reason = "bounded test modulo fits in f32"
        )]
        let val = (((idx * mul + seed) % modulus) as f32 - offset) * 0.025;
        let bits = if quant == QuantType::F16 {
            f32_to_f16_bits(val)
        } else {
            f32_to_bf16_bits(val)
        };
        out[idx * 2..idx * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    out
}

fn fill_q8_0(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    let blocks = (rows * cols) / 32;
    let mut out = vec![0_u8; blocks * 34];
    for b in 0..blocks {
        let off = b * 34;
        #[expect(clippy::cast_precision_loss, reason = "bounded modulo fits in f32")]
        let d = 0.015_f32 + ((b + seed) % 7) as f32 * 0.004;
        out[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for k in 0..32 {
            let signed = i8::try_from((b * 11 + k * 7 + seed) % 81).unwrap_or(0) - 40;
            out[off + 2 + k] = signed.to_ne_bytes()[0];
        }
    }
    out
}

fn fill_q4_0_or_q5_0(rows: usize, cols: usize, quant: QuantType, seed: usize) -> Vec<u8> {
    let blocks = (rows * cols) / 32;
    let b_bytes = quant.block_bytes();
    let mut out = vec![0_u8; blocks * b_bytes];
    for b in 0..blocks {
        let off = b * b_bytes;
        let d = if quant == QuantType::Q4_0 { 0.04 } else { 0.03 };
        out[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        let qs_off = if quant == QuantType::Q5_0 {
            let add = u32::try_from(b * 13).unwrap_or(0);
            let qh = 0x5A3C_96F1_u32.wrapping_add(add);
            out[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
            off + 6
        } else {
            off + 2
        };
        for i in 0..16 {
            let lo = mod_byte(b * 5 + i + seed, 16);
            let hi = mod_byte(b * 3 + i * 3 + seed, 16);
            out[qs_off + i] = lo | (hi << 4);
        }
    }
    out
}

fn fill_k_quant(rows: usize, cols: usize, quant: QuantType, seed: usize) -> Vec<u8> {
    let sblks = (rows * cols) / 256;
    let b_bytes = quant.block_bytes();
    let mut out = vec![0_u8; sblks * b_bytes];
    for sb in 0..sblks {
        let off = sb * b_bytes;
        match quant {
            QuantType::Q4_K_M | QuantType::Q5_K => {
                out[off..off + 2].copy_from_slice(&f32_to_f16_bits(0.015).to_le_bytes());
                out[off + 2..off + 4].copy_from_slice(&f32_to_f16_bits(0.004).to_le_bytes());
                for j in 0..12 {
                    out[off + 4 + j] = mod_byte(sb * 7 + j * 11 + seed, 64) + 1;
                }
                let qs_off = if quant == QuantType::Q5_K {
                    for l in 0..32 {
                        out[off + 16 + l] = mod_byte(sb + l * 13 + seed, 256);
                    }
                    off + 48
                } else {
                    off + 16
                };
                for l in 0..128 {
                    let lo = mod_byte(sb * 2 + l + seed, 16);
                    let hi = mod_byte(sb * 5 + l + seed, 16);
                    out[qs_off + l] = lo | (hi << 4);
                }
            }
            QuantType::Q6_K => {
                out[off + 208..off + 210].copy_from_slice(&f32_to_f16_bits(0.012).to_le_bytes());
                for l in 0..128 {
                    out[off + l] = mod_byte(sb * 7 + l * 3 + seed, 256);
                }
                for l in 0..64 {
                    out[off + 128 + l] = mod_byte(sb * 11 + l * 5 + seed, 256);
                }
                for l in 0..16 {
                    let s = i8::try_from((sb + l * 3 + seed) % 21).unwrap_or(0) - 10;
                    out[off + 192 + l] = s.to_ne_bytes()[0];
                }
            }
            _ => unreachable!(),
        }
    }
    out
}

fn fill_quant_bytes(rows: usize, cols: usize, quant: QuantType, seed: usize) -> Vec<u8> {
    match quant {
        QuantType::F32 => fill_dense_f32(rows, cols, seed),
        QuantType::F16 | QuantType::BF16 => fill_dense_half(rows, cols, quant, seed),
        QuantType::Q8_0 => fill_q8_0(rows, cols, seed),
        QuantType::Q4_0 | QuantType::Q5_0 => fill_q4_0_or_q5_0(rows, cols, quant, seed),
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            fill_k_quant(rows, cols, quant, seed)
        }
        _ => unreachable!(),
    }
}

fn write_finite_gguf_payloads(raw: &mut [u8]) {
    let parsed = GgufFile::parse(raw).unwrap();
    let descriptors = parsed.metadata().tensors().to_vec();
    for (idx, desc) in descriptors.iter().enumerate() {
        let start = desc.absolute_offset();
        let end = start + desc.byte_len();
        let payload = if desc.shape().len() == 1 {
            let elem_count = desc.num_elements();
            let mut buf = Vec::with_capacity(end - start);
            for e in 0..elem_count {
                #[expect(clippy::cast_precision_loss, reason = "bounded modulo fits in f32")]
                let val = if desc.name().ends_with("norm.weight") {
                    0.85 + ((e + idx) % 13) as f32 * 0.02
                } else {
                    (((e * 7 + idx * 11) % 29) as f32 - 14.0) * 0.04
                };
                match desc.quant() {
                    QuantType::F32 => buf.extend_from_slice(&val.to_le_bytes()),
                    QuantType::F16 => buf.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes()),
                    QuantType::BF16 => buf.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes()),
                    other => unreachable!("unexpected 1D quant {other:?}"),
                }
            }
            buf
        } else {
            let cols = desc.shape()[0];
            let rows = desc.shape()[1];
            fill_quant_bytes(rows, cols, desc.quant(), idx + 1)
        };
        assert_eq!(payload.len(), end - start);
        raw[start..end].copy_from_slice(&payload);
    }
}

fn build_wave3_test_gguf(arch: &str, weight_quant: QuantType, hidden: usize) -> Vec<u8> {
    let ffn = hidden;
    let n_q = 4_usize;
    let n_kv = 2_usize;
    let head_dim = hidden / n_q;
    let kv_dim = n_kv * head_dim;
    let vocab = 64_usize;

    let kvs = [
        (format!("{arch}.block_count"), GgufValue::Uint32(2)),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(u32::try_from(hidden).unwrap()),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(u32::try_from(ffn).unwrap()),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(u32::try_from(n_q).unwrap()),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(u32::try_from(n_kv).unwrap()),
        ),
        (
            format!("{arch}.attention.key_length"),
            GgufValue::Uint32(u32::try_from(head_dim).unwrap()),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(10_000.0),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(1e-5),
        ),
        (format!("{arch}.context_length"), GgufValue::Uint32(64)),
    ];
    let kvs_refs: Vec<(&str, GgufValue<'_>)> =
        kvs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

    let norm_quant = if arch == "qwen2" {
        QuantType::BF16
    } else {
        QuantType::F16
    };
    let emb_dims = [hidden, vocab];
    let norm_dims = [hidden];
    let q_dims = [hidden, hidden];
    let kv_dims = [hidden, kv_dim];
    let o_dims = [hidden, hidden];
    let ffn_dims = [hidden, ffn];
    let q_bias_dims = [hidden];
    let kv_bias_dims = [kv_dim];

    let mut specs: Vec<(&str, QuantType, &[usize])> = vec![
        ("token_embd.weight", QuantType::Q8_0, &emb_dims),
        ("blk.0.attn_norm.weight", norm_quant, &norm_dims),
        ("blk.0.attn_q.weight", weight_quant, &q_dims),
        ("blk.0.attn_k.weight", QuantType::Q8_0, &kv_dims),
        ("blk.0.attn_v.weight", QuantType::Q8_0, &kv_dims),
        ("blk.0.attn_output.weight", weight_quant, &o_dims),
        ("blk.0.ffn_norm.weight", QuantType::F32, &norm_dims),
        ("blk.0.ffn_gate.weight", weight_quant, &ffn_dims),
        ("blk.0.ffn_up.weight", weight_quant, &ffn_dims),
        ("blk.0.ffn_down.weight", weight_quant, &ffn_dims),
        ("blk.1.attn_norm.weight", norm_quant, &norm_dims),
        ("blk.1.attn_q.weight", QuantType::Q8_0, &q_dims),
        ("blk.1.attn_k.weight", QuantType::Q4_0, &kv_dims),
        ("blk.1.attn_v.weight", QuantType::Q5_0, &kv_dims),
        ("blk.1.attn_output.weight", QuantType::Q8_0, &o_dims),
        ("blk.1.ffn_norm.weight", norm_quant, &norm_dims),
        ("blk.1.ffn_gate.weight", QuantType::Q4_0, &ffn_dims),
        ("blk.1.ffn_up.weight", QuantType::Q5_0, &ffn_dims),
        ("blk.1.ffn_down.weight", weight_quant, &ffn_dims),
        ("output_norm.weight", QuantType::F32, &norm_dims),
        ("output.weight", QuantType::F32, &emb_dims),
    ];
    if arch == "qwen2" {
        specs.extend_from_slice(&[
            ("blk.0.attn_q.bias", QuantType::F16, &q_bias_dims),
            ("blk.0.attn_k.bias", QuantType::F32, &kv_bias_dims),
            ("blk.0.attn_v.bias", QuantType::BF16, &kv_bias_dims),
            ("blk.0.attn_output.bias", QuantType::F32, &q_bias_dims),
            ("blk.1.attn_q.bias", QuantType::F32, &q_bias_dims),
            ("blk.1.attn_k.bias", QuantType::F16, &kv_bias_dims),
            ("blk.1.attn_v.bias", QuantType::F32, &kv_bias_dims),
        ]);
    }

    let mut raw = create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs_refs, &specs);
    write_finite_gguf_payloads(&mut raw);
    raw
}

fn assert_slice_close(lhs: &[f32], rhs: &[f32], tol: f32, context: &str) {
    assert_eq!(lhs.len(), rhs.len(), "{context}: length mismatch");
    for (idx, (&a, &b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{context}[{idx}] non-finite: {a} vs {b}"
        );
        let rel = (a - b).abs() / b.abs().max(1.0);
        assert!(
            rel < tol,
            "{context}[{idx}] mismatch: {a} vs {b} (rel={rel}, tol={tol})"
        );
    }
}

#[test]
fn test_wave3_parallel_scatter_and_quantize_into_across_m_1_8_32() {
    let rows = 192_usize; // Multiple 64-row panels (> PANEL_ROWS=64) exercising parallel scatter
    let cols = 256_usize;
    assert!(rows > PANEL_ROWS);

    let admitted = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q8_0,
        QuantType::Q4_0,
        QuantType::Q5_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
    ];

    let mut reusable_act = QuantizedBatchAct::new();
    for &m in &[1_usize, 8, 32] {
        let input: Vec<f32> = (0..m * cols)
            .map(|i| {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "bounded test modulo fits in f32"
                )]
                let val = (((i * 23 + m * 11) % 79) as f32 - 39.0) * 0.025;
                val
            })
            .collect();
        reusable_act.quantize_into(&input, m, cols).unwrap();
        let fresh_act = QuantizedBatchAct::try_quantize(&input, m, cols).unwrap();

        for (seed, &quant) in admitted.iter().enumerate() {
            let raw = fill_quant_bytes(rows, cols, quant, seed + 3);
            let dims = [cols, rows];
            let view = TensorView::from_bytes("w.weight", quant, &dims, 0, &raw).unwrap();
            let cached = CachedMatrix::repack(&view, rows, cols).unwrap();

            let mut out_reused = vec![0.0_f32; m * rows];
            let mut out_fresh = vec![0.0_f32; m * rows];
            cached
                .project_batch_with_act(&mut out_reused, &input, Some(&reusable_act), m, true)
                .unwrap();
            cached
                .project_batch_with_act(&mut out_fresh, &input, Some(&fresh_act), m, true)
                .unwrap();
            assert_eq!(out_reused, out_fresh);

            let is_dense = matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16);
            let tol = if is_dense { 1.0e-5 } else { 1.5e-2 };
            for tok in 0..m {
                let x_row = &input[tok * cols..(tok + 1) * cols];
                #[expect(clippy::cast_precision_loss, reason = "cols fits in f32")]
                let scale = (x_row.iter().map(|v| v * v).sum::<f32>().sqrt()
                    * (cols as f32).sqrt())
                .max(1.0);
                for r in 0..rows {
                    let expected = compute_row_dot(r, cols, quant, &raw, x_row);
                    let actual = out_reused[tok * rows + r];
                    let rel = (actual - expected).abs() / scale.max(expected.abs());
                    assert!(
                        rel < tol,
                        "M={m} quant={quant:?} tok={tok} row={r}: {actual} vs {expected} (rel={rel})"
                    );
                }
            }
        }
    }
}

#[test]
fn test_wave3_batched_prefill_matches_sequential_across_m_1_8_32_and_quants() {
    let test_cases = [
        ("llama", QuantType::F32),
        ("qwen2", QuantType::Q4_K_M),
        ("llama", QuantType::Q5_K),
        ("qwen2", QuantType::Q6_K),
    ];

    for (arch, weight_quant) in test_cases {
        let hidden = 256_usize; // > PANEL_ROWS (64) and aligned to 256-elem K-quant superblocks
        let raw = build_wave3_test_gguf(arch, weight_quant, hidden);
        let gguf = GgufFile::parse(&raw).unwrap();
        let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
        let cache = recipe.build_repack_cache(&gguf).unwrap();
        let kv_geom = KvGeometry::new(2, 2, hidden / 4, 64, None).unwrap();
        let kv_len = kv_geom.required_f32_len().unwrap();

        for &m in &[1_usize, 8, 32] {
            let prompt: Vec<u32> = (0..m)
                .map(|i| u32::try_from((i * 7 + 3) % 64).unwrap_or(0))
                .collect();

            let mut logits_batched = vec![0.0_f32; 64];
            let mut kv_batched = vec![0.0_f32; kv_len];
            let mut ctx_batched =
                ExecutionContext::new_prefill(&prompt, 64, &mut logits_batched, &mut kv_batched)
                    .unwrap()
                    .with_kv_geometry(kv_geom)
                    .unwrap();
            let decodes_before = thread_decode_1d_count();
            recipe
                .prefill_step_with_cache(&mut ctx_batched, &gguf, &cache)
                .unwrap();
            assert_eq!(thread_decode_1d_count(), decodes_before);
            assert_eq!(ctx_batched.position(), m);

            let mut logits_seq = vec![0.0_f32; 64];
            let mut kv_seq = vec![0.0_f32; kv_len];
            let mut ctx_seq =
                ExecutionContext::new_prefill(&prompt[..1], 64, &mut logits_seq, &mut kv_seq)
                    .unwrap()
                    .with_kv_geometry(kv_geom)
                    .unwrap();
            recipe
                .prefill_step_with_cache(&mut ctx_seq, &gguf, &cache)
                .unwrap();
            for &tok in &prompt[1..] {
                ctx_seq.prepare_decode(tok).unwrap();
                recipe
                    .decode_step_with_cache(&mut ctx_seq, &gguf, &cache)
                    .unwrap();
            }

            let tol = if weight_quant == QuantType::F32 {
                1.0e-5
            } else {
                2.0e-3
            };
            assert_slice_close(
                &logits_batched,
                &logits_seq,
                tol,
                &format!("{arch} {weight_quant:?} M={m} logits"),
            );
            assert_slice_close(
                &kv_batched,
                &kv_seq,
                tol,
                &format!("{arch} {weight_quant:?} M={m} kv"),
            );
        }
    }
}

fn assert_chunked_16x2_matches_single_shot_32(
    arch: &str,
    weight_quant: QuantType,
    sliding_window: Option<usize>,
) {
    let hidden = 256_usize;
    let raw = build_wave3_test_gguf(arch, weight_quant, hidden);
    let gguf = GgufFile::parse(&raw).unwrap();
    let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
    let cache = recipe.build_repack_cache(&gguf).unwrap();

    let kv_geom = KvGeometry::new(2, 2, hidden / 4, 64, sliding_window).unwrap();
    let kv_len = kv_geom.required_f32_len().unwrap();
    let prompt: Vec<u32> = (0..32_u32).map(|i| (i * 5 + 1) % 64).collect();

    let mut logits_single = vec![0.0_f32; 64];
    let mut kv_single = vec![0.0_f32; kv_len];
    let mut ctx_single =
        ExecutionContext::new_prefill(&prompt, 64, &mut logits_single, &mut kv_single)
            .unwrap()
            .with_kv_geometry(kv_geom)
            .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_single, &gguf, &cache)
        .unwrap();
    assert_eq!(ctx_single.position(), 32);

    let mut kv_chunked = vec![0.0_f32; kv_len];
    let sentinel = -777.25_f32;
    let mut sentinel_logits = vec![sentinel; 64];
    let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
        &prompt[..16],
        64,
        0,
        false,
        &mut sentinel_logits,
        &mut kv_chunked,
    )
    .unwrap()
    .with_kv_geometry(kv_geom)
    .unwrap();
    let before = thread_decode_1d_count();
    recipe
        .prefill_step_with_cache(&mut ctx_c1, &gguf, &cache)
        .unwrap();
    assert_eq!(ctx_c1.position(), 16);
    assert!(
        sentinel_logits
            .iter()
            .all(|&v| v.to_bits() == sentinel.to_bits()),
        "compute_logits=false must leave logits buffer untouched"
    );

    let mut logits_chunked = vec![0.0_f32; 64];
    let mut ctx_c2 = ExecutionContext::new_prefill_chunk(
        &prompt[16..32],
        64,
        16,
        true,
        &mut logits_chunked,
        &mut kv_chunked,
    )
    .unwrap()
    .with_kv_geometry(kv_geom)
    .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_c2, &gguf, &cache)
        .unwrap();
    assert_eq!(thread_decode_1d_count(), before);
    assert_eq!(ctx_c2.position(), 32);

    let tol = if weight_quant == QuantType::F32 {
        1.0e-5
    } else {
        2.0e-3
    };
    assert_slice_close(
        ctx_single.logits(),
        ctx_c2.logits(),
        tol,
        &format!("{arch} {weight_quant:?} sw={sliding_window:?} 32 vs 16+16 logits"),
    );
    assert_slice_close(
        ctx_single.kv_cache(),
        ctx_c2.kv_cache(),
        tol,
        &format!("{arch} {weight_quant:?} sw={sliding_window:?} 32 vs 16+16 kv"),
    );

    ctx_single.prepare_decode(19).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_single, &gguf, &cache)
        .unwrap();
    ctx_c2.prepare_decode(19).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_c2, &gguf, &cache)
        .unwrap();
    assert_slice_close(
        &logits_single,
        &logits_chunked,
        tol,
        &format!("{arch} {weight_quant:?} sw={sliding_window:?} post-chunk decode"),
    );
}

#[test]
fn test_wave3_chunked_start_position_prefill_n_ctx_64_single_shot_vs_two_16_chunks() {
    for (arch, weight_quant, sliding_window) in [
        ("llama", QuantType::F32, None),
        ("qwen2", QuantType::Q4_K_M, None),
        ("llama", QuantType::Q5_0, Some(8)),
    ] {
        assert_chunked_16x2_matches_single_shot_32(arch, weight_quant, sliding_window);
    }
}

#[test]
fn test_wave3_uneven_multi_chunk_prefill_with_empty_logits() {
    let hidden = 256_usize;
    let raw = build_wave3_test_gguf("llama", QuantType::F32, hidden);
    let gguf = GgufFile::parse(&raw).unwrap();
    let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
    let kv_geom = KvGeometry::new(2, 2, hidden / 4, 64, None).unwrap();
    let kv_len = kv_geom.required_f32_len().unwrap();
    let prompt: Vec<u32> = (0..32_u32).map(|i| (i * 3 + 9) % 64).collect();

    let mut logits_full = vec![0.0_f32; 64];
    let mut kv_full = vec![0.0_f32; kv_len];
    let mut ctx_full = ExecutionContext::new_prefill(&prompt, 64, &mut logits_full, &mut kv_full)
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap();
    recipe.prefill_step(&mut ctx_full, &gguf).unwrap();

    let mut kv_uneven = vec![0.0_f32; kv_len];
    let mut logits_uneven = vec![0.0_f32; 64];
    let mut empty = Vec::new();
    for (start, end, compute) in [(0_usize, 7_usize, false), (7, 19, false), (19, 32, true)] {
        let out_slice: &mut [f32] = if compute {
            &mut logits_uneven
        } else {
            &mut empty
        };
        let mut ctx = ExecutionContext::new_prefill_chunk(
            &prompt[start..end],
            64,
            start,
            compute,
            out_slice,
            &mut kv_uneven,
        )
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap();
        recipe.prefill_step(&mut ctx, &gguf).unwrap();
        assert_eq!(ctx.position(), end);
    }
    assert_slice_close(
        &logits_full,
        &logits_uneven,
        1.0e-5,
        "uneven 3-chunk logits",
    );
    assert_slice_close(&kv_full, &kv_uneven, 1.0e-5, "uneven 3-chunk kv");
}
