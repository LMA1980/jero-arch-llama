//! Wave 2 verification suite: shared `jero-kernels-rvv::RepackedWeight` parity across all
//! nine admitted quantizations, fail-closed rejection of un-admitted formats, model-scoped
//! `ModelRepackCache` 1D `Aligned128F32` norm/bias caching with zero per-step `decode_1d_f32`,
//! and compact zero-copy `F16`/`BF16` LM head row streaming for `128,256` & `151,936` vocabularies.

use jero_arch_llama::{
    CachedMatrix, LlamaRecipe, LmHeadCache, MAX_DENSE_HALF_ELEMENTS, PhaseLayout, compute_row_dot,
    is_oversized_dense, project_lm_head_row_split, thread_decode_1d_count,
};
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, KvGeometry, QuantType, TensorView,
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
        let val = (((idx * 19 + seed) % 97) as f32 - 48.0) * 0.025;
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
        let val = (((idx * mul + seed) % modulus) as f32 - offset) * 0.03;
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
        let d = if quant == QuantType::Q4_0 { 0.05 } else { 0.04 };
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
                out[off..off + 2].copy_from_slice(&f32_to_f16_bits(0.02).to_le_bytes());
                out[off + 2..off + 4].copy_from_slice(&f32_to_f16_bits(0.006).to_le_bytes());
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
                out[off + 208..off + 210].copy_from_slice(&f32_to_f16_bits(0.015).to_le_bytes());
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

#[test]
fn test_wave2_all_nine_quants_repacked_gemv_gemm_and_unadmitted_rejection() {
    let rows = 73_usize; // Non-multiple of PANEL_ROWS=64 and TILE_N=8
    let cols = 256_usize;
    let m = 3_usize;
    let mut act = vec![0.0_f32; m * cols];
    for (i, slot) in act.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "bounded test modulo fits in f32"
        )]
        {
            *slot = (((i * 29 + 7) % 89) as f32 - 44.0) * 0.02;
        }
    }

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
    let dims = [cols, rows];
    for (seed, &quant) in admitted.iter().enumerate() {
        let raw = fill_quant_bytes(rows, cols, quant, seed + 1);
        let view = TensorView::from_bytes("w.weight", quant, &dims, 0, &raw).unwrap();
        let mat = CachedMatrix::repack(&view, rows, cols).expect("admitted quant must repack");
        assert!(mat.uses_repacked_storage());

        let is_dense = matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16);
        let tol = if is_dense { 1.0e-5 } else { 1.0e-2 };
        let mut gemm_out = vec![0.0_f32; m * rows];
        mat.gemm_rows(&mut gemm_out, &act, m, true).unwrap();

        for tok in 0..m {
            let x_tok = &act[tok * cols..(tok + 1) * cols];
            #[expect(clippy::cast_precision_loss, reason = "cols fits in f32")]
            let bound =
                (x_tok.iter().map(|v| v * v).sum::<f32>().sqrt() * (cols as f32).sqrt()).max(1.0);
            for r in 0..rows {
                let expected = compute_row_dot(r, cols, quant, &raw, x_tok);
                let actual = gemm_out[tok * rows + r];
                let rel = (actual - expected).abs() / bound.max(expected.abs());
                assert!(
                    rel < tol,
                    "GEMM parity failed for {quant:?} tok={tok} row={r}: {actual} vs {expected} (rel={rel})"
                );
            }
        }
    }

    let rejected = [
        QuantType::Q2_K,
        QuantType::Q3_K,
        QuantType::Q4_1,
        QuantType::Q5_1,
        QuantType::Q8_1,
        QuantType::IQ4_NL,
        QuantType::TQ1_0,
        QuantType::TQ2_0,
        QuantType::I2_S,
        QuantType::MXFP4,
    ];
    for &quant in &rejected {
        let blk_bytes = quant.block_bytes();
        let blk_elems = quant.block_elements().max(256);
        let blocks = blk_elems / quant.block_elements();
        let raw = vec![0_u8; blocks * blk_bytes + quant.scale_trailer_bytes()];
        let bad_dims = [blk_elems, 1];
        let view = TensorView::from_bytes("bad.weight", quant, &bad_dims, 0, &raw).unwrap();
        let err = CachedMatrix::repack(&view, 1, blk_elems)
            .expect_err("un-admitted quant must fail closed");
        assert!(
            matches!(err, K3Error::CapabilityMismatch { .. }),
            "expected CapabilityMismatch for {quant:?}, got {err:?}"
        );
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
                    (((e * 7 + idx * 11) % 29) as f32 - 14.0) * 0.05
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

fn build_wave2_e2e_gguf(arch: &str) -> Vec<u8> {
    let norm_type = if arch == "llama" {
        QuantType::F16
    } else {
        QuantType::BF16
    };
    let kvs = [
        (format!("{arch}.block_count"), GgufValue::Uint32(2)),
        (format!("{arch}.embedding_length"), GgufValue::Uint32(64)),
        (format!("{arch}.feed_forward_length"), GgufValue::Uint32(96)),
        (format!("{arch}.attention.head_count"), GgufValue::Uint32(2)),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(1),
        ),
        (
            format!("{arch}.attention.key_length"),
            GgufValue::Uint32(32),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(10_000.0),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(1e-5),
        ),
    ];
    let kvs_refs: Vec<(&str, GgufValue<'_>)> =
        kvs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

    let mut specs: Vec<(&str, QuantType, &[usize])> = vec![
        ("token_embd.weight", QuantType::Q8_0, &[64, 24]),
        ("blk.0.attn_norm.weight", norm_type, &[64]),
        ("blk.0.attn_q.weight", QuantType::Q5_0, &[64, 64]),
        ("blk.0.attn_k.weight", QuantType::Q4_0, &[64, 32]),
        ("blk.0.attn_v.weight", QuantType::Q4_0, &[64, 32]),
        ("blk.0.attn_output.weight", QuantType::Q8_0, &[64, 64]),
        ("blk.0.ffn_norm.weight", norm_type, &[64]),
        ("blk.0.ffn_gate.weight", QuantType::Q5_0, &[64, 96]),
        ("blk.0.ffn_up.weight", QuantType::Q5_0, &[64, 96]),
        ("blk.0.ffn_down.weight", QuantType::Q4_0, &[96, 64]),
        ("blk.1.attn_norm.weight", QuantType::F32, &[64]),
        ("blk.1.attn_q.weight", QuantType::Q8_0, &[64, 64]),
        ("blk.1.attn_k.weight", QuantType::Q5_0, &[64, 32]),
        ("blk.1.attn_v.weight", QuantType::Q5_0, &[64, 32]),
        ("blk.1.attn_output.weight", QuantType::Q4_0, &[64, 64]),
        ("blk.1.ffn_norm.weight", QuantType::F16, &[64]),
        ("blk.1.ffn_gate.weight", QuantType::Q8_0, &[64, 96]),
        ("blk.1.ffn_up.weight", QuantType::Q8_0, &[64, 96]),
        ("blk.1.ffn_down.weight", QuantType::Q5_0, &[96, 64]),
        ("output_norm.weight", norm_type, &[64]),
        ("output.weight", QuantType::Q8_0, &[64, 24]),
    ];
    if arch == "qwen2" {
        specs.extend_from_slice(&[
            ("blk.0.attn_q.bias", QuantType::F16, &[64]),
            ("blk.0.attn_k.bias", QuantType::BF16, &[32]),
            ("blk.0.attn_v.bias", QuantType::F32, &[32]),
            ("blk.1.attn_q.bias", QuantType::F32, &[64]),
            ("blk.1.attn_k.bias", QuantType::F16, &[32]),
            ("blk.1.attn_v.bias", QuantType::BF16, &[32]),
        ]);
    }

    let mut raw = create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs_refs, &specs);
    write_finite_gguf_payloads(&mut raw);
    raw
}

fn run_model_repack_cache_e2e_for_arch(arch: &str) {
    let raw = build_wave2_e2e_gguf(arch);
    let gguf = GgufFile::parse(&raw).unwrap();
    let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
    recipe.validate_model(gguf.metadata(), &gguf).unwrap();

    let cache = recipe.build_repack_cache(&gguf).unwrap();
    assert_eq!(cache.num_layers(), 2);
    assert_eq!(cache.norm_count(), 5);
    assert_eq!(cache.bias_count(), if arch == "qwen2" { 6 } else { 0 });
    assert_eq!(cache.repacked_matrix_count(), 15);
    assert!(cache.total_aligned_bytes() > 0);

    let kv_geom = KvGeometry::new(2, 1, 32, 8, None).unwrap();
    let kv_len = kv_geom.required_f32_len().unwrap();

    let mut logits_cached = vec![0.0_f32; 24];
    let mut kv_cached = vec![0.0_f32; kv_len];
    let mut ctx_cached = ExecutionContext::new(&[1, 3, 5], 8, &mut logits_cached, &mut kv_cached)
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap();

    let before = thread_decode_1d_count();
    recipe
        .prefill_step_with_cache(&mut ctx_cached, &gguf, &cache)
        .unwrap();
    for tok in [7_u32, 11, 13] {
        ctx_cached.prepare_decode(tok).unwrap();
        recipe
            .decode_step_with_cache(&mut ctx_cached, &gguf, &cache)
            .unwrap();
    }
    assert_eq!(
        thread_decode_1d_count(),
        before,
        "cached model execution must emit zero 1D decodes across prefill + 3 decode steps"
    );

    let mut logits_trait = vec![0.0_f32; 24];
    let mut kv_trait = vec![0.0_f32; kv_len];
    let mut ctx_trait = ExecutionContext::new(&[1, 3, 5], 8, &mut logits_trait, &mut kv_trait)
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap();
    recipe.prefill_step(&mut ctx_trait, &gguf).unwrap();
    for tok in [7_u32, 11, 13] {
        ctx_trait.prepare_decode(tok).unwrap();
        recipe.decode_step(&mut ctx_trait, &gguf).unwrap();
    }

    for (i, (&a, &b)) in logits_cached.iter().zip(&logits_trait).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() < 1.0e-5,
            "{arch} logit {i} mismatch between ModelRepackCache and trait fallback: {a} vs {b}"
        );
    }
    for (i, (&a, &b)) in kv_cached.iter().zip(&kv_trait).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() < 1.0e-5,
            "{arch} KV element {i} mismatch between ModelRepackCache and trait fallback: {a} vs {b}"
        );
    }
}

#[test]
fn test_wave2_model_repack_cache_e2e_llama_and_qwen2_zero_per_step_decodes() {
    run_model_repack_cache_e2e_for_arch("llama");
    run_model_repack_cache_e2e_for_arch("qwen2");
}

#[test]
fn test_wave2_compact_lm_head_streaming_llama3_and_qwen25() {
    // 1. Llama-3 vocabulary size (128,256) in F16 exceeding MAX_DENSE_HALF_ELEMENTS (33.55M):
    // 128,256 * 288 = 36,937,728 > 33,554,432 (~70.5 MiB raw F16 slice; zero copy on dispatch)
    let llama3_vocab = 128_256_usize;
    let llama3_cols = 288_usize;
    assert!(llama3_vocab * llama3_cols > MAX_DENSE_HALF_ELEMENTS);
    assert!(is_oversized_dense(
        QuantType::F16,
        llama3_vocab,
        llama3_cols
    ));
    {
        let mut raw = vec![0_u8; llama3_vocab * llama3_cols * 2];
        for row in [0_usize, 64_128, llama3_vocab - 1] {
            let base = row * llama3_cols * 2;
            for c in 0..llama3_cols {
                let bits = f32_to_f16_bits(if (row + c) % 5 == 0 { 0.0625 } else { -0.03125 });
                raw[base + c * 2..base + c * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let hidden = vec![0.25_f32; llama3_cols];
        let mut logits = vec![0.0_f32; llama3_vocab];
        let l3_dims = [llama3_cols, llama3_vocab];
        let view =
            TensorView::from_bytes("output.weight", QuantType::F16, &l3_dims, 0, &raw).unwrap();
        project_lm_head_row_split(&mut logits, &hidden, &view, &PhaseLayout::default()).unwrap();
        for row in [0_usize, 64_128, llama3_vocab - 1] {
            let expected = compute_row_dot(row, llama3_cols, QuantType::F16, &raw, &hidden);
            assert!(
                (logits[row] - expected).abs() < 1.0e-5,
                "Llama-3 F16 streaming row {row}: got {} vs {expected}",
                logits[row]
            );
        }
    }

    // 2. Qwen2.5 vocabulary size (151,936) in BF16 exceeding MAX_DENSE_HALF_ELEMENTS:
    // 151,936 * 256 = 38,895,616 > 33,554,432 (~74.2 MiB raw BF16 slice; zero copy on dispatch)
    let qwen_vocab = 151_936_usize;
    let qwen_cols = 256_usize;
    assert!(qwen_vocab * qwen_cols > MAX_DENSE_HALF_ELEMENTS);
    assert!(is_oversized_dense(QuantType::BF16, qwen_vocab, qwen_cols));
    {
        let mut raw = vec![0_u8; qwen_vocab * qwen_cols * 2];
        for row in [0_usize, 75_968, qwen_vocab - 1] {
            let base = row * qwen_cols * 2;
            for c in 0..qwen_cols {
                let bits = f32_to_bf16_bits(if (row + c) % 3 == 0 { 0.125 } else { -0.0625 });
                raw[base + c * 2..base + c * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let hidden = vec![0.5_f32; qwen_cols];
        let mut logits = vec![0.0_f32; qwen_vocab];
        let q_dims = [qwen_cols, qwen_vocab];
        let view =
            TensorView::from_bytes("output.weight", QuantType::BF16, &q_dims, 0, &raw).unwrap();
        project_lm_head_row_split(&mut logits, &hidden, &view, &PhaseLayout::default()).unwrap();
        for row in [0_usize, 75_968, qwen_vocab - 1] {
            let expected = compute_row_dot(row, qwen_cols, QuantType::BF16, &raw, &hidden);
            assert!(
                (logits[row] - expected).abs() < 1.0e-5,
                "Qwen2.5 BF16 streaming row {row}: got {} vs {expected}",
                logits[row]
            );
        }
    }

    // 3. Verify ModelRepackCache retains StreamingUntied without copying oversized F16 LM head
    let lm_hint = LmHeadCache::StreamingUntied;
    assert!(matches!(lm_hint, LmHeadCache::StreamingUntied));
}
