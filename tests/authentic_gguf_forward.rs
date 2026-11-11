//! Integration tests for authentic `llama`, `llama4`, and `qwen2` GGUF execution,
//! cached `128-byte`-aligned weight repacking, batched multi-token `prefill_step` GEMM,
//! `0..=pos` causal GQA KV-cache residency, QKV biases, tied/untied `output.weight`,
//! and multi-threaded `rayon` row-parallel projections.

use jero_arch_llama::{
    CachedMatrix, LlamaRecipe, PhaseLayout, clear_repack_cache, compute_row_dot,
    project_lm_head_row_split,
};
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, QuantType, TensorView,
};

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0) - 127 + 15;
    let mant = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7BFF
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounded f16 fields"
        )]
        {
            sign | ((exp as u16) << 10) | (mant as u16)
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "test fixture builder covers all GGUF quant formats"
)]
fn build_realistic_matrix_bytes(
    rows: usize,
    cols: usize,
    quant: QuantType,
    seed: usize,
) -> Vec<u8> {
    match quant {
        QuantType::F32 => {
            let mut out = Vec::with_capacity(rows * cols * 4);
            for i in 0..rows * cols {
                #[expect(clippy::cast_precision_loss, reason = "test index fits in f32")]
                let v = (((i * 13 + seed * 7) % 97) as f32 - 48.0) * 0.015;
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        QuantType::F16 => {
            let mut out = Vec::with_capacity(rows * cols * 2);
            for i in 0..rows * cols {
                #[expect(clippy::cast_precision_loss, reason = "test index fits in f32")]
                let v = (((i * 11 + seed * 5) % 89) as f32 - 44.0) * 0.02;
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            out
        }
        QuantType::BF16 => {
            let mut out = Vec::with_capacity(rows * cols * 2);
            for i in 0..rows * cols {
                #[expect(clippy::cast_precision_loss, reason = "test index fits in f32")]
                let v = (((i * 17 + seed * 3) % 83) as f32 - 41.0) * 0.02;
                let bf16 = (v.to_bits() >> 16) as u16;
                out.extend_from_slice(&bf16.to_le_bytes());
            }
            out
        }
        QuantType::Q8_0 => {
            let blocks = rows * (cols / 32);
            let mut out = vec![0_u8; blocks * 34];
            let d = f32_to_f16_bits(0.025).to_le_bytes();
            for b in 0..blocks {
                let off = b * 34;
                out[off..off + 2].copy_from_slice(&d);
                for k in 0..32 {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_possible_wrap,
                        clippy::cast_sign_loss,
                        reason = "signed i8 pattern"
                    )]
                    {
                        let q = (((b * 19 + k * 7 + seed) % 121) as i32 - 60) as i8;
                        out[off + 2 + k] = q as u8;
                    }
                }
            }
            out
        }
        QuantType::Q4_0 => {
            let blocks = rows * (cols / 32);
            let mut out = vec![0_u8; blocks * 18];
            let d = f32_to_f16_bits(0.125).to_le_bytes();
            for b in 0..blocks {
                let off = b * 18;
                out[off..off + 2].copy_from_slice(&d);
                for k in 0..16 {
                    #[expect(clippy::cast_possible_truncation, reason = "4-bit nibble pair")]
                    {
                        let lo = ((b * 7 + k * 3 + seed) & 0x0F) as u8;
                        let hi = ((b * 11 + k * 5 + seed) & 0x0F) as u8;
                        out[off + 2 + k] = lo | (hi << 4);
                    }
                }
            }
            out
        }
        QuantType::Q4_K_M => {
            let blocks = rows * (cols / 256);
            let mut out = vec![0_u8; blocks * 144];
            let d = f32_to_f16_bits(0.05).to_le_bytes();
            let dmin = f32_to_f16_bits(0.02).to_le_bytes();
            for b in 0..blocks {
                let off = b * 144;
                out[off..off + 2].copy_from_slice(&d);
                out[off + 2..off + 4].copy_from_slice(&dmin);
                for s in 0..12 {
                    #[expect(clippy::cast_possible_truncation, reason = "packed 6-bit scale table")]
                    {
                        out[off + 4 + s] = (((b + s * 9 + seed) % 61) + 1) as u8;
                    }
                }
                for q in 0..128 {
                    #[expect(clippy::cast_possible_truncation, reason = "4-bit nibble pair")]
                    {
                        let lo = ((b * 5 + q * 3 + seed) & 0x0F) as u8;
                        let hi = ((b * 13 + q * 7 + seed) & 0x0F) as u8;
                        out[off + 16 + q] = lo | (hi << 4);
                    }
                }
            }
            out
        }
        QuantType::Q6_K => {
            let blocks = rows * (cols / 256);
            let mut out = vec![0_u8; blocks * 210];
            let d = f32_to_f16_bits(0.02).to_le_bytes();
            for b in 0..blocks {
                let off = b * 210;
                for i in 0..128 {
                    #[expect(clippy::cast_possible_truncation, reason = "ql byte")]
                    {
                        out[off + i] = ((b * 7 + i * 11 + seed) & 0xFF) as u8;
                    }
                }
                for i in 0..64 {
                    #[expect(clippy::cast_possible_truncation, reason = "qh byte")]
                    {
                        out[off + 128 + i] = ((b * 3 + i * 5 + seed) & 0xFF) as u8;
                    }
                }
                for s in 0..16 {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_possible_wrap,
                        clippy::cast_sign_loss,
                        reason = "signed scale"
                    )]
                    {
                        let sc = (((b + s * 3 + seed) % 15) as i32 - 7) as i8;
                        out[off + 192 + s] = sc as u8;
                    }
                }
                out[off + 208..off + 210].copy_from_slice(&d);
            }
            out
        }
        _ => panic!("unsupported test quant {quant:?}"),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "constructs complete 2-layer GGUF fixture with optional biases and output head"
)]
fn build_two_layer_gguf(
    arch: &str,
    quant: QuantType,
    with_biases: bool,
    with_untied_output: bool,
) -> Vec<u8> {
    let hidden = 256_usize;
    let ffn = 512_usize; // Strictly > hidden_dim to verify zero ffn_dim truncation
    let n_q = 8_usize;
    let n_kv = 2_usize;
    let head_dim = 32_usize;
    let vocab = 67_usize; // Non-multiple of TILE_N = 8

    let block_key = format!("{arch}.block_count");
    let emb_key = format!("{arch}.embedding_length");
    let ffn_key = format!("{arch}.feed_forward_length");
    let hq_key = format!("{arch}.attention.head_count");
    let hkv_key = format!("{arch}.attention.head_count_kv");
    let klen_key = format!("{arch}.attention.key_length");
    let rope_key = format!("{arch}.rope.freq_base");
    let eps_key = format!("{arch}.attention.layer_norm_rms_epsilon");
    let ctx_key = format!("{arch}.context_length");

    let kvs = [
        (block_key.as_str(), GgufValue::Uint32(2)),
        (emb_key.as_str(), GgufValue::Uint32(256)),
        (ffn_key.as_str(), GgufValue::Uint32(512)),
        (hq_key.as_str(), GgufValue::Uint32(8)),
        (hkv_key.as_str(), GgufValue::Uint32(2)),
        (klen_key.as_str(), GgufValue::Uint32(32)),
        (
            rope_key.as_str(),
            GgufValue::Float32(if arch == "qwen2" {
                1_000_000.0
            } else {
                500_000.0
            }),
        ),
        (
            eps_key.as_str(),
            GgufValue::Float32(if arch == "qwen2" { 1e-6 } else { 1e-5 }),
        ),
        (ctx_key.as_str(), GgufValue::Uint32(64)),
    ];

    let emb_shape = [hidden, vocab];
    let norm_shape = [hidden];
    let q_shape = [hidden, n_q * head_dim];
    let kv_shape = [hidden, n_kv * head_dim];
    let q_bias_shape = [n_q * head_dim];
    let kv_bias_shape = [n_kv * head_dim];
    let o_shape = [n_q * head_dim, hidden];
    let gate_up_shape = [hidden, ffn];
    let down_shape = [ffn, hidden];

    let mut specs: Vec<(&str, QuantType, &[usize])> = vec![
        ("token_embd.weight", quant, &emb_shape),
        ("blk.0.attn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.0.attn_q.weight", quant, &q_shape),
        ("blk.0.attn_k.weight", quant, &kv_shape),
        ("blk.0.attn_v.weight", quant, &kv_shape),
        ("blk.0.attn_output.weight", quant, &o_shape),
        ("blk.0.ffn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.0.ffn_gate.weight", quant, &gate_up_shape),
        ("blk.0.ffn_up.weight", quant, &gate_up_shape),
        ("blk.0.ffn_down.weight", quant, &down_shape),
        ("blk.1.attn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.1.attn_q.weight", quant, &q_shape),
        ("blk.1.attn_k.weight", quant, &kv_shape),
        ("blk.1.attn_v.weight", quant, &kv_shape),
        ("blk.1.attn_output.weight", quant, &o_shape),
        ("blk.1.ffn_norm.weight", QuantType::F32, &norm_shape),
        ("blk.1.ffn_gate.weight", quant, &gate_up_shape),
        ("blk.1.ffn_up.weight", quant, &gate_up_shape),
        ("blk.1.ffn_down.weight", quant, &down_shape),
        ("output_norm.weight", QuantType::F32, &norm_shape),
    ];

    if with_biases {
        specs.push(("blk.0.attn_q.bias", QuantType::F32, &q_bias_shape));
        specs.push(("blk.0.attn_k.bias", QuantType::F32, &kv_bias_shape));
        specs.push(("blk.0.attn_v.bias", QuantType::F32, &kv_bias_shape));
        specs.push(("blk.1.attn_q.bias", QuantType::F32, &q_bias_shape));
        specs.push(("blk.1.attn_k.bias", QuantType::F32, &kv_bias_shape));
        specs.push(("blk.1.attn_v.bias", QuantType::F32, &kv_bias_shape));
    }

    if with_untied_output {
        specs.push(("output.weight", quant, &emb_shape));
    }

    let mut raw = create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs, &specs);

    // Overwrite tensor payloads with clean finite FP16/FP32/quantized blocks
    let parsed = GgufFile::parse(&raw).expect("parse template gguf");
    let descriptors = parsed.metadata().tensors().to_vec();
    for (idx, desc) in descriptors.iter().enumerate() {
        let start = desc.absolute_offset();
        let end = start + desc.byte_len();
        if desc.shape().len() == 1 {
            let elem_count = desc.num_elements();
            for e in 0..elem_count {
                #[expect(clippy::cast_precision_loss, reason = "small index")]
                let val = if desc.name().ends_with("norm.weight") {
                    0.85 + ((e % 13) as f32) * 0.02
                } else {
                    // Non-trivial QKV bias
                    (((e * 7 + idx * 11) % 29) as f32 - 14.0) * 0.08
                };
                let off = start + e * 4;
                raw[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
        } else {
            let cols = desc.shape()[0];
            let rows = desc.shape()[1];
            let payload = build_realistic_matrix_bytes(rows, cols, desc.quant(), idx + 1);
            assert_eq!(payload.len(), end - start);
            raw[start..end].copy_from_slice(&payload);
        }
    }

    raw
}

fn vector_l2(xs: &[f32]) -> f32 {
    xs.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let denom = (vector_l2(a) * vector_l2(b)).max(1.0e-12);
    dot / denom
}

#[test]
fn test_repacked_gemv_and_batched_gemm_numerical_parity_all_quants() {
    clear_repack_cache();
    let rows = 133_usize; // Spans 3 panels (64 + 64 + 5) and non-multiple of TILE_N = 8
    let cols = 256_usize;
    let m = 5_usize;

    let mut batch_in = vec![0.0_f32; m * cols];
    for (i, slot) in batch_in.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "test index fits in f32")]
        {
            *slot = (((i * 23 + 5) % 101) as f32 - 50.0) * 0.03;
        }
    }

    let formats = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q8_0,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q6_K,
    ];

    for (f_idx, &quant) in formats.iter().enumerate() {
        let raw = build_realistic_matrix_bytes(rows, cols, quant, f_idx + 10);
        let name = format!("blk.{f_idx}.attn_q.weight");
        let shape = [cols, rows];
        let view =
            TensorView::from_bytes(&name, quant, &shape, 0, &raw).expect("create TensorView");
        let cached = CachedMatrix::repack(&view, rows, cols).expect("repack succeeds");
        let is_dense = matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16);
        let norm_tol = if is_dense { 1.0e-5 } else { 1.0e-2 };

        // 1. Single-token GEMV (M = 1)
        let mut single_out = vec![0.0_f32; rows];
        cached
            .gemv_rows(&mut single_out, &batch_in[..cols], true)
            .expect("gemv succeeds");
        let mut ref_single = vec![0.0_f32; rows];
        let first_norm = vector_l2(&batch_in[..cols]);
        #[expect(clippy::cast_precision_loss, reason = "cols fits in f32")]
        let scale_bound_0 = (first_norm * (cols as f32).sqrt()).max(1.0);
        for (r, &actual) in single_out.iter().enumerate() {
            let expected = compute_row_dot(r, cols, quant, &raw, &batch_in[..cols]);
            ref_single[r] = expected;
            let norm_err = (actual - expected).abs() / scale_bound_0.max(expected.abs());
            assert!(
                norm_err < norm_tol,
                "GEMV parity failed for {quant:?} row {r}: got {actual}, expected {expected} (norm_err={norm_err})"
            );
        }
        assert!(
            cosine_similarity(&single_out, &ref_single) > 0.9998,
            "GEMV cosine similarity too low for {quant:?}"
        );

        // 2. Batched multi-token GEMM (M = 5)
        let mut batched_out = vec![0.0_f32; m * rows];
        cached
            .gemm_rows(&mut batched_out, &batch_in, m, true)
            .expect("gemm succeeds");
        for r in 0..rows {
            assert!(
                (single_out[r] - batched_out[r]).abs() < 1.0e-5,
                "GEMV vs GEMM tok 0 mismatch for {quant:?} row {r}: {} vs {}",
                single_out[r],
                batched_out[r]
            );
        }
        for tok in 0..m {
            let x_row = &batch_in[tok * cols..(tok + 1) * cols];
            let token_norm = vector_l2(x_row);
            let actual_slice = &batched_out[tok * rows..(tok + 1) * rows];
            let mut ref_slice = vec![0.0_f32; rows];
            for r in 0..rows {
                let expected = compute_row_dot(r, cols, quant, &raw, x_row);
                ref_slice[r] = expected;
                let actual = actual_slice[r];
                #[expect(clippy::cast_precision_loss, reason = "cols fits in f32")]
                let scale_bound = (token_norm * (cols as f32).sqrt())
                    .max(expected.abs())
                    .max(1.0);
                let norm_err = (actual - expected).abs() / scale_bound;
                assert!(
                    norm_err < norm_tol,
                    "GEMM parity failed for {quant:?} tok {tok} row {r}: got {actual}, expected {expected} (norm_err={norm_err})"
                );
            }
            assert!(
                cosine_similarity(actual_slice, &ref_slice) > 0.9998,
                "GEMM cosine similarity too low for {quant:?} tok {tok}"
            );
        }
    }
}

#[test]
fn test_unhydrated_default_llama_recipe_live_geometry_and_ffn_expansion() {
    clear_repack_cache();
    let raw_gguf = build_two_layer_gguf("llama", QuantType::Q4_K_M, false, true);
    let gguf = GgufFile::parse(&raw_gguf).expect("parse llama gguf");

    // 1. Deliberately use unhydrated `LlamaRecipe::new()` (32 layers, 4096 hidden, 14336 ffn, 128 head_dim)
    // to verify live tensor-shape resolution handles 2 layers, 256 hidden, 512 ffn (> hidden), and 67 vocab!
    let default_recipe = LlamaRecipe::new();
    let hydrated_from_gguf = LlamaRecipe::from_gguf(&gguf).expect("hydrate llama recipe from gguf");
    let hydrated_from_meta = LlamaRecipe::from_metadata(gguf.metadata());
    assert_eq!(hydrated_from_gguf.num_layers, 2);
    assert_eq!(hydrated_from_gguf.hidden_dim, 256);
    assert_eq!(hydrated_from_gguf.ffn_dim, 512);
    assert_eq!(hydrated_from_gguf.n_heads_q, 8);
    assert_eq!(hydrated_from_gguf.n_heads_kv, 2);
    assert_eq!(hydrated_from_gguf.head_dim, 32);
    assert_eq!(hydrated_from_gguf.vocab_size, 67);

    let prompt = [1_u32, 7, 13, 22];
    let n_ctx = 16_usize;
    let kv_len = n_ctx * 2 * 2 * (2 * 32);

    let mut logits_default = vec![0.0_f32; 67];
    let mut kv_default = vec![0.0_f32; kv_len];
    let mut ctx_default =
        ExecutionContext::new(&prompt, n_ctx, &mut logits_default, &mut kv_default)
            .expect("ctx default");
    default_recipe
        .prefill_step(&mut ctx_default, &gguf)
        .expect("default recipe resolves live tensor geometry cleanly");
    ctx_default.prepare_decode(5).expect("stage decode");
    default_recipe
        .decode_step(&mut ctx_default, &gguf)
        .expect("default recipe decode succeeds");
    assert!(logits_default.iter().all(|v| v.is_finite()));
    assert!(logits_default.iter().any(|&v| v.abs() > 1e-4));

    // 2. Verify `from_gguf` and `from_metadata` produce bit-identical hydrated outputs
    let mut logits_gguf = vec![0.0_f32; 67];
    let mut kv_gguf = vec![0.0_f32; kv_len];
    let mut ctx_gguf = ExecutionContext::new(&prompt, n_ctx, &mut logits_gguf, &mut kv_gguf)
        .expect("ctx hydrated gguf");
    hydrated_from_gguf
        .prefill_step(&mut ctx_gguf, &gguf)
        .expect("hydrated from_gguf prefill succeeds");

    let mut logits_meta = vec![0.0_f32; 67];
    let mut kv_meta = vec![0.0_f32; kv_len];
    let mut ctx_meta = ExecutionContext::new(&prompt, n_ctx, &mut logits_meta, &mut kv_meta)
        .expect("ctx hydrated meta");
    hydrated_from_meta
        .prefill_step(&mut ctx_meta, &gguf)
        .expect("hydrated from_metadata prefill succeeds");

    for (a, b) in logits_gguf.iter().zip(&logits_meta) {
        assert!(a.is_finite());
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_qwen2_biases_causal_kv_history_and_tied_vs_untied_embeddings() {
    clear_repack_cache();
    let raw_biased_tied = build_two_layer_gguf("qwen2", QuantType::Q8_0, true, false);
    let raw_unbiased_tied = build_two_layer_gguf("qwen2", QuantType::Q8_0, false, false);
    let raw_biased_untied = build_two_layer_gguf("qwen2", QuantType::Q8_0, true, true);

    let gguf_biased_tied = GgufFile::parse(&raw_biased_tied).expect("parse biased tied");
    let gguf_unbiased_tied = GgufFile::parse(&raw_unbiased_tied).expect("parse unbiased tied");
    let gguf_biased_untied = GgufFile::parse(&raw_biased_untied).expect("parse biased untied");

    let recipe = LlamaRecipe::from_gguf(&gguf_biased_tied).expect("from_gguf qwen2");
    assert_eq!(recipe.architecture_id(), "qwen2");
    assert!((recipe.rope_theta - 1_000_000.0).abs() < 1.0);
    assert!((recipe.rms_eps - 1e-6).abs() < 1e-10);

    let n_ctx = 16_usize;
    let kv_len = n_ctx * 2 * 2 * (2 * 32);

    // 1. QKV biases produce measurably distinct logits from unbiased Qwen2
    let prompt_a = [2_u32, 9, 4, 15];
    let mut logits_biased = vec![0.0_f32; 67];
    let mut kv_a = vec![0.0_f32; kv_len];
    let mut ctx_a =
        ExecutionContext::new(&prompt_a, n_ctx, &mut logits_biased, &mut kv_a).expect("ctx a");
    recipe
        .prefill_step(&mut ctx_a, &gguf_biased_tied)
        .expect("biased prefill");

    let mut logits_unbiased = vec![0.0_f32; 67];
    let mut kv_b = vec![0.0_f32; kv_len];
    let mut ctx_b =
        ExecutionContext::new(&prompt_a, n_ctx, &mut logits_unbiased, &mut kv_b).expect("ctx b");
    recipe
        .prefill_step(&mut ctx_b, &gguf_unbiased_tied)
        .expect("unbiased prefill");

    let bias_diff: f32 = logits_biased
        .iter()
        .zip(&logits_unbiased)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        bias_diff > 1e-2,
        "expected Qwen2 QKV biases to alter logits, got total diff {bias_diff}"
    );

    // 2. Full 0..=pos causal GQA + K-RoPE: permuting prefix [2, 9, 4, 15] -> [9, 2, 4, 15]
    // keeps final token (15 at pos=3) identical, yet changes logits via 0..=pos KV attention & K-RoPE!
    let prompt_permuted = [9_u32, 2, 4, 15];
    let mut logits_perm = vec![0.0_f32; 67];
    let mut kv_perm = vec![0.0_f32; kv_len];
    let mut ctx_perm =
        ExecutionContext::new(&prompt_permuted, n_ctx, &mut logits_perm, &mut kv_perm)
            .expect("ctx perm");
    recipe
        .prefill_step(&mut ctx_perm, &gguf_biased_tied)
        .expect("permuted prefill");

    let perm_diff: f32 = logits_biased
        .iter()
        .zip(&logits_perm)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        perm_diff > 1e-4,
        "expected prefix permutation with K-RoPE and 0..=pos KV cache to alter logits, got {perm_diff}"
    );

    // 3. Untied `output.weight` produces distinct logits from tied `token_embd.weight`
    let mut logits_untied = vec![0.0_f32; 67];
    let mut kv_untied = vec![0.0_f32; kv_len];
    let mut ctx_untied =
        ExecutionContext::new(&prompt_a, n_ctx, &mut logits_untied, &mut kv_untied)
            .expect("ctx untied");
    recipe
        .prefill_step(&mut ctx_untied, &gguf_biased_untied)
        .expect("untied prefill");
    let untied_diff: f32 = logits_biased
        .iter()
        .zip(&logits_untied)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        untied_diff > 1e-2,
        "expected untied output.weight to differ from tied token_embd.weight, got {untied_diff}"
    );

    // 4. Out-of-bounds token ID fails closed with TensorBoundsError
    let mut logits_oob = vec![0.0_f32; 67];
    let mut kv_oob = vec![0.0_f32; kv_len];
    let mut ctx_oob =
        ExecutionContext::new(&[999_u32], n_ctx, &mut logits_oob, &mut kv_oob).expect("ctx oob");
    let err = recipe
        .prefill_step(&mut ctx_oob, &gguf_biased_tied)
        .expect_err("OOB token must fail");
    assert!(matches!(err, K3Error::TensorBoundsError { .. }));
}

#[test]
fn test_batched_prefill_vs_sequential_decode_across_all_quants() {
    let quants = [
        QuantType::F32,
        QuantType::F16,
        QuantType::Q8_0,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q6_K,
    ];

    let prompt = [3_u32, 11, 27, 42];
    let n_ctx = 16_usize;
    let kv_len = n_ctx * 2 * 2 * (2 * 32);

    for &quant in &quants {
        clear_repack_cache();
        let raw_gguf = build_two_layer_gguf("llama", quant, false, true);
        let gguf = GgufFile::parse(&raw_gguf).expect("parse gguf");
        let recipe = LlamaRecipe::from_gguf(&gguf).expect("hydrate recipe");

        let mut logits_batch = vec![0.0_f32; 67];
        let mut kv_batch = vec![0.0_f32; kv_len];
        let mut ctx_batch = ExecutionContext::new(&prompt, n_ctx, &mut logits_batch, &mut kv_batch)
            .expect("batch ctx");
        recipe
            .prefill_step(&mut ctx_batch, &gguf)
            .expect("batched prefill");

        // Follow up with an autoregressive decode token at pos = 4
        ctx_batch.prepare_decode(19).expect("stage 19");
        recipe
            .decode_step(&mut ctx_batch, &gguf)
            .expect("post-batch decode");

        // Sequential token-by-token reference: 1-token prefill + 4 decode steps
        let mut logits_seq = vec![0.0_f32; 67];
        let mut kv_seq = vec![0.0_f32; kv_len];
        let mut ctx_seq = ExecutionContext::new(&prompt[..1], n_ctx, &mut logits_seq, &mut kv_seq)
            .expect("seq ctx");
        recipe
            .prefill_step(&mut ctx_seq, &gguf)
            .expect("seq prefill");
        for &tok in &prompt[1..] {
            ctx_seq.prepare_decode(tok).expect("stage prompt tok");
            recipe.decode_step(&mut ctx_seq, &gguf).expect("seq decode");
        }
        ctx_seq.prepare_decode(19).expect("stage 19 seq");
        recipe
            .decode_step(&mut ctx_seq, &gguf)
            .expect("post-seq decode");

        for (idx, (&a, &b)) in logits_batch.iter().zip(&logits_seq).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "quant {quant:?} logit[{idx}] mismatch between batched prefill and sequential decode: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_rayon_worker_pool_row_parallel_equivalence() {
    clear_repack_cache();
    let raw_gguf = build_two_layer_gguf("qwen2", QuantType::Q4_K_M, true, true);
    let gguf = GgufFile::parse(&raw_gguf).expect("parse gguf");
    let recipe = LlamaRecipe::from_gguf(&gguf).expect("hydrate recipe");

    let pool_1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread pool");
    let pool_4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build 4-thread pool");

    let prompt = [1_u32, 5, 9, 13, 17, 21];
    let n_ctx = 16_usize;
    let kv_len = n_ctx * 2 * 2 * (2 * 32);

    let (logits_1, logits_4) = (
        pool_1.install(|| {
            let mut logits = vec![0.0_f32; 67];
            let mut kv = vec![0.0_f32; kv_len];
            let mut ctx = ExecutionContext::new(&prompt, n_ctx, &mut logits, &mut kv).unwrap();
            recipe.prefill_step(&mut ctx, &gguf).unwrap();
            ctx.prepare_decode(8).unwrap();
            recipe.decode_step(&mut ctx, &gguf).unwrap();
            logits
        }),
        pool_4.install(|| {
            let mut logits = vec![0.0_f32; 67];
            let mut kv = vec![0.0_f32; kv_len];
            let mut ctx = ExecutionContext::new(&prompt, n_ctx, &mut logits, &mut kv).unwrap();
            recipe.prefill_step(&mut ctx, &gguf).unwrap();
            ctx.prepare_decode(8).unwrap();
            recipe.decode_step(&mut ctx, &gguf).unwrap();
            logits
        }),
    );

    for (a, b) in logits_1.iter().zip(&logits_4) {
        assert!(
            (a - b).abs() < 1e-5,
            "1-thread vs 4-thread mismatch: {a} vs {b}"
        );
    }

    // Also verify large multi-panel LM head row-split inside 4-thread pool
    let vocab_rows = 263_usize;
    let cols = 256_usize;
    let lm_raw = build_realistic_matrix_bytes(vocab_rows, cols, QuantType::Q4_K_M, 77);
    let lm_shape = [cols, vocab_rows];
    let lm_view = TensorView::from_bytes("output.weight", QuantType::Q4_K_M, &lm_shape, 0, &lm_raw)
        .expect("lm view");
    let hidden = vec![0.25_f32; cols];
    let out_1 = pool_1.install(|| {
        let mut out = vec![0.0_f32; vocab_rows];
        project_lm_head_row_split(&mut out, &hidden, &lm_view, &PhaseLayout::row_split(0.5))
            .unwrap();
        out
    });
    let out_4 = pool_4.install(|| {
        let mut out = vec![0.0_f32; vocab_rows];
        project_lm_head_row_split(&mut out, &hidden, &lm_view, &PhaseLayout::row_split(0.5))
            .unwrap();
        out
    });
    assert_eq!(out_1, out_4);
}
