//! `jero-arch-llama`: Meta Llama and Alibaba `Qwen2` architecture recipes for `SpacemiT` K3.
//!
//! Implements [`jero_core::ArchRecipe`] for `llama`, `llama4` (`iRoPE`), and `qwen2`, supporting dynamic
//! GGUF hyperparameter hydration, authentic `token_embd.weight` lookup, Pre-RMSNorm, optional
//! `Qwen2` QKV biases, Grouped-Query Attention (`GQA`) over `0..=pos` KV-cache slices, smooth
//! frequency-scaled `RoPE` on both `Q` and `K`, `128-byte`-aligned cached weight repacking,
//! batched multi-token `prefill_step` GEMMs, and row-parallel multi-threaded projections.

pub mod decode_gemv;
pub mod dispatch;
pub mod recipe;
pub mod repack_cache;

pub use decode_gemv::{
    BlockQ8_0, BlockQ8K, DecodeQ8Scratch, LayerKeyTable, LayerTensorKeys, bytes_as_q8_0_blocks_mut,
    bytes_as_q8_k_blocks_mut, ensure_global_layer_keys, gemv_decode_projection,
    gemv_decode_projection_with_workers, gemv_q4_0_q8_0, gemv_q4_k_q8_k, gemv_q5_0_q8_0,
    gemv_q5_k_q8_k, gemv_q6_k_q8_k, gemv_q8_0_q8_0, gemv_tq1_0_q8_k, gemv_tq2_0_q8_k,
    probe_live_layer_count, quantize_row_q8_0, quantize_row_q8_k,
};

pub use dispatch::{Lane, PhaseLayout, compute_row_dot, project_lm_head_row_split};
pub use recipe::{LlamaRecipe, RopeScalingConfig, RopeType};
pub use repack_cache::{
    Aligned128, Aligned128F32, CachedMatrix, DecodeActQ8View, LayerRepackCache, LmHeadCache,
    MAX_DENSE_F32_ELEMENTS, MAX_DENSE_HALF_ELEMENTS, ModelRepackCache, PANEL_ROWS,
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS, QuantizedBatchAct, cached_matrix_count, cached_vector_1d_count,
    clear_repack_cache, decode_1d_f32, dequantize_q4_0_block, dequantize_q5_0_block,
    dequantize_q5_k_block, get_or_decode_1d_f32, get_or_repack, is_oversized_dense,
    par_for_each_row_tile, thread_decode_1d_count,
};

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{
        ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvGeometry,
        ModelWeights, QuantType, TensorView,
    };

    struct TestWeights<'a> {
        tensors: Vec<(&'static str, TensorView<'a>)>,
    }

    impl ModelWeights for TestWeights<'_> {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            self.tensors
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, view)| *view)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "not found in test weights".to_owned(),
                })
        }

        fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.tensors.iter().map(|(_, v)| v.byte_len()).sum()
        }
    }

    #[test]
    fn test_llama_and_qwen2_recipe_capabilities() {
        let recipe = LlamaRecipe::new();
        assert_eq!(recipe.architecture_id(), "llama");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::Q5_K));
        assert!(recipe.supports_quant(QuantType::Q5_0));
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::Q4_0));
        assert!(recipe.supports_quant(QuantType::Q6_K));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert!(!recipe.supports_quant(QuantType::TQ1_0));
        assert!(!recipe.supports_quant(QuantType::Q2_K));
        assert_eq!(recipe.recipe_version(), "1.0.0");
        assert_eq!(recipe.target_layouts(), &["native", "ime_tile_8x16"]);

        let qwen2 = LlamaRecipe::new_qwen2();
        assert_eq!(qwen2.architecture_id(), "qwen2");
        assert!((qwen2.rms_eps - 1e-6).abs() < 1e-10);
        assert!((qwen2.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(qwen2.supports_quant(QuantType::Q4_0));
    }

    #[test]
    fn test_llama4_recipe_irope_masking() {
        let recipe = LlamaRecipe::new_llama4().with_irope_mask(vec![false, true, false, true]);
        assert_eq!(recipe.architecture_id(), "llama4");
        assert!(!recipe.should_skip_rope(0));
        assert!(recipe.should_skip_rope(1));
        assert!(!recipe.should_skip_rope(2));
        assert!(recipe.should_skip_rope(3));
        assert!(!recipe.should_skip_rope(10));
    }

    #[test]
    fn test_project_lm_head_row_split_large_vocab() {
        let rows = if cfg!(miri) { 256 } else { 128_256 };
        let cols = 16;
        let mut raw_bytes = vec![0_u8; rows * cols * 4];
        for i in 0..cols {
            let val = 1.0_f32;
            raw_bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
            let last_row_off = (rows - 1) * cols * 4 + i * 4;
            let val_last = 2.0_f32;
            raw_bytes[last_row_off..last_row_off + 4].copy_from_slice(&val_last.to_le_bytes());
        }

        let shape = [rows, cols];
        let tensor = TensorView::from_bytes("output.weight", QuantType::F32, &shape, 0, &raw_bytes)
            .expect("valid tensor");

        let hidden = vec![1.0_f32; cols];
        let mut logits = vec![0.0_f32; rows];

        let layout = PhaseLayout::row_split(0.5);
        project_lm_head_row_split(&mut logits, &hidden, &tensor, &layout).expect("split succeeds");

        assert!((logits[0] - 16.0).abs() < 1e-4);
        assert!((logits[100] - 0.0).abs() < 1e-4);
        assert!((logits[rows - 1] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_project_lm_head_row_split_tile_m_remainder_edge_case15() {
        let rows = if cfg!(miri) { 133 } else { 128_005 };
        let cols = 16;
        let mut raw_bytes = vec![0_u8; rows * cols * 4];
        for i in 0..cols {
            let last_off = (rows - 1) * cols * 4 + i * 4;
            raw_bytes[last_off..last_off + 4].copy_from_slice(&3.0_f32.to_le_bytes());
        }

        let shape = [rows, cols];
        let tensor = TensorView::from_bytes("output.weight", QuantType::F32, &shape, 0, &raw_bytes)
            .expect("valid tensor");

        let hidden = vec![1.0_f32; cols];
        let mut logits = vec![0.0_f32; rows];

        let layout = PhaseLayout::row_split(0.5);
        project_lm_head_row_split(&mut logits, &hidden, &tensor, &layout)
            .expect("remainder partition succeeds");

        assert_eq!(logits.len(), rows);
        assert!((logits[rows - 1] - 48.0).abs() < 1e-4);
    }

    #[test]
    fn test_project_lm_head_row_split_dimension_mismatch() {
        let shape = [100, 16];
        let raw = vec![0_u8; 100 * 16 * 4];
        let tensor =
            TensorView::from_bytes("output.weight", QuantType::F32, &shape, 0, &raw).unwrap();

        let hidden = vec![1.0_f32; 16];
        let mut wrong_logits = vec![0.0_f32; 50];
        let layout = PhaseLayout::default();

        let err = project_lm_head_row_split(&mut wrong_logits, &hidden, &tensor, &layout)
            .expect_err("dimension mismatch");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_llama_recipe_prefill_and_decode_step_synthetic_fallback() {
        let mut recipe = LlamaRecipe::new();
        recipe.num_layers = 1;
        let weights = TestWeights {
            tensors: Vec::new(),
        };

        let prompt = [10_u32, 20, 30];
        let mut logits = [0.0_f32; 16];
        let mut kv_cache = [0.0_f32; 64];

        let mut ctx = ExecutionContext::new(&prompt, 16, &mut logits, &mut kv_cache)
            .expect("create execution context");

        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds");
        assert_eq!(ctx.position(), 3);
        assert!(ctx.logits()[9] >= 1.0 && ctx.logits()[9] < 1.1);

        ctx.prepare_decode(5).expect("prepare decode");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds");
        assert_eq!(ctx.position(), 4);
        assert!(ctx.logits()[6] >= 2.0 && ctx.logits()[6] < 2.1);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "end-to-end 2-layer Qwen2 GGUF fixture and batched vs sequential verification"
    )]
    fn test_from_gguf_and_batched_vs_sequential_equivalence_miri_safe() {
        clear_repack_cache();
        // Compact 2-layer Qwen2 GGUF with Q8_0 weights, QKV biases, tied embeddings, and ffn_dim > hidden_dim
        let hidden = 32_usize;
        let ffn = 64_usize;
        let n_q = 4_usize;
        let n_kv = 2_usize;
        let head_dim = 8_usize;
        let vocab = 24_usize;

        let kvs = [
            ("qwen2.block_count", GgufValue::Uint32(2)),
            ("qwen2.embedding_length", GgufValue::Uint32(32)),
            ("qwen2.feed_forward_length", GgufValue::Uint32(64)),
            ("qwen2.attention.head_count", GgufValue::Uint32(4)),
            ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
            ("qwen2.attention.key_length", GgufValue::Uint32(8)),
            ("qwen2.rope.freq_base", GgufValue::Float32(1_000_000.0)),
            (
                "qwen2.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(1e-6),
            ),
        ];

        let emb_shape = [hidden, vocab];
        let norm_shape = [hidden];
        let q_shape = [hidden, n_q * head_dim];
        let q_bias_shape = [n_q * head_dim];
        let kv_shape = [hidden, n_kv * head_dim];
        let kv_bias_shape = [n_kv * head_dim];
        let o_shape = [n_q * head_dim, hidden];
        let gate_up_shape = [hidden, ffn];
        let down_shape = [ffn, hidden];

        let specs: Vec<(&str, QuantType, &[usize])> = vec![
            ("token_embd.weight", QuantType::Q8_0, &emb_shape),
            ("blk.0.attn_norm.weight", QuantType::F32, &norm_shape),
            ("blk.0.attn_q.weight", QuantType::Q8_0, &q_shape),
            ("blk.0.attn_q.bias", QuantType::F32, &q_bias_shape),
            ("blk.0.attn_k.weight", QuantType::Q8_0, &kv_shape),
            ("blk.0.attn_k.bias", QuantType::F32, &kv_bias_shape),
            ("blk.0.attn_v.weight", QuantType::Q8_0, &kv_shape),
            ("blk.0.attn_v.bias", QuantType::F32, &kv_bias_shape),
            ("blk.0.attn_output.weight", QuantType::Q8_0, &o_shape),
            ("blk.0.ffn_norm.weight", QuantType::F32, &norm_shape),
            ("blk.0.ffn_gate.weight", QuantType::Q4_0, &gate_up_shape),
            ("blk.0.ffn_up.weight", QuantType::Q4_0, &gate_up_shape),
            ("blk.0.ffn_down.weight", QuantType::Q8_0, &down_shape),
            ("blk.1.attn_norm.weight", QuantType::F32, &norm_shape),
            ("blk.1.attn_q.weight", QuantType::Q8_0, &q_shape),
            ("blk.1.attn_k.weight", QuantType::Q8_0, &kv_shape),
            ("blk.1.attn_v.weight", QuantType::Q8_0, &kv_shape),
            ("blk.1.attn_output.weight", QuantType::Q8_0, &o_shape),
            ("blk.1.ffn_norm.weight", QuantType::F32, &norm_shape),
            ("blk.1.ffn_gate.weight", QuantType::Q4_0, &gate_up_shape),
            ("blk.1.ffn_up.weight", QuantType::Q4_0, &gate_up_shape),
            ("blk.1.ffn_down.weight", QuantType::Q8_0, &down_shape),
            ("output_norm.weight", QuantType::F32, &norm_shape),
        ];

        let raw_gguf = create_synthetic_gguf_v3_with_kvs("qwen2", 32, &kvs, &specs);
        let gguf = GgufFile::parse(&raw_gguf).expect("parse synthetic qwen2 gguf");
        let recipe = LlamaRecipe::from_gguf(&gguf).expect("hydrate qwen2 recipe");
        assert_eq!(recipe.architecture_id(), "qwen2");
        assert_eq!(recipe.num_layers, 2);
        assert_eq!(recipe.hidden_dim, 32);
        assert_eq!(recipe.ffn_dim, 64);
        assert_eq!(recipe.n_heads_q, 4);
        assert_eq!(recipe.n_heads_kv, 2);
        assert_eq!(recipe.head_dim, 8);
        assert_eq!(recipe.vocab_size, 24);

        let n_ctx = 8_usize;
        let kv_len = n_ctx * 2 * 2 * (n_kv * head_dim);

        // Path A: Batched multi-token prefill [1, 3, 5]
        let prompt = [1_u32, 3, 5];
        let mut logits_batched = vec![0.0_f32; vocab];
        let mut kv_batched = vec![0.0_f32; kv_len];
        let mut ctx_batched =
            ExecutionContext::new(&prompt, n_ctx, &mut logits_batched, &mut kv_batched)
                .expect("batched ctx");
        recipe
            .prefill_step(&mut ctx_batched, &gguf)
            .expect("batched prefill");

        // Path B: Sequential 1-token prefill [1], then decode(3), then decode(5)
        let mut logits_seq = vec![0.0_f32; vocab];
        let mut kv_seq = vec![0.0_f32; kv_len];
        let mut ctx_seq =
            ExecutionContext::new(&[1_u32], n_ctx, &mut logits_seq, &mut kv_seq).expect("seq ctx");
        recipe
            .prefill_step(&mut ctx_seq, &gguf)
            .expect("seq prefill 0");
        ctx_seq.prepare_decode(3).expect("stage 3");
        recipe.decode_step(&mut ctx_seq, &gguf).expect("decode 1");
        ctx_seq.prepare_decode(5).expect("stage 5");
        recipe.decode_step(&mut ctx_seq, &gguf).expect("decode 2");

        for (a, b) in logits_batched.iter().zip(&logits_seq) {
            assert!(a.is_finite() && b.is_finite());
            let rel = (a - b).abs() / b.abs().max(1.0);
            assert!(
                rel < 1e-4,
                "batched vs sequential logit mismatch: {a} vs {b} (rel={rel})"
            );
        }
        for (a, b) in kv_batched.iter().zip(&kv_seq) {
            let rel = (a - b).abs() / b.abs().max(1.0);
            assert!(
                rel < 1e-4,
                "batched vs sequential KV mismatch: {a} vs {b} (rel={rel})"
            );
        }
    }

    fn build_compact_arch_gguf(
        arch: &str,
        omit_tensor: Option<&str>,
        extra_kvs: &[(&str, GgufValue<'_>)],
    ) -> Vec<u8> {
        let hidden = 16_usize;
        let ffn = 24_usize;
        let n_q = 2_usize;
        let n_kv = 1_usize;
        let head_dim = 8_usize;
        let vocab = 12_usize;
        let blk_key = format!("{arch}.block_count");
        let emb_key = format!("{arch}.embedding_length");
        let ffn_key = format!("{arch}.feed_forward_length");
        let hq_key = format!("{arch}.attention.head_count");
        let hkv_key = format!("{arch}.attention.head_count_kv");
        let klen_key = format!("{arch}.attention.key_length");
        let eps_key = format!("{arch}.attention.layer_norm_rms_epsilon");
        let theta_key = format!("{arch}.rope.freq_base");
        let ctx_key = format!("{arch}.context_length");

        let mut kvs: Vec<(&str, GgufValue<'_>)> = vec![
            (blk_key.as_str(), GgufValue::Uint32(1)),
            (emb_key.as_str(), GgufValue::Uint32(16)),
            (ffn_key.as_str(), GgufValue::Uint32(24)),
            (hq_key.as_str(), GgufValue::Uint32(2)),
            (hkv_key.as_str(), GgufValue::Uint32(1)),
            (klen_key.as_str(), GgufValue::Uint32(8)),
            (eps_key.as_str(), GgufValue::Float32(1e-5)),
            (theta_key.as_str(), GgufValue::Float32(10_000.0)),
            (ctx_key.as_str(), GgufValue::Uint32(16)),
        ];
        kvs.extend_from_slice(extra_kvs);

        let emb_shape = [hidden, vocab];
        let norm_shape = [hidden];
        let q_shape = [hidden, n_q * head_dim];
        let kv_shape = [hidden, n_kv * head_dim];
        let o_shape = [n_q * head_dim, hidden];
        let gate_up_shape = [hidden, ffn];
        let down_shape = [ffn, hidden];

        let all_specs: [(&str, QuantType, &[usize]); 12] = [
            ("token_embd.weight", QuantType::F32, &emb_shape),
            ("blk.0.attn_norm.weight", QuantType::F32, &norm_shape),
            ("blk.0.attn_q.weight", QuantType::F32, &q_shape),
            ("blk.0.attn_k.weight", QuantType::F32, &kv_shape),
            ("blk.0.attn_v.weight", QuantType::F32, &kv_shape),
            ("blk.0.attn_output.weight", QuantType::F32, &o_shape),
            ("blk.0.ffn_norm.weight", QuantType::F32, &norm_shape),
            ("blk.0.ffn_gate.weight", QuantType::F32, &gate_up_shape),
            ("blk.0.ffn_up.weight", QuantType::F32, &gate_up_shape),
            ("blk.0.ffn_down.weight", QuantType::F32, &down_shape),
            ("output_norm.weight", QuantType::F32, &norm_shape),
            ("output.weight", QuantType::F32, &emb_shape),
        ];
        let filtered: Vec<_> = all_specs
            .into_iter()
            .filter(|(name, _, _)| Some(*name) != omit_tensor)
            .collect();
        create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs, &filtered)
    }

    #[test]
    fn test_from_metadata_non_8b_scales_and_rotary_dim() {
        let cases = [
            (
                "qwen2",
                896_u32,
                4864_u32,
                14_u32,
                2_u32,
                64_u32,
                151_936_u32,
                24_u32,
                None,
                RopeType::NeoX,
            ),
            (
                "qwen2",
                1536,
                8960,
                12,
                2,
                128,
                151_936,
                28,
                None,
                RopeType::NeoX,
            ),
            (
                "llama",
                3072,
                8192,
                24,
                8,
                128,
                128_256,
                28,
                Some(64_u32),
                RopeType::Norm,
            ),
            (
                "llama4",
                5120,
                16384,
                40,
                8,
                128,
                202_048,
                48,
                Some(96_u32),
                RopeType::Norm,
            ),
        ];
        for (arch, hidden, ffn, hq, hkv, hd, vocab, layers, rot_dim, expected_rope) in cases {
            let blk_k = format!("{arch}.block_count");
            let emb_k = format!("{arch}.embedding_length");
            let ffn_k = format!("{arch}.feed_forward_length");
            let hq_k = format!("{arch}.attention.head_count");
            let hkv_k = format!("{arch}.attention.head_count_kv");
            let key_len_k = format!("{arch}.attention.key_length");
            let v_k = format!("{arch}.vocab_size");
            let eps_k = format!("{arch}.attention.layer_norm_rms_epsilon");
            let theta_k = format!("{arch}.rope.freq_base");
            let rot_k = format!("{arch}.rope.dimension_count");

            let mut kvs = vec![
                (blk_k.as_str(), GgufValue::Uint32(layers)),
                (emb_k.as_str(), GgufValue::Uint32(hidden)),
                (ffn_k.as_str(), GgufValue::Uint32(ffn)),
                (hq_k.as_str(), GgufValue::Uint32(hq)),
                (hkv_k.as_str(), GgufValue::Uint32(hkv)),
                (key_len_k.as_str(), GgufValue::Uint32(hd)),
                (v_k.as_str(), GgufValue::Uint32(vocab)),
                (eps_k.as_str(), GgufValue::Float32(1e-6)),
                (theta_k.as_str(), GgufValue::Float32(10_000.0)),
            ];
            if let Some(rd) = rot_dim {
                kvs.push((rot_k.as_str(), GgufValue::Uint32(rd)));
            }
            let raw = create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs, &[]);
            let gguf = GgufFile::parse(&raw).expect("parse scale metadata");
            let from_meta = LlamaRecipe::from_metadata(gguf.metadata());
            let from_full = LlamaRecipe::from_gguf(&gguf).expect("from_gguf");
            for recipe in [&from_meta, &from_full] {
                assert_eq!(recipe.architecture_id(), arch);
                assert_eq!(recipe.hidden_dim, hidden as usize);
                assert_eq!(recipe.ffn_dim, ffn as usize);
                assert_eq!(recipe.n_heads_q, hq as usize);
                assert_eq!(recipe.n_heads_kv, hkv as usize);
                assert_eq!(recipe.head_dim, hd as usize);
                assert_eq!(recipe.vocab_size, vocab as usize);
                assert_eq!(recipe.num_layers, layers as usize);
                assert_eq!(recipe.rope_type, expected_rope);
                assert_eq!(
                    recipe.effective_rotary_dim(),
                    rot_dim.map_or(hd as usize, |v| v as usize)
                );
            }
        }
    }

    #[test]
    fn test_rope_norm_vs_neox_and_partial_rotary_dim_parity() {
        clear_repack_cache();
        // Direct kernel/compat formula test for Norm vs NeoX & partial rotary_dim=4 < head_dim=8
        let inv_freq = recipe::precompute_inv_freq_compat(4, 10_000.0, None, 8).unwrap();
        assert_eq!(inv_freq.len(), 2);
        let orig = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];
        let mut norm_vec = orig;
        let mut neox_vec = orig;
        recipe::apply_rope_compat(&mut norm_vec, 8, 4, RopeType::Norm, 3, &inv_freq).unwrap();
        recipe::apply_rope_compat(&mut neox_vec, 8, 4, RopeType::NeoX, 3, &inv_freq).unwrap();

        // Untouched non-rotary dims [4..8]
        assert_eq!(&norm_vec[4..8], &orig[4..8]);
        assert_eq!(&neox_vec[4..8], &orig[4..8]);

        for (i, &freq) in inv_freq.iter().enumerate() {
            let (s, c) = (3.0_f32 * freq).sin_cos();
            let exp_norm_0 = orig[2 * i] * c - orig[2 * i + 1] * s;
            let exp_norm_1 = orig[2 * i] * s + orig[2 * i + 1] * c;
            assert!((norm_vec[2 * i] - exp_norm_0).abs() < 1e-5);
            assert!((norm_vec[2 * i + 1] - exp_norm_1).abs() < 1e-5);

            let exp_neox_0 = orig[i] * c - orig[i + 2] * s;
            let exp_neox_1 = orig[i] * s + orig[i + 2] * c;
            assert!((neox_vec[i] - exp_neox_0).abs() < 1e-5);
            assert!((neox_vec[i + 2] - exp_neox_1).abs() < 1e-5);
        }

        // End-to-end GGUF test with partial rope.dimension_count = 4 on head_dim = 8
        for (arch, exp_rope) in [("llama", RopeType::Norm), ("qwen2", RopeType::NeoX)] {
            let rot_key = format!("{arch}.rope.dimension_count");
            let raw =
                build_compact_arch_gguf(arch, None, &[(rot_key.as_str(), GgufValue::Uint32(4))]);
            let gguf = GgufFile::parse(&raw).unwrap();
            let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
            assert_eq!(recipe.rope_type, exp_rope);
            assert_eq!(recipe.rotary_dim, Some(4));

            if !cfg!(miri) {
                let mut logits_a = vec![0.0_f32; 12];
                let mut kv_a = vec![0.0_f32; 8 * 2 * 8];
                let prefill_toks: &[u32] = &[1, 3, 5];
                let mut ctx_a =
                    ExecutionContext::new(prefill_toks, 8, &mut logits_a, &mut kv_a).unwrap();
                recipe.prefill_step(&mut ctx_a, &gguf).unwrap();

                let mut logits_b = vec![0.0_f32; 12];
                let mut kv_b = vec![0.0_f32; 8 * 2 * 8];
                let mut ctx_b = ExecutionContext::new(&[1], 8, &mut logits_b, &mut kv_b).unwrap();
                recipe.prefill_step(&mut ctx_b, &gguf).unwrap();
                for &tok in &prefill_toks[1..] {
                    ctx_b.prepare_decode(tok).unwrap();
                    recipe.decode_step(&mut ctx_b, &gguf).unwrap();
                }
                for (a, b) in logits_a.iter().zip(&logits_b) {
                    assert!(
                        (a - b).abs() < 1e-5,
                        "{arch} batched vs seq mismatch: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_llama3_smooth_rope_scaling_from_gguf_and_llama_cpp_parity() {
        // Ignore large llama.context_length=131072 when original_context_length=8192 is set
        let kvs_31 = [
            ("llama.context_length", GgufValue::Uint32(131_072)),
            ("llama.attention.key_length", GgufValue::Uint32(128)),
            ("llama.rope.freq_base", GgufValue::Float32(500_000.0)),
            ("llama.rope.scaling.type", GgufValue::String("llama3")),
            ("llama.rope.scaling.factor", GgufValue::Float32(8.0)),
            (
                "llama.rope.scaling.low_freq_factor",
                GgufValue::Float32(1.0),
            ),
            (
                "llama.rope.scaling.high_freq_factor",
                GgufValue::Float32(4.0),
            ),
            (
                "llama.rope.scaling.original_context_length",
                GgufValue::Uint32(8192),
            ),
        ];
        let raw_31 = create_synthetic_gguf_v3_with_kvs("llama", 32, &kvs_31, &[]);
        let gguf_31 = GgufFile::parse(&raw_31).unwrap();
        let recipe_31 = LlamaRecipe::from_gguf(&gguf_31).unwrap();
        assert_eq!(
            recipe_31.rope_scaling,
            Some(RopeScalingConfig::Llama3Smooth {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            })
        );

        let inv_freq = recipe::precompute_inv_freq_compat(
            128,
            500_000.0,
            recipe_31.rope_scaling.as_ref(),
            16_384,
        )
        .unwrap();
        assert_eq!(inv_freq.len(), 64);
        let orig_ctx = 8192.0_f32;
        for (i, &actual) in inv_freq.iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "pair index fits in f32")]
            let base = 1.0_f32 / 500_000.0_f32.powf((2 * i) as f32 / 128.0_f32);
            let wavelen = core::f32::consts::TAU / base;
            let expected = if wavelen < orig_ctx / 4.0 {
                base
            } else if wavelen > orig_ctx / 1.0 {
                base / 8.0
            } else {
                let smooth = (orig_ctx / wavelen - 1.0) / 3.0;
                (1.0 - smooth) * (base / 8.0) + smooth * base
            };
            assert!(
                (actual - expected).abs() < 1e-6,
                "Llama-3.1 inv_freq[{i}] mismatch: {actual} vs {expected}"
            );
        }

        // Fallback orig_ctx_len alias & Llama-3.2 factor=32, plus per-dim rope.scaling.factors
        let kvs_32 = [
            ("llama.rope.scaling.type", GgufValue::String("llama3")),
            ("llama.rope.scaling.factor", GgufValue::Float32(32.0)),
            ("llama.rope.scaling.orig_ctx_len", GgufValue::Uint32(4096)),
        ];
        let raw_32 = create_synthetic_gguf_v3_with_kvs("llama", 32, &kvs_32, &[]);
        let gguf_32 = GgufFile::parse(&raw_32).unwrap();
        let recipe_32 = LlamaRecipe::from_gguf(&gguf_32).unwrap();
        assert_eq!(
            recipe_32.rope_scaling,
            Some(RopeScalingConfig::Llama3Smooth {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 4096,
            })
        );

        let kvs_factors = [(
            "llama.rope.scaling.factors",
            GgufValue::Array(GgufArray::Float32(vec![1.5, 2.5])),
        )];
        let raw_factors = create_synthetic_gguf_v3_with_kvs("llama", 32, &kvs_factors, &[]);
        let gguf_factors = GgufFile::parse(&raw_factors).unwrap();
        let recipe_factors = LlamaRecipe::from_gguf(&gguf_factors).unwrap();
        let custom_freq = recipe::precompute_inv_freq_compat(
            4,
            10_000.0,
            recipe_factors.rope_scaling.as_ref(),
            16,
        )
        .unwrap();
        assert!((custom_freq[0] - 1.0 / 1.5).abs() < 1e-6);
        assert!((custom_freq[1] - (1.0 / 10_000.0_f32.powf(0.5)) / 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_q5_0_dequant_repack_and_lm_head() {
        clear_repack_cache();
        // Verify Q5_0 single block high-bit decoding, RepackCache GEMM, and LM head projection
        let mut row_q50 = [0_u8; Q5_0_BLOCK_BYTES];
        row_q50[0..2].copy_from_slice(&0x3800_u16.to_le_bytes()); // d = 0.5
        row_q50[2..6].copy_from_slice(&0xA55A_5AA5_u32.to_le_bytes());
        for (j, byte) in row_q50[6..22].iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "4-bit nibble pattern")]
            {
                *byte = (((j * 3 + 1) & 0x0F) | (((j * 5 + 2) & 0x0F) << 4)) as u8;
            }
        }
        let mut deq_q50 = [0.0_f32; Q5_0_BLOCK_ELEMENTS];
        dequantize_q5_0_block(&row_q50, &mut deq_q50);
        let qh = 0xA55A_5AA5_u32;
        for j in 0..16 {
            let p = row_q50[6 + j];
            #[expect(
                clippy::cast_possible_truncation,
                reason = "5-bit value fits in signed i8 (-16..=15 after offset)"
            )]
            let exp0 = 0.5
                * f32::from(
                    ((i16::from(p & 0x0F) | (i16::try_from((qh >> j) & 1).unwrap() << 4)) - 16)
                        as i8,
                );
            #[expect(
                clippy::cast_possible_truncation,
                reason = "5-bit value fits in signed i8 (-16..=15 after offset)"
            )]
            let exp1 = 0.5
                * f32::from(
                    ((i16::from(p >> 4) | (i16::try_from((qh >> (j + 16)) & 1).unwrap() << 4)) - 16)
                        as i8,
                );
            assert!((deq_q50[j] - exp0).abs() < 1e-6);
            assert!((deq_q50[j + 16] - exp1).abs() < 1e-6);
        }

        #[expect(clippy::cast_precision_loss, reason = "index fits in f32")]
        let x_32: Vec<f32> = (0..32).map(|i| (i as f32 - 15.0) * 0.1).collect();
        let ref_dot_0: f32 = deq_q50.iter().zip(&x_32).map(|(a, b)| a * b).sum();
        assert!(
            (compute_row_dot(0, 32, QuantType::Q5_0, &row_q50, &x_32) - ref_dot_0).abs() < 1e-5
        );
        let view_q50 =
            TensorView::from_bytes("output.weight", QuantType::Q5_0, &[32, 1], 0, &row_q50)
                .unwrap();
        let mut out_0 = [0.0_f32; 1];
        project_lm_head_row_split(&mut out_0, &x_32, &view_q50, &PhaseLayout::default()).unwrap();
        let l2_32 = x_32.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let bound_0 = (l2_32 * (32.0_f32).sqrt()).max(ref_dot_0.abs()).max(1.0);
        let rel_0 = (out_0[0] - ref_dot_0).abs() / bound_0;
        assert!(
            rel_0 < 1.0e-2,
            "Q5_0 LM head mismatch: got {}, expected {ref_dot_0} (rel={rel_0})",
            out_0[0]
        );
    }

    #[test]
    fn test_q5_k_dequant_repack_and_unsupported_quant_rejection() {
        clear_repack_cache();
        // Verify Q5_K superblock high-bit decoding + RepackCache + LM head projection
        let mut kblock = [0_u8; Q5_K_BLOCK_BYTES];
        kblock[0..2].copy_from_slice(&0x3400_u16.to_le_bytes()); // d = 0.25
        kblock[2..4].copy_from_slice(&0x2E00_u16.to_le_bytes()); // dmin = 0.09375
        for i in 0..12 {
            #[expect(clippy::cast_possible_truncation, reason = "bounded scale byte")]
            {
                kblock[4 + i] = ((i * 7 + 3) % 60 + 1) as u8;
            }
        }
        for i in 0..32 {
            #[expect(clippy::cast_possible_truncation, reason = "masked byte")]
            {
                kblock[16 + i] = ((i * 29 + 11) & 0xFF) as u8;
            }
        }
        for i in 0..128 {
            #[expect(clippy::cast_possible_truncation, reason = "4-bit nibble pattern")]
            {
                kblock[48 + i] = (((i * 3 + 1) & 0x0F) | (((i * 7 + 2) & 0x0F) << 4)) as u8;
            }
        }
        let mut kout = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
        dequantize_q5_k_block(&kblock, &mut kout);
        #[expect(clippy::cast_precision_loss, reason = "index fits in f32")]
        let x_256: Vec<f32> = (0..256).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
        let ref_dot_k: f32 = kout.iter().zip(&x_256).map(|(a, b)| a * b).sum();
        assert!(
            (compute_row_dot(0, 256, QuantType::Q5_K, &kblock, &x_256) - ref_dot_k).abs() < 1e-4
        );
        let kview = TensorView::from_bytes("output.weight", QuantType::Q5_K, &[256, 1], 0, &kblock)
            .unwrap();
        let cached = CachedMatrix::repack(&kview, 1, 256).unwrap();
        let mut gemv_out = [0.0_f32; 1];
        cached.gemv_rows(&mut gemv_out, &x_256, false).unwrap();
        let l2_256 = x_256.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let bound_k = (l2_256 * 16.0_f32).max(ref_dot_k.abs()).max(1.0);
        assert!(
            (gemv_out[0] - ref_dot_k).abs() / bound_k < 1.0e-2,
            "Q5_K GEMV mismatch: {} vs {ref_dot_k}",
            gemv_out[0]
        );
        let mut lm_k = [0.0_f32; 1];
        project_lm_head_row_split(&mut lm_k, &x_256, &kview, &PhaseLayout::default()).unwrap();
        assert!(
            (lm_k[0] - ref_dot_k).abs() / bound_k < 1.0e-2,
            "Q5_K LM head mismatch: {} vs {ref_dot_k}",
            lm_k[0]
        );

        // Verify unsupported quant NEVER silently fills zeros in LM head
        let raw_q2k = [0_u8; 84];
        let tensor_q2k =
            TensorView::from_bytes("output.weight", QuantType::Q2_K, &[256, 1], 0, &raw_q2k)
                .unwrap();
        let mut dummy_logits = [9.0_f32; 1];
        let err = project_lm_head_row_split(
            &mut dummy_logits,
            &x_256,
            &tensor_q2k,
            &PhaseLayout::default(),
        )
        .expect_err("Q2_K LM head must fail closed");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }

    #[test]
    fn test_fail_closed_missing_tensors_validate_model_and_kv_view() {
        clear_repack_cache();
        // 1. Missing required tensor when tensor_count > 0 fails closed
        for omit in [
            "blk.0.attn_k.weight",
            "blk.0.ffn_gate.weight",
            "output_norm.weight",
        ] {
            let raw = build_compact_arch_gguf("llama", Some(omit), &[]);
            let gguf = GgufFile::parse(&raw).unwrap();
            let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
            assert!(
                recipe.validate_model(gguf.metadata(), &gguf).is_err(),
                "validate_model must reject GGUF missing {omit}"
            );
            if !cfg!(miri) || omit == "blk.0.attn_k.weight" {
                let mut logits = vec![0.0_f32; 12];
                let mut kv = vec![0.0_f32; 8 * 2 * 8];
                let mut ctx = ExecutionContext::new(&[1], 8, &mut logits, &mut kv).unwrap();
                let err = recipe
                    .prefill_step(&mut ctx, &gguf)
                    .expect_err("forward pass must fail closed on missing tensor");
                assert!(matches!(err, K3Error::TensorBoundsError { .. }));
            }
        }

        // 2. Undersized KV cache fails closed when tensor_count > 0
        let raw_ok = build_compact_arch_gguf("llama", None, &[]);
        let gguf_ok = GgufFile::parse(&raw_ok).unwrap();
        let recipe_ok = LlamaRecipe::from_gguf(&gguf_ok).unwrap();
        recipe_ok
            .validate_model(gguf_ok.metadata(), &gguf_ok)
            .unwrap();
        let mut logits = vec![0.0_f32; 12];
        let mut tiny_kv = vec![0.0_f32; 8];
        let mut ctx_tiny = ExecutionContext::new(&[1], 8, &mut logits, &mut tiny_kv).unwrap();
        assert!(matches!(
            recipe_ok.prefill_step(&mut ctx_tiny, &gguf_ok),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 3. validate_model rejects unsupported arch, bad rotary_dim, and unsupported Q2_K metadata
        let bad_arch_raw = build_compact_arch_gguf("gpt2", None, &[]);
        let bad_arch = GgufFile::parse(&bad_arch_raw).unwrap();
        assert!(matches!(
            recipe_ok.validate_model(bad_arch.metadata(), &bad_arch),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            recipe_ok
                .clone()
                .with_rotary_dim(6)
                .validate_model(gguf_ok.metadata(), &gguf_ok),
            Ok(())
        ));
        assert!(matches!(
            recipe_ok
                .clone()
                .with_rotary_dim(5)
                .validate_model(gguf_ok.metadata(), &gguf_ok),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        let q2k_raw = create_synthetic_gguf_v3_with_kvs(
            "llama",
            32,
            &[],
            &[("blk.0.attn_q.weight", QuantType::Q2_K, &[256, 1])],
        );
        let q2k_meta = GgufFile::parse(&q2k_raw).unwrap();
        assert!(matches!(
            recipe_ok.validate_model(q2k_meta.metadata(), &TestWeights { tensors: vec![] }),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // 4. ExecutionContext::with_kv_geometry + sliding_window works end-to-end
        let kv_geom = KvGeometry::new(1, 1, 8, 8, Some(2)).unwrap();
        let mut sw_kv = vec![0.0_f32; kv_geom.required_f32_len().unwrap()];
        let mut ctx_sw = ExecutionContext::new(&[1], 8, &mut logits, &mut sw_kv)
            .unwrap()
            .with_kv_geometry(kv_geom)
            .unwrap();
        recipe_ok.prefill_step(&mut ctx_sw, &gguf_ok).unwrap();
        if !cfg!(miri) {
            ctx_sw.prepare_decode(3).unwrap();
            recipe_ok.decode_step(&mut ctx_sw, &gguf_ok).unwrap();
        }
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_wave3_chunked_start_position_and_compute_logits_miri_safe() {
        let mut raw = build_compact_arch_gguf("llama", None, &[]);
        let parsed = GgufFile::parse(&raw).unwrap();
        let descs = parsed.metadata().tensors().to_vec();
        for (idx, desc) in descs.iter().enumerate() {
            let start = desc.absolute_offset();
            let elems = desc.num_elements();
            for e in 0..elems {
                #[expect(clippy::cast_precision_loss, reason = "small bounded pattern")]
                let val = (((e * 7 + idx * 5) % 19) as f32 - 9.0) * 0.04;
                raw[start + e * 4..start + e * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        let gguf = GgufFile::parse(&raw).unwrap();
        let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
        let cache = recipe.build_repack_cache(&gguf).unwrap();

        let prompt: &[u32] = if cfg!(miri) { &[1, 2] } else { &[1, 2, 3, 4] };
        let windows: &[Option<usize>] = if cfg!(miri) {
            &[None]
        } else {
            &[None, Some(2)]
        };
        let split = prompt.len() / 2;
        for &sw in windows {
            let kv_geom = KvGeometry::new(1, 1, 8, 8, sw).unwrap();
            let kv_len = kv_geom.required_f32_len().unwrap();

            let mut logits_single = vec![0.0_f32; 12];
            let mut kv_single = vec![0.0_f32; kv_len];
            let mut ctx_single =
                ExecutionContext::new_prefill(prompt, 8, &mut logits_single, &mut kv_single)
                    .unwrap()
                    .with_kv_geometry(kv_geom)
                    .unwrap();
            recipe
                .prefill_step_with_cache(&mut ctx_single, &gguf, &cache)
                .unwrap();

            let mut kv_chunked = vec![0.0_f32; kv_len];
            let mut empty_logits = Vec::new();
            let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
                &prompt[..split],
                8,
                0,
                false,
                &mut empty_logits,
                &mut kv_chunked,
            )
            .unwrap()
            .with_kv_geometry(kv_geom)
            .unwrap();
            recipe
                .prefill_step_with_cache(&mut ctx_c1, &gguf, &cache)
                .unwrap();
            assert_eq!(ctx_c1.position(), split);

            let mut logits_chunked = vec![0.0_f32; 12];
            let mut ctx_c2 = ExecutionContext::new_prefill_chunk(
                &prompt[split..],
                8,
                split,
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
            assert_eq!(ctx_c2.position(), prompt.len());

            for (i, (&a, &b)) in logits_single.iter().zip(&logits_chunked).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() < 1.0e-5,
                    "chunked logit {i} mismatch (sw={sw:?}): {a} vs {b}"
                );
            }
            for (i, (&a, &b)) in kv_single.iter().zip(&kv_chunked).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() < 1.0e-5,
                    "chunked KV {i} mismatch (sw={sw:?}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_wave4_decode_step_with_explicit_scratch_workspace_and_fail_closed_miri_safe() {
        let mut raw = build_compact_arch_gguf("llama", None, &[]);
        let parsed = GgufFile::parse(&raw).unwrap();
        let descs = parsed.metadata().tensors().to_vec();
        for (idx, desc) in descs.iter().enumerate() {
            let start = desc.absolute_offset();
            let elems = desc.num_elements();
            for e in 0..elems {
                #[expect(clippy::cast_precision_loss, reason = "small bounded pattern")]
                let val = (((e * 5 + idx * 3) % 13) as f32 - 6.0) * 0.03;
                raw[start + e * 4..start + e * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        let gguf = GgufFile::parse(&raw).unwrap();
        let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
        let cache = recipe.build_repack_cache(&gguf).unwrap();
        let kv_geom = KvGeometry::new(1, 1, 8, 8, None).unwrap();
        let kv_len = kv_geom.required_f32_len().unwrap();

        let scratch_bytes = recipe.decode_scratch_bytes(8);
        assert!(scratch_bytes > 0);
        let mut scratch = jero_core::ScratchWorkspace::new(scratch_bytes).unwrap();

        let mut logits_ws = vec![0.0_f32; 12];
        let mut kv_ws = vec![0.0_f32; kv_len];
        let mut ctx_ws = ExecutionContext::new(&[1, 2], 8, &mut logits_ws, &mut kv_ws)
            .unwrap()
            .with_kv_geometry(kv_geom)
            .unwrap()
            .with_scratch(&mut scratch);
        recipe
            .prefill_step_with_cache(&mut ctx_ws, &gguf, &cache)
            .unwrap();
        ctx_ws.prepare_decode(4).unwrap();
        recipe
            .decode_step_with_cache(&mut ctx_ws, &gguf, &cache)
            .unwrap();
        assert_eq!(ctx_ws.position(), 3);
        assert!(ctx_ws.scratch().unwrap().peak_used_bytes() <= scratch_bytes);
        assert_eq!(ctx_ws.scratch().unwrap().used_bytes(), 0);

        let mut logits_fb = vec![0.0_f32; 12];
        let mut kv_fb = vec![0.0_f32; kv_len];
        let mut ctx_fb = ExecutionContext::new(&[1, 2], 8, &mut logits_fb, &mut kv_fb)
            .unwrap()
            .with_kv_geometry(kv_geom)
            .unwrap();
        recipe.prefill_step(&mut ctx_fb, &gguf).unwrap();
        ctx_fb.prepare_decode(4).unwrap();
        recipe.decode_step(&mut ctx_fb, &gguf).unwrap();

        for (i, (&a, &b)) in logits_ws.iter().zip(&logits_fb).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() < 1.0e-5,
                "explicit ScratchWorkspace decode logit {i}: {a} vs {b}"
            );
        }

        // Undersized ScratchWorkspace fails closed without advancing position
        let mut tiny = jero_core::ScratchWorkspace::new(128).unwrap();
        let mut tiny_ctx = ExecutionContext::new(&[], 8, &mut logits_fb, &mut kv_fb)
            .unwrap()
            .with_kv_geometry(kv_geom)
            .unwrap()
            .with_scratch(&mut tiny);
        tiny_ctx.prepare_decode(4).unwrap();
        let before_pos = tiny_ctx.position();
        let err = recipe
            .decode_step(&mut tiny_ctx, &gguf)
            .expect_err("undersized ScratchWorkspace must fail closed");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
        assert_eq!(tiny_ctx.position(), before_pos);
    }
}
