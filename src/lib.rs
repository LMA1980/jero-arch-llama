//! `jero-arch-llama`: Meta Llama and Alibaba `Qwen2` architecture recipes for `SpacemiT` K3.
//!
//! Implements [`ArchRecipe`] for `llama`, `llama4` (`iRoPE`), and `qwen2`, supporting dynamic
//! GGUF hyperparameter hydration, authentic `token_embd.weight` lookup, Pre-RMSNorm, optional
//! `Qwen2` QKV biases, Grouped-Query Attention (`GQA`) over `0..=pos` KV-cache slices, smooth
//! frequency-scaled `RoPE` on both `Q` and `K`, `128-byte`-aligned cached weight repacking,
//! batched multi-token `prefill_step` GEMMs, and row-parallel multi-threaded projections.

pub mod dispatch;
pub mod recipe;
pub mod repack_cache;

pub use dispatch::{Lane, PhaseLayout, compute_row_dot, project_lm_head_row_split};
pub use recipe::{LlamaRecipe, RopeScalingConfig};
pub use repack_cache::{
    Aligned128, CachedMatrix, PANEL_ROWS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, QuantizedBatchAct,
    cached_matrix_count, clear_repack_cache, dequantize_q4_0_block, get_or_repack,
    par_for_each_row_tile,
};

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{
        ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
        TensorView,
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
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::Q4_0));
        assert!(recipe.supports_quant(QuantType::Q6_K));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert!(!recipe.supports_quant(QuantType::TQ1_0));
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
}
