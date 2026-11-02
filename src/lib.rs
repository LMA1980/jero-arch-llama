//! `k3-arch-llama`: Meta Llama architecture recipe for the `SpacemiT` K3 platform.
//!
//! Implements [`ArchRecipe`] for Meta Llama family models (`llama`, `llama4`),
//! supporting Pre-RMSNorm, Grouped-Query Attention (GQA), `SwiGLU` FFN, smooth
//! frequency-scaled `RoPE`, interleaved attention without positional embeddings (`iRoPE`),
//! and heterogeneous row-split parallel LM head projection across `X100` and `A100` clusters.

pub mod dispatch;
pub mod recipe;

pub use dispatch::{Lane, PhaseLayout, project_lm_head_row_split};
pub use recipe::{LlamaRecipe, RopeScalingConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::{ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

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
    fn test_llama_recipe_basic() {
        use k3_core::ArchRecipe;

        let recipe = LlamaRecipe::new();
        assert_eq!(recipe.architecture_id(), "llama");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::Q6_K));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert_eq!(recipe.recipe_version(), "1.0.0");
        assert_eq!(recipe.target_layouts(), &["native", "ime_tile_8x16"]);
    }

    #[test]
    fn test_llama4_recipe_irope_masking() {
        use k3_core::ArchRecipe;

        let recipe = LlamaRecipe::new_llama4().with_irope_mask(vec![false, true, false, true]);
        assert_eq!(recipe.architecture_id(), "llama4");
        assert!(!recipe.should_skip_rope(0));
        assert!(recipe.should_skip_rope(1));
        assert!(!recipe.should_skip_rope(2));
        assert!(recipe.should_skip_rope(3));
        // Out-of-bounds layer index defaults to false (standard RoPE applied)
        assert!(!recipe.should_skip_rope(10));
    }

    #[test]
    fn test_project_lm_head_row_split_large_vocab() {
        // 128,256 vocabulary rows with hidden_dim = 16 (F32)
        let rows = if cfg!(miri) { 1024 } else { 128_256 };
        let cols = 16;
        let mut raw_bytes = vec![0_u8; rows * cols * 4];
        // Populate first row and last row with distinct values
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

        // First row dot product: cols * (1.0 * 1.0) = 16.0
        assert!((logits[0] - 16.0).abs() < 1e-4);
        // Middle rows with zero weights should be 0.0
        assert!((logits[100] - 0.0).abs() < 1e-4);
        // Last row dot product: cols * (2.0 * 1.0) = 32.0
        assert!((logits[rows - 1] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_project_lm_head_row_split_tile_m_remainder_edge_case15() {
        // Edge Case 15: Vocabulary size not evenly divisible by TILE_M = 8 (e.g. 128_005 rows)
        let rows = if cfg!(miri) { 1005 } else { 128_005 };
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
        let mut wrong_logits = vec![0.0_f32; 50]; // Expected 100
        let layout = PhaseLayout::default();

        let err = project_lm_head_row_split(&mut wrong_logits, &hidden, &tensor, &layout)
            .expect_err("dimension mismatch");
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_llama_recipe_prefill_and_decode_step() {
        use k3_core::ArchRecipe;

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
}
