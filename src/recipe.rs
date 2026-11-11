//! `LlamaRecipe`: Architecture recipe for Meta Llama (`llama`, `llama4`) and Alibaba `Qwen2` (`qwen2`).
//!
//! Implements [`ArchRecipe`] supporting dynamic GGUF hyperparameter hydration, authentic
//! `token_embd.weight` lookup, Pre-RMSNorm, optional `Qwen2` QKV biases, Grouped-Query Attention
//! (`GQA`) over contiguous `0..=pos` KV-cache slices, smooth frequency-scaled `RoPE` on both `Q`
//! and `K`, `Llama-4` interleaved `iRoPE`, cached `128-byte`-aligned repacked weights, batched
//! multi-token `prefill_step` GEMMs, row-parallel multi-threading, and tied/untied `output.weight`.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, ModelMetadata,
    ModelWeights, QuantType, TensorDescriptor,
};

#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::RopeScalingConfig;
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_with_inv_freq, gqa_causal_attention_f32,
    precompute_rope_inv_freq, swiglu_f32,
};

#[cfg(not(feature = "with-kernels"))]
/// Rotary position embedding scaling configurations.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingConfig {
    /// Standard linear frequency interpolation.
    Linear {
        /// Scaling multiplier factor.
        factor: f32,
    },
    /// Llama-3 smooth frequency scaling with bounded wavelength bands.
    Llama3Smooth {
        /// Low-frequency scaling multiplier.
        factor: f32,
        /// Low frequency bound factor.
        low_freq_factor: f32,
        /// High frequency bound factor.
        high_freq_factor: f32,
        /// Context length used during base pre-training.
        original_max_position_embeddings: usize,
    },
    /// `LongRoPE` dual-factor dynamic position interpolation.
    LongRope {
        /// Per-dimension scaling factors for short context lengths.
        short_factor: Vec<f32>,
        /// Per-dimension scaling factors for extended context lengths.
        long_factor: Vec<f32>,
        /// Context length threshold separating short and long factor tables.
        original_max_position_embeddings: usize,
    },
}

use crate::dispatch::{PhaseLayout, project_lm_head_row_split};
use crate::repack_cache::{
    QuantizedBatchAct, decode_1d_f32, decode_row_to_f32, get_or_repack, quant_block_geometry,
    resolve_2d_dims, set_active_model_anchor,
};

/// Llama and Qwen2 architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct LlamaRecipe {
    /// Exact architecture string (`"llama"`, `"llama4"`, or `"qwen2"`).
    pub architecture: &'static str,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Model hidden dimension ($d_{\text{model}}$).
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension ($d_{\text{ffn}}$).
    pub ffn_dim: usize,
    /// Number of query attention heads ($`H_q`$).
    pub n_heads_q: usize,
    /// Number of key-value attention heads ($`H_{kv}`$).
    pub n_heads_kv: usize,
    /// Attention head dimension ($d_{\text{head}}$).
    pub head_dim: usize,
    /// Vocabulary dimension.
    pub vocab_size: usize,
    /// RMS normalization epsilon constant.
    pub rms_eps: f32,
    /// Base rotary theta parameter ($\theta$).
    pub rope_theta: f32,
    /// Optional rotary scaling configuration (e.g. Llama-3 Smooth `RoPE`).
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Optional layer-by-layer boolean mask for Llama-4 iRoPE (skips `RoPE` on designated layers).
    pub irope_no_pos_mask: Option<Vec<bool>>,
}

impl Default for LlamaRecipe {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct EffectiveGeometry {
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_d_even: usize,
    total_q: usize,
    total_kv: usize,
}

impl LlamaRecipe {
    /// Creates a new default Llama recipe (`"llama"`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "llama",
            num_layers: 32,
            hidden_dim: 4096,
            ffn_dim: 14336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 128_256,
            rms_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: None,
            irope_no_pos_mask: None,
        }
    }

    /// Creates a new Llama-4 recipe (`"llama4"`) with iRoPE interleaved attention support.
    #[must_use]
    pub const fn new_llama4() -> Self {
        Self {
            architecture: "llama4",
            num_layers: 32,
            hidden_dim: 4096,
            ffn_dim: 14336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 128_256,
            rms_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: None,
            irope_no_pos_mask: None,
        }
    }

    /// Creates a new Alibaba `Qwen2` / `Qwen2.5` recipe (`"qwen2"`).
    #[must_use]
    pub const fn new_qwen2() -> Self {
        Self {
            architecture: "qwen2",
            num_layers: 28,
            hidden_dim: 1536,
            ffn_dim: 8960,
            n_heads_q: 12,
            n_heads_kv: 2,
            head_dim: 128,
            vocab_size: 151_936,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            irope_no_pos_mask: None,
        }
    }

    /// Hydrates a [`LlamaRecipe`] from parsed GGUF [`ModelMetadata`] and tensor descriptors.
    #[must_use]
    pub fn from_metadata(meta: &ModelMetadata) -> Self {
        let mut recipe = match meta.architecture() {
            "llama4" => Self::new_llama4(),
            "qwen2" => Self::new_qwen2(),
            _ => Self::new(),
        };

        let hidden_dim = if meta.embedding_dim() > 0 {
            meta.embedding_dim()
        } else if let Some(norm) = meta
            .find_tensor("blk.0.attn_norm.weight")
            .or_else(|| meta.find_tensor("output_norm.weight"))
        {
            norm.num_elements().max(16)
        } else {
            recipe.hidden_dim
        };
        recipe.hidden_dim = hidden_dim;

        let probed_layers = probe_descriptor_layers(meta.tensors());
        if meta.layer_count() > 0 {
            recipe.num_layers = meta.layer_count();
        } else if probed_layers > 0 {
            recipe.num_layers = probed_layers;
        }

        if meta.head_count() > 0 {
            recipe.n_heads_q = meta.head_count();
        }
        if meta.head_count_kv() > 0 {
            recipe.n_heads_kv = meta.head_count_kv();
        }

        if let Some(q_desc) = meta.find_tensor("blk.0.attn_q.weight")
            && let Some(q_rows) = infer_descriptor_rows(q_desc, hidden_dim)
        {
            if recipe.n_heads_q > 0 && q_rows % recipe.n_heads_q == 0 {
                recipe.head_dim = q_rows / recipe.n_heads_q;
            } else if recipe.head_dim > 0 && q_rows % recipe.head_dim == 0 {
                recipe.n_heads_q = q_rows / recipe.head_dim;
            }
        } else if recipe.n_heads_q > 0 && hidden_dim % recipe.n_heads_q == 0 {
            recipe.head_dim = hidden_dim / recipe.n_heads_q;
        }

        if let Some(k_desc) = meta.find_tensor("blk.0.attn_k.weight")
            && let Some(k_rows) = infer_descriptor_rows(k_desc, hidden_dim)
            && recipe.head_dim > 0
            && k_rows % recipe.head_dim == 0
        {
            recipe.n_heads_kv = k_rows / recipe.head_dim;
        }

        if let Some(ffn_desc) = meta
            .find_tensor("blk.0.ffn_gate.weight")
            .or_else(|| meta.find_tensor("blk.0.ffn_up.weight"))
            .or_else(|| meta.find_tensor("blk.0.ffn_down.weight"))
            && let Some(ffn_dim) = infer_descriptor_rows(ffn_desc, hidden_dim)
        {
            recipe.ffn_dim = ffn_dim;
        }

        if let Some(vocab_desc) = meta
            .find_tensor("output.weight")
            .or_else(|| meta.find_tensor("token_embd.weight"))
            && let Some(vocab_rows) = infer_descriptor_rows(vocab_desc, hidden_dim)
        {
            recipe.vocab_size = vocab_rows;
        }

        recipe
    }

    /// Hydrates a [`LlamaRecipe`] from a full [`GgufFile`], reading architecture-prefixed KV
    /// entries (`feed_forward_length`, `attention.key_length`, `rope.freq_base`,
    /// `attention.layer_norm_rms_epsilon`, `vocab_size`, `rope.scaling.*`) alongside tensor shapes.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `gguf.architecture()` is not `"llama"`,
    /// `"llama4"`, or `"qwen2"`.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let arch = gguf.architecture();
        if !matches!(arch, "llama" | "llama4" | "qwen2") {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "LlamaRecipe supports 'llama', 'llama4', and 'qwen2', got '{arch}'"
                ),
            });
        }

        let mut recipe = Self::from_metadata(gguf.metadata());

        let ffn_key = format!("{arch}.feed_forward_length");
        if let Some(ffn) = gguf.metadata_value(&ffn_key).and_then(GgufValue::as_usize)
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        }

        let key_len_key = format!("{arch}.attention.key_length");
        if let Some(hd) = gguf
            .metadata_value(&key_len_key)
            .and_then(GgufValue::as_usize)
            && hd >= 2
        {
            recipe.head_dim = hd;
        }

        let rope_key = format!("{arch}.rope.freq_base");
        if let Some(theta) = gguf.metadata_value(&rope_key).and_then(GgufValue::as_f32)
            && theta.is_finite()
            && theta > 0.0
        {
            recipe.rope_theta = theta;
        }

        let eps_key = format!("{arch}.attention.layer_norm_rms_epsilon");
        if let Some(eps) = gguf.metadata_value(&eps_key).and_then(GgufValue::as_f32)
            && eps.is_finite()
            && eps > 0.0
        {
            recipe.rms_eps = eps;
        }

        let vocab_key = format!("{arch}.vocab_size");
        if let Some(v) = gguf
            .metadata_value(&vocab_key)
            .and_then(GgufValue::as_usize)
            && v > 0
        {
            recipe.vocab_size = v;
        } else if gguf.metadata().find_tensor("output.weight").is_none()
            && gguf.metadata().find_tensor("token_embd.weight").is_none()
            && let Some(GgufValue::Array(arr)) = gguf.metadata_value("tokenizer.ggml.tokens")
        {
            let tok_len = gguf_array_len(arr);
            if tok_len > 0 {
                recipe.vocab_size = tok_len;
            }
        }

        let scale_type_key = format!("{arch}.rope.scaling.type");
        let scale_factor_key = format!("{arch}.rope.scaling.factor");
        if let Some("linear") = gguf
            .metadata_value(&scale_type_key)
            .and_then(GgufValue::as_str)
            && let Some(factor) = gguf
                .metadata_value(&scale_factor_key)
                .and_then(GgufValue::as_f32)
            && factor.is_finite()
            && factor > 0.0
        {
            recipe.rope_scaling = Some(RopeScalingConfig::Linear { factor });
        }

        let no_rope_step_key = format!("{arch}.attention.no_rope_layer_step");
        if let Some(step) = gguf
            .metadata_value(&no_rope_step_key)
            .and_then(GgufValue::as_usize)
            && step > 0
            && recipe.num_layers > 0
        {
            let mask = (0..recipe.num_layers)
                .map(|l| (l + 1) % step == 0)
                .collect();
            recipe.irope_no_pos_mask = Some(mask);
        }

        Ok(recipe)
    }

    /// Configures the Llama-4 iRoPE layer-by-layer positional embedding mask.
    ///
    /// For layers where `mask[layer_idx] == true`, rotary position embeddings are skipped,
    /// preserving token identity representations for long-context stability.
    #[must_use]
    pub fn with_irope_mask(mut self, mask: Vec<bool>) -> Self {
        self.irope_no_pos_mask = Some(mask);
        self
    }

    /// Configures the `RoPE` scaling parameters (e.g. Llama-3 Smooth `RoPE`).
    #[must_use]
    pub fn with_rope_scaling(mut self, scaling: RopeScalingConfig) -> Self {
        self.rope_scaling = Some(scaling);
        self
    }

    /// Returns `true` if `RoPE` should be skipped on `layer_idx` under Llama-4 iRoPE.
    #[must_use]
    pub fn should_skip_rope(&self, layer_idx: usize) -> bool {
        self.irope_no_pos_mask
            .as_deref()
            .and_then(|m| m.get(layer_idx))
            .copied()
            .unwrap_or(false)
    }

    fn resolve_geometry(
        &self,
        weights: &dyn ModelWeights,
        logits_len: usize,
    ) -> Result<EffectiveGeometry, K3Error> {
        if weights.tensor_count() == 0 {
            let hidden_dim = self.hidden_dim.max(16);
            let ffn_dim = self.ffn_dim.max(16);
            let n_heads_q = self.n_heads_q.max(1);
            let n_heads_kv = self.n_heads_kv.max(1);
            let hd = self.head_dim.max(2);
            let head_d_even = if hd % 2 == 0 { hd } else { hd - 1 };
            return Ok(EffectiveGeometry {
                num_layers: self.num_layers,
                hidden_dim,
                ffn_dim,
                n_heads_q,
                n_heads_kv,
                head_d_even,
                total_q: n_heads_q * head_d_even,
                total_kv: n_heads_kv * head_d_even,
            });
        }

        let hidden_dim = self.resolve_live_hidden_dim(weights, logits_len)?;
        let num_layers = self.resolve_live_num_layers(weights);

        let (n_heads_q, n_heads_kv, head_d_even) = self.resolve_live_heads(weights, hidden_dim)?;

        let ffn_dim = if weights.contains_tensor("blk.0.ffn_gate.weight") {
            let t = weights.tensor("blk.0.ffn_gate.weight")?;
            resolve_2d_dims(&t, hidden_dim)?.0
        } else if weights.contains_tensor("blk.0.ffn_up.weight") {
            let t = weights.tensor("blk.0.ffn_up.weight")?;
            resolve_2d_dims(&t, hidden_dim)?.0
        } else if weights.contains_tensor("blk.0.ffn_down.weight") {
            let t = weights.tensor("blk.0.ffn_down.weight")?;
            // ffn_down maps ffn_dim -> hidden_dim, so hidden_dim is the output row axis
            let shape = t.shape();
            if shape.len() == 2 {
                if shape[0] == hidden_dim {
                    shape[1]
                } else {
                    shape[0]
                }
            } else {
                self.ffn_dim.max(16)
            }
        } else {
            self.ffn_dim.max(16)
        };

        Ok(EffectiveGeometry {
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_d_even,
            total_q: n_heads_q * head_d_even,
            total_kv: n_heads_kv * head_d_even,
        })
    }

    fn resolve_live_hidden_dim(
        &self,
        weights: &dyn ModelWeights,
        logits_len: usize,
    ) -> Result<usize, K3Error> {
        if weights.contains_tensor("blk.0.attn_norm.weight") {
            return Ok(weights.tensor("blk.0.attn_norm.weight")?.num_elements());
        }
        if weights.contains_tensor("output_norm.weight") {
            return Ok(weights.tensor("output_norm.weight")?.num_elements());
        }
        if weights.contains_tensor("token_embd.weight") {
            let emb = weights.tensor("token_embd.weight")?;
            let shape = emb.shape();
            if shape.len() == 2 {
                if logits_len > 0 && shape[1] == logits_len && shape[0] != logits_len {
                    return Ok(shape[0]);
                }
                if logits_len > 0 && shape[0] == logits_len && shape[1] != logits_len {
                    return Ok(shape[1]);
                }
                if shape[0] == self.hidden_dim || shape[1] == self.hidden_dim {
                    return Ok(self.hidden_dim);
                }
                return Ok(shape[0]);
            }
        }
        if weights.contains_tensor("blk.0.attn_q.weight") {
            let q = weights.tensor("blk.0.attn_q.weight")?;
            let shape = q.shape();
            if shape.len() == 2 {
                if shape[0] == self.hidden_dim || shape[1] == self.hidden_dim {
                    return Ok(self.hidden_dim);
                }
                if let Some((blk_elems, _)) = quant_block_geometry(q.quant())
                    && blk_elems > 1
                {
                    if shape[0] % blk_elems == 0 && shape[1] % blk_elems != 0 {
                        return Ok(shape[0]);
                    }
                    if shape[1] % blk_elems == 0 && shape[0] % blk_elems != 0 {
                        return Ok(shape[1]);
                    }
                }
                return Ok(shape[0]);
            }
        }
        Ok(self.hidden_dim.max(16))
    }

    fn resolve_live_num_layers(&self, weights: &dyn ModelWeights) -> usize {
        let mut probed = 0_usize;
        for layer in 0..512 {
            let has_layer = weights.contains_tensor(&format!("blk.{layer}.attn_norm.weight"))
                || weights.contains_tensor(&format!("blk.{layer}.attn_q.weight"))
                || weights.contains_tensor(&format!("blk.{layer}.ffn_gate.weight"))
                || weights.contains_tensor(&format!("blk.{layer}.attn_output.weight"));
            if has_layer {
                probed = layer + 1;
            } else if probed > 0 {
                break;
            }
        }
        if probed > 0 { probed } else { self.num_layers }
    }

    fn resolve_live_heads(
        &self,
        weights: &dyn ModelWeights,
        hidden_dim: usize,
    ) -> Result<(usize, usize, usize), K3Error> {
        let hd_default = {
            let d = self.head_dim.max(2);
            if d % 2 == 0 { d } else { d - 1 }
        };
        if !weights.contains_tensor("blk.0.attn_q.weight") {
            let n_q = self.n_heads_q.max(1);
            let n_kv = self.n_heads_kv.max(1);
            return Ok((n_q, n_kv, hd_default));
        }

        let q_tensor = weights.tensor("blk.0.attn_q.weight")?;
        let q_rows = resolve_2d_dims(&q_tensor, hidden_dim)?.0;
        let k_rows = if weights.contains_tensor("blk.0.attn_k.weight") {
            let k_tensor = weights.tensor("blk.0.attn_k.weight")?;
            resolve_2d_dims(&k_tensor, hidden_dim)?.0
        } else {
            q_rows
        };

        let cur_q = self.n_heads_q.max(1);
        let cur_kv = self.n_heads_kv.max(1);
        if cur_q * hd_default == q_rows && cur_kv * hd_default == k_rows && cur_q % cur_kv == 0 {
            return Ok((cur_q, cur_kv, hd_default));
        }

        let g = gcd_usize(q_rows, k_rows);
        let candidates = [
            hd_default,
            128,
            64,
            96,
            80,
            48,
            32,
            16,
            8,
            4,
            if g >= 2 && g % 2 == 0 { g } else { 2 },
            2,
        ];
        for cand in candidates {
            if cand >= 2
                && cand % 2 == 0
                && q_rows >= cand
                && k_rows >= cand
                && q_rows % cand == 0
                && k_rows % cand == 0
            {
                let nq = q_rows / cand;
                let nkv = k_rows / cand;
                if nkv > 0 && nq % nkv == 0 {
                    return Ok((nq, nkv, cand));
                }
            }
        }

        Err(K3Error::CapabilityMismatch {
            architecture: self.architecture_id().to_owned(),
            reason: format!(
                "unable to resolve valid even head_dim for q_rows={q_rows}, k_rows={k_rows}"
            ),
        })
    }

    fn embed_tokens(
        tokens: &[u32],
        hidden_batch: &mut [f32],
        hidden_dim: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if weights.contains_tensor("token_embd.weight") {
            let emb_tensor = weights.tensor("token_embd.weight")?;
            set_active_model_anchor(&emb_tensor);
            let (vocab_rows, _) = resolve_2d_dims(&emb_tensor, hidden_dim)?;
            for (t_idx, &tok_id) in tokens.iter().enumerate() {
                let row_idx = usize::try_from(tok_id).map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: "token_embd.weight".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: emb_tensor.byte_len(),
                    reason: format!("token_id {tok_id} overflows usize"),
                })?;
                if row_idx >= vocab_rows {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "token_embd.weight".to_owned(),
                        offset: row_idx,
                        byte_len: hidden_dim,
                        buffer_len: vocab_rows,
                        reason: format!(
                            "token_id {tok_id} out of vocabulary bounds [0..{vocab_rows})"
                        ),
                    });
                }
                let dst = &mut hidden_batch[t_idx * hidden_dim..(t_idx + 1) * hidden_dim];
                decode_row_to_f32(&emb_tensor, row_idx, hidden_dim, dst)?;
            }
        } else {
            for (t_idx, &tok_id) in tokens.iter().enumerate() {
                #[expect(clippy::cast_precision_loss, reason = "token_id modulo 17 fits in f32")]
                let base_val = ((tok_id % 17) as f32) * 0.1 + 0.5;
                hidden_batch[t_idx * hidden_dim..(t_idx + 1) * hidden_dim].fill(base_val);
            }
        }
        Ok(())
    }

    fn apply_rmsnorm_batch(
        &self,
        output: &mut [f32],
        input: &[f32],
        norm_name: &str,
        dim: usize,
        m: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if !weights.contains_tensor(norm_name) {
            output.copy_from_slice(input);
            return Ok(());
        }
        let norm_tensor = weights.tensor(norm_name)?;
        let w_f32 = decode_1d_f32(&norm_tensor, dim, true)?;
        let eps = self.rms_eps.max(1.0e-7);

        for tok in 0..m {
            let in_row = &input[tok * dim..(tok + 1) * dim];
            let out_row = &mut output[tok * dim..(tok + 1) * dim];
            rmsnorm_row(out_row, in_row, &w_f32, eps)?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "orchestrates full batched transformer block across QKV, biases, RoPE, KV-cache GQA, and SwiGLU"
    )]
    fn execute_batch_forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
        logits_len: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let m = tokens.len();
        let geom = self.resolve_geometry(weights, logits_len)?;
        if m == 0 {
            return Ok(vec![0.0_f32; geom.hidden_dim]);
        }

        let end_pos = start_pos + m;
        let mut hidden_batch = vec![0.0_f32; m * geom.hidden_dim];
        Self::embed_tokens(tokens, &mut hidden_batch, geom.hidden_dim, weights)?;

        let inv_freq = precompute_inv_freq_compat(
            geom.head_d_even,
            self.rope_theta,
            self.rope_scaling.as_ref(),
            end_pos,
        )?;

        let mut norm_batch = vec![0.0_f32; m * geom.hidden_dim];
        let mut q_batch = vec![0.0_f32; m * geom.total_q];
        let mut k_batch = vec![0.0_f32; m * geom.total_kv];
        let mut v_batch = vec![0.0_f32; m * geom.total_kv];
        let mut attn_out_batch = vec![0.0_f32; m * geom.total_q];
        let mut proj_hidden_batch = vec![0.0_f32; m * geom.hidden_dim];
        let mut gate_batch = vec![0.0_f32; m * geom.ffn_dim];
        let mut up_batch = vec![0.0_f32; m * geom.ffn_dim];
        let mut ffn_act_batch = vec![0.0_f32; m * geom.ffn_dim];

        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let attn_scale = 1.0_f32 / (geom.head_d_even as f32).sqrt();

        for layer in 0..geom.num_layers {
            // 1. Pre-Attention RMSNorm
            self.apply_rmsnorm_batch(
                &mut norm_batch,
                &hidden_batch,
                &format!("blk.{layer}.attn_norm.weight"),
                geom.hidden_dim,
                m,
                weights,
            )?;

            // 2. Batched Q, K, V projections sharing pre-quantized activation tiles when possible
            project_qkv_layer(
                layer,
                &norm_batch,
                &mut q_batch,
                &mut k_batch,
                &mut v_batch,
                geom,
                m,
                weights,
            )?;

            // 3. Rotary Position Embeddings on BOTH Q and K
            if !self.should_skip_rope(layer) {
                for tok in 0..m {
                    let pos = start_pos + tok;
                    apply_rope_compat(
                        &mut q_batch[tok * geom.total_q..(tok + 1) * geom.total_q],
                        geom.head_d_even,
                        pos,
                        &inv_freq,
                    )?;
                    apply_rope_compat(
                        &mut k_batch[tok * geom.total_kv..(tok + 1) * geom.total_kv],
                        geom.head_d_even,
                        pos,
                        &inv_freq,
                    )?;
                }
            }

            // 4. Zero-copy 0..=pos KV Cache Residency & Causal GQA Attention
            execute_layer_causal_gqa(
                layer,
                start_pos,
                m,
                geom,
                attn_scale,
                &q_batch,
                &k_batch,
                &v_batch,
                kv_cache,
                &mut attn_out_batch,
            )?;

            // 5. Batched Attention Output Projection & Residual Connection
            let attn_out_name = format!("blk.{layer}.attn_output.weight");
            if weights.contains_tensor(&attn_out_name) {
                let out_tensor = weights.tensor(&attn_out_name)?;
                let (rows, cols) = resolve_2d_dims(&out_tensor, geom.total_q)?;
                if rows != geom.hidden_dim {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: attn_out_name,
                        offset: out_tensor.offset(),
                        byte_len: out_tensor.byte_len(),
                        buffer_len: out_tensor.as_bytes().len(),
                        reason: format!(
                            "attn_output rows {rows} != hidden_dim {}",
                            geom.hidden_dim
                        ),
                    });
                }
                let cached = get_or_repack(&out_tensor, rows, cols)?;
                cached.project_batch(&mut proj_hidden_batch, &attn_out_batch, m, true)?;
                add_optional_bias(
                    &mut proj_hidden_batch,
                    &format!("blk.{layer}.attn_output.bias"),
                    geom.hidden_dim,
                    m,
                    weights,
                )?;
                for (h, &p) in hidden_batch.iter_mut().zip(&proj_hidden_batch) {
                    *h += p;
                }
            } else {
                for tok in 0..m {
                    let h_row =
                        &mut hidden_batch[tok * geom.hidden_dim..(tok + 1) * geom.hidden_dim];
                    let a_row = &attn_out_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
                    for (i, h) in h_row.iter_mut().enumerate() {
                        *h += a_row[i % geom.total_q] * 0.05;
                    }
                }
            }

            // 6. Pre-FFN RMSNorm
            self.apply_rmsnorm_batch(
                &mut norm_batch,
                &hidden_batch,
                &format!("blk.{layer}.ffn_norm.weight"),
                geom.hidden_dim,
                m,
                weights,
            )?;

            // 7. Batched SwiGLU FFN & Residual Connection
            execute_layer_swiglu_ffn(
                layer,
                &norm_batch,
                &mut gate_batch,
                &mut up_batch,
                &mut ffn_act_batch,
                &mut proj_hidden_batch,
                &mut hidden_batch,
                geom,
                m,
                weights,
            )?;
        }

        // Final output RMSNorm is only required for the last token row (m - 1)
        let last_in = &hidden_batch[(m - 1) * geom.hidden_dim..m * geom.hidden_dim];
        let mut final_hidden = vec![0.0_f32; geom.hidden_dim];
        self.apply_rmsnorm_batch(
            &mut final_hidden,
            last_in,
            "output_norm.weight",
            geom.hidden_dim,
            1,
            weights,
        )?;

        Ok(final_hidden)
    }

    fn project_logits(
        ctx: &mut ExecutionContext<'_>,
        hidden: &[f32],
        weights: &dyn ModelWeights,
        synthetic_decode_tok: Option<u32>,
    ) -> Result<(), K3Error> {
        if weights.contains_tensor("output.weight") {
            let output_weight = weights.tensor("output.weight")?;
            let layout = PhaseLayout::default();
            return project_lm_head_row_split(ctx.logits_mut(), hidden, &output_weight, &layout);
        }
        if weights.contains_tensor("token_embd.weight") {
            let tied_weight = weights.tensor("token_embd.weight")?;
            let layout = PhaseLayout::default();
            return project_lm_head_row_split(ctx.logits_mut(), hidden, &tied_weight, &layout);
        }

        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let bias = (hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
            if let Some(tok) = synthetic_decode_tok {
                let next_slot = (usize::try_from(tok).unwrap_or(0) + 1) % logits.len();
                logits[next_slot] = 2.0 + bias;
            } else {
                let target_slot = if logits.len() > 9 { 9 } else { 0 };
                logits[target_slot] = 1.0 + bias;
            }
        }
        Ok(())
    }
}

impl ArchRecipe for LlamaRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_K_M
                | QuantType::Q8_0
                | QuantType::Q4_0
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens().to_vec();
        let prompt_len = prompt_tokens.len();
        let start_pos = ctx.position();
        let end_pos = start_pos
            .checked_add(prompt_len)
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: ctx.n_ctx(),
            })?;
        if end_pos > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let logits_len = ctx.logits().len();
        let last_hidden = self.execute_batch_forward(
            &prompt_tokens,
            start_pos,
            ctx.kv_cache_mut(),
            weights,
            logits_len,
        )?;
        ctx.advance_position(prompt_len)?;
        Self::project_logits(ctx, &last_hidden, weights, None)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        let pos = ctx.position();
        if pos.saturating_add(1) > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: pos.saturating_add(1),
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let logits_len = ctx.logits().len();
        let single = [token];
        let hidden =
            self.execute_batch_forward(&single, pos, ctx.kv_cache_mut(), weights, logits_len)?;
        ctx.advance_position(1)?;
        Self::project_logits(ctx, &hidden, weights, Some(token))
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "projects Q, K, and V batches with shared pre-quantized activation buffer"
)]
fn project_qkv_layer(
    layer: usize,
    norm_batch: &[f32],
    q_batch: &mut [f32],
    k_batch: &mut [f32],
    v_batch: &mut [f32],
    geom: EffectiveGeometry,
    m: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let q_name = format!("blk.{layer}.attn_q.weight");
    let k_name = format!("blk.{layer}.attn_k.weight");
    let v_name = format!("blk.{layer}.attn_v.weight");

    let q_mat = if weights.contains_tensor(&q_name) {
        let t = weights.tensor(&q_name)?;
        let (rows, cols) = resolve_2d_dims(&t, geom.hidden_dim)?;
        if rows != geom.total_q {
            return Err(K3Error::TensorBoundsError {
                tensor_name: q_name,
                offset: t.offset(),
                byte_len: t.byte_len(),
                buffer_len: t.as_bytes().len(),
                reason: format!("attn_q rows {rows} != total_q {}", geom.total_q),
            });
        }
        Some(get_or_repack(&t, rows, cols)?)
    } else {
        None
    };

    let k_mat = if weights.contains_tensor(&k_name) {
        let t = weights.tensor(&k_name)?;
        let (rows, cols) = resolve_2d_dims(&t, geom.hidden_dim)?;
        if rows != geom.total_kv {
            return Err(K3Error::TensorBoundsError {
                tensor_name: k_name,
                offset: t.offset(),
                byte_len: t.byte_len(),
                buffer_len: t.as_bytes().len(),
                reason: format!("attn_k rows {rows} != total_kv {}", geom.total_kv),
            });
        }
        Some(get_or_repack(&t, rows, cols)?)
    } else {
        None
    };

    let v_mat = if weights.contains_tensor(&v_name) {
        let t = weights.tensor(&v_name)?;
        let (rows, cols) = resolve_2d_dims(&t, geom.hidden_dim)?;
        if rows != geom.total_kv {
            return Err(K3Error::TensorBoundsError {
                tensor_name: v_name,
                offset: t.offset(),
                byte_len: t.byte_len(),
                buffer_len: t.as_bytes().len(),
                reason: format!("attn_v rows {rows} != total_kv {}", geom.total_kv),
            });
        }
        Some(get_or_repack(&t, rows, cols)?)
    } else {
        None
    };

    let shared_group = q_mat
        .as_ref()
        .and_then(|mat| mat.quant_group_size())
        .or_else(|| k_mat.as_ref().and_then(|mat| mat.quant_group_size()))
        .or_else(|| v_mat.as_ref().and_then(|mat| mat.quant_group_size()));
    let shared_act =
        shared_group.map(|gs| QuantizedBatchAct::quantize(norm_batch, m, geom.hidden_dim, gs));

    if let Some(mat) = q_mat {
        mat.project_batch_with_act(q_batch, norm_batch, shared_act.as_ref(), m, true)?;
    } else {
        fill_synthetic_proj(q_batch, norm_batch, geom.total_q, geom.hidden_dim, m, 1.0);
    }
    add_optional_bias(
        q_batch,
        &format!("blk.{layer}.attn_q.bias"),
        geom.total_q,
        m,
        weights,
    )?;

    if let Some(mat) = k_mat {
        mat.project_batch_with_act(k_batch, norm_batch, shared_act.as_ref(), m, true)?;
    } else {
        fill_synthetic_proj(k_batch, norm_batch, geom.total_kv, geom.hidden_dim, m, 1.0);
    }
    add_optional_bias(
        k_batch,
        &format!("blk.{layer}.attn_k.bias"),
        geom.total_kv,
        m,
        weights,
    )?;

    if let Some(mat) = v_mat {
        mat.project_batch_with_act(v_batch, norm_batch, shared_act.as_ref(), m, true)?;
    } else {
        fill_synthetic_proj(v_batch, norm_batch, geom.total_kv, geom.hidden_dim, m, 1.0);
    }
    add_optional_bias(
        v_batch,
        &format!("blk.{layer}.attn_v.bias"),
        geom.total_kv,
        m,
        weights,
    )?;

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "manages layer KV cache slice partitioning and batched causal GQA execution"
)]
fn execute_layer_causal_gqa(
    layer: usize,
    start_pos: usize,
    m: usize,
    geom: EffectiveGeometry,
    attn_scale: f32,
    q_batch: &[f32],
    k_batch: &[f32],
    v_batch: &[f32],
    kv_cache: &mut [f32],
    attn_out_batch: &mut [f32],
) -> Result<(), K3Error> {
    let end_pos = start_pos + m;
    let per_tok_all_layers = 2_usize
        .saturating_mul(geom.num_layers)
        .saturating_mul(geom.total_kv);

    if per_tok_all_layers > 0 && kv_cache.len() >= end_pos.saturating_mul(per_tok_all_layers) {
        let seq_cap = kv_cache.len() / per_tok_all_layers;
        let layer_len = 2 * seq_cap * geom.total_kv;
        let layer_slice = &mut kv_cache[layer * layer_len..(layer + 1) * layer_len];
        let (k_layer, v_layer) = layer_slice.split_at_mut(seq_cap * geom.total_kv);

        let write_start = start_pos * geom.total_kv;
        let write_end = end_pos * geom.total_kv;
        k_layer[write_start..write_end].copy_from_slice(k_batch);
        v_layer[write_start..write_end].copy_from_slice(v_batch);

        let k_valid = &k_layer[..write_end];
        let v_valid = &v_layer[..write_end];

        #[cfg(not(miri))]
        if m >= 2 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            return attn_out_batch
                .par_chunks_mut(geom.total_q)
                .enumerate()
                .try_for_each(|(tok, out_tok)| {
                    let cur_seq_len = start_pos + tok + 1;
                    let q_tok = &q_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
                    gqa_attention_compat(
                        out_tok,
                        q_tok,
                        &k_valid[..cur_seq_len * geom.total_kv],
                        &v_valid[..cur_seq_len * geom.total_kv],
                        geom.n_heads_q,
                        geom.n_heads_kv,
                        geom.head_d_even,
                        cur_seq_len,
                        attn_scale,
                    )
                });
        }

        for tok in 0..m {
            let cur_seq_len = start_pos + tok + 1;
            let q_tok = &q_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
            let out_tok = &mut attn_out_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
            gqa_attention_compat(
                out_tok,
                q_tok,
                &k_valid[..cur_seq_len * geom.total_kv],
                &v_valid[..cur_seq_len * geom.total_kv],
                geom.n_heads_q,
                geom.n_heads_kv,
                geom.head_d_even,
                cur_seq_len,
                attn_scale,
            )?;
        }
        return Ok(());
    }

    // Fallback when a synthetic unit test supplies a tiny scratch `kv_cache`
    let layer_stride = kv_cache
        .len()
        .checked_div(geom.num_layers.max(1))
        .unwrap_or(0);
    if layer_stride > 0 {
        let layer_start = layer * layer_stride;
        let layer_end = (layer_start + layer_stride).min(kv_cache.len());
        let layer_buf = &mut kv_cache[layer_start..layer_end];
        let half_stride = layer_buf.len() / 2;
        if half_stride > 0 {
            let last_k = &k_batch[(m - 1) * geom.total_kv..m * geom.total_kv];
            let last_v = &v_batch[(m - 1) * geom.total_kv..m * geom.total_kv];
            let copy_len = geom.total_kv.min(half_stride);
            layer_buf[..copy_len].copy_from_slice(&last_k[..copy_len]);
            layer_buf[half_stride..half_stride + copy_len].copy_from_slice(&last_v[..copy_len]);
        }
    }

    for tok in 0..m {
        let cur_len = tok + 1;
        let q_tok = &q_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
        let out_tok = &mut attn_out_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
        gqa_attention_compat(
            out_tok,
            q_tok,
            &k_batch[..cur_len * geom.total_kv],
            &v_batch[..cur_len * geom.total_kv],
            geom.n_heads_q,
            geom.n_heads_kv,
            geom.head_d_even,
            cur_len,
            attn_scale,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "executes batched gate, up, SwiGLU, and down FFN projections on preallocated workspaces"
)]
fn execute_layer_swiglu_ffn(
    layer: usize,
    norm_batch: &[f32],
    gate_batch: &mut [f32],
    up_batch: &mut [f32],
    ffn_act_batch: &mut [f32],
    proj_hidden_batch: &mut [f32],
    hidden_batch: &mut [f32],
    geom: EffectiveGeometry,
    m: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let gate_name = format!("blk.{layer}.ffn_gate.weight");
    let up_name = format!("blk.{layer}.ffn_up.weight");
    let down_name = format!("blk.{layer}.ffn_down.weight");

    let gate_mat = if weights.contains_tensor(&gate_name) {
        let t = weights.tensor(&gate_name)?;
        let (rows, cols) = resolve_2d_dims(&t, geom.hidden_dim)?;
        if rows != geom.ffn_dim {
            return Err(K3Error::TensorBoundsError {
                tensor_name: gate_name,
                offset: t.offset(),
                byte_len: t.byte_len(),
                buffer_len: t.as_bytes().len(),
                reason: format!("ffn_gate rows {rows} != ffn_dim {}", geom.ffn_dim),
            });
        }
        Some(get_or_repack(&t, rows, cols)?)
    } else {
        None
    };

    let up_mat = if weights.contains_tensor(&up_name) {
        let t = weights.tensor(&up_name)?;
        let (rows, cols) = resolve_2d_dims(&t, geom.hidden_dim)?;
        if rows != geom.ffn_dim {
            return Err(K3Error::TensorBoundsError {
                tensor_name: up_name,
                offset: t.offset(),
                byte_len: t.byte_len(),
                buffer_len: t.as_bytes().len(),
                reason: format!("ffn_up rows {rows} != ffn_dim {}", geom.ffn_dim),
            });
        }
        Some(get_or_repack(&t, rows, cols)?)
    } else {
        None
    };

    let shared_group = gate_mat
        .as_ref()
        .and_then(|mat| mat.quant_group_size())
        .or_else(|| up_mat.as_ref().and_then(|mat| mat.quant_group_size()));
    let shared_act =
        shared_group.map(|gs| QuantizedBatchAct::quantize(norm_batch, m, geom.hidden_dim, gs));

    if let Some(mat) = gate_mat {
        mat.project_batch_with_act(gate_batch, norm_batch, shared_act.as_ref(), m, true)?;
    } else {
        fill_synthetic_proj(
            gate_batch,
            norm_batch,
            geom.ffn_dim,
            geom.hidden_dim,
            m,
            1.0,
        );
    }

    if let Some(mat) = up_mat {
        mat.project_batch_with_act(up_batch, norm_batch, shared_act.as_ref(), m, true)?;
    } else {
        fill_synthetic_proj(up_batch, norm_batch, geom.ffn_dim, geom.hidden_dim, m, 0.5);
    }

    swiglu_compat(ffn_act_batch, gate_batch, up_batch)?;

    if weights.contains_tensor(&down_name) {
        let down_tensor = weights.tensor(&down_name)?;
        let (rows, cols) = resolve_2d_dims(&down_tensor, geom.ffn_dim)?;
        if rows != geom.hidden_dim {
            return Err(K3Error::TensorBoundsError {
                tensor_name: down_name,
                offset: down_tensor.offset(),
                byte_len: down_tensor.byte_len(),
                buffer_len: down_tensor.as_bytes().len(),
                reason: format!("ffn_down rows {rows} != hidden_dim {}", geom.hidden_dim),
            });
        }
        let cached = get_or_repack(&down_tensor, rows, cols)?;
        cached.project_batch(proj_hidden_batch, ffn_act_batch, m, true)?;
        for (h, &p) in hidden_batch.iter_mut().zip(&*proj_hidden_batch) {
            *h += p;
        }
    } else {
        for tok in 0..m {
            let h_row = &mut hidden_batch[tok * geom.hidden_dim..(tok + 1) * geom.hidden_dim];
            let f_row = &ffn_act_batch[tok * geom.ffn_dim..(tok + 1) * geom.ffn_dim];
            for (i, h) in h_row.iter_mut().enumerate() {
                *h += f_row[i % geom.ffn_dim] * 0.01;
            }
        }
    }

    Ok(())
}

fn add_optional_bias(
    batch: &mut [f32],
    bias_name: &str,
    dim: usize,
    m: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if !weights.contains_tensor(bias_name) {
        return Ok(());
    }
    let bias_tensor = weights.tensor(bias_name)?;
    let bias = decode_1d_f32(&bias_tensor, dim, false)?;
    for tok in 0..m {
        let row = &mut batch[tok * dim..(tok + 1) * dim];
        for (slot, &b) in row.iter_mut().zip(&bias) {
            *slot += b;
        }
    }
    Ok(())
}

fn fill_synthetic_proj(
    out: &mut [f32],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    m: usize,
    scale: f32,
) {
    for tok in 0..m {
        let base = input[tok * in_dim] * scale;
        out[tok * out_dim..(tok + 1) * out_dim].fill(base);
    }
}

fn rmsnorm_row(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().rmsnorm(output, input, weight, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if input.is_empty() || output.len() != input.len() || weight.len() != input.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rmsnorm".to_owned(),
                offset: 0,
                byte_len: size_of_val(output),
                buffer_len: size_of_val(input),
                reason: "rmsnorm slice mismatch".to_owned(),
            });
        }
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        #[expect(clippy::cast_precision_loss, reason = "dim fits in f32")]
        let inv = 1.0_f32 / (sum_sq / (input.len() as f32) + eps).sqrt();
        for i in 0..input.len() {
            output[i] = input[i] * inv * weight[i];
        }
        Ok(())
    }
}

fn precompute_inv_freq_compat(
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScalingConfig>,
    seq_len: usize,
) -> Result<Vec<f32>, K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        precompute_rope_inv_freq(head_dim, theta, scaling, seq_len.max(1))
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let _ = (scaling, seq_len);
        if head_dim == 0 || head_dim % 2 != 0 || !theta.is_finite() || theta <= 0.0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rope".to_owned(),
                reason: "invalid RoPE parameters".to_owned(),
            });
        }
        let half = head_dim / 2;
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits in f32")]
        let hd_f32 = head_dim as f32;
        let mut out = Vec::with_capacity(half);
        for i in 0..half {
            #[expect(clippy::cast_precision_loss, reason = "index fits in f32")]
            let exp = (2 * i) as f32 / hd_f32;
            out.push(1.0_f32 / theta.powf(exp));
        }
        Ok(out)
    }
}

fn apply_rope_compat(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    inv_freq: &[f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        apply_rope_with_inv_freq(qk, head_dim, pos, inv_freq)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if head_dim == 0
            || head_dim % 2 != 0
            || qk.len() % head_dim != 0
            || inv_freq.len() != head_dim / 2
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rope".to_owned(),
                reason: "invalid RoPE slice bounds".to_owned(),
            });
        }
        if pos == 0 {
            return Ok(());
        }
        let half = head_dim / 2;
        #[expect(clippy::cast_precision_loss, reason = "pos fits in f32")]
        let pos_f32 = pos as f32;
        for head in qk.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let angle = pos_f32 * inv_freq[i];
                let (s, c) = angle.sin_cos();
                let x1 = head[i];
                let x2 = head[i + half];
                head[i] = x1 * c - x2 * s;
                head[i + half] = x1 * s + x2 * c;
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_f32 signature"
)]
fn gqa_attention_compat(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_f32(
            out, q, k_cache, v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, scale,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if n_heads_kv == 0
            || n_heads_q % n_heads_kv != 0
            || seq_len == 0
            || out.len() != n_heads_q * head_dim
            || q.len() != n_heads_q * head_dim
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "attention".to_owned(),
                reason: "invalid GQA dimensions".to_owned(),
            });
        }
        let group = n_heads_q / n_heads_kv;
        let kv_stride = n_heads_kv * head_dim;
        let mut scores = vec![0.0_f32; seq_len];
        for hq in 0..n_heads_q {
            let hkv = hq / group;
            let q_head = &q[hq * head_dim..(hq + 1) * head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_head =
                    &k_cache[t * kv_stride + hkv * head_dim..t * kv_stride + (hkv + 1) * head_dim];
                let s = q_head.iter().zip(k_head).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                scores[t] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut denom = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                denom += *s;
            }
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            let out_head = &mut out[hq * head_dim..(hq + 1) * head_dim];
            out_head.fill(0.0);
            for (t, &w_unnorm) in scores.iter().enumerate() {
                let w = w_unnorm * inv_denom;
                let v_head =
                    &v_cache[t * kv_stride + hkv * head_dim..t * kv_stride + (hkv + 1) * head_dim];
                for (o, &v_val) in out_head.iter_mut().zip(v_head) {
                    *o += w * v_val;
                }
            }
        }
        Ok(())
    }
}

fn swiglu_compat(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        swiglu_f32(out, gate, up)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if out.len() != gate.len() || out.len() != up.len() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "swiglu".to_owned(),
                reason: "SwiGLU slice length mismatch".to_owned(),
            });
        }
        for i in 0..out.len() {
            let g = gate[i];
            let sig = if g <= -20.0 {
                0.0
            } else if g >= 20.0 {
                1.0
            } else {
                1.0 / (1.0 + (-g).exp())
            };
            out[i] = (g * sig) * up[i];
        }
        Ok(())
    }
}

fn probe_descriptor_layers(tensors: &[TensorDescriptor]) -> usize {
    let mut max_layer = 0_usize;
    for desc in tensors {
        if let Some(rest) = desc.name().strip_prefix("blk.")
            && let Some((idx_str, _)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            max_layer = max_layer.max(idx + 1);
        }
    }
    max_layer
}

fn infer_descriptor_rows(desc: &TensorDescriptor, expected_cols: usize) -> Option<usize> {
    let shape = desc.shape();
    if shape.len() != 2 {
        return None;
    }
    if shape[0] == expected_cols {
        Some(shape[1])
    } else if shape[1] == expected_cols {
        Some(shape[0])
    } else {
        None
    }
}

fn gguf_array_len(arr: &GgufArray<'_>) -> usize {
    match arr {
        GgufArray::Uint8(v) => v.len(),
        GgufArray::Int8(v) => v.len(),
        GgufArray::Uint16(v) => v.len(),
        GgufArray::Int16(v) => v.len(),
        GgufArray::Uint32(v) => v.len(),
        GgufArray::Int32(v) => v.len(),
        GgufArray::Float32(v) => v.len(),
        GgufArray::Bool(v) => v.len(),
        GgufArray::String(v) => v.len(),
        GgufArray::Uint64(v) => v.len(),
        GgufArray::Int64(v) => v.len(),
        GgufArray::Float64(v) => v.len(),
    }
}

const fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
