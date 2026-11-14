//! `LlamaRecipe`: Architecture recipe for Meta Llama (`llama`, `llama4`) and Alibaba `Qwen2` (`qwen2`).
//!
//! Implements [`ArchRecipe`] supporting dynamic GGUF hyperparameter hydration, authentic
//! `token_embd.weight` lookup, Pre-RMSNorm, optional `Qwen2` QKV biases, Grouped-Query Attention
//! (`GQA`) over contiguous `0..=pos` KV-cache slices, smooth frequency-scaled `RoPE` on both `Q`
//! and `K`, `Llama-4` interleaved `iRoPE`, cached `128-byte`-aligned repacked weights, batched
//! multi-token `prefill_step` GEMMs, row-parallel multi-threading, and tied/untied `output.weight`.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, TensorDescriptor,
};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    AttentionMaskConfig, BaselineKernel, VectorKernel, apply_rope_with_inv_freq_and_layout,
    gqa_causal_attention_prefill_f32, precompute_rope_inv_freq, swiglu_f32,
};
#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::{RopeScalingConfig, RopeType};

#[cfg(not(feature = "with-kernels"))]
/// Memory layout convention for paired Rotary Position Embedding (`RoPE`) dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeType {
    /// Interleaved adjacent pairs: `(v[2 * i], v[2 * i + 1])` (`LLAMA_ROPE_TYPE_NORM`).
    Norm,
    /// Split-half pairs: `(v[i], v[i + rotary_dim / 2])` (`LLAMA_ROPE_TYPE_NEOX`).
    #[default]
    NeoX,
}

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

#[cfg(not(feature = "with-kernels"))]
#[derive(Clone, Copy)]
struct AttentionMaskConfig {
    start_pos: usize,
    sliding_window: Option<usize>,
    scale: f32,
}

#[cfg(not(feature = "with-kernels"))]
impl AttentionMaskConfig {
    const fn causal(start_pos: usize, scale: f32) -> Self {
        Self {
            start_pos,
            sliding_window: None,
            scale,
        }
    }

    const fn with_sliding_window(mut self, sliding_window: Option<usize>) -> Self {
        self.sliding_window = sliding_window;
        self
    }
}

use crate::dispatch::{PhaseLayout, project_lm_head_row_split};
use crate::repack_cache::{
    Aligned128F32, LayerRepackCache, LmHeadCache, ModelRepackCache, QuantizedBatchAct,
    decode_row_to_f32, get_or_decode_1d_f32, get_or_repack, quant_block_geometry, resolve_2d_dims,
    set_active_model_anchor,
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
    /// Rotary embedding pair convention (`Norm` for `llama`/`llama4`, `NeoX` for `qwen2`).
    pub rope_type: RopeType,
    /// Optional partial rotary dimension count (`{arch}.rope.dimension_count`, defaults to `head_dim`).
    pub rotary_dim: Option<usize>,
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
    rotary_dim: usize,
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
            rope_type: RopeType::Norm,
            rotary_dim: None,
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
            rope_type: RopeType::Norm,
            rotary_dim: None,
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
            rope_type: RopeType::NeoX,
            rotary_dim: None,
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
        } else if let Some(&first_kv) = meta.head_count_kv_per_layer().iter().find(|&&v| v > 0) {
            recipe.n_heads_kv = first_kv;
        }

        if let Some(kd) = meta.key_length().filter(|&kd| kd >= 2) {
            recipe.head_dim = kd;
            if meta.head_count() == 0
                && let Some(q_desc) = meta.find_tensor("blk.0.attn_q.weight")
                && let Some(q_rows) = infer_descriptor_rows(q_desc, hidden_dim)
                && q_rows % kd == 0
            {
                recipe.n_heads_q = q_rows / kd;
            }
        } else if let Some(q_desc) = meta.find_tensor("blk.0.attn_q.weight")
            && let Some(q_rows) = infer_descriptor_rows(q_desc, hidden_dim)
        {
            if meta.head_count() > 0 && q_rows % recipe.n_heads_q == 0 {
                recipe.head_dim = q_rows / recipe.n_heads_q;
            } else if recipe.head_dim > 0 && q_rows % recipe.head_dim == 0 {
                recipe.n_heads_q = q_rows / recipe.head_dim;
            }
        } else if recipe.n_heads_q > 0 && hidden_dim % recipe.n_heads_q == 0 {
            recipe.head_dim = hidden_dim / recipe.n_heads_q;
        }

        if meta.head_count_kv() == 0
            && let Some(k_desc) = meta.find_tensor("blk.0.attn_k.weight")
            && let Some(k_rows) = infer_descriptor_rows(k_desc, hidden_dim)
            && recipe.head_dim > 0
            && k_rows % recipe.head_dim == 0
        {
            recipe.n_heads_kv = k_rows / recipe.head_dim;
        }

        if let Some(ffn) = meta.feed_forward_length().filter(|&v| v > 0) {
            recipe.ffn_dim = ffn;
        } else if let Some(ffn_desc) = meta
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
        } else if let Some(vocab) = meta.vocab_size().filter(|&v| v > 0) {
            recipe.vocab_size = vocab;
        }

        if let Some(eps) = meta.rms_norm_eps().filter(|e| e.is_finite() && *e > 0.0) {
            recipe.rms_eps = eps;
        }

        if let Some(theta) = meta.rope_freq_base().filter(|t| t.is_finite() && *t > 0.0) {
            recipe.rope_theta = theta;
        }

        if let Some(rd) = meta.rope_dimension_count() {
            recipe.rotary_dim = Some(rd);
        }

        recipe
    }

    /// Hydrates a [`LlamaRecipe`] from a full [`GgufFile`], reading architecture-prefixed KV
    /// entries (`feed_forward_length`, `attention.key_length`, `rope.freq_base`,
    /// `rope.dimension_count`, `attention.layer_norm_rms_epsilon`, `vocab_size`, `rope.scaling.*`)
    /// alongside tensor shapes.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `gguf.architecture()` is not `"llama"`,
    /// `"llama4"`, or `"qwen2"`, or if explicit `rope.scaling.*` metadata is malformed.
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

        let rot_dim_key = format!("{arch}.rope.dimension_count");
        if let Some(rd) = gguf
            .metadata_value(&rot_dim_key)
            .and_then(GgufValue::as_usize)
        {
            recipe.rotary_dim = Some(rd);
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

        recipe.rope_scaling = parse_rope_scaling_from_gguf(gguf, arch)?;

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

    /// Configures the `RoPE` pairing convention (`Norm` or `NeoX`).
    #[must_use]
    pub const fn with_rope_type(mut self, rope_type: RopeType) -> Self {
        self.rope_type = rope_type;
        self
    }

    /// Configures a partial or explicit rotary dimension count (`<= head_dim`).
    #[must_use]
    pub const fn with_rotary_dim(mut self, rotary_dim: usize) -> Self {
        self.rotary_dim = Some(rotary_dim);
        self
    }

    /// Returns the effective rotary dimension (`rotary_dim` if set, otherwise `head_dim`).
    #[must_use]
    pub const fn effective_rotary_dim(&self) -> usize {
        match self.rotary_dim {
            Some(rd) => rd,
            None => self.head_dim,
        }
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

    /// Pre-repacks layer projections and pre-decodes `128-byte`-aligned 1D `RMSNorm` / `qwen2`
    /// bias vectors once into a model-scoped [`ModelRepackCache`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] when required
    /// weights are missing, un-admitted, or shape-mismatched.
    pub fn build_repack_cache(
        &self,
        weights: &dyn ModelWeights,
    ) -> Result<ModelRepackCache, K3Error> {
        if weights.tensor_count() == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "build_repack_cache requires resident model tensors".to_owned(),
            });
        }
        let geom = self.resolve_geometry(weights, 0)?;
        let lm_name = if weights.contains_tensor("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_tensor = weights.tensor(lm_name)?;
        let (vocab_size, _) = resolve_2d_dims(&lm_tensor, geom.hidden_dim)?;
        ModelRepackCache::build(
            self.architecture,
            geom.num_layers,
            geom.hidden_dim,
            geom.ffn_dim,
            geom.total_q,
            geom.total_kv,
            vocab_size,
            weights,
        )
    }

    /// Executes batched prefill using a caller-owned model-scoped [`ModelRepackCache`],
    /// eliminating per-step `decode_1d_f32` allocations and global cache lookups.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] on context overflow, cache geometry mismatch, or tensor errors.
    pub fn prefill_step_with_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &ModelRepackCache,
    ) -> Result<(), K3Error> {
        self.prefill_step_inner(ctx, weights, Some(cache))
    }

    /// Executes single-token autoregressive decode using a caller-owned [`ModelRepackCache`],
    /// eliminating per-step `decode_1d_f32` allocations and global cache lookups.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if no decode token is staged, context overflows, or tensors mismatch.
    pub fn decode_step_with_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &ModelRepackCache,
    ) -> Result<(), K3Error> {
        self.decode_step_inner(ctx, weights, Some(cache))
    }

    fn prefill_step_inner(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: Option<&ModelRepackCache>,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens();
        let prompt_len = prompt_tokens.len();
        let start_pos = ctx.start_position();
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

        let compute_logits = ctx.compute_logits();
        let logits_len = if compute_logits {
            ctx.logits().len()
        } else {
            0
        };
        let kv_geometry = ctx.kv_geometry();
        let last_hidden = self.execute_batch_forward(
            prompt_tokens,
            start_pos,
            ctx.kv_cache_mut(),
            kv_geometry,
            weights,
            logits_len,
            compute_logits,
            cache,
        )?;
        ctx.advance_position(prompt_len)?;
        if compute_logits {
            Self::project_logits(ctx, &last_hidden, weights, None, cache)?;
        }
        Ok(())
    }

    fn decode_step_inner(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: Option<&ModelRepackCache>,
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
        let kv_geometry = ctx.kv_geometry();
        let single = [token];
        let hidden = self.execute_batch_forward(
            &single,
            pos,
            ctx.kv_cache_mut(),
            kv_geometry,
            weights,
            logits_len,
            true,
            cache,
        )?;
        ctx.advance_position(1)?;
        Self::project_logits(ctx, &hidden, weights, Some(token), cache)
    }

    fn resolve_rotary_dim(&self, head_d_even: usize) -> Result<usize, K3Error> {
        let Some(rd) = self.rotary_dim else {
            return Ok(head_d_even);
        };
        if rd < 2 || rd % 2 != 0 || rd > head_d_even {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: format!(
                    "invalid rotary_dim {rd} for head_dim {head_d_even} (must be even and within 2..={head_d_even})"
                ),
            });
        }
        Ok(rd)
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
            let rotary_dim = self.resolve_rotary_dim(head_d_even)?;
            return Ok(EffectiveGeometry {
                num_layers: self.num_layers,
                hidden_dim,
                ffn_dim,
                n_heads_q,
                n_heads_kv,
                head_d_even,
                rotary_dim,
                total_q: n_heads_q * head_d_even,
                total_kv: n_heads_kv * head_d_even,
            });
        }

        let hidden_dim = self.resolve_live_hidden_dim(weights, logits_len)?;
        let num_layers = Self::resolve_live_num_layers(weights)?;
        let (n_heads_q, n_heads_kv, head_d_even) = self.resolve_live_heads(weights, hidden_dim)?;
        let rotary_dim = self.resolve_rotary_dim(head_d_even)?;

        let ffn_dim = if weights.contains_tensor("blk.0.ffn_gate.weight") {
            let t = weights.tensor("blk.0.ffn_gate.weight")?;
            resolve_2d_dims(&t, hidden_dim)?.0
        } else if weights.contains_tensor("blk.0.ffn_up.weight") {
            let t = weights.tensor("blk.0.ffn_up.weight")?;
            resolve_2d_dims(&t, hidden_dim)?.0
        } else if weights.contains_tensor("blk.0.ffn_down.weight") {
            let t = weights.tensor("blk.0.ffn_down.weight")?;
            let shape = t.shape();
            if shape.len() == 2 {
                if shape[0] == hidden_dim {
                    shape[1]
                } else if shape[1] == hidden_dim {
                    shape[0]
                } else {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "blk.0.ffn_down.weight".to_owned(),
                        offset: t.offset(),
                        byte_len: t.byte_len(),
                        buffer_len: t.as_bytes().len(),
                        reason: format!(
                            "ffn_down shape {shape:?} does not match hidden_dim {hidden_dim}"
                        ),
                    });
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "blk.0.ffn_down.weight".to_owned(),
                    offset: t.offset(),
                    byte_len: t.byte_len(),
                    buffer_len: t.as_bytes().len(),
                    reason: format!("expected 2D tensor for blk.0.ffn_down.weight, got {shape:?}"),
                });
            }
        } else {
            return Err(missing_tensor_err("blk.0.ffn_gate.weight"));
        };

        Ok(EffectiveGeometry {
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_d_even,
            rotary_dim,
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
        Err(missing_tensor_err("blk.0.attn_norm.weight"))
    }

    fn resolve_live_num_layers(weights: &dyn ModelWeights) -> Result<usize, K3Error> {
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
        if probed > 0 {
            Ok(probed)
        } else {
            Err(missing_tensor_err("blk.0.attn_norm.weight"))
        }
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
            return Err(missing_tensor_err("blk.0.attn_q.weight"));
        }
        if !weights.contains_tensor("blk.0.attn_k.weight") {
            return Err(missing_tensor_err("blk.0.attn_k.weight"));
        }

        let q_tensor = weights.tensor("blk.0.attn_q.weight")?;
        let q_rows = resolve_2d_dims(&q_tensor, hidden_dim)?.0;
        let k_tensor = weights.tensor("blk.0.attn_k.weight")?;
        let k_rows = resolve_2d_dims(&k_tensor, hidden_dim)?.0;

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
        bind_fallback_anchor: bool,
    ) -> Result<(), K3Error> {
        if weights.contains_tensor("token_embd.weight") {
            let emb_tensor = weights.tensor("token_embd.weight")?;
            if bind_fallback_anchor {
                set_active_model_anchor(&emb_tensor);
            }
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
        } else if weights.tensor_count() == 0 {
            for (t_idx, &tok_id) in tokens.iter().enumerate() {
                #[expect(clippy::cast_precision_loss, reason = "token_id modulo 17 fits in f32")]
                let base_val = ((tok_id % 17) as f32) * 0.1 + 0.5;
                hidden_batch[t_idx * hidden_dim..(t_idx + 1) * hidden_dim].fill(base_val);
            }
        } else {
            return Err(missing_tensor_err("token_embd.weight"));
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "passes explicit cached 1D norm or fallback tensor name alongside batch dimensions"
    )]
    fn apply_rmsnorm_batch(
        &self,
        output: &mut [f32],
        input: &[f32],
        norm_name: &str,
        cached_norm: Option<&Aligned128F32>,
        dim: usize,
        m: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let owned_fallback;
        let w_f32: &[f32] = if let Some(cached) = cached_norm {
            cached.as_slice()
        } else {
            if !weights.contains_tensor(norm_name) {
                if weights.tensor_count() == 0 {
                    output.copy_from_slice(input);
                    return Ok(());
                }
                return Err(missing_tensor_err(norm_name));
            }
            let norm_tensor = weights.tensor(norm_name)?;
            owned_fallback = get_or_decode_1d_f32(&norm_tensor, dim, true)?;
            owned_fallback.as_slice()
        };
        let eps = self.rms_eps.max(1.0e-7);

        for tok in 0..m {
            let in_row = &input[tok * dim..(tok + 1) * dim];
            let out_row = &mut output[tok * dim..(tok + 1) * dim];
            rmsnorm_row(out_row, in_row, w_f32, eps)?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "orchestrates full batched transformer block across QKV, biases, RoPE, KV-cache GQA, and SwiGLU"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "internal batched forward accepts explicit optional model-scoped cache"
    )]
    fn execute_batch_forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_cache: &mut [f32],
        kv_geometry: Option<KvGeometry>,
        weights: &dyn ModelWeights,
        logits_len: usize,
        compute_logits: bool,
        cache: Option<&ModelRepackCache>,
    ) -> Result<Vec<f32>, K3Error> {
        let m = tokens.len();
        let geom = self.resolve_geometry(weights, logits_len)?;
        if let Some(c) = cache
            && (c.num_layers() != geom.num_layers
                || c.hidden_dim() != geom.hidden_dim
                || c.ffn_dim() != geom.ffn_dim
                || c.total_q() != geom.total_q
                || c.total_kv() != geom.total_kv)
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "ModelRepackCache".to_owned(),
                offset: 0,
                byte_len: c.hidden_dim(),
                buffer_len: geom.hidden_dim,
                reason: format!(
                    "ModelRepackCache geometry (layers={}, hidden={}, ffn={}, q={}, kv={}) mismatches model ({}, {}, {}, {}, {})",
                    c.num_layers(),
                    c.hidden_dim(),
                    c.ffn_dim(),
                    c.total_q(),
                    c.total_kv(),
                    geom.num_layers,
                    geom.hidden_dim,
                    geom.ffn_dim,
                    geom.total_q,
                    geom.total_kv,
                ),
            });
        }
        if m == 0 {
            return Ok(if compute_logits {
                vec![0.0_f32; geom.hidden_dim]
            } else {
                Vec::new()
            });
        }

        let synthetic_mode = weights.tensor_count() == 0;
        let end_pos = start_pos + m;
        let mut hidden_batch = vec![0.0_f32; m * geom.hidden_dim];
        Self::embed_tokens(
            tokens,
            &mut hidden_batch,
            geom.hidden_dim,
            weights,
            cache.is_none(),
        )?;

        let inv_freq = precompute_inv_freq_compat(
            geom.rotary_dim,
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
        let mut scores_scratch = vec![0.0_f32; end_pos];
        let mut shared_hidden_act = QuantizedBatchAct::new();
        let mut shared_q_act = QuantizedBatchAct::new();
        let mut shared_ffn_act = QuantizedBatchAct::new();

        #[expect(
            clippy::cast_precision_loss,
            reason = "head_dim fits comfortably in f32"
        )]
        let attn_scale = 1.0_f32 / (geom.head_d_even as f32).sqrt();

        for layer in 0..geom.num_layers {
            let layer_cache = cache.and_then(|c| c.layer(layer));

            // 1. Pre-Attention RMSNorm
            self.apply_rmsnorm_batch(
                &mut norm_batch,
                &hidden_batch,
                &format!("blk.{layer}.attn_norm.weight"),
                layer_cache.map(|lc| &lc.attn_norm),
                geom.hidden_dim,
                m,
                weights,
            )?;

            // 2. Batched Q, K, V projections sharing reusable pre-quantized activation scratch
            project_qkv_layer(
                layer,
                &norm_batch,
                &mut q_batch,
                &mut k_batch,
                &mut v_batch,
                geom,
                m,
                weights,
                layer_cache,
                &mut shared_hidden_act,
            )?;

            // 3. Rotary Position Embeddings on BOTH Q and K
            if !self.should_skip_rope(layer) {
                for tok in 0..m {
                    let pos = start_pos + tok;
                    apply_rope_compat(
                        &mut q_batch[tok * geom.total_q..(tok + 1) * geom.total_q],
                        geom.head_d_even,
                        geom.rotary_dim,
                        self.rope_type,
                        pos,
                        &inv_freq,
                    )?;
                    apply_rope_compat(
                        &mut k_batch[tok * geom.total_kv..(tok + 1) * geom.total_kv],
                        geom.head_d_even,
                        geom.rotary_dim,
                        self.rope_type,
                        pos,
                        &inv_freq,
                    )?;
                }
            }

            // 4. Batched KV Cache Append (`[start_pos..end_pos]`) & Causal GQA Attention
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
                kv_geometry,
                synthetic_mode,
                &mut attn_out_batch,
                &mut scores_scratch,
            )?;

            // 5. Batched Attention Output Projection & Residual Connection
            if let Some(lc) = layer_cache {
                let wo_act = if lc.wo.uses_repacked_storage() {
                    shared_q_act.quantize_into(&attn_out_batch, m, geom.total_q)?;
                    Some(&shared_q_act)
                } else {
                    None
                };
                lc.wo.project_batch_with_act(
                    &mut proj_hidden_batch,
                    &attn_out_batch,
                    wo_act,
                    m,
                    true,
                )?;
                add_optional_bias(
                    &mut proj_hidden_batch,
                    &format!("blk.{layer}.attn_output.bias"),
                    lc.attn_output_bias.as_ref(),
                    true,
                    geom.hidden_dim,
                    m,
                    weights,
                )?;
                for (h, &p) in hidden_batch.iter_mut().zip(&proj_hidden_batch) {
                    *h += p;
                }
            } else {
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
                    let wo_act = if cached.uses_repacked_storage() {
                        shared_q_act.quantize_into(&attn_out_batch, m, geom.total_q)?;
                        Some(&shared_q_act)
                    } else {
                        None
                    };
                    cached.project_batch_with_act(
                        &mut proj_hidden_batch,
                        &attn_out_batch,
                        wo_act,
                        m,
                        true,
                    )?;
                    add_optional_bias(
                        &mut proj_hidden_batch,
                        &format!("blk.{layer}.attn_output.bias"),
                        None,
                        false,
                        geom.hidden_dim,
                        m,
                        weights,
                    )?;
                    for (h, &p) in hidden_batch.iter_mut().zip(&proj_hidden_batch) {
                        *h += p;
                    }
                } else if synthetic_mode {
                    for tok in 0..m {
                        let h_row =
                            &mut hidden_batch[tok * geom.hidden_dim..(tok + 1) * geom.hidden_dim];
                        let a_row = &attn_out_batch[tok * geom.total_q..(tok + 1) * geom.total_q];
                        for (i, h) in h_row.iter_mut().enumerate() {
                            *h += a_row[i % geom.total_q] * 0.05;
                        }
                    }
                } else {
                    return Err(missing_tensor_err(&attn_out_name));
                }
            }

            // 6. Pre-FFN RMSNorm
            self.apply_rmsnorm_batch(
                &mut norm_batch,
                &hidden_batch,
                &format!("blk.{layer}.ffn_norm.weight"),
                layer_cache.map(|lc| &lc.ffn_norm),
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
                layer_cache,
                &mut shared_hidden_act,
                &mut shared_ffn_act,
            )?;
        }

        if !compute_logits {
            return Ok(Vec::new());
        }

        // Final output RMSNorm is only required for the last token row (m - 1)
        let last_in = &hidden_batch[(m - 1) * geom.hidden_dim..m * geom.hidden_dim];
        let mut final_hidden = vec![0.0_f32; geom.hidden_dim];
        self.apply_rmsnorm_batch(
            &mut final_hidden,
            last_in,
            "output_norm.weight",
            cache.map(ModelRepackCache::output_norm),
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
        cache: Option<&ModelRepackCache>,
    ) -> Result<(), K3Error> {
        if let Some(c) = cache {
            return match c.lm_head() {
                LmHeadCache::Repacked(cached) => cached.gemv_rows(ctx.logits_mut(), hidden, true),
                LmHeadCache::StreamingUntied => {
                    let output_weight = weights.tensor("output.weight")?;
                    let layout = PhaseLayout::default();
                    project_lm_head_row_split(ctx.logits_mut(), hidden, &output_weight, &layout)
                }
                LmHeadCache::StreamingTied => {
                    let tied_weight = weights.tensor("token_embd.weight")?;
                    let layout = PhaseLayout::default();
                    project_lm_head_row_split(ctx.logits_mut(), hidden, &tied_weight, &layout)
                }
            };
        }
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
        if weights.tensor_count() > 0 {
            return Err(missing_tensor_err("output.weight"));
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
                | QuantType::Q5_K
                | QuantType::Q5_0
                | QuantType::Q8_0
                | QuantType::Q4_0
                | QuantType::Q6_K
        )
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let arch = metadata.architecture();
        if !matches!(arch, "llama" | "llama4" | "qwen2") {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "LlamaRecipe supports 'llama', 'llama4', and 'qwen2', got '{arch}'"
                ),
            });
        }
        if self.n_heads_kv == 0
            || self.n_heads_q < self.n_heads_kv
            || self.n_heads_q % self.n_heads_kv != 0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "invalid GQA head counts: n_heads_q={} must be divisible by non-zero n_heads_kv={}",
                    self.n_heads_q, self.n_heads_kv
                ),
            });
        }

        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    arch,
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0
                && let Ok(view) = weights.tensor(desc.name())
                && !self.supports_quant(view.quant())
            {
                return Err(K3Error::unsupported_tensor_quant(
                    arch,
                    view.name(),
                    view.quant(),
                ));
            }
        }

        if weights.tensor_count() == 0 {
            let hd = self.head_dim.max(2);
            let hd_even = if hd % 2 == 0 { hd } else { hd - 1 };
            let _ = self.resolve_rotary_dim(hd_even)?;
            return Ok(());
        }

        let geom = self.resolve_geometry(weights, 0)?;
        let check_resident = |name: &str| -> Result<(), K3Error> {
            let view = weights.tensor(name)?;
            if !self.supports_quant(view.quant()) {
                return Err(K3Error::unsupported_tensor_quant(arch, name, view.quant()));
            }
            Ok(())
        };

        check_resident("token_embd.weight")?;
        check_resident("output_norm.weight")?;
        if weights.contains_tensor("output.weight") {
            check_resident("output.weight")?;
        }

        for layer in 0..geom.num_layers {
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                check_resident(&format!("blk.{layer}.{suffix}"))?;
            }
            for opt_bias in [
                "attn_q.bias",
                "attn_k.bias",
                "attn_v.bias",
                "attn_output.bias",
            ] {
                let name = format!("blk.{layer}.{opt_bias}");
                if weights.contains_tensor(&name) {
                    check_resident(&name)?;
                }
            }
        }

        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.prefill_step_inner(ctx, weights, None)
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.decode_step_inner(ctx, weights, None)
    }
}

fn load_optional_proj_matrix(
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
    label: &str,
    weights: &dyn ModelWeights,
) -> Result<Option<std::sync::Arc<crate::repack_cache::CachedMatrix>>, K3Error> {
    if !weights.contains_tensor(name) {
        if weights.tensor_count() == 0 {
            return Ok(None);
        }
        return Err(missing_tensor_err(name));
    }
    let t = weights.tensor(name)?;
    let (rows, cols) = resolve_2d_dims(&t, expected_cols)?;
    if rows != expected_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: t.offset(),
            byte_len: t.byte_len(),
            buffer_len: t.as_bytes().len(),
            reason: format!("{label} rows {rows} != expected {expected_rows}"),
        });
    }
    Ok(Some(get_or_repack(&t, rows, cols)?))
}

#[expect(
    clippy::too_many_arguments,
    reason = "projects Q, K, and V batches with reusable pre-quantized activation scratch"
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
    layer_cache: Option<&LayerRepackCache>,
    shared_act_buf: &mut QuantizedBatchAct,
) -> Result<(), K3Error> {
    let (q_mat, k_mat, v_mat) = if let Some(lc) = layer_cache {
        (
            Some(std::sync::Arc::clone(&lc.wq)),
            Some(std::sync::Arc::clone(&lc.wk)),
            Some(std::sync::Arc::clone(&lc.wv)),
        )
    } else {
        (
            load_optional_proj_matrix(
                &format!("blk.{layer}.attn_q.weight"),
                geom.total_q,
                geom.hidden_dim,
                "attn_q",
                weights,
            )?,
            load_optional_proj_matrix(
                &format!("blk.{layer}.attn_k.weight"),
                geom.total_kv,
                geom.hidden_dim,
                "attn_k",
                weights,
            )?,
            load_optional_proj_matrix(
                &format!("blk.{layer}.attn_v.weight"),
                geom.total_kv,
                geom.hidden_dim,
                "attn_v",
                weights,
            )?,
        )
    };

    let needs_repacked_act = q_mat
        .as_ref()
        .is_some_and(|mat| mat.uses_repacked_storage())
        || k_mat
            .as_ref()
            .is_some_and(|mat| mat.uses_repacked_storage())
        || v_mat
            .as_ref()
            .is_some_and(|mat| mat.uses_repacked_storage());
    let shared_act = if needs_repacked_act {
        shared_act_buf.quantize_into(norm_batch, m, geom.hidden_dim)?;
        Some(&*shared_act_buf)
    } else {
        None
    };

    if let Some(mat) = q_mat {
        mat.project_batch_with_act(q_batch, norm_batch, shared_act, m, true)?;
    } else {
        fill_synthetic_proj(q_batch, norm_batch, geom.total_q, geom.hidden_dim, m, 1.0);
    }
    add_optional_bias(
        q_batch,
        &format!("blk.{layer}.attn_q.bias"),
        layer_cache.and_then(|lc| lc.attn_q_bias.as_ref()),
        layer_cache.is_some(),
        geom.total_q,
        m,
        weights,
    )?;

    if let Some(mat) = k_mat {
        mat.project_batch_with_act(k_batch, norm_batch, shared_act, m, true)?;
    } else {
        fill_synthetic_proj(k_batch, norm_batch, geom.total_kv, geom.hidden_dim, m, 1.0);
    }
    add_optional_bias(
        k_batch,
        &format!("blk.{layer}.attn_k.bias"),
        layer_cache.and_then(|lc| lc.attn_k_bias.as_ref()),
        layer_cache.is_some(),
        geom.total_kv,
        m,
        weights,
    )?;

    if let Some(mat) = v_mat {
        mat.project_batch_with_act(v_batch, norm_batch, shared_act, m, true)?;
    } else {
        fill_synthetic_proj(v_batch, norm_batch, geom.total_kv, geom.hidden_dim, m, 1.0);
    }
    add_optional_bias(
        v_batch,
        &format!("blk.{layer}.attn_v.bias"),
        layer_cache.and_then(|lc| lc.attn_v_bias.as_ref()),
        layer_cache.is_some(),
        geom.total_kv,
        m,
        weights,
    )?;

    Ok(())
}

fn resolve_kv_geometry(
    geom: EffectiveGeometry,
    end_pos: usize,
    kv_cache_len: usize,
    kv_cache_bytes: usize,
    kv_geometry: Option<KvGeometry>,
    synthetic_mode: bool,
) -> Result<Option<KvGeometry>, K3Error> {
    if let Some(ctx_geom) = kv_geometry {
        if ctx_geom.num_layers != geom.num_layers
            || ctx_geom.n_heads_kv != geom.n_heads_kv
            || ctx_geom.head_dim != geom.head_d_even
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "kv_cache".to_owned(),
                reason: format!(
                    "ExecutionContext KvGeometry ({}, {}, {}) does not match model geometry ({}, {}, {})",
                    ctx_geom.num_layers,
                    ctx_geom.n_heads_kv,
                    ctx_geom.head_dim,
                    geom.num_layers,
                    geom.n_heads_kv,
                    geom.head_d_even
                ),
            });
        }
        return Ok(Some(ctx_geom));
    }

    let per_tok_all_layers = 2_usize
        .checked_mul(geom.num_layers)
        .and_then(|v| v.checked_mul(geom.total_kv))
        .unwrap_or(0);
    let required = end_pos.saturating_mul(per_tok_all_layers);
    if per_tok_all_layers > 0 && kv_cache_len >= required {
        let seq_cap = kv_cache_len / per_tok_all_layers;
        Ok(Some(KvGeometry::new(
            geom.num_layers,
            geom.n_heads_kv,
            geom.head_d_even,
            seq_cap,
            None,
        )?))
    } else if !synthetic_mode {
        Err(K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: required.saturating_mul(size_of::<f32>()),
            buffer_len: kv_cache_bytes,
            reason: format!(
                "kv_cache length {kv_cache_len} f32 elements is smaller than required {required} for end_pos={end_pos}"
            ),
        })
    } else {
        Ok(None)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "manages layer KV cache append and batched causal GQA execution with caller scratch"
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
    kv_geometry: Option<KvGeometry>,
    synthetic_mode: bool,
    attn_out_batch: &mut [f32],
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    let end_pos = start_pos + m;
    let resolved_geom = resolve_kv_geometry(
        geom,
        end_pos,
        kv_cache.len(),
        size_of_val(kv_cache),
        kv_geometry,
        synthetic_mode,
    )?;

    if let Some(view_geom) = resolved_geom {
        let mut kv_view = KvCacheView::new(kv_cache, view_geom)?;
        if end_pos > view_geom.seq_capacity {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: view_geom.seq_capacity,
            });
        }
        let dst_start = start_pos * geom.total_kv;
        let dst_end = end_pos * geom.total_kv;
        let batch_kv_elems = m * geom.total_kv;
        {
            let (k_plane, v_plane) = kv_view.layer_kv_mut(layer)?;
            k_plane[dst_start..dst_end].copy_from_slice(&k_batch[..batch_kv_elems]);
            v_plane[dst_start..dst_end].copy_from_slice(&v_batch[..batch_kv_elems]);
        }

        let (k_plane, v_plane) = kv_view.layer_kv(layer)?;
        let active_kv_elems = end_pos * geom.total_kv;
        let mask_cfg = AttentionMaskConfig::causal(start_pos, attn_scale)
            .with_sliding_window(view_geom.sliding_window);
        return gqa_attention_prefill_compat(
            attn_out_batch,
            q_batch,
            &k_plane[..active_kv_elems],
            &v_plane[..active_kv_elems],
            m,
            geom.n_heads_q,
            geom.n_heads_kv,
            geom.head_d_even,
            end_pos,
            mask_cfg,
            scores_scratch,
        );
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

    gqa_attention_prefill_compat(
        attn_out_batch,
        q_batch,
        k_batch,
        v_batch,
        m,
        geom.n_heads_q,
        geom.n_heads_kv,
        geom.head_d_even,
        m,
        AttentionMaskConfig::causal(0, attn_scale),
        scores_scratch,
    )
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
    layer_cache: Option<&LayerRepackCache>,
    shared_hidden_act: &mut QuantizedBatchAct,
    shared_ffn_act: &mut QuantizedBatchAct,
) -> Result<(), K3Error> {
    let (gate_mat, up_mat, down_mat) = if let Some(lc) = layer_cache {
        (
            Some(std::sync::Arc::clone(&lc.wgate)),
            Some(std::sync::Arc::clone(&lc.wup)),
            Some(std::sync::Arc::clone(&lc.wdown)),
        )
    } else {
        (
            load_optional_proj_matrix(
                &format!("blk.{layer}.ffn_gate.weight"),
                geom.ffn_dim,
                geom.hidden_dim,
                "ffn_gate",
                weights,
            )?,
            load_optional_proj_matrix(
                &format!("blk.{layer}.ffn_up.weight"),
                geom.ffn_dim,
                geom.hidden_dim,
                "ffn_up",
                weights,
            )?,
            load_optional_proj_matrix(
                &format!("blk.{layer}.ffn_down.weight"),
                geom.hidden_dim,
                geom.ffn_dim,
                "ffn_down",
                weights,
            )?,
        )
    };

    let needs_hidden_act = gate_mat
        .as_ref()
        .is_some_and(|mat| mat.uses_repacked_storage())
        || up_mat
            .as_ref()
            .is_some_and(|mat| mat.uses_repacked_storage());
    let hidden_act = if needs_hidden_act {
        shared_hidden_act.quantize_into(norm_batch, m, geom.hidden_dim)?;
        Some(&*shared_hidden_act)
    } else {
        None
    };

    if let Some(mat) = gate_mat {
        mat.project_batch_with_act(gate_batch, norm_batch, hidden_act, m, true)?;
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
        mat.project_batch_with_act(up_batch, norm_batch, hidden_act, m, true)?;
    } else {
        fill_synthetic_proj(up_batch, norm_batch, geom.ffn_dim, geom.hidden_dim, m, 0.5);
    }

    swiglu_compat(ffn_act_batch, gate_batch, up_batch)?;

    if let Some(cached) = down_mat {
        let down_act = if cached.uses_repacked_storage() {
            shared_ffn_act.quantize_into(ffn_act_batch, m, geom.ffn_dim)?;
            Some(&*shared_ffn_act)
        } else {
            None
        };
        cached.project_batch_with_act(proj_hidden_batch, ffn_act_batch, down_act, m, true)?;
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
    cached_bias: Option<&Aligned128F32>,
    has_layer_cache: bool,
    dim: usize,
    m: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let owned_fallback;
    let bias: &[f32] = if let Some(cached) = cached_bias {
        cached.as_slice()
    } else if has_layer_cache || !weights.contains_tensor(bias_name) {
        return Ok(());
    } else {
        let bias_tensor = weights.tensor(bias_name)?;
        owned_fallback = get_or_decode_1d_f32(&bias_tensor, dim, false)?;
        owned_fallback.as_slice()
    };
    for tok in 0..m {
        let row = &mut batch[tok * dim..(tok + 1) * dim];
        for (slot, &b) in row.iter_mut().zip(bias) {
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

pub(crate) fn precompute_inv_freq_compat(
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

        match scaling {
            None => {}
            Some(RopeScalingConfig::Linear { factor }) => {
                if !factor.is_finite() || *factor <= 0.0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: "invalid linear RoPE scaling factor".to_owned(),
                    });
                }
                for freq in &mut out {
                    *freq /= *factor;
                }
            }
            Some(RopeScalingConfig::Llama3Smooth {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => {
                if !factor.is_finite()
                    || *factor <= 0.0
                    || !low_freq_factor.is_finite()
                    || *low_freq_factor <= 0.0
                    || !high_freq_factor.is_finite()
                    || *high_freq_factor <= *low_freq_factor
                    || *original_max_position_embeddings == 0
                {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: "invalid Llama-3 smooth RoPE scaling parameters".to_owned(),
                    });
                }
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "context length fits accurately enough in f32 for wavelength band"
                )]
                let orig_ctx = *original_max_position_embeddings as f32;
                let low_freq_wavelen = orig_ctx / *low_freq_factor;
                let high_freq_wavelen = orig_ctx / *high_freq_factor;
                for freq in &mut out {
                    let wavelen = core::f32::consts::TAU / *freq;
                    if wavelen > low_freq_wavelen {
                        *freq /= *factor;
                    } else if wavelen >= high_freq_wavelen {
                        let smooth = (orig_ctx / wavelen - *low_freq_factor)
                            / (*high_freq_factor - *low_freq_factor);
                        *freq = (1.0_f32 - smooth) * (*freq / *factor) + smooth * (*freq);
                    }
                }
            }
            Some(RopeScalingConfig::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
            }) => {
                let factors = if seq_len > *original_max_position_embeddings {
                    long_factor
                } else {
                    short_factor
                };
                if factors.len() != half || factors.iter().any(|f| !f.is_finite() || *f <= 0.0) {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: "invalid LongRoPE per-dimension scaling factor table".to_owned(),
                    });
                }
                for (freq, &factor) in out.iter_mut().zip(factors) {
                    *freq /= factor;
                }
            }
        }
        Ok(out)
    }
}

pub(crate) fn apply_rope_compat(
    qk: &mut [f32],
    head_dim: usize,
    rotary_dim: usize,
    rope_type: RopeType,
    pos: usize,
    inv_freq: &[f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        apply_rope_with_inv_freq_and_layout(qk, head_dim, rotary_dim, rope_type, pos, inv_freq)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if head_dim < 2
            || head_dim % 2 != 0
            || rotary_dim < 2
            || rotary_dim % 2 != 0
            || rotary_dim > head_dim
            || qk.len() % head_dim != 0
            || inv_freq.len() != rotary_dim / 2
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rope".to_owned(),
                reason: "invalid RoPE slice bounds".to_owned(),
            });
        }
        if pos == 0 {
            return Ok(());
        }
        let half = rotary_dim / 2;
        #[expect(clippy::cast_precision_loss, reason = "pos fits in f32")]
        let pos_f32 = pos as f32;
        for head in qk.chunks_exact_mut(head_dim) {
            match rope_type {
                RopeType::Norm => {
                    for i in 0..half {
                        let angle = pos_f32 * inv_freq[i];
                        let (s, c) = angle.sin_cos();
                        let x0 = head[2 * i];
                        let x1 = head[2 * i + 1];
                        head[2 * i] = x0 * c - x1 * s;
                        head[2 * i + 1] = x0 * s + x1 * c;
                    }
                }
                RopeType::NeoX => {
                    for i in 0..half {
                        let angle = pos_f32 * inv_freq[i];
                        let (s, c) = angle.sin_cos();
                        let x0 = head[i];
                        let x1 = head[i + half];
                        head[i] = x0 * c - x1 * s;
                        head[i + half] = x0 * s + x1 * c;
                    }
                }
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_prefill_f32 signature"
)]
fn gqa_attention_prefill_compat(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    config: AttentionMaskConfig,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_prefill_f32(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            config,
            scores_scratch,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        gqa_attention_prefill_portable(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            config,
            scores_scratch,
        )
    }
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "portable GQA prefill mirrors gqa_causal_attention_prefill_f32 signature"
)]
fn validate_gqa_prefill_portable(
    out_len: usize,
    q_len: usize,
    k_len: usize,
    v_len: usize,
    scratch_len: usize,
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    config: AttentionMaskConfig,
) -> Result<(), K3Error> {
    let query_end = config.start_pos.saturating_add(m);
    let min_scratch = config
        .sliding_window
        .map_or(query_end, |w| w.min(query_end));
    let kv_stride = n_heads_kv.saturating_mul(head_dim);
    let expected_q = m.saturating_mul(n_heads_q).saturating_mul(head_dim);
    let min_kv = total_seq_len.saturating_mul(kv_stride);
    if m == 0
        || n_heads_kv == 0
        || n_heads_q == 0
        || head_dim == 0
        || total_seq_len == 0
        || n_heads_q % n_heads_kv != 0
        || !config.scale.is_finite()
        || config.sliding_window == Some(0)
        || query_end > total_seq_len
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: "attention".to_owned(),
            reason: "invalid GQA prefill dimensions or mask config".to_owned(),
        });
    }
    if out_len != expected_q
        || q_len != expected_q
        || k_len < min_kv
        || v_len < min_kv
        || scratch_len < min_scratch
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_prefill".to_owned(),
            offset: 0,
            byte_len: out_len,
            buffer_len: expected_q,
            reason: "GQA prefill slice or scratch length mismatch".to_owned(),
        });
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "portable GQA prefill mirrors gqa_causal_attention_prefill_f32 signature"
)]
fn gqa_attention_prefill_portable(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    config: AttentionMaskConfig,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    validate_gqa_prefill_portable(
        out.len(),
        q.len(),
        k_cache.len(),
        v_cache.len(),
        scores_scratch.len(),
        m,
        n_heads_q,
        n_heads_kv,
        head_dim,
        total_seq_len,
        config,
    )?;

    let kv_stride = n_heads_kv * head_dim;
    let group = n_heads_q / n_heads_kv;
    for hkv in 0..n_heads_kv {
        let kv_head_off = hkv * head_dim;
        for tok in 0..m {
            let query_pos = config.start_pos + tok;
            let valid_end = (query_pos + 1).min(total_seq_len);
            let valid_start = config
                .sliding_window
                .map_or(0, |w| query_pos.saturating_add(1).saturating_sub(w));
            let active_len = valid_end.saturating_sub(valid_start);
            let score_buf = &mut scores_scratch[..active_len];

            for g in 0..group {
                let hq = hkv * group + g;
                let q_head = &q[(tok * n_heads_q + hq) * head_dim..][..head_dim];
                let mut max_score = f32::NEG_INFINITY;
                for (slot, s) in score_buf.iter_mut().zip(valid_start..valid_end) {
                    let k_head = &k_cache[s * kv_stride + kv_head_off..][..head_dim];
                    let score =
                        q_head.iter().zip(k_head).map(|(&a, &b)| a * b).sum::<f32>() * config.scale;
                    *slot = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut denom = 0.0_f32;
                for s in score_buf.iter_mut() {
                    *s = (*s - max_score).exp();
                    denom += *s;
                }
                let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let out_head = &mut out[(tok * n_heads_q + hq) * head_dim..][..head_dim];
                out_head.fill(0.0);
                for (s, &w_unnorm) in (valid_start..valid_end).zip(score_buf.iter()) {
                    let w = w_unnorm * inv_denom;
                    let v_head = &v_cache[s * kv_stride + kv_head_off..][..head_dim];
                    for (o, &v_val) in out_head.iter_mut().zip(v_head) {
                        *o += w * v_val;
                    }
                }
            }
        }
    }
    Ok(())
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

fn missing_tensor_err(tensor_name: &str) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: 0,
        reason: format!("missing required tensor '{tensor_name}'"),
    }
}

fn read_positive_f32_meta(gguf: &GgufFile<'_>, key: &str) -> Result<Option<f32>, K3Error> {
    let Some(val) = gguf.metadata_value(key) else {
        return Ok(None);
    };
    let Some(num) = val.as_f32() else {
        return Err(K3Error::CapabilityMismatch {
            architecture: gguf.architecture().to_owned(),
            reason: format!("expected numeric GGUF metadata value for '{key}'"),
        });
    };
    if !num.is_finite() || num <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: gguf.architecture().to_owned(),
            reason: format!("expected finite positive f32 for '{key}', got {num}"),
        });
    }
    Ok(Some(num))
}

fn read_orig_ctx_len_meta(gguf: &GgufFile<'_>, arch: &str) -> Result<Option<usize>, K3Error> {
    for suffix in [
        "rope.scaling.original_context_length",
        "rope.scaling.orig_ctx_len",
        "rope.scaling.original_max_position_embeddings",
    ] {
        let key = format!("{arch}.{suffix}");
        if let Some(val) = gguf.metadata_value(&key) {
            let Some(len) = val.as_usize() else {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!("expected integer GGUF metadata value for '{key}'"),
                });
            };
            if len == 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!("expected non-zero context length for '{key}'"),
                });
            }
            return Ok(Some(len));
        }
    }
    Ok(None)
}

fn read_f32_array_meta(gguf: &GgufFile<'_>, key: &str) -> Result<Option<Vec<f32>>, K3Error> {
    let Some(val) = gguf.metadata_value(key) else {
        return Ok(None);
    };
    let GgufValue::Array(arr) = val else {
        return Err(K3Error::CapabilityMismatch {
            architecture: gguf.architecture().to_owned(),
            reason: format!("expected array for GGUF metadata key '{key}'"),
        });
    };
    let out: Vec<f32> = match arr {
        GgufArray::Float32(v) => v.clone(),
        GgufArray::Float64(v) =>
        {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "RoPE scaling factors are small positive constants representable in f32"
            )]
            v.iter().map(|&x| x as f32).collect()
        }
        GgufArray::Int32(v) =>
        {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small integer RoPE factors convert to f32"
            )]
            v.iter().map(|&x| x as f32).collect()
        }
        GgufArray::Uint32(v) =>
        {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small integer RoPE factors convert to f32"
            )]
            v.iter().map(|&x| x as f32).collect()
        }
        _ => {
            return Err(K3Error::CapabilityMismatch {
                architecture: gguf.architecture().to_owned(),
                reason: format!("expected numeric array for GGUF metadata key '{key}'"),
            });
        }
    };
    if out.is_empty() || out.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(K3Error::CapabilityMismatch {
            architecture: gguf.architecture().to_owned(),
            reason: format!("GGUF key '{key}' contains empty or invalid non-positive RoPE factors"),
        });
    }
    Ok(Some(out))
}

fn build_long_rope_from_tables(
    arch: &str,
    short: Option<Vec<f32>>,
    long: Option<Vec<f32>>,
    factors: Option<Vec<f32>>,
    orig_ctx: usize,
) -> Result<RopeScalingConfig, K3Error> {
    match (short, long, factors) {
        (Some(short_factor), Some(long_factor), _) => {
            if short_factor.len() != long_factor.len() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!(
                        "LongRoPE short_factor len {} != long_factor len {}",
                        short_factor.len(),
                        long_factor.len()
                    ),
                });
            }
            Ok(RopeScalingConfig::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings: orig_ctx,
            })
        }
        (None, Some(long_factor), _) => Ok(RopeScalingConfig::LongRope {
            short_factor: vec![1.0_f32; long_factor.len()],
            long_factor,
            original_max_position_embeddings: orig_ctx,
        }),
        (Some(short_factor), None, None) => Ok(RopeScalingConfig::LongRope {
            long_factor: short_factor.clone(),
            short_factor,
            original_max_position_embeddings: orig_ctx,
        }),
        (_, _, Some(factors)) => Ok(RopeScalingConfig::LongRope {
            short_factor: factors.clone(),
            long_factor: factors,
            original_max_position_embeddings: orig_ctx,
        }),
        (None, None, None) => Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: "missing LongRoPE scaling factor tables in GGUF metadata".to_owned(),
        }),
    }
}

fn parse_rope_scaling_from_gguf(
    gguf: &GgufFile<'_>,
    arch: &str,
) -> Result<Option<RopeScalingConfig>, K3Error> {
    let type_key = format!("{arch}.rope.scaling.type");
    let scaling_type = match gguf.metadata_value(&type_key) {
        Some(val) => Some(val.as_str().ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("expected string value for GGUF metadata key '{type_key}'"),
        })?),
        None => None,
    };

    let factor = read_positive_f32_meta(gguf, &format!("{arch}.rope.scaling.factor"))?;
    let low_freq = read_positive_f32_meta(gguf, &format!("{arch}.rope.scaling.low_freq_factor"))?;
    let high_freq = read_positive_f32_meta(gguf, &format!("{arch}.rope.scaling.high_freq_factor"))?;
    let orig_ctx_opt = read_orig_ctx_len_meta(gguf, arch)?;
    let short_tbl = read_f32_array_meta(gguf, &format!("{arch}.rope.scaling.short_factor"))?;
    let long_tbl = read_f32_array_meta(gguf, &format!("{arch}.rope.scaling.long_factor"))?;
    let factors_tbl = read_f32_array_meta(gguf, &format!("{arch}.rope.scaling.factors"))?;

    match scaling_type {
        Some("none" | "") => Ok(None),
        Some("linear") => {
            let factor = factor.ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("missing '{arch}.rope.scaling.factor' for linear RoPE scaling"),
            })?;
            Ok(Some(RopeScalingConfig::Linear { factor }))
        }
        Some("llama3") => {
            let factor = factor.ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("missing '{arch}.rope.scaling.factor' for llama3 RoPE scaling"),
            })?;
            let low_freq_factor = low_freq.unwrap_or(1.0);
            let high_freq_factor = high_freq.unwrap_or(4.0);
            if high_freq_factor <= low_freq_factor {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!(
                        "invalid llama3 RoPE frequency bounds: low={low_freq_factor}, high={high_freq_factor}"
                    ),
                });
            }
            Ok(Some(RopeScalingConfig::Llama3Smooth {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings: orig_ctx_opt.unwrap_or(8192),
            }))
        }
        Some("longrope" | "yarn" | "su") => Ok(Some(build_long_rope_from_tables(
            arch,
            short_tbl,
            long_tbl,
            factors_tbl,
            orig_ctx_opt.unwrap_or(8192),
        )?)),
        Some(other) => Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("unsupported RoPE scaling type '{other}'"),
        }),
        None => {
            if short_tbl.is_some() || long_tbl.is_some() || factors_tbl.is_some() {
                return Ok(Some(build_long_rope_from_tables(
                    arch,
                    short_tbl,
                    long_tbl,
                    factors_tbl,
                    orig_ctx_opt.unwrap_or(8192),
                )?));
            }
            if let Some(factor) = factor {
                if low_freq.is_some() || high_freq.is_some() || orig_ctx_opt.is_some() {
                    let low_freq_factor = low_freq.unwrap_or(1.0);
                    let high_freq_factor = high_freq.unwrap_or(4.0);
                    if high_freq_factor <= low_freq_factor {
                        return Err(K3Error::CapabilityMismatch {
                            architecture: arch.to_owned(),
                            reason: format!(
                                "invalid llama3 RoPE frequency bounds: low={low_freq_factor}, high={high_freq_factor}"
                            ),
                        });
                    }
                    return Ok(Some(RopeScalingConfig::Llama3Smooth {
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                        original_max_position_embeddings: orig_ctx_opt.unwrap_or(8192),
                    }));
                }
                return Ok(Some(RopeScalingConfig::Linear { factor }));
            }
            Ok(None)
        }
    }
}
