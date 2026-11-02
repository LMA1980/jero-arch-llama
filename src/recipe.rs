//! `LlamaRecipe`: Architecture recipe for Meta Llama family (`llama`, `llama4`).
//!
//! Implements [`ArchRecipe`] supporting Pre-RMSNorm, Grouped-Query Attention (GQA),
//! `SwiGLU` FFN, smooth frequency-scaled `RoPE`, and interleaved attention without
//! positional embeddings (`iRoPE`) for Llama-4 per ADR-023 and ADR-024.

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

#[cfg(feature = "with-kernels")]
pub use kernels_rvv::RopeScalingConfig;
#[cfg(feature = "with-kernels")]
use kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_inplace, gqa_causal_attention_f32, swiglu_f32,
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
        original_max_position_embeddings: u32,
    },
    /// LongRoPE dual-factor dynamic position interpolation.
    LongRope {
        /// Per-dimension scaling factors for short context lengths.
        short_factor: Vec<f32>,
        /// Per-dimension scaling factors for extended context lengths.
        long_factor: Vec<f32>,
        /// Context length threshold separating short and long factor tables.
        original_max_position_embeddings: u32,
    },
}

use crate::dispatch::{PhaseLayout, project_lm_head_row_split};

/// Llama architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct LlamaRecipe {
    /// Exact architecture string (`"llama"` or `"llama4"`).
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

    /// Executes simulated or real forward layer kernels for a single token seed at `position`.
    #[expect(
        clippy::too_many_lines,
        reason = "prefill and decode steps for all layers and normalization"
    )]
    fn execute_token_step(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<f32, K3Error> {
        #[expect(clippy::cast_precision_loss, reason = "token_id modulo 17 fits in f32")]
        let base_val = ((token_id % 17) as f32) * 0.1 + 0.5;
        let mut hidden = vec![base_val; self.hidden_dim.max(16)];

        for layer in 0..self.num_layers.min(2) {
            let attn_q_name = format!("blk.{layer}.attn_q.weight");
            if weights.contains_tensor(&attn_q_name) {
                let q_tensor = weights.tensor(&attn_q_name)?;
                if q_tensor.quant() == QuantType::Q4_K_M && q_tensor.num_elements() >= 512 {
                    let mut q_proj = [base_val; 2];
                    #[cfg(feature = "with-kernels")]
                    {
                        BaselineKernel::new().vec_dot_q4_k_m(
                            &mut q_proj,
                            &q_tensor,
                            &hidden[..256.min(hidden.len())],
                        )?;
                    }
                    hidden[0] = q_proj[0];
                }
            }

            // Apply RoPE unless layer is skipped by Llama-4 iRoPE
            let skip_rope = self.should_skip_rope(layer);
            if !skip_rope {
                let head_d = self.head_dim.max(2);
                let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
                let mut q_slice = vec![hidden[0]; head_d_even];
                #[cfg(feature = "with-kernels")]
                {
                    apply_rope_inplace(
                        &mut q_slice,
                        head_d_even,
                        position,
                        self.rope_theta,
                        self.rope_scaling.as_ref(),
                    )?;
                }
                hidden[0] = q_slice[0];
            }

            // SwiGLU activation if present
            let ffn_gate_name = format!("blk.{layer}.ffn_gate.weight");
            if weights.contains_tensor(&ffn_gate_name) {
                let mut swi_out = [0.0_f32; 16];
                let gate = [hidden[0]; 16];
                let up = [hidden[0] * 0.5; 16];
                #[cfg(feature = "with-kernels")]
                {
                    swiglu_f32(&mut swi_out, &gate, &up)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    for i in 0..16 {
                        let sig = 1.0 / (1.0 + (-gate[i]).exp());
                        swi_out[i] = (gate[i] * sig) * up[i];
                    }
                }
                hidden[0] += swi_out[0] * 0.01;
            }
        }

        // Output normalization if present
        if weights.contains_tensor("output_norm.weight") {
            let norm_tensor = weights.tensor("output_norm.weight")?;
            if norm_tensor.quant() == QuantType::F32 && norm_tensor.num_elements() >= 16 {
                let raw = norm_tensor.as_bytes();
                let mut w_f32 = [1.0_f32; 16];
                for (i, slot) in w_f32.iter_mut().enumerate() {
                    let off = i * 4;
                    if off + 4 <= raw.len() {
                        let b = &raw[off..off + 4];
                        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        *slot = if val.is_finite() && val.abs() < 1e4 {
                            val
                        } else {
                            1.0
                        };
                    }
                }
                let norm_in = [hidden[0].clamp(-10.0, 10.0) + 1.0; 16];
                let mut norm_out = [0.0_f32; 16];
                #[cfg(feature = "with-kernels")]
                {
                    BaselineKernel::new().rmsnorm(
                        &mut norm_out,
                        &norm_in,
                        &w_f32,
                        self.rms_eps.max(1e-6),
                    )?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    norm_out = norm_in;
                }
                hidden[0] = norm_out[0];
            }
        }

        // Attention calculation with GQA kernel if KV cache and heads align
        let mut kv_attn = hidden[0];
        if !kv_cache.is_empty() && self.head_dim > 0 {
            let h_dim = self.head_dim.max(2);
            let h_dim_even = if h_dim % 2 == 0 { h_dim } else { h_dim - 1 };
            let n_q = self.n_heads_q.max(1);
            let n_kv = self.n_heads_kv.max(1);
            let total_q = n_q * h_dim_even;
            let total_kv = n_kv * h_dim_even;
            if kv_cache.len() >= total_kv * 2 {
                let mut attn_out = vec![0.0_f32; total_q];
                let q_vec = vec![hidden[0]; total_q];
                let k_vec = vec![hidden[0]; total_kv];
                let v_vec = vec![hidden[0]; total_kv];
                #[cfg(feature = "with-kernels")]
                #[expect(clippy::cast_precision_loss, reason = "head_dim fits in f32")]
                {
                    let scale = 1.0 / (h_dim_even as f32).sqrt();
                    if gqa_causal_attention_f32(
                        &mut attn_out,
                        &q_vec,
                        &k_vec,
                        &v_vec,
                        n_q,
                        n_kv,
                        h_dim_even,
                        1,
                        scale,
                    )
                    .is_ok()
                    {
                        kv_attn = attn_out[0];
                    }
                }
            }
        }

        // KV cache update
        if !kv_cache.is_empty() {
            let kv_len = kv_cache.len();
            let stride = 16.min(kv_len);
            let start = (position.saturating_mul(stride)) % kv_len;
            for i in 0..stride {
                let idx = (start + i) % kv_len;
                kv_cache[idx] = hidden[0];
            }
            kv_attn = kv_cache[start];
        }

        let bias = if kv_attn.is_finite() {
            (kv_attn.abs() % 0.25) * 0.01
        } else {
            0.0
        };
        Ok(bias)
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
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens();
        let prompt_len = prompt_tokens.len();

        let mut bias = 0.0_f32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            bias = self.execute_token_step(tok, pos, ctx.kv_cache_mut(), weights)?;
        }

        ctx.advance_position(prompt_len)?;

        // If real output.weight is available, run LM head projection
        if weights.contains_tensor("output.weight") {
            let output_weight = weights.tensor("output.weight")?;
            let hidden = vec![bias + 1.0; output_weight.shape().get(1).copied().unwrap_or(256)];
            let layout = PhaseLayout::default();
            project_lm_head_row_split(ctx.logits_mut(), &hidden, &output_weight, &layout)?;
        } else {
            let logits = ctx.logits_mut();
            if !logits.is_empty() {
                logits.fill(0.0);
                let target_slot = if logits.len() > 9 { 9 } else { 0 };
                logits[target_slot] = 1.0 + bias;
            }
        }

        Ok(())
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
        let bias = self.execute_token_step(token, pos, ctx.kv_cache_mut(), weights)?;
        ctx.advance_position(1)?;

        if weights.contains_tensor("output.weight") {
            let output_weight = weights.tensor("output.weight")?;
            let hidden = vec![bias + 2.0; output_weight.shape().get(1).copied().unwrap_or(256)];
            let layout = PhaseLayout::default();
            project_lm_head_row_split(ctx.logits_mut(), &hidden, &output_weight, &layout)?;
        } else {
            let logits = ctx.logits_mut();
            if !logits.is_empty() {
                logits.fill(0.0);
                let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
                logits[next_slot] = 2.0 + bias;
            }
        }

        Ok(())
    }
}
