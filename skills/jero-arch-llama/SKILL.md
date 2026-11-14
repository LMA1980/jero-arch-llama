---
name: jero-arch-llama
description: Llama (`llama`, `llama4` iRoPE) and Qwen2 (`qwen2`) architecture recipe crate (`jero-arch-llama`) with `ModelRepackCache` delegation to `jero-kernels-rvv::RepackedWeight`, `128-byte`-aligned 1D norm/bias caching, and compact large-vocabulary LM head streaming.
---

# `jero-arch-llama`

Meta `Llama` (`llama`, `llama4` iRoPE) and `Qwen2` (`qwen2`) forward/decode architecture recipe for `Jero-Mu`.

## Key Architecture & Wave 2 APIs

- **`LlamaRecipe::build_repack_cache(&self, weights)` -> `ModelRepackCache`**:
  - Pre-repacks per-layer weights (`wq`, `wk`, `wv`, `wo`, `wgate`, `wup`, `wdown`) once via `CachedMatrix::repack`, delegating `256b` `X100` / `1024b` `A100` `8x16` tile packing and `QuantizedActQ8` quantization to `jero_kernels_rvv::repack::RepackedWeight` when `with-kernels` is enabled (with a parity portable scalar fallback under `--no-default-features`).
  - Enforces fail-closed `K3Error::CapabilityMismatch` rejection for un-admitted formats (`Q2_K`, `Q3_K`, `Q4_1`, `Q5_1`, `Q8_1`, `IQ4_NL`, `TQ1_0`, `TQ2_0`, `I2_S`, `MXFP4`) while supporting `Q4_0`, `Q4_K_M`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, `BF16`, and `F32`.
  - Pre-decodes 1D `RMSNorm` tensors (`attn_norm.weight`, `ffn_norm.weight`, `output_norm.weight`) and optional `qwen2` biases (`attn_{q,k,v,output}.bias`) into `128-byte`-aligned `Aligned128F32` vectors once at cache build time, eliminating per-step `decode_1d_f32` `Vec<f32>` allocations across `prefill_step_with_cache` and `decode_step_with_cache` (as well as repeated trait `prefill_step`/`decode_step` calls via `get_or_decode_1d_f32`).
- **Compact Large-Vocab LM Head Streaming**:
  - Oversized `F16`/`BF16` (`> MAX_DENSE_HALF_ELEMENTS`) and `F32` (`> MAX_DENSE_F32_ELEMENTS`) LM heads (`128,256` Llama-3, `151,936` Qwen2.5) use `LmHeadCache::StreamingUntied`/`StreamingTied` and `project_lm_head_row_split` zero-copy row streaming within `RLIMIT_MEMLOCK` bounds.

## Wave 3 Prefill Throughput APIs

- **Zero-Allocation Direct Column Scatter GEMM**:
  - `CachedMatrix::project_batch_with_act` / `run_repacked_gemm_panels` executes `RepackedWeight::gemm_rows` (`A100` `smt.vmadot` for `M >= 8`, RVV vector, or portable fallback) and scatters `[M, sub_n]` tiles directly into disjoint columns of `[M, out_dim]` activation workspaces without per-panel `panel_major` heap allocations, using caller-reused `QuantizedBatchAct::quantize_into` scratch across transformer layers.
- **Batched Causal GQA & Chunked `start_position` Prefill**:
  - `LlamaRecipe::prefill_step` / `prefill_step_with_cache` honors `ExecutionContext::start_position()` and `ExecutionContext::compute_logits()`, rotating RoPE at `start_pos + tok`, appending contiguous `[start_pos..end_pos]` K/V planes via `KvCacheView::layer_kv_mut`, dispatching `gqa_causal_attention_prefill_f32` (`AttentionMaskConfig`) with preallocated `scores_scratch` (plus `--no-default-features` parity), and skipping final RMSNorm + LM-head `project_logits` on intermediate prefill chunks where `compute_logits() == false`.
