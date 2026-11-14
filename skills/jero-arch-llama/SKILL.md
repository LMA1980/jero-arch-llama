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

## Wave 4 Zero-Allocation Decode & Dedicated `M = 1` RVV GEMV APIs

- **Zero-Allocation `decode_step` & `ScratchWorkspace` (`#41`)**:
  - `LlamaRecipe::decode_scratch_bytes(&self, seq_len_upper_bound)` reports the exact `128-byte`-aligned byte capacity needed by `ExecutionContext::with_scratch(&mut scratch)` (falling back to a reusable thread-local `ScratchWorkspace` when no workspace is attached, allocating zero heap buffers in steady state under both default `with-kernels` and `--no-default-features`).
  - `LayerTensorKeys` / `LayerKeyTable` (`ensure_global_layer_keys`, `probe_live_layer_count`, `ModelRepackCache::layer_keys`) pre-compute all `blk.{layer}.*` key strings once, and `ActiveModelFallbackCache` caches model-wide layer count/head geometry and per-layer Qwen2 bias presence, eliminating steady-state `format!` and missing-tensor probe `String` allocations across `decode_step` and `decode_step_with_cache`.
  - `ZeroAllocBucketMap` inside `ActiveModelFallbackCache` (`get_or_repack`, `get_or_decode_1d_f32`, `set_active_model_anchor`) executes borrow-checked read-lock lookups without cloning strings or building temporary `CacheKey` structs on cache hits.
- **Dedicated `M = 1` Multi-Row RVV GEMV Fast-Path (`#42`)**:
  - `DecodeActQ8View` (`CachedMatrix::gemv_with_act_view`, `CachedMatrix::gemv_in_scope`) quantizes `M = 1` activations directly inside `ScratchScope` via `jero_kernels_rvv::repack::QuantizedActQ8View` and dispatches stack-recursive `rayon::join` panels onto `RepackedWeight::gemv_view_rows` (sharing one quantized view across `Q/K/V` and `Gate/Up` projections, with `< 1e-4` parity and `--no-default-features` portable fallback).
  - `DecodeQ8Scratch` (`bytes_as_q8_0_blocks_mut`, `bytes_as_q8_k_blocks_mut`) reuses pre-quantized `BlockQ8_0` (`32`-element) and `BlockQ8K` (`256`-element) slices across uncached `gemv_decode_projection` / `gemv_decode_projection_with_workers` projections partitioned across `4`-row aligned `rayon::join` slices (`gemv_q4_0_q8_0`, `gemv_q5_0_q8_0`, `gemv_q8_0_q8_0`, `gemv_q4_k_q8_k`, `gemv_q5_k_q8_k`, `gemv_q6_k_q8_k`, and zero-alloc dense `F32`/`F16`/`BF16`), with MuConDe delivery manifest at `.muconde/MuConDe.yml`.
