---
name: jero-arch-llama
description: "Implement and validate Meta Llama (Llama 3 and Llama 4 iRoPE) ArchRecipe execution, GQA KV routing, and SwiGLU FFN forward passes in jero-arch-llama."
---

# `jero-arch-llama` Architecture Recipe Invariants

- Implement `jero_core::ArchRecipe` with zero-copy `TensorView` weight lookup, Grouped-Query Attention (GQA), RoPE / Llama 4 interleaved iRoPE frequency scaling, and optional `jero-kernels-rvv` hardware acceleration.
- Use `usize` for tensor slice indexing and memory offsets, and `u32` for GGUF architectural hyperparameters (`n_layer`, `n_head`, `n_head_kv`, `n_embd`).
