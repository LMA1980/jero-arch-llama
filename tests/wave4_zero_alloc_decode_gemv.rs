#![allow(missing_docs)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use jero_arch_llama::repack_cache::set_active_model_anchor;
use jero_arch_llama::{
    BlockQ8_0, BlockQ8K, DecodeQ8Scratch, LlamaRecipe, bytes_as_q8_0_blocks_mut,
    bytes_as_q8_k_blocks_mut, clear_repack_cache, compute_row_dot, gemv_decode_projection,
    gemv_decode_projection_with_workers, get_or_decode_1d_f32, get_or_repack,
    project_lm_head_row_split,
};
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, KvGeometry, QuantType, ScratchWorkspace,
    TensorView,
};

struct CountingAllocator;

static COUNTING_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegates directly to `std::alloc::System` while maintaining lock-free atomic telemetry.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING_ENABLED.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Same preconditions forwarded directly to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with identical `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING_ENABLED.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Same preconditions forwarded directly to `System::alloc_zeroed`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING_ENABLED.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: `ptr` belongs to `System` with identical prior `layout`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAllocator = CountingAllocator;

struct CountingResetGuard;

impl Drop for CountingResetGuard {
    fn drop(&mut self) {
        COUNTING_ENABLED.store(false, Ordering::SeqCst);
    }
}

fn warm_rayon_pool(pool: &rayon::ThreadPool) {
    // Wake every worker in `pool`, let all workers settle into `WorkerSleepState::condvar.wait`
    // while the pool is completely idle (initializing macOS lazy `OnceBox<pthread::Condvar>`
    // on every worker), and warm a `join` `SpinLatch` sleep before any allocation check runs.
    pool.broadcast(|_| {});
    std::thread::sleep(std::time::Duration::from_millis(40));
    pool.install(|| {
        for _ in 0..4 {
            rayon::join(
                || std::thread::sleep(std::time::Duration::from_millis(2)),
                || std::thread::sleep(std::time::Duration::from_millis(5)),
            );
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(40));
}

fn measure_heap_allocs<R>(op: impl FnOnce() -> R) -> (R, usize, usize) {
    COUNTING_ENABLED.store(false, Ordering::SeqCst);
    ALLOC_CALLS.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    COUNTING_ENABLED.store(true, Ordering::SeqCst);
    let _reset = CountingResetGuard;
    let res = op();
    COUNTING_ENABLED.store(false, Ordering::SeqCst);
    let calls = ALLOC_CALLS.load(Ordering::SeqCst);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    (res, calls, bytes)
}

const fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 {
        return sign | 0x7E00;
    }
    if exp >= 143 {
        return sign | 0x7C00;
    }
    if exp <= 112 {
        return sign;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "exp - 112 is bounded in 1..=30"
    )]
    let exp_bits = ((exp - 112) as u16) << 10;
    sign | exp_bits | ((mant >> 13) as u16)
}

fn fill_quantized_row(
    row: &mut [u8],
    r: usize,
    cols: usize,
    quant: QuantType,
    seed: usize,
    scale_f16: [u8; 2],
    min_f16: [u8; 2],
) {
    match quant {
        QuantType::F32 => {
            for c in 0..cols {
                #[expect(clippy::cast_precision_loss, reason = "bounded synthetic weight")]
                let v = ((((r * 13 + c * 7 + seed) % 23) as f32) - 11.0) * 0.02;
                row[c * 4..(c + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        QuantType::F16 => {
            for c in 0..cols {
                #[expect(clippy::cast_precision_loss, reason = "bounded synthetic weight")]
                let v = ((((r * 11 + c * 5 + seed) % 19) as f32) - 9.0) * 0.025;
                row[c * 2..(c + 1) * 2].copy_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
        }
        QuantType::BF16 => {
            for c in 0..cols {
                #[expect(clippy::cast_precision_loss, reason = "bounded synthetic weight")]
                let v = ((((r * 17 + c * 3 + seed) % 17) as f32) - 8.0) * 0.025;
                let bf16 = (v.to_bits() >> 16) as u16;
                row[c * 2..(c + 1) * 2].copy_from_slice(&bf16.to_le_bytes());
            }
        }
        QuantType::Q4_0 => {
            for (b, blk) in row.chunks_exact_mut(18).enumerate() {
                blk[0..2].copy_from_slice(&scale_f16);
                for j in 0..16 {
                    let lo = u8::try_from((r + b + j + seed) % 15).unwrap_or(0);
                    let hi = u8::try_from((r + b + 2 * j + seed) % 15).unwrap_or(0);
                    blk[2 + j] = lo | (hi << 4);
                }
            }
        }
        QuantType::Q5_0 => {
            for (b, blk) in row.chunks_exact_mut(22).enumerate() {
                blk[0..2].copy_from_slice(&scale_f16);
                blk[2..6].copy_from_slice(&0x55AA_55AA_u32.to_le_bytes());
                for j in 0..16 {
                    let lo = u8::try_from((r + b + j + seed) % 15).unwrap_or(0);
                    let hi = u8::try_from((r + b + 2 * j + seed) % 15).unwrap_or(0);
                    blk[6 + j] = lo | (hi << 4);
                }
            }
        }
        QuantType::Q8_0 => {
            for (b, blk) in row.chunks_exact_mut(34).enumerate() {
                blk[0..2].copy_from_slice(&scale_f16);
                for j in 0..32 {
                    let q = i8::try_from((r + b * 3 + j + seed) % 29).unwrap_or(0) - 14;
                    blk[2 + j] = q.to_le_bytes()[0];
                }
            }
        }
        QuantType::Q4_K_M => {
            for (b, blk) in row.chunks_exact_mut(144).enumerate() {
                blk[0..2].copy_from_slice(&scale_f16);
                blk[2..4].copy_from_slice(&min_f16);
                blk[4..16].fill(0x21);
                for j in 0..128 {
                    let lo = u8::try_from((r + b + j + seed) % 15).unwrap_or(0);
                    let hi = u8::try_from((r + 2 * b + 3 * j + seed) % 15).unwrap_or(0);
                    blk[16 + j] = lo | (hi << 4);
                }
            }
        }
        QuantType::Q5_K => {
            for (b, blk) in row.chunks_exact_mut(176).enumerate() {
                blk[0..2].copy_from_slice(&scale_f16);
                blk[2..4].copy_from_slice(&min_f16);
                blk[4..16].fill(0x18);
                blk[16..48].fill(0x33);
                for j in 0..128 {
                    let lo = u8::try_from((r + b + j + seed) % 15).unwrap_or(0);
                    let hi = u8::try_from((r + b + 3 * j + seed) % 15).unwrap_or(0);
                    blk[48 + j] = lo | (hi << 4);
                }
            }
        }
        QuantType::Q6_K => {
            for (b, blk) in row.chunks_exact_mut(210).enumerate() {
                for (j, slot) in blk.iter_mut().enumerate().take(128) {
                    *slot = u8::try_from((r + b + j * 5 + seed) & 0x77).unwrap_or(0);
                }
                blk[128..192].fill(0x55);
                blk[192..208].fill(0x03);
                blk[208..210].copy_from_slice(&scale_f16);
            }
        }
        other => panic!("unsupported test quant {other:?}"),
    }
}

fn fill_quantized_tensor_bytes(
    buf: &mut [u8],
    rows: usize,
    cols: usize,
    quant: QuantType,
    seed: usize,
) {
    let scale_f16 = f32_to_f16_bits(0.03125).to_le_bytes();
    let min_f16 = f32_to_f16_bits(0.007_812_5).to_le_bytes();
    let row_bytes = match quant {
        QuantType::F32 => cols * 4,
        QuantType::F16 | QuantType::BF16 => cols * 2,
        QuantType::Q4_0 => (cols / 32) * 18,
        QuantType::Q5_0 => (cols / 32) * 22,
        QuantType::Q8_0 => (cols / 32) * 34,
        QuantType::Q4_K_M => (cols / 256) * 144,
        QuantType::Q5_K => (cols / 256) * 176,
        QuantType::Q6_K => (cols / 256) * 210,
        other => panic!("unsupported test quant {other:?}"),
    };
    for r in 0..rows {
        let row = &mut buf[r * row_bytes..(r + 1) * row_bytes];
        fill_quantized_row(row, r, cols, quant, seed, scale_f16, min_f16);
    }
}

struct SynthGgufSpec<'a> {
    arch: &'a str,
    quant: QuantType,
    hidden: usize,
    ffn: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    vocab: usize,
}

fn build_synthetic_quant_gguf(spec: &SynthGgufSpec<'_>) -> Vec<u8> {
    let arch = spec.arch;
    let quant = spec.quant;
    let hidden = spec.hidden;
    let ffn = spec.ffn;
    let n_q = spec.n_q;
    let n_kv = spec.n_kv;
    let head_dim = spec.head_dim;
    let vocab = spec.vocab;
    let blk_k = format!("{arch}.block_count");
    let emb_k = format!("{arch}.embedding_length");
    let ffn_k = format!("{arch}.feed_forward_length");
    let hq_k = format!("{arch}.attention.head_count");
    let hkv_k = format!("{arch}.attention.head_count_kv");
    let key_k = format!("{arch}.attention.key_length");
    let eps_k = format!("{arch}.attention.layer_norm_rms_epsilon");
    let theta_k = format!("{arch}.rope.freq_base");
    let ctx_k = format!("{arch}.context_length");

    let kvs = [
        (blk_k.as_str(), GgufValue::Uint32(1)),
        (
            emb_k.as_str(),
            GgufValue::Uint32(u32::try_from(hidden).unwrap()),
        ),
        (
            ffn_k.as_str(),
            GgufValue::Uint32(u32::try_from(ffn).unwrap()),
        ),
        (
            hq_k.as_str(),
            GgufValue::Uint32(u32::try_from(n_q).unwrap()),
        ),
        (
            hkv_k.as_str(),
            GgufValue::Uint32(u32::try_from(n_kv).unwrap()),
        ),
        (
            key_k.as_str(),
            GgufValue::Uint32(u32::try_from(head_dim).unwrap()),
        ),
        (eps_k.as_str(), GgufValue::Float32(1e-5)),
        (theta_k.as_str(), GgufValue::Float32(10_000.0)),
        (ctx_k.as_str(), GgufValue::Uint32(16)),
    ];

    let total_q = n_q * head_dim;
    let total_kv = n_kv * head_dim;
    let emb_shape = [hidden, vocab];
    let norm_shape = [hidden];
    let q_shape = [hidden, total_q];
    let kv_shape = [hidden, total_kv];
    let o_shape = [total_q, hidden];
    let gate_up_shape = [hidden, ffn];
    let down_shape = [ffn, hidden];
    let q_bias_shape = [total_q];
    let kv_bias_shape = [total_kv];

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
        ("output_norm.weight", QuantType::F32, &norm_shape),
        ("output.weight", quant, &emb_shape),
    ];
    if arch == "qwen2" {
        specs.push(("blk.0.attn_q.bias", QuantType::F32, &q_bias_shape));
        specs.push(("blk.0.attn_k.bias", QuantType::F32, &kv_bias_shape));
        specs.push(("blk.0.attn_v.bias", QuantType::F32, &kv_bias_shape));
    }

    let mut raw = create_synthetic_gguf_v3_with_kvs(arch, 32, &kvs, &specs);
    let parsed = GgufFile::parse(&raw).unwrap();
    let descs = parsed.metadata().tensors().to_vec();
    for (idx, desc) in descs.iter().enumerate() {
        let start = desc.absolute_offset();
        let end = start + desc.byte_len();
        let slice = &mut raw[start..end];
        if desc.quant() == QuantType::F32 && desc.shape().len() == 1 {
            for (i, chunk) in slice.chunks_exact_mut(4).enumerate() {
                #[expect(clippy::cast_precision_loss, reason = "bounded synthetic norm")]
                let val = if desc.name().contains("norm") {
                    0.9 + ((i % 5) as f32) * 0.02
                } else {
                    ((i % 7) as f32 - 3.0) * 0.01
                };
                chunk.copy_from_slice(&val.to_le_bytes());
            }
        } else {
            let cols = desc.shape()[0];
            let rows = desc.shape().get(1).copied().unwrap_or(1);
            fill_quantized_tensor_bytes(slice, rows, cols, desc.quant(), idx + 1);
        }
    }
    raw
}

fn verify_single_quant_decode_zero_alloc(arch: &str, quant: QuantType) {
    let hidden = 256_usize;
    let ffn = 256_usize;
    let n_q = 4_usize;
    let n_kv = 2_usize;
    let head_dim = 64_usize;
    let vocab = 16_usize;

    let raw = build_synthetic_quant_gguf(&SynthGgufSpec {
        arch,
        quant,
        hidden,
        ffn,
        n_q,
        n_kv,
        head_dim,
        vocab,
    });
    let gguf = GgufFile::parse(&raw).unwrap();
    let recipe = LlamaRecipe::from_gguf(&gguf).unwrap();
    let cache = recipe.build_repack_cache(&gguf).unwrap();
    let kv_geom = KvGeometry::new(1, n_kv, head_dim, 16, None).unwrap();
    let kv_len = kv_geom.required_f32_len().unwrap();

    // 1. Explicit caller-owned ScratchWorkspace + ModelRepackCache
    let scratch_bytes = recipe.decode_scratch_bytes(16);
    let mut scratch = ScratchWorkspace::new(scratch_bytes).unwrap();
    let mut logits_cached = vec![0.0_f32; vocab];
    let mut kv_cached = vec![0.0_f32; kv_len];
    let mut ctx_cached = ExecutionContext::new(&[1], 16, &mut logits_cached, &mut kv_cached)
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap()
        .with_scratch(&mut scratch);
    recipe
        .prefill_step_with_cache(&mut ctx_cached, &gguf, &cache)
        .unwrap();
    // Warm 1st decode step before checking steady-state zero-allocations.
    ctx_cached.prepare_decode(2).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_cached, &gguf, &cache)
        .unwrap();

    for step_tok in [3_u32, 4, 5] {
        ctx_cached.prepare_decode(step_tok).unwrap();
        let (res, calls, bytes) =
            measure_heap_allocs(|| recipe.decode_step_with_cache(&mut ctx_cached, &gguf, &cache));
        res.unwrap();
        assert_eq!(
            (calls, bytes),
            (0, 0),
            "decode_step_with_cache allocated on {arch}/{quant:?}: calls={calls}, bytes={bytes}"
        );
        assert!(ctx_cached.logits().iter().all(|v| v.is_finite()));
    }

    // 2. Fallback decode_step with ScratchWorkspace attached (zero allocs after warm hit)
    let mut fallback_scratch = ScratchWorkspace::new(scratch_bytes).unwrap();
    let mut logits_fb = vec![0.0_f32; vocab];
    let mut kv_fb = vec![0.0_f32; kv_len];
    let mut ctx_fb = ExecutionContext::new(&[1], 16, &mut logits_fb, &mut kv_fb)
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap()
        .with_scratch(&mut fallback_scratch);
    recipe.prefill_step(&mut ctx_fb, &gguf).unwrap();
    ctx_fb.prepare_decode(2).unwrap();
    recipe.decode_step(&mut ctx_fb, &gguf).unwrap();

    for step_tok in [3_u32, 4, 5] {
        ctx_fb.prepare_decode(step_tok).unwrap();
        let (res, calls, bytes) = measure_heap_allocs(|| recipe.decode_step(&mut ctx_fb, &gguf));
        res.unwrap();
        assert_eq!(
            (calls, bytes),
            (0, 0),
            "decode_step(with_scratch) allocated on {arch}/{quant:?}: calls={calls}, bytes={bytes}"
        );
        assert!(ctx_fb.logits().iter().all(|v| v.is_finite()));
    }

    // 3. Fallback decode_step WITHOUT ScratchWorkspace attached (reusable thread-local workspace)
    let mut logits_tl = vec![0.0_f32; vocab];
    let mut kv_tl = vec![0.0_f32; kv_len];
    let mut ctx_tl = ExecutionContext::new(&[1], 16, &mut logits_tl, &mut kv_tl)
        .unwrap()
        .with_kv_geometry(kv_geom)
        .unwrap();
    recipe.prefill_step(&mut ctx_tl, &gguf).unwrap();
    ctx_tl.prepare_decode(2).unwrap();
    recipe.decode_step(&mut ctx_tl, &gguf).unwrap();

    for step_tok in [3_u32, 4, 5] {
        ctx_tl.prepare_decode(step_tok).unwrap();
        let (res, calls, bytes) = measure_heap_allocs(|| recipe.decode_step(&mut ctx_tl, &gguf));
        res.unwrap();
        assert_eq!(
            (calls, bytes),
            (0, 0),
            "decode_step(thread-local scratch) allocated on {arch}/{quant:?}: calls={calls}, bytes={bytes}"
        );
        assert!(ctx_tl.logits().iter().all(|v| v.is_finite()));
    }
    assert_eq!(logits_tl, logits_fb);
}

fn check_zero_alloc_decode_step_with_and_without_scratch_across_quants() {
    clear_repack_cache();
    let pool_4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    warm_rayon_pool(&pool_4);
    pool_4.install(|| {
        for (arch, quant) in [
            ("llama", QuantType::F32),
            ("llama", QuantType::F16),
            ("qwen2", QuantType::BF16),
            ("llama", QuantType::Q4_0),
            ("qwen2", QuantType::Q5_0),
            ("llama", QuantType::Q8_0),
            ("qwen2", QuantType::Q4_K_M),
            ("llama", QuantType::Q5_K),
            ("qwen2", QuantType::Q6_K),
        ] {
            verify_single_quant_decode_zero_alloc(arch, quant);
        }
    });
}

fn check_zero_alloc_fallback_bucket_map_hits() {
    clear_repack_cache();
    let cols = 64_usize;
    let rows = 8_usize;
    let emb_bytes = vec![1_u8; rows * cols * 4];
    let mat_bytes = vec![2_u8; rows * cols * 4];
    let norm_bytes = vec![0_u8; cols * 4];

    let shape_2d = [cols, rows];
    let shape_1d = [cols];
    let emb_view = TensorView::from_bytes(
        "token_embd.weight",
        QuantType::F32,
        &shape_2d,
        0,
        &emb_bytes,
    )
    .unwrap();
    let mat_view = TensorView::from_bytes(
        "blk.0.attn_q.weight",
        QuantType::F32,
        &shape_2d,
        0,
        &mat_bytes,
    )
    .unwrap();
    let norm_view = TensorView::from_bytes(
        "blk.0.attn_norm.weight",
        QuantType::F32,
        &shape_1d,
        0,
        &norm_bytes,
    )
    .unwrap();

    set_active_model_anchor(&emb_view);
    let _ = get_or_repack(&mat_view, rows, cols).unwrap();
    let _ = get_or_decode_1d_f32(&norm_view, cols, true).unwrap();

    let ((), calls, bytes) = measure_heap_allocs(|| {
        for _ in 0..16 {
            set_active_model_anchor(&emb_view);
            let _m = get_or_repack(&mat_view, rows, cols).unwrap();
            let _v = get_or_decode_1d_f32(&norm_view, cols, true).unwrap();
        }
    });
    assert_eq!(
        (calls, bytes),
        (0, 0),
        "ActiveModelFallbackCache hit paths must allocate 0 bytes"
    );
}

fn check_multi_worker_1_4_8_quant_gemv(
    pool_1: &rayon::ThreadPool,
    pool_4: &rayon::ThreadPool,
    pool_8: &rayon::ThreadPool,
) {
    for quant in [
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
    ] {
        let cols = 256_usize;
        let rows = 64_usize;
        let row_bytes = (cols / quant.block_elements()) * quant.block_bytes();
        let mut raw = vec![0_u8; rows * row_bytes];
        fill_quantized_tensor_bytes(&mut raw, rows, cols, quant, 9);
        let mat_shape = [cols, rows];
        let view = TensorView::from_bytes("output.weight", quant, &mat_shape, 0, &raw).unwrap();
        let mut hidden = vec![0.0_f32; cols];
        for (i, h) in hidden.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small bounded index")]
            {
                *h = (((i * 7 + 3) % 19) as f32 - 9.0) * 0.05;
            }
        }

        let mut ws = ScratchWorkspace::new(4096).unwrap();
        let mut run_on = |pool: &rayon::ThreadPool, workers: usize| -> Vec<f32> {
            let mut out = vec![0.0_f32; rows];
            pool.install(|| {
                ws.reset();
                let scope = ws.scope();
                let small_raw = scope.alloc_u8(8 * size_of::<BlockQ8_0>()).unwrap();
                let q8_0 = bytes_as_q8_0_blocks_mut(small_raw, 8).unwrap();
                let super_raw = scope.alloc_u8(size_of::<BlockQ8K>()).unwrap();
                let q8_k = bytes_as_q8_k_blocks_mut(super_raw, 1).unwrap();
                let mut scratch = DecodeQ8Scratch::new(q8_0, q8_k);

                // Warm up once, then measure strict 0 allocs across workers
                gemv_decode_projection_with_workers(&mut out, &view, &hidden, &mut scratch, workers)
                    .unwrap();
                let (res, calls, bytes) = measure_heap_allocs(|| {
                    gemv_decode_projection_with_workers(
                        &mut out,
                        &view,
                        &hidden,
                        &mut scratch,
                        workers,
                    )
                });
                res.unwrap();
                assert_eq!(
                    (calls, bytes),
                    (0, 0),
                    "{quant:?} M=1 GEMV allocated on {workers} workers: {calls} calls, {bytes} bytes"
                );
            });
            out
        };

        let o1 = run_on(pool_1, 1);
        let o4 = run_on(pool_4, 4);
        let o8 = run_on(pool_8, 8);
        assert_eq!(o1, o4, "{quant:?} 1 vs 4 worker partition mismatch");
        assert_eq!(o1, o8, "{quant:?} 1 vs 8 worker partition mismatch");

        for (r, &got) in o1.iter().enumerate() {
            let expected = compute_row_dot(r, cols, quant, &raw, &hidden);
            assert!(
                got.is_finite() && (got - expected).abs() < 0.05 + expected.abs() * 0.01,
                "{quant:?} row {r} accuracy regression: {got} vs {expected}"
            );
        }
    }
}

fn check_152k_vocab_gemv_determinism(
    pool_1: &rayon::ThreadPool,
    pool_4: &rayon::ThreadPool,
    pool_8: &rayon::ThreadPool,
) {
    let lm_cols = 256_usize;
    let lm_rows = 151_936_usize;
    let q4_row_bytes = (lm_cols / 32) * 18;
    let mut lm_raw = vec![0_u8; lm_rows * q4_row_bytes];
    for row in [0_usize, 75_968, lm_rows - 1] {
        fill_quantized_tensor_bytes(
            &mut lm_raw[row * q4_row_bytes..(row + 1) * q4_row_bytes],
            1,
            lm_cols,
            QuantType::Q4_0,
            row + 1,
        );
    }
    let lm_shape = [lm_cols, lm_rows];
    let lm_view =
        TensorView::from_bytes("output.weight", QuantType::Q4_0, &lm_shape, 0, &lm_raw).unwrap();
    let hidden = vec![0.125_f32; lm_cols];
    let mut logits_1 = vec![0.0_f32; lm_rows];
    let mut logits_4 = vec![0.0_f32; lm_rows];
    let mut logits_8 = vec![0.0_f32; lm_rows];

    let mut lm_ws = ScratchWorkspace::new(4096).unwrap();
    pool_1.install(|| {
        lm_ws.reset();
        let scope = lm_ws.scope();
        let q8_0 = bytes_as_q8_0_blocks_mut(scope.alloc_u8(8 * size_of::<BlockQ8_0>()).unwrap(), 8)
            .unwrap();
        let q8_k =
            bytes_as_q8_k_blocks_mut(scope.alloc_u8(size_of::<BlockQ8K>()).unwrap(), 1).unwrap();
        let mut scratch = DecodeQ8Scratch::new(q8_0, q8_k);
        gemv_decode_projection(&mut logits_1, &lm_view, &hidden, &mut scratch).unwrap();
    });
    pool_4.install(|| {
        lm_ws.reset();
        let scope = lm_ws.scope();
        let q8_0 = bytes_as_q8_0_blocks_mut(scope.alloc_u8(8 * size_of::<BlockQ8_0>()).unwrap(), 8)
            .unwrap();
        let q8_k =
            bytes_as_q8_k_blocks_mut(scope.alloc_u8(size_of::<BlockQ8K>()).unwrap(), 1).unwrap();
        let mut scratch = DecodeQ8Scratch::new(q8_0, q8_k);
        gemv_decode_projection(&mut logits_4, &lm_view, &hidden, &mut scratch).unwrap();
    });
    pool_8.install(|| {
        project_lm_head_row_split(
            &mut logits_8,
            &hidden,
            &lm_view,
            &jero_arch_llama::PhaseLayout::default(),
        )
        .unwrap();
    });
    assert_eq!(logits_1, logits_4);
    assert_eq!(logits_1, logits_8);
}

fn check_multi_worker_1_4_8_gemv_determinism_and_152k_vocab() {
    let pool_1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool_4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool_8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    // Warm worker pools once so steady-state stack-allocated rayon::join does 0 allocs
    for pool in [&pool_1, &pool_4, &pool_8] {
        warm_rayon_pool(pool);
    }

    check_multi_worker_1_4_8_quant_gemv(&pool_1, &pool_4, &pool_8);
    check_152k_vocab_gemv_determinism(&pool_1, &pool_4, &pool_8);
}

fn check_repacked_gemv_view_matches_gemm_prefill_zero_alloc() {
    let pool_4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    warm_rayon_pool(&pool_4);
    pool_4.install(|| {
        let cols = 256_usize;
        let rows = 96_usize;
        let mut hidden = vec![0.0_f32; cols];
        for (i, h) in hidden.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "bounded synthetic activation")]
            {
                *h = (((i * 11 + 5) % 23) as f32 - 11.0) * 0.04;
            }
        }

        let mut ws = ScratchWorkspace::new(
            jero_arch_llama::DecodeActQ8View::scratch_bytes_for_row(cols) + 512,
        )
        .unwrap();
        for quant in [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4_0,
            QuantType::Q5_0,
            QuantType::Q8_0,
            QuantType::Q4_K_M,
            QuantType::Q5_K,
            QuantType::Q6_K,
        ] {
            let row_bytes = match quant {
                QuantType::F32 => cols * 4,
                QuantType::F16 | QuantType::BF16 => cols * 2,
                _ => (cols / quant.block_elements()) * quant.block_bytes(),
            };
            let mut raw = vec![0_u8; rows * row_bytes];
            fill_quantized_tensor_bytes(&mut raw, rows, cols, quant, 17);
            let shape = [cols, rows];
            let view =
                TensorView::from_bytes("blk.0.attn_q.weight", quant, &shape, 0, &raw).unwrap();
            let cached = jero_arch_llama::CachedMatrix::repack(&view, rows, cols).unwrap();

            let mut prefill_out = vec![0.0_f32; rows];
            cached
                .gemm_rows(&mut prefill_out, &hidden, 1, true)
                .unwrap();

            let mut decode_out = vec![0.0_f32; rows];
            ws.reset();
            {
                let scope = ws.scope();
                cached
                    .gemv_in_scope(&mut decode_out, &hidden, &scope, true)
                    .unwrap();
            }
            ws.reset();
            let (res, calls, bytes) = measure_heap_allocs(|| {
                let scope = ws.scope();
                cached.gemv_in_scope(&mut decode_out, &hidden, &scope, true)
            });
            res.unwrap();
            assert_eq!(
                (calls, bytes),
                (0, 0),
                "{quant:?} gemv_in_scope allocated {calls} calls / {bytes} bytes"
            );
            for (r, (&got, &expected)) in decode_out.iter().zip(&prefill_out).enumerate() {
                assert!(
                    (got - expected).abs() < 1.0e-4,
                    "{quant:?} row {r} gemv_in_scope vs gemm_rows mismatch: {got} vs {expected}"
                );
            }
        }
    });
}

#[test]
fn test_wave4_zero_alloc_decode_gemv_suite() {
    check_zero_alloc_fallback_bucket_map_hits();
    check_zero_alloc_decode_step_with_and_without_scratch_across_quants();
    check_multi_worker_1_4_8_gemv_determinism_and_152k_vocab();
    check_repacked_gemv_view_matches_gemm_prefill_zero_alloc();
}
