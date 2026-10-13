# PART 3 - The Core Idea

## The analogy

Think of a model as a huge library of books (the weights) and the GPU as a reading room with a fixed number of shelves (VRAM). Quantization reprints every book in smaller type: 16-point to 8 or 4. The words stay nearly the same, but two or four times as many books fit, and a reader (the GPU) scans a page faster. Each reader at a desk also needs scratch paper (the KV cache), and the scratch paper grows with how long the conversation is. The inference engine (vLLM) is the librarian who seats as many readers as the room allows. Autoscaling opens more reading rooms when a queue forms at the door. Evaluation checks that the small print did not change the meaning of the books.

## The 20% of ideas that explain 80%

| # | Idea | In one line |
|---|---|---|
| 1 | Memory = weights + KV cache + overhead | Weights are fixed; KV grows with context x concurrent users |
| 2 | weights_GB = params x bits / 8 / 1e9 | 8B at 16 bits = 16 GB; at 4 bits = about 4-6 GB |
| 3 | Generating text is memory-bound | Each new token re-reads all weights: fewer bytes = more tokens/second |
| 4 | Rounding error per weight <= scale / 2 | Smaller groups and outlier handling keep the scale small |
| 5 | Calibration data matters | GPTQ / AWQ / SmoothQuant tune themselves on your sample text: bad samples = bad model |
| 6 | Hardware decides fast formats | FP8 is native from compute capability 8.9 (L4, L40S, RTX 4090, H100); INT8 W8A8 from 7.5 |
| 7 | Never quantize twice | Always start from the 16-bit original; dequantize first if that is all you have |
| 8 | Measure, never assume | Write quality gates BEFORE measuring; compare with identical settings |
| 9 | Scale on the right signal | vLLM pre-allocates 90% VRAM, so scale on KV-cache usage % and queue length, not raw VRAM |
| 10 | Bigger models quantize better | 70B at 4-bit loses less than 1B at 4-bit; 1T MoE models are often shipped already in FP8/INT4 |

## Number formats

| Format | Bits | What it stores | Typical use |
|---|---|---|---|
| FP32 | 32 | 1 sign, 8 exponent, 23 mantissa bits | Training math, never for serving |
| BF16 | 16 | 1 sign, 8 exponent, 7 mantissa (same range as FP32) | The "original" model you download |
| FP16 | 16 | 1 sign, 5 exponent, 10 mantissa | Older GPUs (before Ampere) |
| FP8 E4M3 | 8 | 1 sign, 4 exponent, 3 mantissa + a scale | Weights and activations on L40S/H100/4090 |
| INT8 | 8 | Integer -128..127 + a scale | W8A8 on A100/A10 (no native FP8) |
| INT4 | 4 | Integer 0..15 (or -8..7) + scale per group of 128 | GPTQ / AWQ weight-only (W4A16) |
| NF4 | 4 | 16 levels placed at normal-distribution quantiles | bitsandbytes / QLoRA, no calibration |
| GGUF Q4_K_M etc. | ~4.85 avg | Blocks of 32 inside super-blocks of 256, 6-bit scales | llama.cpp on CPU / Mac / laptop |
| NVFP4 / MXFP4 | 4 | 4-bit float + 8-bit block scale | Blackwell GPUs (B200, RTX 50xx); newest (verify per engine) |

Naming you will see everywhere: **W4A16** = weights 4-bit, activations 16-bit (weight-only; saves memory, fast at low batch). **W8A8** = weights AND activations 8-bit (uses INT8/FP8 tensor cores; fast at high batch). **FP8_DYNAMIC** = FP8 weights, activation scales computed per token at runtime, so no calibration data.

## The quantization formula (with real numbers)

Asymmetric INT4 for a group of weights:

```
scale      = (max - min) / (2^bits - 1)            # 4 bits -> 15 steps
zero_point = round(-min / scale)
q          = clamp(round(w / scale) + zero_point, 0, 15)     # what is stored (4 bits each)
w_hat      = (q - zero_point) * scale                        # what the GPU uses at runtime
error      = |w - w_hat| <= scale / 2
```

Symmetric (used by GPTQ W4A16): `scale = max|w| / 7`, `q = clamp(round(w / scale), -8, 7)`, no zero point.

Worked example (8 weights):

```
weights = [-0.62, -0.11, 0.05, 0.33, 0.91, -0.47, 0.18, 0.72]
scale = (0.91 - (-0.62)) / 15 = 0.1020,  zero point = round(0.62 / 0.1020) = 6
q     = [0, 5, 6, 9, 15, 1, 8, 13]
w_hat = [-0.612, -0.102, 0.000, 0.306, 0.918, -0.510, 0.204, 0.714]
largest error 0.0500 <= scale/2 = 0.0510
```

**Why groups (g128) matter.** One scale per group. If a single huge "outlier" weight sits in the group, the scale becomes large and every other weight in the group gets rounded coarsely. Measured with one outlier among 256 weights: group 256 MSE 0.000346, group 128 MSE 0.000211, group 32 MSE 0.000047. Smaller group = more accurate, but more scales to store (bits per weight: g128 = 4.125, g64 = 4.25, g32 = 4.5).

## What each method does, in plain words

| Method | Idea in plain words | Needs data? |
|---|---|---|
| RTN (round-to-nearest) | Just round every weight. The naive baseline | No |
| FP8 dynamic | Round weights to FP8 (per channel); activations rounded per token at runtime | No |
| GPTQ [20] | Quantize one column at a time; push each column's rounding error onto the columns not yet quantized, using second-order statistics (Hessian H = X^T X) from your calibration text. `dampening` adds a small value to H's diagonal so the math stays stable | Yes |
| AWQ [21] | ~1% of input channels matter most (large activations). Scale those weights UP before rounding (and the activations down) so they lose less precision | Yes |
| SmoothQuant [22] | Activations have outlier channels that ruin INT8. Move that difficulty into the weights: X.W = (X diag(s)^-1)(diag(s) W), with s_j = max|X_j|^a / max|W_j|^(1-a); a = smoothing strength | Yes |
| QuIP / SpinQuant rotations [24][25] | Multiply by an orthogonal (Hadamard) matrix: W x = (W H^T)(H x). Same result, but outliers are spread evenly so rounding hurts less | Yes (with GPTQ) |
| NF4 [23] | 16 levels at normal-distribution quantiles (weights are roughly normal); double quantization also compresses the scales | No |
| GGUF k-quants + imatrix [3] | Block-wise scales; the importance matrix (from sample text) tells the quantizer which weights to protect | Optional |

## Every formula you need (also in the Cheat Sheet)

```
params (dense, Llama-style) = L x [ h*(n_heads*hd) + 2*h*(n_kv*hd) + (n_heads*hd)*h + 3*h*I ] + V*h (+ V*h if not tied)
params (MoE)                = L x [ attention + E*3*h*I_moe + shared experts + router ] + embeddings
weights_GB                  = quantized_params x bits/8 / 1e9 + (embeddings + lm_head) x 2 / 1e9
KV_bytes_per_token          = 2 x L x n_kv x hd x kv_bytes          (2 = K and V; kv_bytes 2 for bf16, 1 for fp8)
KV_GB for your traffic      = KV_bytes_per_token x context x concurrent_users / 1e9
VRAM needed per GPU         = weights_GB / TP + KV_GB / min(TP, n_kv) + ~2 GB       must be <= VRAM x 0.90
GPUs (TP)                   = smallest of 1, 2, 4, 8 that fits AND divides n_heads
single-user tok/s (max)     ~ GPU memory bandwidth (GB/s) / weights_GB
cost per 1M tokens ($)      = hourly price / (tok/s x 3600) x 1e6
speculative decoding        = expected tokens per target pass E = (1 - a^(k+1)) / (1 - a)   (a = acceptance, k = draft tokens)
quality retention %         = quantized score / original score x 100
```

Worked example, Qwen3-8B (L=36, h=4096, heads=32, kv_heads=8, hd=128, I=12288, V=151936, not tied):

```
attention per layer = 4096*4096 + 2*4096*1024 + 4096*4096 = 41.9 M
MLP per layer       = 3 * 4096 * 12288                      = 151.0 M
36 layers           = 36 * 192.9 M                          = 6.95 B
embeddings + head   = 2 * 151936 * 4096                     = 1.24 B
TOTAL               = 8.19 B parameters
weights bf16 = 16.4 GB | fp8 = 9.4 GB | w4a16 = 6.1 GB   (embeddings and lm_head stay 16-bit)
KV per token (bf16) = 2 * 36 * 8 * 128 * 2 bytes = 147,456 bytes = 0.147 MB
16 users x 8192 tokens = 131,072 tokens x 0.147 MB = 19.3 GB
bf16 on one L40S (48 GB): 16.4 + 19.3 + 2 = 37.7 GB <= 43.2 GB  -> fits, 1 GPU
single-user speed on L40S (864 GB/s): bf16 ~ 864/16.4 = 53 tok/s; fp8 ~ 92 tok/s; w4a16 ~ 140 tok/s (upper bounds)
```

(These are exactly the numbers `size_model.py` printed when this manual was tested.)

## What quality loss to expect (starting gates; always measure your own model)

| Format | Typical perplexity increase vs bf16 | Typical task-score retention | Notes |
|---|---|---|---|
| FP8 dynamic | < 0.5% | 99-100% | Safest first quantization |
| W8A8 INT8 (SmoothQuant) | 0.5-1% | 98-100% | Best speed on A100/A10 |
| W4A16 GPTQ / AWQ (g128) | 2-5% | 95-99% | Worse on small models (< 3B) and on maths/code |
| NF4 | 3-6% | 94-98% | No calibration; slower kernels |
| GGUF Q4_K_M with imatrix | 2-4% | 95-99% | For CPU/Mac; Q8_0 is near-lossless |

Everything else in this manual is these ideas applied carefully, one step at a time.
