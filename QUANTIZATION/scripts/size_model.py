"""Size ANY Hugging Face causal LM before you rent a GPU.

Reads only config.json (a few KB, no weights), then prints:
  1. parameter count (dense and MoE)
  2. weight memory for every format (bf16, fp8, int8, 4-bit, GGUF)
  3. KV-cache memory for your context length x concurrent users
  4. which GPU, how many, and the price per hour for each option
  5. what the QUANTIZATION job itself needs (disk, CPU RAM, GPU)

Run:
  python size_model.py --model Qwen/Qwen3-8B --context 8192 --users 16
  python size_model.py --model ./my-local-model --context 4096 --users 64 --kv-fp8
"""
import argparse
import json
import math
import os

# Effective bits per weight INCLUDING the scale numbers stored next to the weights.
# W4A16 group 128: 4 bits + one 16-bit scale per 128 weights = 4 + 16/128 = 4.125
BITS = {
    "bf16": 16.0,
    "fp8": 8.0,
    "w8a8-int8": 8.0,
    "w4a16 (gptq/awq)": 4.125,
    "nf4": 4.13,
    "gguf Q8_0": 8.5,
    "gguf Q5_K_M": 5.69,
    "gguf Q4_K_M": 4.85,
    "gguf IQ4_XS": 4.25,
}

# name: (VRAM GB, CUDA compute capability, example $/hour, where).  Prices change weekly: update them.
GPUS = {
    "RTX 4090":  (24, 8.9, 0.50, "Vast.ai median / RunPod community ~0.34"),
    "L4":        (24, 8.9, 0.80, "AWS g6.xlarge 0.8048 / RunPod ~0.39"),
    "L40S":      (48, 8.9, 0.79, "RunPod community 0.79 / Vast median 0.67 / AWS g6e.xlarge 1.861"),
    "L40":       (48, 8.9, 1.00, "HyperStack 1.00"),
    "A100 80GB": (80, 8.0, 1.35, "HyperStack 1.35 / RunPod 1.19-1.39"),
    "H100 80GB": (80, 9.0, 2.50, "HyperStack 2.50 / RunPod 1.99-2.89 / Vast SXM median 2.25"),
    "H200":      (141, 9.0, 3.59, "RunPod community 3.59 / Vast median 4.71"),
    "B200":      (180, 10.0, 6.00, "HyperStack ~6.00 / RunPod 5.98"),
}


def load_config(model: str) -> dict:
    if os.path.isdir(model):
        path = os.path.join(model, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, "config.json", token=os.environ.get("HF_TOKEN"))
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    # Multimodal models keep the language model under text_config.
    return cfg.get("text_config", cfg)


def shape(cfg: dict) -> dict:
    h = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    return {
        "hidden": h,
        "layers": cfg["num_hidden_layers"],
        "heads": nh,
        "kv_heads": cfg.get("num_key_value_heads") or nh,
        "head_dim": cfg.get("head_dim") or h // nh,
        "vocab": cfg["vocab_size"],
        "inter": cfg.get("intermediate_size") or 4 * h,
        "tied": bool(cfg.get("tie_word_embeddings", False)),
        "experts": cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0,
        "experts_per_tok": cfg.get("num_experts_per_tok") or 0,
        "moe_inter": cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or 4 * h,
        "shared_inter": (cfg.get("shared_expert_intermediate_size")
                         or (cfg.get("n_shared_experts") or 0) * (cfg.get("moe_intermediate_size") or 0)),
        "mla_kv": (cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)) if cfg.get("kv_lora_rank") else 0,
    }


def param_counts(s: dict) -> tuple[float, float, float]:
    """Returns (total params, active params per token, params inside Linear layers that get quantized)."""
    h, hd = s["hidden"], s["head_dim"]
    attn = h * s["heads"] * hd + 2 * h * s["kv_heads"] * hd + s["heads"] * hd * h  # q, k, v, o
    if s["experts"]:
        expert = 3 * h * s["moe_inter"]                     # gate, up, down of ONE expert
        shared = 3 * h * s["shared_inter"]
        mlp_total = s["experts"] * expert + shared + h * s["experts"]  # + router
        mlp_active = s["experts_per_tok"] * expert + shared + h * s["experts"]
    else:
        mlp_total = mlp_active = 3 * h * s["inter"]         # SwiGLU: gate, up, down
    embed = s["vocab"] * h
    lm_head = 0 if s["tied"] else s["vocab"] * h
    linear = s["layers"] * (attn + mlp_total)
    total = linear + embed + lm_head
    active = s["layers"] * (attn + mlp_active) + embed + lm_head
    return total, active, linear


def weights_gb(total: float, linear: float, fmt: str) -> float:
    bits = BITS[fmt]
    if fmt.startswith("gguf"):
        return total * bits / 8 / 1e9                       # GGUF quantizes embeddings too
    # llm-compressor / bitsandbytes keep embeddings and lm_head in 16-bit
    return (linear * bits / 8 + (total - linear) * 2) / 1e9


def kv_bytes_per_token(s: dict, kv_bytes: float) -> float:
    if s["mla_kv"]:                                         # DeepSeek-style latent attention
        return s["layers"] * s["mla_kv"] * kv_bytes
    return 2 * s["layers"] * s["kv_heads"] * s["head_dim"] * kv_bytes  # 2 = K and V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local folder")
    ap.add_argument("--context", type=int, default=8192, help="max tokens per request (prompt + answer)")
    ap.add_argument("--users", type=int, default=16, help="requests served AT THE SAME TIME")
    ap.add_argument("--kv-fp8", action="store_true", help="store the KV cache in fp8 (1 byte)")
    ap.add_argument("--gpu-util", type=float, default=0.90, help="vLLM --gpu-memory-utilization")
    ap.add_argument("--overhead-gb", type=float, default=2.0, help="CUDA graphs + activations per GPU")
    a = ap.parse_args()

    s = shape(load_config(a.model))
    total, active, linear = param_counts(s)
    kv_b = 1.0 if a.kv_fp8 else 2.0
    kv_tok = kv_bytes_per_token(s, kv_b)
    kv_req_gb = kv_tok * a.context / 1e9
    kv_all_gb = kv_req_gb * a.users

    print(f"\nMODEL {a.model}")
    print(f"  layers={s['layers']} hidden={s['hidden']} heads={s['heads']} kv_heads={s['kv_heads']} "
          f"head_dim={s['head_dim']} vocab={s['vocab']} experts={s['experts']}")
    print(f"  params total = {total/1e9:.2f} B   active per token = {active/1e9:.2f} B")
    print(f"\nKV CACHE ({'fp8' if a.kv_fp8 else 'bf16'})")
    print(f"  per token   = {kv_tok/1e6:.4f} MB")
    print(f"  per request = {kv_req_gb:.2f} GB  ({a.context} tokens)")
    print(f"  all users   = {kv_all_gb:.2f} GB  ({a.users} x {a.context} tokens)")

    print("\nWEIGHTS PER FORMAT")
    for fmt in BITS:
        print(f"  {fmt:<18} {weights_gb(total, linear, fmt):8.2f} GB")

    print("\nWHICH GPU (serving, vLLM-style memory)")
    print(f"  {'gpu':<10}{'format':<18}{'GPUs':>5}{'$/hour':>8}{'max users':>11}  note")
    options = []
    for gname, (vram, cc, price, where) in GPUS.items():
        usable = vram * a.gpu_util - a.overhead_gb
        for fmt in ["bf16", "fp8", "w8a8-int8", "w4a16 (gptq/awq)", "nf4"]:
            if fmt == "fp8" and cc < 8.9:
                note = "no native FP8 (weight-only, slower)"
            elif fmt == "bf16" and cc < 8.0:
                continue
            else:
                note = ""
            w = weights_gb(total, linear, fmt)
            tps = [1] if fmt == "nf4" else [t for t in (1, 2, 4, 8) if s["heads"] % t == 0]
            for tp in tps:
                per_gpu_kv_room = usable - w / tp
                if per_gpu_kv_room <= 0:
                    continue
                shard = min(tp, s["kv_heads"])
                max_users = int(per_gpu_kv_room * shard / kv_req_gb) if kv_req_gb else 0
                if max_users >= a.users:
                    options.append((tp * price, gname, fmt, tp, max_users, note))
                    break
    quality = ["bf16", "fp8", "w8a8-int8", "w4a16 (gptq/awq)", "nf4"]  # best first
    options.sort(key=lambda o: (o[0], quality.index(o[2])))
    for cost, gname, fmt, tp, mu, note in options[:20]:
        print(f"  {gname:<10}{fmt:<18}{tp:>5}{cost:>8.2f}{mu:>11}  {note}")
    if not options:
        print("  nothing fits on one 8-GPU node: lower --context/--users, use --kv-fp8, or plan multi-node")
    else:
        best = options[0]
        print(f"\nCHEAPEST THAT FITS: {best[3]} x {best[1]} with {best[2]}  (~${best[0]:.2f}/hour)")
        print("  Prices are examples; check the provider page before renting.")

    bf16_gb = weights_gb(total, linear, "bf16")
    print("\nQUANTIZATION JOB NEEDS (one-time)")
    print(f"  disk      >= {bf16_gb * 3 + 30:.0f} GB  (original + 2 variants + cache)")
    print(f"  CPU RAM   >= {bf16_gb * 1.3 + 16:.0f} GB  (llm-compressor loads the bf16 model in RAM)")
    whole = bf16_gb * 1.15
    if whole <= 24:
        print(f"  GPU VRAM  any 24 GB GPU holds the whole model ({whole:.0f} GB needed)")
    else:
        print(f"  GPU VRAM  >= 24 GB works (layers are moved to the GPU one at a time);"
              f" >= {whole:.0f} GB = whole model on GPU = faster")
    print(f"  FP8 needs a GPU with compute capability >= 8.9 to SERVE fast (L4/L40S/4090/H100)\n")


if __name__ == "__main__":
    main()
