"""Plan GPU memory for serving MANY LoRA adapters on one base model (vLLM / SGLang multi-LoRA).

For each rank it prints the size of ONE adapter, and what the engine reserves:
  GPU:  max_loras slots, each sized for max_lora_rank   (vLLM --max-loras, --max-lora-rank)
  CPU:  max_cpu_loras adapters cached in RAM            (vLLM --max-cpu-loras)
LoRA adds two matrices per adapted layer: A (r x in) and B (out x r) -> r x (in + out) numbers.

Run:
  python lora_plan.py --model Qwen/Qwen3-8B --ranks 8 16 32 64 --max-loras 4 --max-lora-rank 64 --max-cpu-loras 16
"""
import argparse

from size_model import load_config, shape


def lora_params_per_layer(s: dict, rank: int, targets: list[str]) -> int:
    h, hd, inter = s["hidden"], s["head_dim"], s["inter"]
    q_out, kv_out = s["heads"] * hd, s["kv_heads"] * hd
    dims = {  # module: (in_features, out_features)
        "q_proj": (h, q_out), "k_proj": (h, kv_out), "v_proj": (h, kv_out), "o_proj": (q_out, h),
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h),
    }
    return sum(rank * (dims[t][0] + dims[t][1]) for t in targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--max-loras", type=int, default=4, help="adapters usable in ONE batch (GPU slots)")
    ap.add_argument("--max-lora-rank", type=int, default=64, help="every GPU slot is sized for this rank")
    ap.add_argument("--max-cpu-loras", type=int, default=16, help="adapters cached in CPU RAM")
    a = ap.parse_args()

    s = shape(load_config(a.model))
    if s["experts"]:
        print("NOTE: MoE model: this counts attention + dense MLP shapes only; expert LoRA support varies by engine (verify).")
    print(f"\nLoRA targets: {', '.join(a.targets)}  | layers={s['layers']}")
    print(f"{'rank':>6}{'params':>14}{'MB (bf16)':>12}")
    for r in a.ranks:
        p = lora_params_per_layer(s, r, a.targets) * s["layers"]
        print(f"{r:>6}{p:>14,}{p * 2 / 1e6:>12.1f}")
    slot = lora_params_per_layer(s, a.max_lora_rank, a.targets) * s["layers"] * 2 / 1e9
    print(f"\nGPU reserved by the engine = max_loras x size(max_lora_rank) = {a.max_loras} x {slot:.3f} GB = "
          f"{a.max_loras * slot:.2f} GB  (taken from the KV-cache budget)")
    print(f"CPU RAM cache              = max_cpu_loras x size(max_lora_rank) = {a.max_cpu_loras} x {slot:.3f} GB = "
          f"{a.max_cpu_loras * slot:.2f} GB")
    print("Rule: set --max-lora-rank to the LARGEST adapter rank you serve, not higher (every slot pays for it).\n")


if __name__ == "__main__":
    main()
