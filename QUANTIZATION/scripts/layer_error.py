"""WHAT CHANGED inside the model: compare every quantized weight matrix with the original, layer by layer.

For each Linear layer it prints
  rel_err  = ||W_original - W_quantized|| / ||W_original||   (0.00 = identical; 4-bit is often 0.03-0.12)
  max_abs  = the biggest single-number change
and an average per layer TYPE (q_proj, k_proj, ..., down_proj) and per depth (first / middle / last layers).
Layers with the largest error are your candidates to keep in 16-bit (--ignore) if quality is not good enough.

Works for llm-compressor checkpoints (fp8, gptq, awq, w8a8). Not for nf4 (bitsandbytes) or GGUF.
Not meaningful for quip_gptq (its weights are stored in a rotated basis).

Run (CPU is fine, needs RAM ~2x the bf16 model size):
  python layer_error.py --original Qwen/Qwen3-8B --quantized ~/work/qwen3-8b/gptq --top 15
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, CompressedTensorsConfig


def original_tensors(model: str) -> dict[str, str]:
    """Map tensor name -> safetensors file, without loading anything."""
    if os.path.isdir(os.path.expanduser(model)):
        root = Path(os.path.expanduser(model))
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"]))
    index = {}
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                index[k] = str(f)
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    qdir = os.path.expanduser(a.quantized)
    qcfg = json.loads(Path(qdir, "config.json").read_text()).get("quantization_config", {})
    if qcfg.get("transform_config"):
        print("WARNING: this checkpoint uses rotations; per-weight errors are not comparable.")
    # run_compressed=False -> transformers unpacks the low-bit weights back to bf16 numbers (the values the
    # GPU effectively multiplies with), so we can subtract them from the original.
    qmodel = AutoModelForCausalLM.from_pretrained(qdir, dtype=torch.bfloat16,
                                                  quantization_config=CompressedTensorsConfig(run_compressed=False))
    orig = original_tensors(a.original)

    rows = []
    for name, p in qmodel.named_parameters():
        if not name.endswith(".weight") or p.dim() != 2 or name not in orig:
            continue
        with safe_open(orig[name], framework="pt") as sf:
            w = sf.get_tensor(name).float()
        q = p.detach().float()
        if w.shape != q.shape:
            continue
        diff = w - q
        rel = (diff.norm() / w.norm()).item()
        if rel == 0.0:
            continue  # untouched layer (ignored, e.g. lm_head / embeddings)
        parts = name.split(".")
        layer = int(parts[parts.index("layers") + 1]) if "layers" in parts else -1
        rows.append({"name": name, "type": parts[-2], "layer": layer, "rel_err": rel,
                     "max_abs": diff.abs().max().item()})

    if not rows:
        raise SystemExit("no comparable quantized weights found")
    rows.sort(key=lambda r: r["rel_err"], reverse=True)
    print(f"\nTOP {a.top} MOST CHANGED LAYERS")
    for r in rows[: a.top]:
        print(f"  {r['rel_err']:.4f}  max_abs={r['max_abs']:.4f}  {r['name']}")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["rel_err"])
    print("\nAVERAGE BY LAYER TYPE")
    for t, v in sorted(by_type.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {t:<14} {sum(v)/len(v):.4f}  ({len(v)} layers)")

    n = max(r["layer"] for r in rows) + 1
    thirds = {"first third": [], "middle third": [], "last third": []}
    for r in rows:
        key = "first third" if r["layer"] < n / 3 else ("middle third" if r["layer"] < 2 * n / 3 else "last third")
        thirds[key].append(r["rel_err"])
    print("\nAVERAGE BY DEPTH")
    for k, v in thirds.items():
        if v:
            print(f"  {k:<14} {sum(v)/len(v):.4f}")
    print(f"\nOVERALL mean rel_err = {sum(r['rel_err'] for r in rows)/len(rows):.4f} over {len(rows)} layers")
    if a.out:
        Path(os.path.expanduser(a.out)).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
