"""Bake ONE LoRA adapter into its 16-bit base model BEFORE quantizing (then quantize the merged folder).

Rule: merge into the bf16 base, never into a 4-bit model. The adapter must be trained on this exact base.

Run:
  python merge_lora.py --base Qwen/Qwen3-8B --adapter ./adapters/support --out ~/work/qwen3-8b/merged
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, a.adapter).eval()
    probe = tok("The most important thing to remember is", return_tensors="pt")
    with torch.no_grad():
        before = peft_model(**probe).logits.float()
    merged = peft_model.merge_and_unload()           # W_new = W + (B @ A) * scale, adapter disappears
    with torch.no_grad():
        after = merged(**probe).logits.float()
    diff = (before - after).abs().max().item()
    print(f"verify: max |logit diff| adapter vs merged = {diff:.3e}  (should be < 1e-2)")
    if diff > 1e-2:
        raise SystemExit("merged model disagrees with adapter model: wrong base or broken adapter")
    out = os.path.expanduser(a.out)
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
