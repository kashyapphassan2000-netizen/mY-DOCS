"""ELITE LAB: measure why GPTQ, rotations (QuIP/SpinQuant) and SmoothQuant exist, on a REAL small model.

Quantizes the same model five ways and measures the damage against the original on held-out text:
  KL(orig || quant)  lower = closer to the original
  top-1 agree        % of positions where both pick the same next token (higher = better)

Recipes:
  RTN W4A16            plain round-to-nearest (the naive baseline)
  GPTQ W4A16           second-order error compensation
  QuIP + GPTQ W4A16    Hadamard rotation first (spreads outlier channels), then GPTQ
  GPTQ W8A8            INT8 weights AND activations, no smoothing
  SmoothQuant + W8A8   move activation outliers into the weights first

Run (GPU: ~5 min; CPU works for tiny models):
  python elite_lab.py --model Qwen/Qwen2.5-0.5B-Instruct
  python elite_lab.py --model HuggingFaceTB/SmolLM2-135M-Instruct --group-size 64   (width 576 needs g64)
"""
import argparse
import copy

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier, SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def w4(group_size: int) -> dict:
    if group_size == 128:
        return {"scheme": "W4A16"}
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    return {"config_groups": {"g": QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        num_bits=4, type="int", symmetric=True, strategy="group", group_size=group_size))}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib-samples", type=int, default=64)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    wiki = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if len(t) > 400]
    calib = tok(wiki[: a.calib_samples], max_length=256, truncation=True)
    from datasets import Dataset
    calib = Dataset.from_dict({"input_ids": calib["input_ids"], "attention_mask": calib["attention_mask"]})
    test = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    eval_ids = tok(test, return_tensors="pt").input_ids[:, :2048].to(dev)

    def load():
        return AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    with torch.no_grad():
        ref_lp = torch.log_softmax(load().to(dev)(eval_ids).logits.float(), -1)

    def damage(model) -> tuple[float, float]:
        with torch.no_grad():
            q_lp = torch.log_softmax(model.to(dev)(eval_ids).logits.float(), -1)
        kl = (ref_lp.exp() * (ref_lp - q_lp)).sum(-1).mean().item()
        agree = (ref_lp.argmax(-1) == q_lp.argmax(-1)).float().mean().item()
        return kl, agree

    g = a.group_size
    recipes = {
        "RTN W4A16": [QuantizationModifier(targets="Linear", ignore=["lm_head"], **w4(g))],
        "GPTQ W4A16": [GPTQModifier(targets="Linear", ignore=["lm_head"], **w4(g))],
        "QuIP + GPTQ W4A16": [QuIPModifier(rotations=["v"], transform_type="random-hadamard"),
                              GPTQModifier(targets="Linear", ignore=["lm_head"], **w4(g))],
        "GPTQ W8A8": [GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])],
        "SmoothQuant + GPTQ W8A8": [SmoothQuantModifier(smoothing_strength=0.8),
                                    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])],
    }
    print(f"\n{'recipe':<26}{'KL(orig||quant)':>17}{'top-1 agree':>13}")
    for name, recipe in recipes.items():
        try:
            model = load()
            oneshot(model=model, dataset=copy.deepcopy(calib), recipe=recipe,
                    num_calibration_samples=a.calib_samples, max_seq_length=256)
            kl, agree = damage(model.eval())
            print(f"{name:<26}{kl:>17.5f}{agree:>13.3f}", flush=True)
        except Exception as e:  # report, do not hide
            print(f"{name:<26}  FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)


if __name__ == "__main__":
    main()
