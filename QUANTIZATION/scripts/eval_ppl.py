"""Perplexity of ANY model folder (original, fp8, gptq, awq, w8a8, nf4) measured the SAME way.

Perplexity = exp(average negative log-likelihood of the next token). Lower = better.
Compare models ONLY when text, window, stride and max-tokens are identical.

Run:
  python eval_ppl.py --model Qwen/Qwen3-8B            --out ~/work/qwen3-8b/eval/ppl-bf16.json
  python eval_ppl.py --model ~/work/qwen3-8b/gptq     --out ~/work/qwen3-8b/eval/ppl-gptq.json
  python eval_ppl.py --model ... --text my_domain.txt   (your own text = more honest for your use case)
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_text(name: str) -> str:
    if name == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])
    return Path(os.path.expanduser(name)).read_text(encoding="utf-8")


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, window: int, stride: int) -> float:
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + window, ids.size(1))
        target_len = end - prev_end            # only score tokens not scored in the previous window
        chunk = ids[:, begin:end].to(model.device)
        labels = chunk.clone()
        labels[:, :-target_len] = -100         # -100 = ignore (context only)
        loss = model(chunk, labels=labels).loss
        scored = (labels[:, 1:] != -100).sum().item()
        nll_sum += loss.float().item() * scored
        n_tokens += scored
        prev_end = end
        if end == ids.size(1):
            break
    return math.exp(nll_sum / n_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="wikitext2")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=40960, help="cap for speed; keep identical across models")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model_path = os.path.expanduser(a.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto",
                                                 device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    ids = tok(load_text(a.text), return_tensors="pt").input_ids[:, : a.max_tokens]
    t0 = time.time()
    ppl = perplexity(model, ids, a.window, a.stride)
    r = {"model": a.model, "text": a.text, "perplexity": round(ppl, 4), "tokens": ids.size(1),
         "window": a.window, "stride": a.stride, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(r, indent=2))
    if a.out:
        out = Path(os.path.expanduser(a.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
