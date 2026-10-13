"""HOW the predictions changed: token-level comparison of original vs quantized on YOUR text.

For every position in your text, both models predict the next token. We measure
  KL(orig || quant)  how different the two probability distributions are (0 = identical). Elite labs use this.
  top-1 agreement    % of positions where both models would pick the same next token
  ppl orig / quant   perplexity of each model on the same text
KL catches damage that perplexity hides (a model can keep perplexity but change WHICH answer it prefers).

Run (GPU recommended; models are loaded one after the other, never both at once):
  python kl_compare.py --original Qwen/Qwen3-8B --quantized ~/work/qwen3-8b/gptq --text data/domain.txt
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def chunks(tok, text: str, n: int, length: int) -> list[torch.Tensor]:
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = [ids[i: i + length] for i in range(0, len(ids) - length, length)][:n]
    if not out:
        raise SystemExit("text too short: give at least n * length tokens")
    return out


@torch.no_grad()
def logprobs(model_path: str, seqs: list[torch.Tensor], topk: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Store only the top-k log-probs per position (full vocab x tokens would not fit in RAM)."""
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto",
                                                 device_map="auto" if torch.cuda.is_available() else None).eval()
    res = []
    for s in seqs:
        lp = torch.log_softmax(model(s[None].to(model.device)).logits[0].float(), -1)
        v, i = lp.topk(topk, -1)
        res.append((v.cpu(), i.cpu(), lp.gather(-1, s[1:, None].to(lp.device)).squeeze(-1).cpu()))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--text", default=None, help="a .txt file of YOUR domain text (default: wikitext-2)")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    orig_path, q_path = os.path.expanduser(a.original), os.path.expanduser(a.quantized)
    tok = AutoTokenizer.from_pretrained(orig_path)
    if a.text:
        text = Path(os.path.expanduser(a.text)).read_text(encoding="utf-8")
    else:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    seqs = chunks(tok, text, a.samples, a.length)

    ref = logprobs(orig_path, seqs, a.topk)
    qnt = logprobs(q_path, seqs, a.topk)

    kl_sum, agree, n, nll_o, nll_q = 0.0, 0, 0, 0.0, 0.0
    for (rv, ri, rt), (qv, qi, qt) in zip(ref, qnt):
        # KL over the original's top-k tokens (covers ~all probability mass for k=64)
        q_on_ref = torch.full_like(rv, -30.0)
        for pos in range(ri.size(0)):
            lookup = dict(zip(qi[pos].tolist(), qv[pos].tolist()))
            q_on_ref[pos] = torch.tensor([lookup.get(t, qv[pos, -1].item()) for t in ri[pos].tolist()])
        p = rv.exp()
        kl_sum += (p * (rv - q_on_ref)).sum().item()
        agree += (ri[:, 0] == qi[:, 0]).sum().item()
        n += ri.size(0)
        nll_o -= rt.sum().item()
        nll_q -= qt.sum().item()
    tokens = sum(len(s) - 1 for s in seqs)
    r = {"positions": n, "kl_orig_quant": round(kl_sum / n, 5), "top1_agreement_%": round(100 * agree / n, 2),
         "ppl_original": round(math.exp(nll_o / tokens), 3), "ppl_quantized": round(math.exp(nll_q / tokens), 3)}
    r["ppl_change_%"] = round(100 * (r["ppl_quantized"] / r["ppl_original"] - 1), 2)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(os.path.expanduser(a.out)).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
