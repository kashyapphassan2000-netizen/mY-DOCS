"""Collect EVERY measurement for one model into a single comparison report: ORIGINAL vs each QUANTIZED variant.

Reads (whatever exists) from the model's work folder:
  <variant>/quant_info.json          size, method, settings
  eval/ppl-<variant>.json            perplexity          (eval_ppl.py)
  eval/kl-<variant>.json             KL + top-1 agree    (kl_compare.py)
  eval/lmeval-<variant>/**.json      benchmark scores    (lm_eval --output_path)
  compare-<variant>/summary.json     quality retention % (compare_answers.py)
  bench_results.jsonl                speed per label     (bench.py)
Writes REPORT.md next to them and prints it.

Run:
  python report.py --workdir $QLAB/work/qwen3-8b --variants bf16 fp8 gptq awq w8a8
"""
import argparse
import glob
import json
import os
from pathlib import Path


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def lmeval_scores(folder: Path) -> dict:
    out = {}
    for f in glob.glob(str(folder / "**" / "results*.json"), recursive=True):
        for task, m in json.load(open(f)).get("results", {}).items():
            for key in ("exact_match,strict-match", "acc_norm,none", "acc,none", "exact_match,flexible-extract"):
                if key in m:
                    out[task] = round(100 * m[key], 1)
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--variants", nargs="+", required=True, help="first one = the ORIGINAL (e.g. bf16)")
    a = ap.parse_args()

    wd = Path(os.path.expanduser(a.workdir)).resolve()
    bench = [json.loads(l) for l in open(wd / "bench_results.jsonl")] if (wd / "bench_results.jsonl").exists() else []
    rows, tasks = [], set()
    for v in a.variants:
        r = {"variant": v,
             "size_gb": load(wd / v / "quant_info.json").get("size_gb", ""),
             "ppl": load(wd / "eval" / f"ppl-{v}.json").get("perplexity"),
             **{k: load(wd / "eval" / f"kl-{v}.json").get(k) for k in ("kl_orig_quant", "top1_agreement_%")},
             "retention_%": load(wd / f"compare-{v}" / "summary.json").get("quality_retention_percent"),
             "bench": lmeval_scores(wd / "eval" / f"lmeval-{v}")}
        tasks |= set(r["bench"])
        speed = [b for b in bench if b.get("label") == v]
        r["tok_s"] = {b["concurrency"]: b["tok_per_s"] for b in speed}
        r["ttft_p95"] = {b["concurrency"]: b["ttft_p95_ms"] for b in speed}
        rows.append(r)

    base = rows[0]
    tasks = sorted(tasks)
    head = ["variant", "size GB", "ppl", "ppl change %", "KL", "top-1 %", "retention %"] + tasks + ["tok/s", "TTFT p95 ms"]
    lines = [f"# Comparison report: {wd.name}", "", f"Original = {base['variant']}", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        change = (f"{100 * (r['ppl'] / base['ppl'] - 1):+.2f}" if r["ppl"] and base["ppl"] else "")
        cells = [r["variant"], r["size_gb"], r["ppl"], change, r["kl_orig_quant"], r["top1_agreement_%"],
                 r["retention_%"]]
        for t in tasks:
            s, b = r["bench"].get(t), base["bench"].get(t)
            cells.append(f"{s} ({s - b:+.1f})" if s is not None and b is not None and r is not base else s)
        cells += [" / ".join(f"{c}:{v}" for c, v in sorted(r["tok_s"].items())),
                  " / ".join(f"{c}:{v}" for c, v in sorted(r["ttft_p95"].items()))]
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    lines += ["", "tok/s and TTFT are shown as concurrency:value.",
              "Decision rule: pick the SMALLEST/FASTEST variant that passes EVERY gate in GATES.md."]
    text = "\n".join(lines) + "\n"
    (wd / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
