"""'Hyperparameter tuning' for quantization: try several quantization settings, measure each, keep the best.

Quantization has no learning rate or epochs. Its knobs are:
  method       gptq / awq / quip_gptq / w8a8 / fp8
  group size   128 (small file) -> 64 -> 32 (more accurate, slightly bigger)
  dampening    0.01 -> 0.05 -> 0.1 (stabilises GPTQ; higher helps when you see NaN)
  samples      128 / 256 / 512 / 1024 calibration samples
  seq len      1024 / 2048 / 4096 tokens per sample
  actorder     static / group / weight (order in which GPTQ processes columns)
  ignore       layers kept in 16-bit (lm_head always; add the worst layers from layer_error.py)

Each trial = quantize.py + eval_ppl.py (+ kl_compare.py with --kl). Results -> sweep/results.csv, sorted.

Run:
  python sweep.py --model Qwen/Qwen3-8B --calib data/calib.jsonl --workdir ~/work/qwen3-8b/sweep --kl
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

GRID = {
    "method": ["gptq", "awq"],
    "group_size": [128, 64],
    "dampening": [0.01, 0.05],
    "num_samples": [256, 512],
}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--text", default="wikitext2", help="eval text for perplexity")
    ap.add_argument("--kl", action="store_true", help="also run kl_compare.py (slower, better signal)")
    ap.add_argument("--keep", action="store_true", help="keep every quantized folder (needs disk)")
    a = ap.parse_args()

    wd = Path(os.path.expanduser(a.workdir))
    wd.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    base_ppl = wd / "ppl-original.json"
    if not base_ppl.exists():
        run([py, str(HERE / "eval_ppl.py"), "--model", a.model, "--text", a.text, "--out", str(base_ppl)])
    base = json.loads(base_ppl.read_text())["perplexity"]

    results = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        t = dict(zip(keys, values))
        if t["method"] == "awq" and (t["group_size"] != 128 or t["dampening"] != 0.01):
            continue  # those knobs belong to GPTQ; skip duplicate AWQ trials
        name = f"{t['method']}-g{t['group_size']}-d{t['dampening']}-n{t['num_samples']}"
        out = wd / name
        try:
            cmd = [py, str(HERE / "quantize.py"), "--model", a.model, "--method", t["method"], "--out", str(out),
                   "--group-size", str(t["group_size"]), "--dampening", str(t["dampening"]),
                   "--num-samples", str(t["num_samples"])]
            if a.calib:
                cmd += ["--calib", a.calib]
            run(cmd)
            run([py, str(HERE / "eval_ppl.py"), "--model", str(out), "--text", a.text, "--out", str(out / "ppl.json")])
            row = {"trial": name, **t, "ppl": json.loads((out / "ppl.json").read_text())["perplexity"],
                   "size_gb": json.loads((out / "quant_info.json").read_text())["size_gb"]}
            row["ppl_change_%"] = round(100 * (row["ppl"] / base - 1), 2)
            if a.kl:
                run([py, str(HERE / "kl_compare.py"), "--original", a.model, "--quantized", str(out),
                     "--out", str(out / "kl.json")] + (["--text", a.text] if a.text != "wikitext2" else []))
                kl = json.loads((out / "kl.json").read_text())
                row.update(kl=kl["kl_orig_quant"], top1=kl["top1_agreement_%"])
        except subprocess.CalledProcessError as e:
            row = {"trial": name, **t, "error": f"exit {e.returncode}"}
        results.append(row)
        print(row, flush=True)
        if not a.keep and "error" not in row:
            for f in out.glob("*.safetensors"):
                f.unlink()  # keep json results, drop weights; rebuild the winner with quantize.py

    results.sort(key=lambda r: (r.get("kl", r.get("ppl_change_%", 1e9)) if "error" not in r else 1e9))
    cols = sorted({k for r in results for k in r}, key=lambda k: (k != "trial", k))
    with open(wd / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    print(f"\noriginal ppl = {base}\nbest first -> {wd / 'results.csv'}")
    for r in results[:5]:
        print(r)


if __name__ == "__main__":
    main()
