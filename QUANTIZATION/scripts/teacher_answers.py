"""Make training data for accuracy recovery: the ORIGINAL (teacher) model answers your prompts.

Input  prompts.jsonl   one {"prompt": "..."} per line (real user questions from your domain, 500-5000 lines)
Output teacher.jsonl   one {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} per line

Run (original bf16 model served on port 8001):
  python teacher_answers.py --prompts data/prompts.jsonl --url http://localhost:8001 --model qwen3-8b-bf16 \
      --out data/teacher.jsonl
"""
import argparse
import concurrent.futures as cf
import json
import os
import re

import httpx


def answer(url: str, model: str, key: str, prompt: str, max_tokens: int) -> str:
    r = httpx.post(f"{url.rstrip('/')}/v1/chat/completions",
                   headers={"Authorization": f"Bearer {key}"} if key else {},
                   json={"model": model, "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.0, "max_tokens": max_tokens}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-env", default="VLLM_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-think", action="store_true",
                    help="keep <think>...</think> reasoning (default: train on the final answer only)")
    a = ap.parse_args()

    key = os.environ.get(a.key_env, "")
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts, encoding="utf-8") if l.strip()]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        answers = list(ex.map(lambda p: answer(a.url, a.model, key, p, a.max_tokens), prompts))
    if not a.keep_think:
        answers = [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip() for t in answers]
    with open(a.out, "w", encoding="utf-8") as f:
        for p, ans in zip(prompts, answers):
            f.write(json.dumps({"messages": [{"role": "user", "content": p},
                                             {"role": "assistant", "content": ans}]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} teacher examples to {a.out}")


if __name__ == "__main__":
    main()
