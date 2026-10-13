"""Ask the ORIGINAL model and the QUANTIZED model the same questions, then score both.

Input  questions.jsonl, one per line:  {"id": "q1", "question": "...", "reference": "optional correct answer"}
Output compare/answers.jsonl  (every answer + scores)  and  compare/summary.json

Scores:
  similarity      how close the quantized answer is to the original answer (0-100, text overlap)
  exact_match     only if you gave a reference: normalized answer contains the reference (0/1)
  judge score     1-10 from a judge model that grades each answer against the question (+ reference)
  retention %     quantized judge average / original judge average x 100   <- the headline number
  win/tie/loss    per question: is the quantized answer better, equal or worse than the original

Run (both servers running, see Part 5):
  python compare_answers.py --questions data/questions.jsonl \
      --orig-url http://localhost:8001 --orig-model qwen3-8b-bf16 \
      --quant-url http://localhost:8000 --quant-model qwen3-8b-gptq \
      --judge-url http://localhost:8001 --judge-model qwen3-8b-bf16
The judge should be the original model or a STRONGER one. A model judging itself is biased: treat the
retention % as a screening number and read 20-50 answers yourself before you ship.
"""
import argparse
import concurrent.futures as cf
import difflib
import json
import os
import re
from pathlib import Path

import httpx

JUDGE_PROMPT = """You are a strict grader. Grade the ANSWER to the QUESTION from 1 to 10.
10 = fully correct, complete and clear. 5 = partly correct or missing important parts. 1 = wrong or useless.
{ref}
QUESTION:
{q}

ANSWER:
{a}

Reply with ONLY a JSON object like {{"score": 7, "reason": "one short sentence"}}."""


def chat(url: str, model: str, key: str, prompt: str, max_tokens: int) -> str:
    r = httpx.post(f"{url.rstrip('/')}/v1/chat/completions",
                   headers={"Authorization": f"Bearer {key}"} if key else {},
                   json={"model": model, "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.0, "max_tokens": max_tokens},
                   timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def strip_think(text: str) -> str:
    """Reasoning models (Qwen3, DeepSeek-R1) put hidden thinking in <think>...</think>; grade the final answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def judge(a, question: str, answer: str, reference: str | None) -> tuple[int | None, str]:
    ref = f"REFERENCE (correct answer):\n{reference}\n" if reference else ""
    raw = chat(a.judge_url, a.judge_model, os.environ.get(a.judge_key_env, ""),
               JUDGE_PROMPT.format(ref=ref, q=question, a=answer), 200)
    m = re.search(r"\{.*\}", strip_think(raw), flags=re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
        return int(obj.get("score")), str(obj.get("reason", ""))
    except (ValueError, TypeError):
        return None, raw[:200]


def process(a, item: dict) -> dict:
    q, ref = item["question"], item.get("reference")
    orig = strip_think(chat(a.orig_url, a.orig_model, os.environ.get(a.orig_key_env, ""), q, a.max_tokens))
    quant = strip_think(chat(a.quant_url, a.quant_model, os.environ.get(a.quant_key_env, ""), q, a.max_tokens))
    row = {"id": item.get("id"), "question": q, "reference": ref, "original": orig, "quantized": quant,
           "similarity": round(difflib.SequenceMatcher(None, orig, quant).ratio() * 100, 1)}
    if ref:
        row["em_original"] = int(norm(ref) in norm(orig))
        row["em_quantized"] = int(norm(ref) in norm(quant))
    if a.judge_url:
        row["judge_original"], row["why_original"] = judge(a, q, orig, ref)
        row["judge_quantized"], row["why_quantized"] = judge(a, q, quant, ref)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--orig-url", required=True)
    ap.add_argument("--orig-model", required=True)
    ap.add_argument("--quant-url", required=True)
    ap.add_argument("--quant-model", required=True)
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--orig-key-env", default="VLLM_API_KEY")
    ap.add_argument("--quant-key-env", default="VLLM_API_KEY")
    ap.add_argument("--judge-key-env", default="VLLM_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="compare")
    a = ap.parse_args()

    items = [json.loads(l) for l in Path(a.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda it: process(a, it), items))
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    s = {"questions": len(rows),
         "avg_similarity_to_original": round(sum(r["similarity"] for r in rows) / len(rows), 1)}
    if any("em_original" in r for r in rows):
        em = [r for r in rows if "em_original" in r]
        s["exact_match_original_%"] = round(100 * sum(r["em_original"] for r in em) / len(em), 1)
        s["exact_match_quantized_%"] = round(100 * sum(r["em_quantized"] for r in em) / len(em), 1)
    judged = [r for r in rows if r.get("judge_original") and r.get("judge_quantized")]
    if judged:
        jo = sum(r["judge_original"] for r in judged) / len(judged)
        jq = sum(r["judge_quantized"] for r in judged) / len(judged)
        s.update(judge_avg_original=round(jo, 2), judge_avg_quantized=round(jq, 2),
                 quality_retention_percent=round(100 * jq / jo, 1),
                 win=sum(r["judge_quantized"] > r["judge_original"] for r in judged),
                 tie=sum(r["judge_quantized"] == r["judge_original"] for r in judged),
                 loss=sum(r["judge_quantized"] < r["judge_original"] for r in judged),
                 worst_5_ids=[r["id"] for r in sorted(judged, key=lambda r: r["judge_quantized"] - r["judge_original"])[:5]])
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
