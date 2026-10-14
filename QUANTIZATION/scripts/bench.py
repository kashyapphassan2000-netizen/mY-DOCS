"""Load-test ANY OpenAI-compatible server (vLLM, SGLang, llama-server, your gateway, a K8s load balancer).

Measures at each concurrency level:
  tok/s      total generated tokens per second (throughput)
  TTFT       time to first token (how long a user waits before text starts), p50 and p95
  ITL        inter-token latency (gap between tokens while streaming), p50
  failed     requests that errored (must be 0)

Run:
  python bench.py --url http://localhost:8000 --model qwen3-8b --label fp8 --concurrency 1 8 32
Results are appended to bench_results.jsonl so you can compare variants later.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time

import httpx

TOPICS = ["databases", "black holes", "Indian history", "GPU memory", "cooking rice", "contract law",
          "football tactics", "photosynthesis", "Kubernetes", "the stock market", "Python decorators"]


def make_prompts(n: int, seed: int = 0) -> list[str]:
    rnd = random.Random(seed)
    return [f"Explain {rnd.choice(TOPICS)} to a new employee in about 200 words. Use one example." for _ in range(n)]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def shared_prefix(words: int) -> str:
    """A long system prompt that EVERY request shares (like a RAG/agent instruction block): tests prefix caching."""
    base = ("You are the support assistant of Example Corp. Follow the policy handbook exactly. "
            "Section {i}: answers must be polite, accurate, cite the handbook section, and never invent facts. ")
    out, i = [], 0
    while sum(len(x.split()) for x in out) < words:
        out.append(base.format(i=i)); i += 1
    return "".join(out)


async def one(client: httpx.AsyncClient, url: str, key: str, model: str, prompt: str, max_tokens: int,
              system: str | None = None) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    t0 = time.perf_counter()
    first, last, gaps, tokens = None, None, [], 0
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body, headers=headers) as r:
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    tokens = obj["usage"].get("completion_tokens", tokens)
                if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    elif last is not None:
                        gaps.append(now - last)
                    last = now
    except Exception as e:  # report, do not hide
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if first is None:
        return {"ok": False, "error": "no tokens received"}
    return {"ok": True, "ttft": first - t0, "gaps": gaps, "tokens": tokens, "total": time.perf_counter() - t0}


async def level(url, key, model, prompts, conc, max_tokens, system=None) -> dict:
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        async def run(p):
            async with sem:
                return await one(client, url, key, model, p, max_tokens, system)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(run(p) for p in prompts))
        wall = time.perf_counter() - t0
    ok = [r for r in res if r["ok"]]
    gaps = [g for r in ok for g in r["gaps"]]
    return {"concurrency": conc, "requests": len(res), "failed": len(res) - len(ok),
            "tok_per_s": round(sum(r["tokens"] for r in ok) / wall, 1),
            "ttft_p50_ms": round(pct([r["ttft"] for r in ok], 50) * 1000, 1),
            "ttft_p95_ms": round(pct([r["ttft"] for r in ok], 95) * 1000, 1),
            "itl_p50_ms": round(statistics.median(gaps) * 1000, 2) if gaps else None,
            "errors": sorted({r["error"] for r in res if not r["ok"]})[:3]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--label", default="run")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--key-env", default="VLLM_API_KEY", help="env var holding the API key")
    ap.add_argument("--shared-prefix-words", type=int, default=0,
                    help="add a shared system prompt of ~N words to every request (prefix-caching test)")
    ap.add_argument("--out", default="bench_results.jsonl")
    a = ap.parse_args()

    key = os.environ.get(a.key_env, "")
    prompts = make_prompts(a.num_prompts)
    print(f"| label | conc | tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | failed |")
    print(f"|---|---|---|---|---|---|---|")
    for c in a.concurrency:
        system = shared_prefix(a.shared_prefix_words) if a.shared_prefix_words else None
        r = asyncio.run(level(a.url.rstrip("/"), key, a.model, prompts, c, a.max_tokens, system))
        r["label"] = a.label
        print(f"| {a.label} | {c} | {r['tok_per_s']} | {r['ttft_p50_ms']} | {r['ttft_p95_ms']} | "
              f"{r['itl_p50_ms']} | {r['failed']} |", flush=True)
        if r["errors"]:
            print("   errors:", r["errors"])
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
