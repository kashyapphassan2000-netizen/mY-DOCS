"""Call the base model and every LoRA adapter on ONE server with the same question, side by side.

The adapter is chosen PER REQUEST by the "model" field:
  vLLM    "model": "<adapter name from --lora-modules>"        base = "<served-model-name>"
  SGLang  "model": "<served-model-name>:<adapter name>"        base = "<served-model-name>"

Run:
  python multi_lora_client.py --url http://localhost:8000 --models qwen3-8b pirate json-only \
      --prompt "Explain what a GPU is in 2 sentences."
"""
import argparse
import os
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--models", nargs="+", required=True, help="base name first, then adapter names")
    ap.add_argument("--prompt", default="Explain what a GPU is in 2 sentences.")
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--key-env", default="VLLM_API_KEY")
    a = ap.parse_args()

    key = os.environ.get(a.key_env, "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    listed = httpx.get(f"{a.url}/v1/models", headers=headers, timeout=30).json()
    print("server lists:", [m["id"] for m in listed.get("data", [])])
    for m in a.models:
        t0 = time.perf_counter()
        r = httpx.post(f"{a.url}/v1/chat/completions", headers=headers, timeout=300,
                       json={"model": m, "temperature": 0.0, "max_tokens": a.max_tokens,
                             "messages": [{"role": "user", "content": a.prompt}]})
        took = time.perf_counter() - t0
        if r.status_code != 200:
            print(f"\n[{m}] HTTP {r.status_code}: {r.text[:200]}")
            continue
        print(f"\n[{m}] ({took:.2f}s)\n{r.json()['choices'][0]['message']['content'].strip()}")


if __name__ == "__main__":
    main()
