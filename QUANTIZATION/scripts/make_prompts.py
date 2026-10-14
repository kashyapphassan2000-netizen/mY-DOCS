"""Generate N varied practice prompts (when you do not have real user prompts yet).

Writes one {"prompt": "..."} per line. Real prompts from your users are always better.
Run:
  python make_prompts.py --n 300 --out data/prompts.jsonl
"""
import argparse
import json
import random

TOPICS = ["password resets", "invoices", "GPU memory", "Python lists", "SQL joins", "cloud costs", "team meetings",
          "customer refunds", "data privacy", "Kubernetes pods", "healthy breakfast", "time management",
          "API rate limits", "machine learning", "electric cars", "exam preparation", "job interviews", "budgets"]
TASKS = ["Explain {t} to a beginner.", "Give three tips about {t}.", "Write a short email about {t}.",
         "What are common mistakes with {t}?", "Summarize the key facts about {t} in 4 bullet points.",
         "Compare two approaches to {t}.", "Write a step-by-step guide for {t}.", "Answer briefly: why does {t} matter?"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for _ in range(a.n):
            f.write(json.dumps({"prompt": rnd.choice(TASKS).format(t=rnd.choice(TOPICS))}) + "\n")
    print(f"wrote {a.n} prompts to {a.out}")


if __name__ == "__main__":
    main()
