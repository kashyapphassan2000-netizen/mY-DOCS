"""Read speculative-decoding counters from a running server's /metrics and compute the acceptance rate.

acceptance rate a   = accepted draft tokens / proposed draft tokens
mean accepted len   = accepted tokens / number of draft rounds  (+1 bonus token per round)
Expected tokens per target pass E = (1 - a^(k+1)) / (1 - a)   (k = num_speculative_tokens)

Run AFTER a benchmark (counters are cumulative since server start):
  python spec_metrics.py --url http://localhost:8000 --k 4
"""
import argparse
import re
from collections import defaultdict

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--k", type=int, default=4, help="num_speculative_tokens you configured")
    a = ap.parse_args()

    text = httpx.get(f"{a.url}/metrics", timeout=30).text
    totals = defaultdict(float)
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if m and "per_pos" not in m.group(1):
            totals[m.group(1)] += float(m.group(3))
    if not totals:
        raise SystemExit("no spec_decode metrics found: is speculative decoding enabled on this server?")
    for k, v in sorted(totals.items()):
        print(f"{k:<60} {v:,.0f}")
    acc = next((v for k, v in totals.items() if "accepted_tokens" in k), None)
    drafted = next((v for k, v in totals.items() if "draft_tokens" in k), None)
    rounds = next((v for k, v in totals.items() if k.endswith("num_drafts_total") or k.endswith("num_drafts")), None)
    if acc is not None and drafted:
        rate = acc / drafted
        e = (1 - rate ** (a.k + 1)) / (1 - rate) if rate < 1 else a.k + 1
        print(f"\nacceptance rate a = {rate:.3f}")
        if rounds:
            print(f"mean accepted tokens per round = {acc / rounds:.2f} (+1 bonus token)")
        print(f"theory: tokens per target pass E = {e:.2f}  (k = {a.k})")


if __name__ == "__main__":
    main()
