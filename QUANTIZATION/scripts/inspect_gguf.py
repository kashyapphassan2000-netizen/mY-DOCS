"""Look INSIDE a GGUF file: header, metadata, and how many bytes each quantization type really uses.

GGUF layout (version 3):  magic "GGUF" | version | tensor count | metadata count
                          -> metadata key/values (architecture, context length, tokenizer, chat template...)
                          -> tensor infos (name, shape, type, offset)
                          -> padding to general.alignment (default 32 bytes)
                          -> tensor data (memory-mapped straight from disk by llama.cpp)

Run (gguf environment, which has the gguf package from llama.cpp):
  python inspect_gguf.py $QLAB/work/SLUG/gguf/SLUG-Q4_K_M.gguf
"""
import argparse
from collections import Counter, defaultdict

from gguf import GGUFReader

SKIP = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.merges")


def field_value(f):
    try:
        return f.contents()
    except Exception:  # older gguf versions
        return "<binary>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all-metadata", action="store_true")
    a = ap.parse_args()

    r = GGUFReader(a.path)
    print(f"file: {a.path}")
    print(f"tensors: {len(r.tensors)}   metadata keys: {len(r.fields)}")
    print("\nMETADATA")
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or (name in SKIP and not a.all_metadata):
            continue
        v = field_value(f)
        s = str(v)
        print(f"  {name:<45} {s[:90] + ('...' if len(s) > 90 else '')}")

    counts, nbytes, nweights = Counter(), defaultdict(int), defaultdict(int)
    for t in r.tensors:
        q = t.tensor_type.name
        counts[q] += 1
        nbytes[q] += int(t.n_bytes)
        nweights[q] += int(t.n_elements)
    total_b, total_w = sum(nbytes.values()), sum(nweights.values())
    print("\nTENSOR TYPES (real bits per weight = bytes x 8 / weights)")
    print(f"  {'type':<10}{'tensors':>8}{'weights':>16}{'MB':>10}{'bits/w':>8}")
    for q in sorted(counts, key=lambda k: -nbytes[k]):
        print(f"  {q:<10}{counts[q]:>8}{nweights[q]:>16,}{nbytes[q] / 1e6:>10.1f}{nbytes[q] * 8 / nweights[q]:>8.3f}")
    print(f"\n  TOTAL {total_w:,} weights, {total_b / 1e6:.1f} MB, average {total_b * 8 / total_w:.3f} bits/weight")
    print("\nFIRST 8 TENSORS")
    for t in r.tensors[:8]:
        print(f"  {t.name:<40} {t.tensor_type.name:<8} shape={[int(x) for x in t.shape]}")


if __name__ == "__main__":
    main()
