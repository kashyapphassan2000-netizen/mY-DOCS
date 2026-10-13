"""Save a quantized model to YOUR Hugging Face account, with a model card and a version tag.

The Hub is your model registry: every upload is a git commit; tags (v1, v2...) are versions you can
deploy and roll back to (vLLM: --revision v1).

Run (HF_TOKEN must be a WRITE token):
  python push_to_hub.py --folder ~/work/qwen3-8b/gptq --repo YOUR_HF_USER/qwen3-8b-gptq-w4a16 \
      --base Qwen/Qwen3-8B --tag v1 --eval-dir ~/work/qwen3-8b/eval
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
base_model: {base}
tags:
- quantized
- {method}
---
# {repo}

Quantized version of [{base}](https://huggingface.co/{base}).

| Item | Value |
|---|---|
| Method | {method} |
| Settings | {settings} |
| Size on disk | {size} GB |
| Version tag | {tag} |

## Measured quality
{evals}

## Serve
```
vllm serve {repo} --revision {tag} --max-model-len 8192
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--eval-dir", default=None, help="folder with ppl-*.json / summary.json to show in the card")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    folder = Path(os.path.expanduser(a.folder))
    info = json.loads((folder / "quant_info.json").read_text()) if (folder / "quant_info.json").exists() else {}
    evals = "Not measured yet."
    if a.eval_dir:
        lines = ["| File | Result |", "|---|---|"]
        for f in sorted(Path(os.path.expanduser(a.eval_dir)).glob("*.json")):
            d = json.loads(f.read_text())
            keys = ["perplexity", "quality_retention_percent", "kl_orig_quant", "top1_agreement_%"]
            lines.append(f"| {f.name} | " + ", ".join(f"{k}={d[k]}" for k in keys if k in d) + " |")
        evals = "\n".join(lines)
    settings = {k: info[k] for k in ("group_size", "dampening", "num_samples", "max_seq_len", "calib") if k in info}
    (folder / "README.md").write_text(CARD.format(
        base=a.base, repo=a.repo, method=info.get("method", "unknown"), settings=settings,
        size=info.get("size_gb", "?"), tag=a.tag, evals=evals), encoding="utf-8")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, private=not a.public, exist_ok=True)
    commit = api.upload_folder(repo_id=a.repo, folder_path=str(folder),
                               commit_message=f"{info.get('method', 'model')} {a.tag}")
    api.create_tag(a.repo, tag=a.tag, revision=commit.oid, exist_ok=False)
    print(f"uploaded https://huggingface.co/{a.repo}  tag={a.tag}  commit={commit.oid}")


if __name__ == "__main__":
    main()
