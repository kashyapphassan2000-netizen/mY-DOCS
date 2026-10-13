"""Quantize ANY Hugging Face causal LM with ONE command.

Methods (llm-compressor 0.13 -> "compressed-tensors" checkpoints that vLLM / SGLang load):
  fp8        FP8 weights + FP8 activations, scales computed at runtime. No calibration data.
  fp8_free   same as fp8 but never loads the model: converts shard by shard (huge models, small RAM).
  gptq       INT4 weights (W4A16), GPTQ error compensation. Needs calibration data.
  awq        INT4 weights (W4A16 asym), AWQ activation-aware scaling. Needs data + NVIDIA GPU.
  w8a8       SmoothQuant + GPTQ INT8 weights AND INT8 activations. Fast on A100/A10 (no FP8).
  quip_gptq  Hadamard rotation (QuIP) + GPTQ W4A16. Elite: spreads outlier channels first.
Method (transformers + bitsandbytes):
  nf4        NF4 4-bit, no calibration data, needs a CUDA GPU. Serve with SGLang.

Examples:
  python quantize.py --model Qwen/Qwen3-8B --method fp8  --out ~/work/qwen3-8b/fp8
  python quantize.py --model Qwen/Qwen3-8B --method gptq --out ~/work/qwen3-8b/gptq --calib data/calib.jsonl
  python quantize.py --model Qwen/Qwen3-8B --method gptq --group-size 64 --dampening 0.05 --out ...
"""
import argparse
import json
import os
import time
from pathlib import Path

CALIBRATED = {"gptq", "awq", "w8a8", "quip_gptq"}


def folder_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def build_calibration(tok, calib: str | None, n: int, max_len: int):
    """Returns a tokenized datasets.Dataset with n samples (input_ids, attention_mask)."""
    from datasets import load_dataset

    if calib:  # your own JSONL: one {"messages": [...]} or {"text": "..."} per line
        ds = load_dataset("json", data_files=calib, split="train")
    else:      # public chat data: fine for general chat, NOT tuned to your task
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    def to_text(ex):
        if ex.get("text"):
            return {"text": ex["text"]}
        if tok.chat_template:
            return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
        return {"text": "\n".join(m["content"] for m in ex["messages"])}

    ds = ds.map(to_text)

    def tokenize(ex):
        return tok(ex["text"], padding=False, max_length=max_len, truncation=True, add_special_tokens=False)

    return ds.map(tokenize, remove_columns=ds.column_names)


def w4_scheme(group_size: int) -> dict:
    """INT4 weight-only scheme. 128 = the standard preset; 32/64 = smaller groups, better accuracy, bigger file."""
    if group_size == 128:
        return {"scheme": "W4A16"}
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    return {"config_groups": {"group_0": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                                 strategy="group", group_size=group_size))}}


def build_recipe(a) -> list:
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    ignore = a.ignore
    if a.method == "fp8":
        return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)]
    if a.method == "gptq":
        return [GPTQModifier(targets="Linear", ignore=ignore, block_size=128,
                             dampening_frac=a.dampening, actorder=a.actorder, **w4_scheme(a.group_size))]
    if a.method == "awq":
        from llmcompressor.modifiers.transform import AWQModifier
        return [AWQModifier(duo_scaling=True),
                QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=ignore)]
    if a.method == "w8a8":
        from llmcompressor.modifiers.transform import SmoothQuantModifier
        return [SmoothQuantModifier(smoothing_strength=a.smoothing),
                GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore, dampening_frac=a.dampening)]
    if a.method == "quip_gptq":
        from llmcompressor.modifiers.transform import QuIPModifier
        return [QuIPModifier(rotations=["v"], transform_type="random-hadamard"),
                GPTQModifier(targets="Linear", ignore=ignore, dampening_frac=a.dampening,
                             **w4_scheme(a.group_size))]
    raise SystemExit(f"unknown method {a.method}")


def quantize_llmcompressor(a, out: Path) -> None:
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    # dtype="auto" keeps the checkpoint's own precision (bf16); the model is loaded in CPU RAM
    # and llm-compressor moves one layer at a time to the GPU while it calibrates.
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto", trust_remote_code=a.trust_remote_code)
    kwargs = {}
    if a.method in CALIBRATED:
        kwargs = dict(dataset=build_calibration(tok, a.calib, a.num_samples, a.max_seq_len),
                      num_calibration_samples=a.num_samples, max_seq_length=a.max_seq_len)
    oneshot(model=model, recipe=build_recipe(a), **kwargs)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)  # writes packed low-bit weights + quantization_config
    tok.save_pretrained(out)


def quantize_fp8_free(a, out: Path) -> None:
    from llmcompressor import model_free_ptq

    model_free_ptq(model_stub=a.model, save_directory=str(out), scheme="FP8_DYNAMIC", ignore=a.ignore)


def quantize_nf4(a, out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_skip_modules=a.ignore)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto",
                                                 trust_remote_code=a.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local folder (always the 16-bit original)")
    ap.add_argument("--method", required=True,
                    choices=["fp8", "fp8_free", "gptq", "awq", "w8a8", "quip_gptq", "nf4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None, help="your JSONL; default = ultrachat_200k")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--group-size", type=int, default=128, help="gptq/quip_gptq: 32, 64 or 128 (must divide every layer width)")
    ap.add_argument("--dampening", type=float, default=0.01, help="gptq/w8a8/quip: 0.01-0.1")
    ap.add_argument("--actorder", default="static", choices=["static", "group", "weight"])
    ap.add_argument("--smoothing", type=float, default=0.8, help="w8a8 only: SmoothQuant strength 0.5-0.9")
    ap.add_argument("--ignore", nargs="*", default=["lm_head"],
                    help="layers kept in 16-bit; MoE: add 're:.*mlp.gate$'")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    t0 = time.time()
    if a.method == "nf4":
        quantize_nf4(a, out)
    elif a.method == "fp8_free":
        quantize_fp8_free(a, out)
    else:
        quantize_llmcompressor(a, out)
    info = {k: v for k, v in vars(a).items()}
    info.update(seconds=round(time.time() - t0, 1), size_gb=round(folder_gb(out), 3))
    (out / "quant_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
