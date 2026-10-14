"""Win back accuracy lost in quantization: train a small LoRA adapter ON TOP OF the quantized model,
so it learns to answer like the original (teacher) model. This is distillation-based recovery.

How it works:
  1. The quantized weights W_q are unpacked to bf16 numbers and FROZEN (they never change).
  2. A LoRA adapter (two small matrices per layer) is trained so that W_q + B@A answers like the teacher.
  3. You SERVE the original quantized checkpoint + this adapter (vLLM --enable-lora). Memory cost: tiny.
Because the frozen weights are exactly the values the quantized GPU kernels use, what you train is what you serve.

Also trains ordinary task/style adapters on a plain 16-bit base: use --base instead of --quantized.

Input data: teacher.jsonl from teacher_answers.py.
Run:
  python recovery_lora.py --quantized ~/work/qwen3-8b/gptq --data data/teacher.jsonl \
      --out ~/work/qwen3-8b/recovery-lora --epochs 1 --rank 16
Serve:
  vllm serve ~/work/qwen3-8b/gptq --enable-lora --max-lora-rank 16 \
      --lora-modules recovered=~/work/qwen3-8b/recovery-lora
  (clients send "model": "recovered")
"""
import argparse
import json
import math
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig


def encode(tok, messages: list[dict], max_len: int) -> dict:
    """Tokenize one conversation; only the assistant answer is learned (prompt tokens get label -100)."""
    prompt_ids = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tok.apply_chat_template(messages, tokenize=True)
    if hasattr(prompt_ids, "keys"):  # some tokenizers return a BatchEncoding
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": torch.tensor(full_ids), "labels": torch.tensor(labels[: len(full_ids)])}


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--quantized", help="llm-compressor folder (gptq/awq/fp8/w8a8): recovery adapter")
    src.add_argument("--base", help="plain 16-bit model (HF id or folder): normal task/style adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()

    qdir = os.path.expanduser(a.quantized or a.base)
    tok = AutoTokenizer.from_pretrained(qdir)
    extra = {"quantization_config": CompressedTensorsConfig(run_compressed=False)} if a.quantized else {}
    model = AutoModelForCausalLM.from_pretrained(
        qdir, dtype=torch.bfloat16, device_map="auto" if torch.cuda.is_available() else None, **extra)
    if a.quantized:
        model.hf_quantizer = None      # weights are now plain (frozen) bf16 values of W_q
        model.is_quantized = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    rows = [json.loads(l)["messages"] for l in open(a.data, encoding="utf-8") if l.strip()]
    examples = [encode(tok, m, a.max_len) for m in rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total_steps = math.ceil(len(examples) * a.epochs / a.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total_steps))

    model.train()
    step, running = 0, 0.0
    for epoch in range(a.epochs):
        random.Random(epoch).shuffle(examples)
        for i, ex in enumerate(examples):
            ids = ex["input_ids"][None].to(model.device)
            labels = ex["labels"][None].to(model.device)
            loss = model(input_ids=ids, labels=labels).loss / a.grad_accum
            loss.backward()
            running += loss.item()
            if (i + 1) % a.grad_accum == 0 or i == len(examples) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0 or step == total_steps:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running:.4f}", flush=True)
                running = 0.0
    out = os.path.expanduser(a.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"LoRA adapter saved to {out}")


if __name__ == "__main__":
    main()
