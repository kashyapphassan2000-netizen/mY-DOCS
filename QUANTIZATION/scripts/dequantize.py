"""Turn an ALREADY-QUANTIZED llm-compressor checkpoint back into a plain bf16 model.

Why: never quantize a quantized model directly (errors stack, packed formats cannot be re-read as numbers).
If you only have a quantized model (e.g. somebody's GPTQ/FP8 upload), unpack it to bf16 first, then quantize
that with quantize.py. The result carries BOTH rounding errors, so always prefer the original bf16 model.

Works for compressed-tensors checkpoints (fp8, gptq, awq, w8a8 made by llm-compressor / vLLM community).
Other formats: GGUF -> use the original HF model; DeepSeek-style block FP8 -> use the vendor's bf16 cast script.

Run:
  python dequantize.py --model RedHatAI/SOME-MODEL-quantized.w4a16 --out ~/work/some-model-bf16
"""
import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src, out = os.path.expanduser(a.model), os.path.expanduser(a.out)
    # 1. load and UNPACK the low-bit weights to bf16 numbers
    qmodel = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, quantization_config=CompressedTensorsConfig(run_compressed=False))
    # 2. build an empty plain bf16 model of the same architecture (config WITHOUT quantization_config)
    cfg = AutoConfig.from_pretrained(src)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    # 3. copy the unpacked numbers in; every weight must find its place
    state = {k: v for k, v in qmodel.state_dict().items() if k in model.state_dict()}
    missing, _ = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    if missing:
        raise SystemExit(f"{len(missing)} weights missing, e.g. {missing[:3]} -> not a supported checkpoint")
    model.tie_weights()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    print(f"bf16 model written to {out}")


if __name__ == "__main__":
    main()
