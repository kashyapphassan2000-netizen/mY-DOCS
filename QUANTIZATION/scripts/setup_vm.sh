#!/usr/bin/env bash
# Prepare a fresh rented GPU machine (HyperStack / RunPod / Vast.ai / AWS) for the whole manual.
# Run:  bash setup_vm.sh 2>&1 | tee setup.log      (15-40 minutes; run it inside tmux)
set -euo pipefail

QLAB="${QLAB:-$HOME/qlab}"            # everything lives here; on RunPod/Vast set QLAB=/workspace/qlab first
mkdir -p "$QLAB/envs" "$QLAB/work"
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "== 1. system packages"
$SUDO apt-get update -y
$SUDO apt-get install -y git git-lfs tmux htop build-essential cmake curl wget unzip python3-dev

echo "== 2. GPU check"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo "== 3. uv (fast Python installer)"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"

echo "== 4. quant environment (quantize, evaluate, upload)"
uv venv -p 3.12 "$QLAB/envs/quant"
uv pip install --python "$QLAB/envs/quant/bin/python" \
  "llmcompressor==0.13.0" "transformers==5.14.1" "peft==0.21.0" "bitsandbytes==0.50.2" \
  datasets accelerate huggingface_hub httpx "lm-eval[api]"

echo "== 5. vllm environment (serving)"
uv venv -p 3.12 "$QLAB/envs/vllm"
uv pip install --python "$QLAB/envs/vllm/bin/python" "vllm==0.29.0" httpx

echo "== 6. llama.cpp (GGUF files for laptops / CPU)"
if [ ! -d "$QLAB/llama.cpp" ]; then git clone https://github.com/ggml-org/llama.cpp "$QLAB/llama.cpp"; fi
CUDA_FLAG="-DGGML_CUDA=ON"
if ! command -v nvcc >/dev/null; then
  echo "WARNING: nvcc (CUDA toolkit) not found -> building llama.cpp for CPU only"
  echo "         fix: sudo apt-get install -y nvidia-cuda-toolkit   then rerun this script"
  CUDA_FLAG=""
fi
cmake -S "$QLAB/llama.cpp" -B "$QLAB/llama.cpp/build" $CUDA_FLAG -DCMAKE_BUILD_TYPE=Release
cmake --build "$QLAB/llama.cpp/build" -j "$(nproc)"
uv venv -p 3.12 "$QLAB/envs/gguf"
uv pip install --python "$QLAB/envs/gguf/bin/python" -r "$QLAB/llama.cpp/requirements.txt"

echo "== 7. verify"
"$QLAB/envs/quant/bin/python" -c "import torch, llmcompressor, transformers; print('quant OK  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| llmcompressor', llmcompressor.__version__)"
"$QLAB/envs/vllm/bin/python" -c "import vllm; print('vllm OK ', vllm.__version__)"
ls "$QLAB/llama.cpp/build/bin/llama-server" >/dev/null && echo "llama.cpp OK"
echo "DONE. Activate with: source $QLAB/envs/<quant|vllm|gguf>/bin/activate"
