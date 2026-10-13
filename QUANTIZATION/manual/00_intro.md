# QUANTIZATION LAB MANUAL: ZERO TO ELITE

Any open-source model (Hugging Face, Llama, Qwen, Mistral, DeepSeek, your own) -> any bit width (16 / 8 / 4) -> evaluate against the original -> save to Hugging Face -> serve -> Kubernetes + Terraform autoscaling (60-80% rule, up to 50 GPU machines) -> update safely. Works the same for 1B, 70B and 1-trillion-parameter models; Part 7 covers the very large ones.

Assumptions (stated once): you are a complete beginner; your PC runs Windows 11; you rent NVIDIA GPUs by the hour (HyperStack, RunPod or Vast.ai) and use AWS only for the autoscaling part; budget is a few dollars per learning session. Versions used: vLLM 0.29.0, llm-compressor 0.13.0, transformers 5.14.1, PEFT 0.21.0, bitsandbytes 0.50.2, Terraform 1.16, EKS module 20.37. Every script in this manual was syntax-checked; the quantize / perplexity / Terraform steps were executed (CPU tiny-model run and terraform validate); GPU serving steps are written from the official docs and the source manual: if a menu or flag differs, look for the same idea and mark it (verify).

All code files are also in your GitHub repo: github.com/kashyapphassan2000-netizen/mY-DOCS, folder QUANTIZATION (scripts, infra, manual). Copy from there when a box is long.

## How to read every section

| Box | Meaning |
|---|---|
| WHERE / GOAL | The exact place you work, and what the section achieves |
| Step N. | Exactly one action. Do them in order |
| Code box | Copy the whole box and press Enter. Never type the $ or > prompt characters |
| YOU SHOULD SEE | What success looks like. Values in <angle brackets> depend on your machine |
| WATCH OUT | A mistake people often make at that point |
| If you see... table | What to do when the result differs |
| CHECKPOINT | Must all be true before you continue. "If not" gives the most likely cause and fix |
| [n] | Source number; the full list is Part 14 |

## Placeholders used everywhere

Wherever you see one of these CAPITAL_NAMES, replace it with your own value.

| Placeholder | Replace with | Example |
|---|---|---|
| MODEL_ID | Hugging Face id (or local folder) of the ORIGINAL 16-bit model | Qwen/Qwen3-8B |
| SLUG | Short lower-case name for folders | qwen3-8b |
| SERVED_NAME | Name clients send as "model" | qwen3-8b |
| HF_USER | Your Hugging Face user name | karthik-ai |
| VM_IP | Public IP of your rented GPU machine | 203.0.113.45 |
| SSH_PORT | SSH port of the machine (22 on HyperStack; RunPod/Vast show their own) | 22 |
| QLAB | Work folder on the GPU machine | /home/ubuntu/qlab or /workspace/qlab |
| YOUR_EMAIL | Where AWS budget alerts go | you@example.com |
| REPO_BRANCH | Branch of your notes repo that holds the QUANTIZATION folder | main (after you merge) or claude/ecstatic-noether-zkrpxd |

# PART 0 - Start Here

## What it is

LLM quantization stores a language model's weights (the numbers it learned in training) with fewer bits: 16 -> 8 -> 4. The model then needs 2-4x less GPU memory, runs faster, and costs less per answer, while answering almost as well. Deployment turns that quantized model into an always-on HTTP endpoint that many users call at once, with an inference engine (vLLM), monitoring and autoscaling.

## Why people use it, and where

- **Cost**: an 8B model needs ~16 GB at 16-bit but ~6 GB at 4-bit, so it fits a cheaper GPU and serves more users per GPU.
- **Fit**: a 70B model needs 140 GB at 16-bit (two 80 GB GPUs) but ~40 GB at 4-bit (one 48 GB GPU). A 1-trillion-parameter model needs 2 TB at 16-bit but ~550 GB at 4-bit (one 8-GPU node).
- **Speed**: generating each new token re-reads all weights from GPU memory. Fewer bytes = more tokens per second.
- **Privacy and control**: your data never leaves your servers; your own fine-tunes; your own limits.
- **Where**: company chat assistants, RAG over private documents, coding assistants, on-device and laptop apps (GGUF), every company that runs open models behind an OpenAI-compatible API.

## The enterprise flow you will learn (your 13 stages)

| Stage | What happens | Part |
|---|---|---|
| 1 Choose a model | Any Hugging Face / local / your own fine-tuned model | 4.1 |
| 2 Size it | Formulas + script + AI prompt -> GPU type, count, cost | 3, 4.1, 4.2 |
| 3 Rent GPUs | HyperStack, RunPod or Vast.ai | 2.4 |
| 4 Connect from your PC | SSH, VS Code Remote-SSH, WinSCP for files | 2.5 |
| 5 Install | One setup script: 3 Python environments + llama.cpp | 2.6 |
| 6 Baseline | Measure the ORIGINAL model first (you cannot compare without it) | 4.4 |
| 7 Quantize | FP8, GPTQ, AWQ, W8A8, QuIP, NF4, GGUF | 4.5, 5.2-5.6 |
| 8 Tune | The real "hyperparameter tuning" of quantization: sweep the knobs | 5.8 |
| 9 Evaluate and compare | Perplexity, KL, benchmarks, your own questions, quality retention %, speed | 5.7, 7.2 |
| 10 Save | Push to your Hugging Face account with a version tag | 5.9 |
| 11 Serve | New GPU, pull your model from Hugging Face, vLLM | 6.1 |
| 12 Scale | Terraform creates EKS; Kubernetes + KEDA autoscale 1-50 GPUs on the 60-80% rule | 6.3-6.8 |
| 13 Update | New version -> gates -> canary -> promote or roll back | 7.6 |

## The parts of this manual

| Part | What you will do | Time estimate |
|---|---|---|
| 0 Start Here | Understand the goal and the map | 15 min |
| 1 Prerequisites | Check skills, create accounts | 1-2 h |
| 2 Workspace | PC tools, rent a GPU, connect, install, first test | 2-3 h |
| 3 Core idea | The 20% of theory and every formula | 45 min |
| 4 Basics | Size a real model, baseline, FP8, serve, first comparison | 2-3 h |
| 5 Intermediate | Calibration, GPTQ/AWQ/W8A8/NF4/GGUF, full evaluation, tuning sweep, push to HF | 4-6 h |
| 6 Advanced | Serve from HF on fresh GPUs, Terraform EKS, Kubernetes, autoscaling to 50 | 4-6 h |
| 7 Elite | Rotations/SmoothQuant lab, what-changed analysis, accuracy recovery, trillion-scale, update flow | 4-8 h |
| 8-14 | Tools map, maintenance, troubleshooting, projects, cheat sheet, self-test, sources | reference |

## What you will be able to do at the end

- Size ANY model (dense or Mixture-of-Experts, 0.5B to 1T+) from its config.json and choose GPU type, count and cost with numbers.
- Quantize any Hugging Face causal LM to FP8, INT8 (W8A8), INT4 (GPTQ, AWQ, QuIP), NF4 and GGUF, and tune the quantization settings.
- Prove what changed: per-layer weight error, KL divergence, top-1 agreement, perplexity, benchmarks, and quality retention % on your own questions.
- Win back lost accuracy with mixed precision, better calibration and a recovery LoRA trained from the original model.
- Version models on Hugging Face, serve them with vLLM, and run them on Kubernetes with Terraform and autoscaling from 1 to 50 GPUs.
- Update a production model with a canary and roll back in one command.

## Environment map: every place you will work

| Place | What happens there |
|---|---|
| Browser | Hugging Face, GPU provider console (HyperStack / RunPod / Vast.ai), AWS console, GitHub |
| PC: PowerShell + VS Code | SSH into the GPU machine, edit files, tunnels, Terraform and kubectl for AWS |
| PC: WinSCP | Drag-and-drop files between PC and GPU machine |
| GPU machine: VS Code Remote-SSH terminal (bash) | Install, quantize, evaluate, serve, benchmark |
| Hugging Face Hub | Source of original models; your private registry of quantized versions |
| AWS (EKS) | Production: Kubernetes cluster with 1-50 GPU machines, load balancer, autoscaling |

ONE-LINE FLOW:

```
PC (PowerShell/VS Code) -> SSH -> rented GPU (quantize + evaluate) -> Hugging Face (versioned model)
  -> Terraform creates AWS EKS -> Kubernetes runs vLLM pods -> Prometheus metrics -> KEDA scales pods 1-50
  -> cluster-autoscaler adds/removes GPU machines -> users call the load balancer
```

## What each piece does

| Component | Its job |
|---|---|
| Hugging Face Hub | Download original models; store your quantized versions with tags v1, v2 |
| uv | Creates Python environments and installs packages fast |
| llm-compressor [4] | Makes FP8, GPTQ, AWQ, W8A8 (SmoothQuant), QuIP checkpoints in compressed-tensors format |
| bitsandbytes [6] | NF4 4-bit without calibration data |
| llama.cpp [3] | GGUF files (Q8_0, Q4_K_M...) for laptops, CPUs, Macs |
| transformers / PEFT [7][8] | Load models, merge LoRA adapters, train recovery adapters |
| vLLM [1] | High-throughput GPU server with an OpenAI-compatible API and Prometheus metrics |
| lm-evaluation-harness | Standard benchmarks (GSM8K, MMLU, ARC, HellaSwag) against your server |
| Terraform | Creates the AWS network, EKS cluster and GPU machine pool from code |
| Kubernetes (EKS) | Runs and restarts vLLM pods, rolls out updates |
| Prometheus | Collects vLLM metrics (KV-cache usage, queue length) |
| KEDA | Turns those metrics into "how many vLLM pods" (your 60-80% rule) |
| cluster-autoscaler | Adds a GPU machine when a pod has nowhere to run; removes idle ones |

# PART 1 - Prerequisites

| What | Level | Refresher (1-3 lines) | 1-minute check |
|---|---|---|---|
| Basic terminal | MUST | Commands are typed and run with Enter; cd changes folder; ls lists; Ctrl+C stops a program | Open a terminal, run pwd, then cd .. and pwd again: the path got shorter |
| Python basics | MUST | Variables, functions, import, running a .py file | Run: python -c "print(sum(range(10)))" and get 45 |
| Virtual environment | MUST | A private folder with its own Python and libraries so projects cannot break each other | Say why this manual uses 3 of them (vLLM and llm-compressor pin different library versions) |
| git basics | MUST | clone, add, commit, push | Clone any public repo |
| Tokens and context | MUST | Tokens are word pieces (~0.75 words each); context = tokens one request can hold | Estimate tokens in 300 words (about 400) |
| SSH | MUST | Encrypted login to a remote machine with a key pair | ssh to a machine and run hostname |
| Matrix multiply | NICE-TO-HAVE | A layer computes y = W x; W is the matrix we quantize | Shape of (4x3) @ (3x2)? (4x2) |
| Docker / containers | NICE-TO-HAVE | A packaged program; Kubernetes runs containers | docker run hello-world prints a greeting |
| AWS basics | NICE-TO-HAVE | Pay-per-use cloud; regions; IAM users | Name your region (us-east-1 here) |
| LoRA | NICE-TO-HAVE | Small trained add-on matrices on top of a base model | Can an adapter for Llama-3.1-8B be used with Qwen3-8B? (no) |

## Install, buy or sign up

| Item | Official link | Cost | Choose |
|---|---|---|---|
| Hugging Face account + tokens | https://huggingface.co | Free (private storage has limits on free plan, verify) | One READ token, one WRITE token |
| GitHub account | https://github.com | Free | Free personal plan |
| Git for Windows | https://git-scm.com (or winget) | Free | Default options |
| Python 3.12 (PC) | https://www.python.org (or winget) | Free | 3.12 64-bit |
| VS Code + Remote-SSH extension | https://code.visualstudio.com | Free | User installer |
| WinSCP (file transfer) | https://winscp.net | Free | Default install |
| GPU cloud: HyperStack | https://www.hyperstack.cloud | L40 48 GB $1.00/h, A100 80 GB $1.35/h, H100 80 GB $2.50/h on-demand [12] | Full VM; easiest for beginners |
| GPU cloud: RunPod | https://www.runpod.io | L40S 48 GB ~$0.79-1.09/h, H100 PCIe ~$1.99-2.89/h, RTX 4090 ~$0.34/h [31] | Container "pod"; per-second billing |
| GPU cloud: Vast.ai | https://vast.ai | Marketplace: L40S median ~$0.67/h, H100 SXM median ~$2.25/h, RTX 4090 median ~$0.50/h [32] | Cheapest; host quality varies |
| AWS account (Part 6-7) | https://aws.amazon.com | g6e.xlarge (1x L40S) $1.861/h, g6.xlarge (1x L4) $0.8048/h, us-east-1 [13][14] | Enable MFA; create a budget first |
| Terraform (PC) | https://developer.hashicorp.com/terraform/install | Free | Latest 1.x (1.16.5 tested) |
| AWS CLI v2, kubectl, Helm (PC) | aws.amazon.com/cli, kubernetes.io, helm.sh | Free | Windows installers or winget |

PRO TIP: Cheapest path through the whole manual: Parts 0-3 are free. Parts 4-5 on one rented 48 GB GPU (L40/L40S) for a few hours per session (~$5-15). Part 7's elite lab runs on the same GPU. Part 6 on AWS: build it, load-test for one hour, then destroy it (~$5-10 if you destroy it the same day).

WATCH OUT: GPU billing runs every second the machine EXISTS, not only while you type. Stop or delete machines when you finish a session. On AWS, destroy the cluster the same day (Part 6.8).
