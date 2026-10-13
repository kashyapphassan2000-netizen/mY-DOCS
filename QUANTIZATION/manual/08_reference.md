# PART 8 - Tools and Resources Map (researched October 2026)

Prices change weekly: treat them as a snapshot and check the provider page before renting.

| Tool | What it does | Free / Open-source / Paid | Cost | Who uses it | Platform | Link | Best for | Verdict |
|---|---|---|---|---|---|---|---|---|
| vLLM | High-throughput GPU server, OpenAI API, multi-LoRA, metrics | Open source | Free | Industry-wide, most hosted-API providers | Linux, NVIDIA/AMD | github.com/vllm-project/vllm [1] | Default GPU serving | Start here |
| SGLang | GPU server with RadixAttention prefix caching, EAGLE | Open source | Free | Industry and research labs | Linux | github.com/sgl-project/sglang [2] | Agents/RAG with shared prefixes; NF4 | Benchmark against vLLM |
| llama.cpp | GGUF conversion, k-quants, imatrix, CPU/GPU/Metal server | Open source (MIT) | Free | Hobbyists to companies shipping on-device | Everything | github.com/ggml-org/llama.cpp [3] | CPU, Mac, laptops, offline | Essential |
| llm-compressor | FP8, GPTQ, AWQ, SmoothQuant, QuIP recipes -> compressed-tensors | Open source | Free | Red Hat AI / vLLM community | Linux, CUDA | github.com/vllm-project/llm-compressor [4] | Quantizing for vLLM/SGLang | Main quantizer of this manual |
| GPTQModel | Maintained successor of AutoGPTQ | Open source | Free | Community model publishers | Linux, CUDA | github.com/ModelCloud/GPTQModel [5] | Classic GPTQ checkpoints | Optional |
| bitsandbytes | NF4 / INT8 loading, QLoRA | Open source (MIT) | Free | Fine-tuners, researchers | Linux, Windows | github.com/bitsandbytes-foundation/bitsandbytes [6] | QLoRA training, quick 4-bit | Train with it, serve elsewhere |
| PEFT | LoRA adapters: train, load, merge | Open source (Apache-2.0) | Free | Everyone fine-tuning | Everything | github.com/huggingface/peft [7] | Adapters, recovery LoRA | Essential |
| transformers | Model loading, tokenizers, quantization configs | Open source (Apache-2.0) | Free | Everyone | Everything | github.com/huggingface/transformers [8] | Loading and evaluation | Essential |
| lm-evaluation-harness | Standard benchmarks (GSM8K, MMLU, ARC...) | Open source (MIT) | Free | Open LLM leaderboards, labs | Everything | github.com/EleutherAI/lm-evaluation-harness [33] | Comparable benchmark numbers | Essential |
| uv | Fast Python environments | Open source | Free | Growing industry standard | Everything | github.com/astral-sh/uv [9] | Clean separate envs | Use it |
| Terraform | Infrastructure as code | Source-available (BSL) | Free CLI | Industry standard | Everything | developer.hashicorp.com/terraform | AWS/GCP/Azure from code | Essential |
| KEDA | Event/metric-driven autoscaling for Kubernetes | Open source (CNCF) | Free | Microsoft, many SaaS companies | Kubernetes | github.com/kedacore/keda [34] | Scaling on Prometheus metrics | Use it |
| kube-prometheus-stack | Prometheus + Grafana + operator | Open source | Free | Industry standard | Kubernetes | github.com/prometheus-community/helm-charts [35] | Metrics and dashboards | Use it |
| HyperStack | GPU VMs by the hour | Paid | L40 $1.00/h, A100 80GB $1.35/h, H100 $2.50/h [12] | Startups, researchers | Cloud | hyperstack.cloud | Learning, full VMs | Good value |
| RunPod | GPU pods, per-second billing | Paid | L40S ~$0.79-1.09/h, H100 PCIe ~$1.99-2.89/h, RTX 4090 ~$0.34/h [31] | Indie devs, startups | Cloud | runpod.io | Cheap quick experiments | Good; use the volume |
| Vast.ai | GPU marketplace | Paid | L40S median ~$0.67/h, H100 SXM median ~$2.25/h [32] | Hobbyists, researchers | Cloud | vast.ai | Lowest price | Check host reliability |
| AWS EC2 GPU + EKS | GPU instances, Kubernetes, load balancers | Paid | g6.xlarge $0.8048/h, g6e.xlarge $1.861/h, EKS $0.10/h [13][14] | Industry | Cloud | aws.amazon.com | Production autoscaling | Budget alert first |

## Recommended setups

| Setup | What | Cost |
|---|---|---|
| Best free | PC + Part 2.7 tiny model on a free/cheap GPU credit, GGUF on your PC CPU, the elite lab on a 135M model on CPU (works, tested) | $0 |
| Best paid | H100/H200 machine for quantization + vLLM; AWS EKS with g6e GPUs autoscaled behind a gateway | Tens to hundreds of $ per day |
| Best within a learning budget | One L40/L40S 48 GB for a few hours per session (~$1/h); AWS only for a 1-hour Part 6 run, then destroy | ~$5-15 per session |

## Repositories worth studying

| Repository | Why |
|---|---|
| vllm-project/llm-compressor: examples/ | Reference recipes for every scheme, kept in sync with vLLM |
| ggml-org/llama.cpp: tools/quantize and tools/imatrix | How k-quants and importance matrices really work |
| vllm-project/vllm: vllm/model_executor/layers/quantization | How each format becomes a GPU kernel; where capability checks live |
| EleutherAI/lm-evaluation-harness: lm_eval/tasks | Exactly how each benchmark is scored |
| huggingface/peft: src/peft/tuners/lora | What merge_and_unload actually computes |

# PART 9 - Keep It Working

## SECTION 9.1 - Save and back up what you built

WHERE: GPU machine terminal
GOAL: Settings and results are in git; weights are on Hugging Face; nothing important lives only on a rented disk.

**Step 1.** Copy your results into your notes repo. Type exactly this, then press Enter:

```
mkdir -p $QLAB/mY-DOCS/QUANTIZATION/results/SLUG && cp $QLAB/work/SLUG/{GATES.md,REPORT.md,bench_results.jsonl} $QLAB/mY-DOCS/QUANTIZATION/results/SLUG/ 2>/dev/null; cp -r $QLAB/work/SLUG/eval $QLAB/mY-DOCS/QUANTIZATION/results/SLUG/
```

**Step 2.** Type exactly this, then press Enter:

```
cd $QLAB/mY-DOCS && git add QUANTIZATION/results && git commit -m "results: SLUG" && git push
```

WATCH OUT: Never git-add model folders (*.safetensors, *.gguf). Weights go to Hugging Face (5.9). git push needs your GitHub login (use a personal access token as the password).

CHECKPOINT: The results folder is on GitHub; every shipped variant has a tag on Hugging Face.

## 9.2 Restart after a machine reboot

| What stopped | Restart with |
|---|---|
| vLLM server | The tmux + vllm serve command from 4.7 or 6.1 |
| Environment variables | source ~/.bashrc (loads QLAB, S and the .env secrets) |
| Kubernetes pods | Nothing: Kubernetes restarts them; check kubectl get pods -n llm |

## 9.3 Update safely

- Change ONE version at a time (e.g. vLLM 0.29 -> 0.30) in a NEW environment; rerun Part 2.7, then your gates (5.7) with the same settings.
- Rebuild llama.cpp after `git pull`: `cmake --build $QLAB/llama.cpp/build -j`.
- After any engine, driver or model change: rerun eval + bench and compare with REPORT.md before switching production.
- In Kubernetes, change the image tag (vllm/vllm-openai:vX) in vllm.yaml and roll out with a canary (7.7).

## 9.4 Undo and roll back

| Situation | Undo |
|---|---|
| Bad quantized variant | Delete $QLAB/work/SLUG/<variant> and rebuild; the original is untouched |
| Bad production model | kubectl rollout undo deployment/vllm -n llm (previous --revision comes back) |
| Bad config edit | git checkout -- <file> |
| Runaway AWS spend | kubectl delete -f infra/k8s/ then terraform destroy (6.8) |
| Leaked token | Hugging Face Settings > Access Tokens > Invalidate; new token in $QLAB/.env and the Kubernetes secret |

# PART 10 - Troubleshooting

| Symptom | Likely cause | Exact fix (steps) |
|---|---|---|
| ValueError: weight column counts not divisible by group_size | Layer width not a multiple of the group size (seen in testing: width 576 with group 128) | 1. --group-size 64 or 32  2. or add the layer to --ignore |
| Process killed during quantization, no traceback | Linux OOM killer: bf16 model does not fit CPU RAM | 1. dmesg / tail  2. bigger-RAM machine  3. FP8: --method fp8_free |
| Absurd perplexity (millions) for quip_gptq | transformers does not apply the stored rotation | Evaluate through vLLM (5.7 C/D/E) |
| 'Found no NVIDIA driver' during AWQ | llm-compressor AWQ pins memory; needs CUDA | Run on the GPU machine; nvidia-smi must work |
| CUDA out of memory when starting vLLM | Weights + KV + overhead exceed VRAM | 1. size_model.py for your GPU  2. lower --max-model-len  3. --kv-cache-dtype fp8  4. 4-bit variant |
| vLLM cannot load NF4 or GGUF | Removed / not the right engine | NF4: SGLang; GGUF: llama.cpp |
| 401 Unauthorized | Missing or different API key / HF token | 1. source ~/.bashrc  2. same key on client and server  3. HF: licence accepted with the same account |
| llama.cpp: unknown pre-tokenizer hash | llama.cpp older than the model | 1. git pull  2. rebuild  3. convert again |
| Autoscaler scales to max with no traffic | Scaling on raw VRAM % | Use vllm:kv_cache_usage_perc (6.2) |
| Autoscaler never adds machines | No GPU quota, metric missing, or wrong Prometheus address | 1. Service Quotas  2. Prometheus query vllm:kv_cache_usage_perc  3. kubectl describe scaledobject -n llm |
| terraform destroy hangs | A Kubernetes load balancer still exists | Delete Kubernetes objects first (6.8 Step 5) |
| Quantized model answers in the wrong language / format | Calibration data unlike real traffic, or broken chat template | Recalibrate on real prompts (5.1); check the tokenizer folder was saved |

# PART 11 - Practice Projects

## Easy: Laptop assistant

- **Goal:** a 1-3B model running on your PC CPU as an OpenAI-compatible endpoint.
- **Requirements:** size_model.py output for your PC RAM; GGUF Q4_K_M with imatrix; llama-server with an API key; bench.py at concurrency 1 and 2.
- **Success:** failed = 0; a Python client gets answers; results committed to your repo.

## Medium: Choose the cheapest GPU that passes

- **Goal:** for one 7-14B model, prove which variant and GPU give the lowest cost per million tokens while passing your gates.
- **Requirements:** your own calibration data; bf16, FP8, GPTQ, AWQ variants; perplexity + KL + 50 own questions + GSM8K; bench at 1/16/32; cost per 1M tokens computed.
- **Success:** a filled REPORT.md, gates written before measuring, a two-sentence decision with numbers.

## Hard: Multi-tenant autoscaled service with recovery

- **Goal:** a 4-bit model + recovery LoRA on EKS behind a load balancer, autoscaling 1-10 GPUs on your rule, with a canary update.
- **Requirements:** recovery LoRA that raises retention % measurably; Terraform cluster; KEDA band 60-80%; load test that triggers scale-out; v2 canary and a rollback.
- **Success:** replicas rise under load and TTFT p95 recovers; rollback demonstrated; terraform destroy leaves no instances.

# PART 12 - Cheat Sheet

```
PART 2 SETUP (PC PowerShell)
  ssh-keygen -t ed25519 -C "qlab" -f $HOME\.ssh\qlab ; ssh gpu hostname ; ssh -N -L 8000:localhost:8000 gpu
PART 2 SETUP (GPU machine)
  export QLAB=$HOME/qlab (RunPod/Vast: /workspace/qlab) ; S=$QLAB/mY-DOCS/QUANTIZATION/scripts
  bash $S/setup_vm.sh 2>&1 | tee $QLAB/setup.log ; source $QLAB/envs/<quant|vllm|gguf>/bin/activate
  tmux new -s NAME ; Ctrl+B then D (detach) ; tmux attach -t NAME ; tmux kill-session -t NAME

PART 3 FORMULAS
  weights_GB = quantized_params x bits/8/1e9 + (embeddings + lm_head) x 2/1e9
  bits: bf16 16 | fp8 8 | int8 8 | w4a16 g128 4.125 (g64 4.25, g32 4.5) | nf4 ~4.13 | Q4_K_M ~4.85 | Q8_0 8.5
  KV_bytes_per_token = 2 x layers x kv_heads x head_dim x (2 bf16 | 1 fp8)
  per GPU: weights/TP + KV_total/min(TP, kv_heads) + 2 GB <= 0.90 x VRAM ; TP divides heads
  INT4 asym: scale = (max-min)/15, zp = round(-min/scale), q = round(w/scale)+zp, error <= scale/2
  single-user tok/s ~ bandwidth / weights_GB ; cost per 1M tokens = $/hour / (tok/s x 3600) x 1e6
  quality retention % = quantized judge avg / original judge avg x 100

PART 4 BASICS
  python $S/size_model.py --model MODEL_ID --context 8192 --users 16 [--kv-fp8]
  python $S/eval_ppl.py --model DIR --out eval/ppl-V.json
  python $S/quantize.py --model MODEL_ID --method fp8 --out $QLAB/work/SLUG/fp8
  vllm serve DIR --served-model-name N --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key $VLLM_API_KEY
  python $S/bench.py --url http://localhost:8000 --model N --label V --concurrency 1 8 32

PART 5 INTERMEDIATE
  quantize.py --method gptq|awq|w8a8|quip_gptq|nf4|fp8_free --calib data/calib.jsonl --num-samples 512
              [--group-size 64] [--dampening 0.05] [--actorder group] [--ignore lm_head "re:..."]
  merge_lora.py --base MODEL_ID --adapter PATH --out merged
  GGUF: convert_hf_to_gguf.py DIR --outtype bf16 ; llama-imatrix -m F -f calib.txt -o imatrix.gguf ;
        llama-quantize --imatrix imatrix.gguf IN.gguf OUT.gguf Q4_K_M ; llama-server -m F -ngl 99 --jinja
  kl_compare.py --original MODEL_ID --quantized DIR ; compare_answers.py ... ; report.py --variants bf16 fp8 gptq
  lm_eval --model local-completions --model_args model=N,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=MODEL_ID --tasks gsm8k,arc_challenge,hellaswag
  sweep.py --model MODEL_ID --calib data/calib.jsonl --workdir sweep --kl
  push_to_hub.py --folder DIR --repo HF_USER/NAME --base MODEL_ID --tag v1

PART 6 ADVANCED
  vllm serve HF_USER/NAME --revision v1 ... ; curl localhost:8000/metrics | grep vllm:kv_cache_usage_perc
  terraform init ; terraform plan -out tfplan ; terraform apply tfplan ; terraform destroy
  aws eks update-kubeconfig --name llm-serving --region us-east-1
  helm upgrade -i nvdp | kube-prometheus-stack | keda | cluster-autoscaler (see 6.5)
  kubectl apply -f vllm.yaml ; kubectl apply -f scaledobject.yaml ; kubectl get hpa -n llm -w
  scale rule: avg(vllm:kv_cache_usage_perc) target 0.70 (band 63-77%), queue 5/pod, min 1, max 50

PART 7 ELITE
  python $S/elite_lab.py --model Qwen/Qwen2.5-0.5B-Instruct
  python $S/layer_error.py --original MODEL_ID --quantized DIR --top 15
  quantize.py ... --ignore lm_head "re:.*layers\.(0|1)\..*" (mixed precision)
  teacher_answers.py -> recovery_lora.py -> vllm serve DIR --enable-lora --lora-modules recovered=ADAPTER
  dequantize.py --model QUANTIZED --out bf16 ; then quantize.py on that
  canary: kubectl apply -f canary.yaml ; promote: edit tag + kubectl apply ; rollback: kubectl rollout undo deployment/vllm -n llm

KEY FACTS
  Never quantize twice | merge 1 adapter, multiplex 2+ | FP8 native from cc 8.9, INT8 W8A8 from 7.5
  vLLM 0.29: no bitsandbytes, no GGUF loader | AWQ (llm-compressor 0.13) needs CUDA
  group size must divide every layer width | calibration data decides GPTQ/AWQ quality
  vLLM pre-allocates 90% VRAM -> never autoscale on raw VRAM %
```

# PART 13 - Self-Test

| # | Question |
|---|---|
| 1 | What happens if you set your autoscaler to "add a machine when nvidia-smi VRAM > 78%" for vLLM? |
| 2 | Do this: compute bf16 weight memory for a 14B dense model, and its size at W4A16 g128. |
| 3 | What happens to the maximum number of concurrent users if you halve --max-model-len, all else equal? |
| 4 | Do this: Qwen3-8B KV per token is 0.147 MB in bf16. How many GB of KV do 64 users x 4096 tokens need with --kv-cache-dtype fp8? |
| 5 | What happens if you GPTQ-quantize a model whose hidden size is 576 with group size 128? |
| 6 | You only have somebody's GPTQ 4-bit upload and want FP8. What exact steps do you take, and what is the cost? |
| 7 | Do this: with HPA target 0.70 and 4 pods at 0.80 average KV usage, how many pods does the HPA ask for? |
| 8 | Why does SmoothQuant help INT8 W8A8 but matter less for W4A16? |
| 9 | Perplexity of your 4-bit model is only +1.5%, but GSM8K dropped 9 points. What do you do next? |
| 10 | What happens if your calibration data is generic web text but production traffic is legal contracts? |

# PART 14 - Sources

| [n] | Source |
|---|---|
| 1 | vLLM - https://github.com/vllm-project/vllm ; Docker image vllm/vllm-openai:v0.29.0 |
| 2 | SGLang - https://github.com/sgl-project/sglang |
| 3 | llama.cpp - https://github.com/ggml-org/llama.cpp |
| 4 | llm-compressor - https://github.com/vllm-project/llm-compressor (0.13.0, APIs tested in this manual) |
| 5 | GPTQModel - https://github.com/ModelCloud/GPTQModel |
| 6 | bitsandbytes - https://github.com/bitsandbytes-foundation/bitsandbytes |
| 7 | PEFT - https://github.com/huggingface/peft (0.21.0 tested) |
| 8 | transformers - https://github.com/huggingface/transformers (5.14.1 tested) |
| 9 | uv - https://docs.astral.sh/uv |
| 11 | NVIDIA CUDA GPU compute capability - https://developer.nvidia.com/cuda-gpus |
| 12 | HyperStack pricing - https://www.hyperstack.cloud/gpu-pricing (checked 2026-10-13) |
| 13 | AWS g6.xlarge price - https://instances.vantage.sh/aws/ec2/g6.xlarge |
| 14 | AWS g6e.xlarge price - https://instances.vantage.sh/aws/ec2/g6e.xlarge (verify) |
| 20 | Frantar et al., GPTQ - arXiv:2210.17323 |
| 21 | Lin et al., AWQ - arXiv:2306.00978 |
| 22 | Xiao et al., SmoothQuant - arXiv:2211.10438 |
| 23 | Dettmers et al., LLM.int8() arXiv:2208.07339 ; QLoRA arXiv:2305.14314 |
| 24 | Chee et al., QuIP arXiv:2307.13304 ; Tseng et al., QuIP# arXiv:2402.04396 |
| 25 | Liu et al., SpinQuant - arXiv:2405.16406 |
| 26 | Kwon et al., PagedAttention / vLLM - arXiv:2309.06180 |
| 27 | Zheng et al., SGLang - arXiv:2312.07104 |
| 28 | Leviathan et al. arXiv:2211.17192 ; Chen et al. arXiv:2302.01318 (speculative decoding) |
| 29 | Li et al., EAGLE arXiv:2401.15077 ; EAGLE-3 arXiv:2503.01840 |
| 30 | Micikevicius et al., FP8 Formats for Deep Learning - arXiv:2209.05433 |
| 31 | RunPod pricing - https://www.runpod.io/pricing (checked 2026-10-13) |
| 32 | Vast.ai live pricing - https://vast.ai/pricing (checked 2026-10-13) |
| 33 | lm-evaluation-harness - https://github.com/EleutherAI/lm-evaluation-harness |
| 34 | KEDA Prometheus scaler - https://keda.sh/docs/latest/scalers/prometheus/ |
| 35 | kube-prometheus-stack - https://github.com/prometheus-community/helm-charts |
| 36 | terraform-aws-modules/eks (20.37.2 validated) - https://github.com/terraform-aws-modules/terraform-aws-eks |
| 37 | Kubernetes HPA algorithm and tolerance - https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/ |
| 38 | Source manual: LLM Quantization & Deployment Zero-to-Elite Lab Manual (qdeploy-kit), versions verified 2026-09-16 |

# ANSWER KEY

| # | Answer |
|---|---|
| 1 | It scales to the maximum (50) with zero traffic: vLLM pre-allocates ~90% of VRAM at startup, so the rule is always true. Use vllm:kv_cache_usage_perc and the waiting-queue length. |
| 2 | 14e9 x 16 / 8 = 28 GB. W4A16 g128: 14e9 x 4.125 / 8 = ~7.2 GB plus the 16-bit embeddings/lm_head (often 1-2 GB more). |
| 3 | KV per request halves, so roughly twice as many full-length requests fit in the same free memory. |
| 4 | fp8 halves KV: 0.0737 MB per token x 64 x 4096 = 262,144 tokens x 0.0737 MB = ~19.3 GB. |
| 5 | llm-compressor refuses with "weight column counts not divisible by group_size" (seen in the test run). Use --group-size 64 (576 = 9 x 64) or ignore those layers. |
| 6 | dequantize.py to get a bf16 copy, then quantize.py --method fp8 on it. The result carries both rounding errors (test run: direct FP8 ppl 19.93 vs dequantized-W8A8-then-FP8 19.99). Always prefer the original bf16 model. |
| 7 | ceil(4 x 0.80 / 0.70) = ceil(4.57) = 5 pods (0.80/0.70 = 1.14 is outside the 10% tolerance). |
| 8 | W8A8 quantizes activations, whose outlier channels ruin INT8 scales; SmoothQuant moves that range into the weights. W4A16 keeps activations in 16-bit, so activation outliers matter less (rotations help its weights instead). |
| 9 | Perplexity is necessary, not sufficient. Read the failing GSM8K answers, recalibrate with maths-heavy and real prompts, try smaller groups / GPTQ vs AWQ / FP8, keep sensitive layers in 16-bit, or train a recovery LoRA; re-measure with identical settings. |
| 10 | GPTQ/AWQ optimise for the wrong activations: perplexity may look fine while contract answers degrade. Recalibrate on real contract prompts and check with 20-50 real task questions before shipping. |
