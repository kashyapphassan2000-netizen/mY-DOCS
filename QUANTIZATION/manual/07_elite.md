# PART 7 - ELITE: how top practitioners and researchers do it

WHY / THEORY. **What:** five things separate elite quantization work: (1) handling activation/weight OUTLIERS with rotations and smoothing; (2) proving WHAT changed, layer by layer and token by token; (3) mixed precision and accuracy RECOVERY instead of accepting the loss; (4) the same method scaling to 100B-1T+ parameter models; (5) operating it: re-quantizing correctly and updating production with canaries. **Why:** plain rounding breaks on outlier channels (real LLMs have them), and a single box is neither reliable nor cost-efficient. **When:** 4-bit quality is not good enough with GPTQ/AWQ alone, the model is huge, or the model is in production.

## Who defines the top level today

| Idea | People / organisation | Paper or repository |
|---|---|---|
| GPTQ (second-order weight quantization) | Elias Frantar, Dan Alistarh (IST Austria) | arXiv 2210.17323 [20] |
| AWQ, SmoothQuant | Song Han's lab (MIT HAN Lab) | arXiv 2306.00978 [21]; 2211.10438 [22] |
| LLM.int8(), QLoRA, NF4, bitsandbytes | Tim Dettmers et al. | arXiv 2208.07339, 2305.14314 [23] |
| QuIP / QuIP# (incoherence, Hadamard rotations) | Chris De Sa's group, Cornell | arXiv 2307.13304, 2402.04396 [24] |
| SpinQuant (learned rotations) | Meta | arXiv 2405.16406 [25] |
| llama.cpp / GGUF / k-quants / imatrix | Georgi Gerganov and contributors | github.com/ggml-org/llama.cpp [3] |
| PagedAttention, vLLM | Woosuk Kwon, Ion Stoica et al. (UC Berkeley Sky Lab) | arXiv 2309.06180 [26] |
| SGLang, RadixAttention | Lianmin Zheng, Ying Sheng et al. | arXiv 2312.07104 [27] |
| Speculative decoding; EAGLE | Leviathan et al. (Google), Chen et al. (DeepMind); Yuhui Li et al. | [28] [29] |
| FP8 formats (E4M3 / E5M2) | Micikevicius et al. (NVIDIA, Arm, Intel) | arXiv 2209.05433 [30] |
| Production quantization tooling | Red Hat AI (formerly Neural Magic) and the vLLM community | llm-compressor [4] |
| Quantization-aware training at trillion scale | Model labs shipping native FP8/INT4 releases (e.g. DeepSeek FP8 training, Moonshot Kimi INT4 QAT) (verify per model card) | Model cards on Hugging Face |

## SECTION 7.1 - Measure why rotations and smoothing matter

WHERE: GPU machine terminal, quant environment
GOAL: A measured table showing each elite technique reducing quantization damage on a real model.

**Step 1.** Type exactly this, then press Enter (about 5 minutes on a GPU):

```
source $QLAB/envs/quant/bin/activate && python $S/elite_lab.py --model Qwen/Qwen2.5-0.5B-Instruct 2>/dev/null | tail -7
```

YOU SHOULD SEE a table like this one (REAL output from the test run on HuggingFaceTB/SmolLM2-135M-Instruct on CPU, group size 64, 32 calibration samples; your numbers will differ):

ELITE_RESULTS_PLACEHOLDER

How to read it: KL(orig||quant) = how far the quantized model's next-token probabilities moved from the original (lower = better); top-1 agree = how often both pick the same next token (higher = better). Read the rows top to bottom: GPTQ's error compensation beats plain rounding (RTN); a Hadamard rotation (QuIP) spreads outlier channels before rounding; for INT8 activations, SmoothQuant moves activation outliers into the weights first. Tiny models with tiny calibration sets exaggerate the differences: the ranking is the lesson, not the size of the gain.

**Every non-obvious line, explained:** `QuantizationModifier(... W4A16)` without GPTQ = plain round-to-nearest. `QuIPModifier(rotations=["v"], transform_type="random-hadamard")` multiplies weights by a random Hadamard matrix (and the inverse on the input side at runtime), so no single channel dominates a group's scale. `SmoothQuantModifier(smoothing_strength=0.8)` divides activations by s and multiplies weights by s per channel. `damage()` compares full next-token distributions against the original, which catches changes perplexity can hide.

{{include scripts/elite_lab.py}}

WATCH OUT: Rotated (QuIP/SpinQuant) checkpoints carry online transforms. vLLM contains compressed-tensors transform support, but not every kernel supports every transform type: if serving a quip_gptq checkpoint fails, keep it as a research result and ship GPTQ/AWQ (verify on your GPU).

CHECKPOINT: You reproduced the table and can say which technique targets weights (GPTQ, rotations) and which targets activations (SmoothQuant).

## SECTION 7.2 - Prove WHAT changed, layer by layer

WHERE: GPU machine (CPU RAM ~2x the bf16 model), quant environment
GOAL: A list of the most-changed layers, the error per layer type and per depth.

**Step 1.** Type exactly this, then press Enter:

```
python $S/layer_error.py --original MODEL_ID --quantized $QLAB/work/SLUG/gptq --top 15 --out $QLAB/work/SLUG/eval/layer-error-gptq.json
```

YOU SHOULD SEE (REAL output from the test run, GPTQ 4-bit g64 on the 135M model):

```
TOP 5 MOST CHANGED LAYERS
  0.1897  max_abs=6.8828  model.layers.11.mlp.down_proj.weight
  0.1559  max_abs=0.8906  model.layers.29.mlp.gate_proj.weight
  0.1527  max_abs=4.4688  model.layers.28.mlp.down_proj.weight
  0.1459  max_abs=0.1904  model.layers.20.self_attn.v_proj.weight
  0.1444  max_abs=0.2754  model.layers.11.self_attn.o_proj.weight
AVERAGE BY LAYER TYPE
  o_proj         0.1339  (30 layers)
  gate_proj      0.1319  (30 layers)
  ...
AVERAGE BY DEPTH
  first third    0.1283
  middle third   0.1316
  last third     0.1323
OVERALL mean rel_err = 0.1307 over 210 layers
```

How to read it: rel_err 0.13 = the quantized matrix differs from the original by 13% of its size (high, because this is a tiny model at 4-bit with toy calibration; 7B+ models with good data are usually far lower). `max_abs=6.88` on layers.11.mlp.down_proj means that layer has a huge outlier weight: exactly the kind of layer to keep in 16-bit (7.3). down_proj layers often show the biggest max_abs because they receive the largest activations.

**Explained:** the quantized model is loaded with `CompressedTensorsConfig(run_compressed=False)`, which unpacks every 4-bit weight back to the bf16 value the GPU really multiplies with (q x scale). The original tensors are read one by one from the .safetensors files (no second full model in memory), and for each Linear layer the script computes ||W - W_hat|| / ||W||.

{{include scripts/layer_error.py}}

CHECKPOINT: You know your 5-15 worst layers by name.

## SECTION 7.3 - Mixed precision: keep the sensitive layers in 16-bit

WHERE: GPU machine terminal, quant environment
GOAL: A GPTQ model that keeps its worst layers in bf16 and loses less quality for a little more memory.

**Step 1.** Pick the worst 2-6 layers from 7.2 (or whole first/last blocks). Build a regular expression for them. Example for layers 0, 1 and 35 down_proj:

```
re:.*layers\.(0|1|35)\.mlp\.down_proj$
```

**Step 2.** Type exactly this (your expression), then press Enter:

```
python $S/quantize.py --model MODEL_ID --method gptq --calib $QLAB/data/calib.jsonl --num-samples 512 --ignore lm_head "re:.*layers\.(0|1|35)\.mlp\.down_proj$" --out $QLAB/work/SLUG/gptq-mixed
```

**Step 3.** Measure it exactly like the others (5.7 A, B, D) and add `gptq-mixed` to report.py --variants.

| Option | Memory cost | Typical effect |
|---|---|---|
| Keep lm_head 16-bit (default) | small | Always do it |
| Keep 2-6 worst layers | +1-3% size | Removes most of the outlier damage |
| Keep first and last 2 blocks | +5-10% size | Common recipe when early/late layers are sensitive |
| MoE models: keep router gates (`re:.*mlp.gate$`) | tiny | Required: quantized routers pick the wrong experts |

CHECKPOINT: gptq-mixed is in REPORT.md and you can say whether the extra memory bought enough quality.

## SECTION 7.4 - Win back accuracy (recovery ladder + recovery LoRA)

WHERE: GPU machine terminal
GOAL: Higher quality retention % than plain GPTQ, measured on your questions.

Climb this ladder from the cheapest fix; re-measure after each step (same settings):

| Rung | Fix | Cost |
|---|---|---|
| 1 | Better calibration data: real prompts, 512-1024 samples, real lengths (5.1) | Minutes |
| 2 | Try the other method (GPTQ vs AWQ), smaller group (64/32), higher dampening (5.8 sweep) | Hours of GPU |
| 3 | Mixed precision: keep the worst layers in 16-bit (7.3) | A little memory |
| 4 | Higher precision for the whole model: FP8 or W8A8 instead of 4-bit | 2x memory vs 4-bit |
| 5 | Rotations (QuIP) before GPTQ (7.1) | Check serving support |
| 6 | Recovery LoRA: teach the quantized model to answer like the original (below) | ~1 GPU-hour for 8B |
| 7 | Quantization-aware training / full distillation (research-level; vendor QAT releases) | Many GPU-hours |

### The recovery LoRA (distillation from the original model)

The idea: the quantized weights stay frozen exactly as the GPU will use them; a small LoRA adapter on top is trained on the ORIGINAL model's answers, so the pair "quantized + adapter" behaves like the original. You serve the same quantized checkpoint plus a tiny adapter (a few tens of MB).

**Step 1.** Create `$QLAB/data/prompts.jsonl` with 500-5000 real prompts from your domain, one `{"prompt": "..."}` per line (DIFFERENT from questions.jsonl, which stays your untouched test set).

**Step 2.** Serve the ORIGINAL on port 8001 (localhost only). Type exactly this, then press Enter:

```
tmux new -d -s orig "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name orig --max-model-len 4096 --host 127.0.0.1 --port 8001"
```

**Step 3.** Let the teacher answer. Type exactly this, then press Enter:

```
sleep 120; source $QLAB/envs/quant/bin/activate && python $S/teacher_answers.py --prompts $QLAB/data/prompts.jsonl --url http://127.0.0.1:8001 --model orig --out $QLAB/data/teacher.jsonl
```

{{include scripts/teacher_answers.py}}

**Step 4.** Stop the teacher (free the GPU): `tmux kill-session -t orig`.

**Step 5.** Train the recovery adapter. Type exactly this, then press Enter:

```
python $S/recovery_lora.py --quantized $QLAB/work/SLUG/gptq --data $QLAB/data/teacher.jsonl --out $QLAB/work/SLUG/recovery-lora --epochs 1 --rank 16
```

YOU SHOULD SEE (the test run on the 135M model):

```
trainable params: 4,884,480 || all params: 141,208,548 || trainable%: 3.4591
epoch 0 step 10/12 loss 2.7958
epoch 0 step 12/12 loss 2.8280
recovery adapter saved to /home/ubuntu/qlab/work/SLUG/recovery-lora
```

**Every non-obvious line, explained:** `CompressedTensorsConfig(run_compressed=False)` unpacks the 4-bit weights to the exact bf16 values the GPU kernel uses; `model.hf_quantizer = None` then treats them as ordinary frozen weights. `LoraConfig(target_modules="all-linear")` adds trainable A and B matrices to every Linear layer except lm_head. `encode()` masks the prompt tokens with -100, so the adapter only learns to produce the teacher's ANSWERS. Gradient checkpointing trades compute for memory; the learning-rate schedule warms up for 10 steps then decays linearly.

{{include scripts/recovery_lora.py}}

**Step 6.** Serve the quantized model WITH the adapter. Type exactly this, then press Enter:

```
tmux new -d -s quant "source $QLAB/envs/vllm/bin/activate; vllm serve $QLAB/work/SLUG/gptq --served-model-name quant --enable-lora --max-lora-rank 16 --lora-modules recovered=$QLAB/work/SLUG/recovery-lora --max-model-len 4096 --gpu-memory-utilization 0.40 --host 127.0.0.1 --port 8000"
```

**Step 7.** Start the original next to it again (0.45 of the GPU, port 8001, as in 5.7 Step 7), then compare "recovered" vs the original on your UNTOUCHED test questions:

```
cd $QLAB/work/SLUG && python $S/compare_answers.py --questions $QLAB/data/questions.jsonl --orig-url http://127.0.0.1:8001 --orig-model orig --quant-url http://127.0.0.1:8000 --quant-model recovered --judge-url http://127.0.0.1:8001 --judge-model orig --out-dir compare-gptq-recovered
```

YOU SHOULD SEE: `quality_retention_percent` higher than compare-gptq/summary.json. If it is not higher, the prompts were too few or unlike the test questions: add more real prompts.

WATCH OUT: Never train on your test questions (questions.jsonl) or the retention % becomes meaningless. Clients choose the adapter with "model": "recovered"; "model": "quant" still gives the plain 4-bit model.

CHECKPOINT: compare-gptq-recovered/summary.json shows higher retention than compare-gptq.

## SECTION 7.5 - Very large models: 100B, 1 trillion parameters and beyond

WHERE: Big multi-GPU machines (8x H100 / H200 / B200), terminal
GOAL: Apply the SAME pipeline to any model size, including future ones, and know where it changes.

The method does not change with size. Three things do: the quantization job's RAM/disk, the number of GPUs to serve, and MoE details. Always start with size_model.py (it reads Mixture-of-Experts fields: total parameters decide MEMORY, active parameters decide SPEED).

| Model size class | Memory at bf16 | At FP8 | At 4-bit | Serving hardware (typical) |
|---|---|---|---|---|
| 7-14B | 14-28 GB | 7-14 GB | 4-8 GB | 1x 24-48 GB GPU |
| 30-70B | 60-140 GB | 30-70 GB | 17-40 GB | 1-2x 80 GB (4-bit fits one 48-80 GB) |
| 100-250B (often MoE) | 200-500 GB | 100-250 GB | 55-140 GB | 2-8x 80 GB, one node |
| 400-700B MoE | 0.8-1.4 TB | 0.4-0.7 TB | 0.22-0.38 TB | 8x H200 (1.1 TB) for FP8; 8x H100 for 4-bit |
| 1T+ MoE | 2+ TB | 1+ TB | ~0.55+ TB | 8x B200 / 8x H200 at 4-bit, or 2+ nodes (pipeline parallel) |

**Step 1.** Size it. Type exactly this (a big MoE model id), then press Enter:

```
python $S/size_model.py --model BIG_MODEL_ID --context 32768 --users 8 --kv-fp8
```

**Step 2.** Check the model card FIRST: many very large models are RELEASED already in FP8 or INT4 (trained or fine-tuned for it). If an official FP8/INT4 release exists, serve that: it is better than anything you can do post-training, and it saves a multi-TB quantization job.

**Step 3.** If you must quantize it yourself:

| Goal | Method | Why it scales |
|---|---|---|
| FP8 of any size | `--method fp8_free` | Converts one .safetensors shard at a time; never loads the model; tested in this manual |
| 4-bit GPTQ/AWQ | `--method gptq` with `--ignore lm_head "re:.*mlp.gate$"` and `--num-samples 1024` | Layers are moved to the GPU one at a time; needs CPU RAM >= bf16 size (TBs) or a disk-offload setup; routers stay 16-bit; more samples so every expert sees calibration tokens |
| GGUF | convert + llama-quantize | Streams tensors; needs disk ~3x model |

**Step 4.** Serve across GPUs and machines with vLLM:

```
# one node, 8 GPUs (tensor parallel inside the node)
vllm serve BIG_MODEL_DIR --tensor-parallel-size 8 --max-model-len 32768 --kv-cache-dtype fp8 --api-key $VLLM_API_KEY
# two nodes x 8 GPUs (pipeline parallel across nodes; needs a Ray cluster first) (verify for your vLLM version)
vllm serve BIG_MODEL_DIR --tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend ray
# MoE: spread experts across GPUs instead of splitting every expert (verify flag on your version)
vllm serve BIG_MODEL_DIR --tensor-parallel-size 8 --enable-expert-parallel
```

On Kubernetes a multi-node replica is one "group" of pods (LeaderWorkerSet is the common pattern); your KEDA rule then scales groups, and gpu_max must count whole 8-GPU machines (verify).

WATCH OUT: The bottleneck for 1T models is not the GPU math but moving terabytes: download time, disk, CPU RAM and cold-start time (tens of minutes). Keep a warm minimum (min replicas >= 1) and long startup probes.

CHECKPOINT: You can state, for any model id: total vs active parameters, memory per format, GPUs per replica, and whether an official FP8/INT4 release exists.

## SECTION 7.6 - Re-quantizing an already-quantized model (the right way)

WHERE: GPU machine terminal, quant environment
GOAL: Turn somebody's quantized checkpoint into a different format without stacking errors blindly.

The rule: never feed a quantized model into a quantizer. Packed 4-bit integers are not "weights" a quantizer understands, and rounding already-rounded values adds a second error. The correct order is: original bf16 (best) -> if you do not have it, dequantize to bf16 first -> quantize to the new format -> measure against the best reference you have.

**Step 1.** Dequantize. Type exactly this, then press Enter:

```
python $S/dequantize.py --model SOMEONE/MODEL-quantized-w8a8 --out $QLAB/work/SLUG/deq-bf16
```

**Step 2.** Quantize the dequantized copy (any method). Type exactly this, then press Enter:

```
python $S/quantize.py --model $QLAB/work/SLUG/deq-bf16 --method fp8 --out $QLAB/work/SLUG/deq-fp8
```

**Step 3.** Measure (5.7) against the original if it exists anywhere.

REAL chain from the test run (SmolLM2-135M, perplexity on the same text):

```
original bf16                           19.49
-> W8A8                                 19.76
-> dequantized back to bf16             19.55   (activation rounding gone, weight rounding stays)
-> FP8 of the dequantized copy          19.99
direct FP8 from the original            19.93   <- always better: start from the original when you can
```

**Explained:** dequantize.py loads the checkpoint with `run_compressed=False` (unpacks to bf16 numbers), builds an EMPTY plain bf16 model from the same config without `quantization_config`, copies every weight in (it refuses if any weight is missing), and saves a normal model any tool can read.

{{include scripts/dequantize.py}}

| You have | Do this |
|---|---|
| compressed-tensors (llm-compressor / vLLM community uploads) | dequantize.py, then quantize.py |
| GGUF only | Find the original HF model; GGUF -> HF conversion is lossy and not recommended |
| Vendor block-FP8 (e.g. DeepSeek-style) | Use the vendor's bf16 cast script or the bf16 release (verify), then quantize.py |
| GGUF Q8_0 -> Q4_K_M | llama-quantize --allow-requantize works but stacks errors; prefer converting from the bf16 GGUF |

CHECKPOINT: deq-bf16/config.json has no quantization_config; the re-quantized model is measured and recorded in REPORT.md.

## SECTION 7.7 - Update production safely: new version -> gates -> canary -> promote or roll back

WHERE: GPU machine (build), PC PowerShell (Kubernetes)
GOAL: Version v2 replaces v1 without risking all users, with a one-command rollback.

The enterprise update flow:

```
new base model / new fine-tune / better calibration data / new vLLM
  -> quantize (Part 4-5, same scripts, same settings file quant_info.json)
  -> evaluate v2 vs ORIGINAL and vs v1 (report.py): every gate must pass, and v2 must not be worse than v1
  -> push_to_hub.py --tag v2
  -> canary: 1 pod with v2 next to the v1 pods (~10% of traffic)
  -> watch errors, TTFT, kv usage, and sample real answers for 1-24 hours
  -> promote (all pods to v2)  OR  delete the canary / roll back
```

**Step 1.** Upload v2 (GPU machine). Type exactly this, then press Enter:

```
python $S/push_to_hub.py --folder $QLAB/work/SLUG/gptq --repo HF_USER/SLUG-gptq-w4a16 --base MODEL_ID --tag v2 --eval-dir $QLAB/work/SLUG/eval
```

**Step 2.** On the PC, open infra\k8s\canary.yaml in VS Code, replace HF_REPO, NEW_TAG (v2) and SERVED_NAME, save. Type exactly this, then press Enter:

```
kubectl apply -f canary.yaml; kubectl get pods -n llm -L track -w
```

YOU SHOULD SEE: stable pods with track=stable and one vllm-canary pod with track=canary, all Running.

{{include infra/k8s/canary.yaml}}

**Step 3.** Watch the canary: `kubectl logs -n llm -l track=canary --tail 50`, Prometheus query `vllm:num_requests_running{pod=~"vllm-canary.*"}`, and run compare_answers.py against the load balancer for a sample of questions.

**Step 4a (promote).** In vllm.yaml change `--revision=v1` to `--revision=v2`, save, then type exactly this and press Enter:

```
kubectl apply -f vllm.yaml; kubectl rollout status deployment/vllm -n llm; kubectl delete -f canary.yaml
```

**Step 4b (abort the canary).** Type exactly this, then press Enter:

```
kubectl delete -f canary.yaml
```

**Step 4c (roll back after a bad promote).** Type exactly this, then press Enter:

```
kubectl rollout undo deployment/vllm -n llm
```

**Explained:** the Service sends traffic to every pod with app=vllm, so traffic splits by pod count (1 canary among 9 stable pods = ~10%). The stable Deployment selects only track=stable pods, so the canary is never adopted or scaled by KEDA. `rollout undo` restores the previous pod template (the old --revision) with the same zero-downtime rolling update. Because models are pinned by Hub TAG, "v1" always means exactly the bytes you tested.

Continuous improvement loop: every week, collect the questions where users complained or the judge scored low, add them to questions.jsonl (test) and similar ones to calib.jsonl / prompts.jsonl (training), and run the flow again.

CHECKPOINT: v2 went live through a canary; you practised a rollback.

## Common mistake at the Elite level

| Mistake | How to spot it | Fix |
|---|---|---|
| Reporting a tiny-model lab gain as a production gain | Lab numbers quoted for your real model | Re-measure on your model (5.7) |
| Training the recovery LoRA on the test questions | Retention % suddenly ~100% | Separate prompts.jsonl (train) from questions.jsonl (test) |
| Quantizing MoE routers | Answers degrade unpredictably | --ignore "re:.*mlp.gate$" |
| Re-quantizing a quantized model directly | Errors, or silently worse quality | dequantize.py first; prefer the original |
| Promoting without a canary | All users hit a bad version | canary.yaml + gates, then promote |
