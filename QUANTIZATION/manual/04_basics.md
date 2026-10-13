# PART 4 - BASICS: size a real model, measure it, quantize to FP8, serve it, compare

WHY / THEORY. **What:** take one real model, decide what hardware it needs, measure the ORIGINAL, make an FP8 copy, serve it and compare. **Why:** FP8_DYNAMIC halves weight memory, needs no calibration data and is near-lossless: the safest first quantization. **When:** whenever your GPU has compute capability 8.9 or higher (L4, L40, L40S, RTX 4090, H100, H200, B200) [11].

| Placeholder | Replace with | Example |
|---|---|---|
| MODEL_ID | Hugging Face id of your model (16-bit original) | Qwen/Qwen3-8B |
| SLUG | Short lower-case folder name | qwen3-8b |
| SERVED_NAME | Name clients send as "model" | qwen3-8b |

## SECTION 4.1 - Size your model (script + by hand)

WHERE: GPU machine terminal (or your PC with Python), quant environment
GOAL: You know the parameter count, memory per format, GPU type and count, and price per hour BEFORE renting.

**Step 1.** Type exactly this, then press Enter:

```
source $QLAB/envs/quant/bin/activate
```

**Step 2.** Type exactly this (your model, your traffic), then press Enter:

```
python $S/size_model.py --model MODEL_ID --context 8192 --users 16
```

YOU SHOULD SEE (real output for Qwen/Qwen3-8B):

```
MODEL Qwen/Qwen3-8B
  layers=36 hidden=4096 heads=32 kv_heads=8 head_dim=128 vocab=151936 experts=0
  params total = 8.19 B   active per token = 8.19 B
KV CACHE (bf16)
  per token   = 0.1475 MB
  per request = 1.21 GB  (8192 tokens)
  all users   = 19.33 GB  (16 x 8192 tokens)
WEIGHTS PER FORMAT
  bf16                  16.38 GB
  fp8                    9.44 GB
  w4a16 (gptq/awq)       6.07 GB
  gguf Q4_K_M            4.97 GB
WHICH GPU (serving, vLLM-style memory)
  gpu       format             GPUs  $/hour  max users
  L40S      bf16                  1    0.79         20
  L40S      fp8                   1    0.79         26
  ...
CHEAPEST THAT FITS: 1 x L40S with bf16  (~$0.79/hour)
QUANTIZATION JOB NEEDS (one-time)
  disk      >= 79 GB
  CPU RAM   >= 37 GB
  GPU VRAM  any 24 GB GPU holds the whole model (19 GB needed)
```

How to read it: **max users** = how many full-length (8192-token) conversations fit at the same time. Quantizing does not change the GPU count here, but it raises max users from 20 (bf16) to 26 (fp8) on the same GPU: that is cheaper per answer. For a 70B model the table changes completely: bf16 needs 2-4 GPUs, 4-bit fits one.

**Every non-obvious part of size_model.py, explained:** `load_config` downloads only config.json (a few KB). `param_counts` applies the formula from Part 3: q/k/v/o attention matrices plus gate/up/down MLP matrices per layer, times layers, plus embeddings. For Mixture-of-Experts models (Mixtral, Qwen3-MoE, DeepSeek, Kimi) it multiplies the expert MLP by the number of experts for TOTAL memory and by experts-per-token for ACTIVE compute. `weights_gb` keeps embeddings and lm_head at 16-bit, because llm-compressor recipes skip them. `kv_bytes_per_token` is 2 (K and V) x layers x kv_heads x head_dim x bytes; DeepSeek-style latent attention (MLA) uses its compressed size instead. The GPU loop tries tensor-parallel sizes 1, 2, 4, 8 (they must divide the head count) and keeps the first that leaves room for your users' KV cache. Prices are examples from October 2026: edit the GPUS table before trusting the dollars.

{{include scripts/size_model.py}}

**Step 3.** Do one check by hand so you trust the script: open https://huggingface.co/MODEL_ID/blob/main/config.json in the browser, copy num_hidden_layers, num_key_value_heads, head_dim, and compute KV per token = 2 x layers x kv_heads x head_dim x 2 bytes. It must match the "per token" line.

| If you see... | It means... | Do this next |
|---|---|---|
| 401 / 403 / GatedRepoError | Gated model, licence not accepted or token missing | Section 2.3 Step 5; check `echo $HF_TOKEN` |
| KeyError: 'hidden_size' | Unusual architecture field names | Read config.json and pass the numbers to the AI prompt in 4.2 |
| nothing fits on one 8-GPU node | Model + KV too big | Lower --context or --users, add --kv-fp8, or see Part 7.5 (very large models) |

WATCH OUT: Two different jobs, two different machines. QUANTIZING needs lots of CPU RAM and disk (it holds the 16-bit original). SERVING needs VRAM for weights + KV cache. You may quantize on one big machine once, then serve on many cheaper ones.

CHECKPOINT: You can say: parameters, KV per token, which GPU and how many for serving, and RAM/disk for the quantization job.

## SECTION 4.2 - The AI-assistant prompt for GPU sizing and cost

WHERE: Any AI chat (Claude, ChatGPT, Gemini) with web search turned on
GOAL: A second opinion with live prices, in one paste.

**Step 1.** Copy this prompt, replace the CAPITAL words, paste it into the AI chat:

```
You are a senior ML infrastructure engineer. Size GPU hardware for serving and for quantizing an LLM.

MODEL: MODEL_ID  (Hugging Face; read its config.json: num_hidden_layers, hidden_size, num_attention_heads,
num_key_value_heads, head_dim, intermediate_size, vocab_size, tie_word_embeddings, and MoE fields
num_experts / num_local_experts / n_routed_experts, num_experts_per_tok, moe_intermediate_size, kv_lora_rank)
TRAFFIC: CONTEXT tokens per request, USERS concurrent requests, target p95 time-to-first-token TTFT seconds
FORMATS TO COMPARE: bf16, fp8, int8 w8a8, int4 w4a16 (gptq/awq), gguf q4_k_m
PROVIDERS: HyperStack, RunPod, Vast.ai, AWS (on-demand, my region REGION)

Do this, showing every formula with numbers:
1. Total and active parameters (show the per-layer breakdown).
2. Weight memory per format (keep embeddings + lm_head at 16-bit for gptq/awq/fp8).
3. KV cache per token = 2 x layers x kv_heads x head_dim x bytes; total for CONTEXT x USERS (bf16 and fp8 KV).
4. For each GPU type (L4 24GB, RTX 4090 24GB, L40S 48GB, A100 80GB, H100 80GB, H200 141GB, B200 180GB):
   smallest tensor-parallel size (1/2/4/8, must divide attention heads) where
   weights/TP + KV/min(TP, kv_heads) + 2 GB <= 0.90 x VRAM. If one 8-GPU node is not enough, give the
   pipeline-parallel node count.
5. Note FP8 needs compute capability >= 8.9 for fast kernels; INT8 W8A8 >= 7.5.
6. Search the web for CURRENT on-demand prices of each option at each provider; give $/hour and an estimated
   cost per 1M output tokens = price / (tokens_per_second x 3600) x 1e6, with your tokens/second assumption.
7. The one-time QUANTIZATION job: disk, CPU RAM (holds the 16-bit model), GPU.
8. Final recommendation table: cheapest option that fits, best quality option, and a balanced pick, with risks.
Mark any number you could not verify with (verify).
```

**Step 2.** Compare the AI's KV-per-token and weight numbers with size_model.py. If they differ by more than ~10%, trust the script (it reads the real config) and ask the AI where its formula differs.

WATCH OUT: AI chats often invent prices or forget that GQA (kv_heads < heads) shrinks the KV cache. Always check the price on the provider's page before renting.

CHECKPOINT: Script and AI agree on parameters and KV within ~10%; you have a GPU choice and a price.

## SECTION 4.3 - Download the original model

WHERE: GPU machine terminal, quant environment, inside tmux
GOAL: The 16-bit original is on the GPU machine's disk.

**Step 1.** Type exactly this, then press Enter:

```
tmux new -s work
```

**Step 2.** Type exactly this, then press Enter:

```
source $QLAB/envs/quant/bin/activate && df -h $QLAB
```

YOU SHOULD SEE: Avail at least the "disk" number from 4.1.

**Step 3.** Type exactly this, then press Enter:

```
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('MODEL_ID'))"
```

YOU SHOULD SEE: download progress, then a path like `/home/ubuntu/qlab/hf-cache/hub/models--.../snapshots/<hash>`.

WATCH OUT: Downloads go to $HF_HOME (set in 2.6) on the big disk. If you see "No space left", check `echo $HF_HOME`.

CHECKPOINT: The path printed; `du -sh $HF_HOME/hub/*` shows roughly the bf16 size from 4.1.

## SECTION 4.4 - Baseline: measure the ORIGINAL first and write your gates

WHERE: GPU machine terminal, quant environment
GOAL: A perplexity number for the original model, and written quality gates, BEFORE you quantize anything.

**Step 1.** Type exactly this, then press Enter:

```
mkdir -p $QLAB/work/SLUG/eval
```

**Step 2.** Type exactly this, then press Enter:

```
python $S/eval_ppl.py --model MODEL_ID --out $QLAB/work/SLUG/eval/ppl-bf16.json
```

YOU SHOULD SEE:

```
{
  "model": "MODEL_ID",
  "text": "wikitext2",
  "perplexity": <n>,
  "tokens": 40960,
  "window": 2048,
  "stride": 512,
  ...
}
```

**Every non-obvious line of eval_ppl.py, explained:** perplexity = exp(average "surprise" per next token); lower is better. The text is cut to 40,960 tokens so every model sees exactly the same tokens. A sliding window of 2048 tokens moves 512 at a time; `labels[:, :-target_len] = -100` tells the model "use these tokens as context but do not score them", so no token is scored twice. `device_map="auto"` spreads big models over all GPUs. The same script loads bf16, FP8, GPTQ, AWQ, W8A8 and NF4 folders because transformers reads `quantization_config` from config.json.

{{include scripts/eval_ppl.py}}

**Step 3.** Write your gates NOW (before seeing any quantized number). Create the file $QLAB/work/SLUG/GATES.md in VS Code with this content:

```
# Quality and performance gates for SLUG (written before measuring)
| Gate                                   | Threshold                 |
|----------------------------------------|---------------------------|
| Perplexity increase vs bf16            | FP8 <= 1%, 4-bit <= 4%    |
| Quality retention on my questions      | >= 97%                    |
| Benchmark (GSM8K / MMLU) drop          | <= 2 points               |
| Failed requests under load             | 0                         |
| TTFT p95 at my concurrency             | <= 2000 ms                |
```

WATCH OUT: Perplexity on Wikipedia text is a smoke test, not proof. Your own domain text (`--text my_domain.txt`) and your own questions (Part 5.7) matter more.

CHECKPOINT: ppl-bf16.json exists; GATES.md exists with numbers.
If not: CUDA out of memory = the bf16 model does not fit this GPU: add `--window 1024`, or measure on a bigger GPU once.

## SECTION 4.5 - Quantize to FP8

WHERE: GPU machine terminal, quant environment, inside tmux
GOAL: $QLAB/work/SLUG/fp8 holds an FP8 checkpoint about half the bf16 size.

**Step 1.** Type exactly this, then press Enter:

```
python $S/quantize.py --model MODEL_ID --method fp8 --out $QLAB/work/SLUG/fp8
```

YOU SHOULD SEE:

```
... oneshot ... FP8_DYNAMIC ...
{
  "method": "fp8",
  "seconds": <n>,
  "size_gb": <about half of bf16>
}
```

**Every non-obvious line of quantize.py, explained:**

- `AutoModelForCausalLM.from_pretrained(a.model, dtype="auto")` loads the original in its own precision (bf16) into CPU RAM. llm-compressor moves one layer at a time to the GPU during calibration ("sequential onloading"), which is why a 24 GB GPU can quantize a 70B model if the machine has enough CPU RAM.
- `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])`: every Linear layer (q, k, v, o, gate, up, down) gets FP8 weights with one scale per output channel; activations are scaled per token at runtime, so no calibration data. `lm_head` (the final vocabulary projection) stays 16-bit because it is very sensitive.
- `GPTQModifier(... block_size=128, dampening_frac, actorder)`: block_size = how many columns are processed together; dampening_frac = stabiliser added to the Hessian diagonal; actorder = process the most important columns first.
- `oneshot(model=..., recipe=..., dataset=...)`: runs the recipe once (post-training quantization, no training).
- `model.save_pretrained(out, save_compressed=True)`: writes PACKED low-bit tensors plus a `quantization_config` block in config.json. vLLM and SGLang read that block and pick the right GPU kernel automatically.
- `build_calibration`: turns your JSONL (or ultrachat) into tokenized samples. Chat data is rendered with the model's OWN chat template, so calibration looks like real traffic.
- `quant_info.json`: every setting + time + size, so you can reproduce the result later.

{{include scripts/quantize.py}}

WATCH OUT: If the process vanishes with no error, Linux killed it for lack of CPU RAM: run `dmesg | tail`. Use a machine with more RAM, or `--method fp8_free` (converts file by file, never loads the full model).

CHECKPOINT: `du -sh $QLAB/work/SLUG/fp8` is roughly half the bf16 size; quant_info.json exists.
If not: "401 Client Error" = token/licence (2.3). "CUDA out of memory" = another process holds the GPU: `nvidia-smi`.

## SECTION 4.6 - Look at WHAT CHANGED inside the files

WHERE: GPU machine terminal, quant environment
GOAL: You can see with your own eyes how the weights are stored now.

**Step 1.** Compare sizes. Type exactly this, then press Enter:

```
du -sh $HF_HOME/hub/models--*/ $QLAB/work/SLUG/fp8
```

**Step 2.** Print the quantization recipe stored in config.json. Type exactly this, then press Enter:

```
python -c "import json,os; q=json.load(open(os.path.expandvars('$QLAB/work/SLUG/fp8/config.json')))['quantization_config']; g=q['config_groups']['group_0']; print(q['format'], g['weights'], g.get('input_activations'), q['ignore'], sep='\n')"
```

**Step 3.** List the stored tensors of the first attention layer. Type exactly this, then press Enter:

```
python -c "from safetensors import safe_open; import glob,os; f=sorted(glob.glob(os.path.expandvars('$QLAB/work/SLUG/fp8/*.safetensors')))[0]; s=safe_open(f,'pt'); [print(k, s.get_slice(k).get_dtype(), s.get_slice(k).get_shape()) for k in s.keys() if 'layers.0.self_attn.q_proj' in k]"
```

YOU SHOULD SEE (real output from the test run, small model; your shapes are bigger):

```
model.layers.0.self_attn.q_proj.weight        F8_E4M3 [576, 576]   <- the weights, now 1 byte each
model.layers.0.self_attn.q_proj.weight_scale  BF16    [576, 1]     <- one scale per output row
format: float-quantized | weights: 8-bit float, strategy channel | activations: 8-bit float, per token, dynamic
```

For comparison, the same layer after GPTQ 4-bit (Part 5) looks like this:

```
q_proj.weight_packed  I32  [576, 72]    <- 8 four-bit numbers packed into each 32-bit integer (576/8 = 72)
q_proj.weight_scale   BF16 [576, 9]     <- one scale per group of 64 weights (576/64 = 9)
q_proj.weight_g_idx   I32  [576]        <- column order used by actorder
q_proj.weight_shape   I64  [2]          <- original shape, needed to unpack
```

This IS the change quantization makes: the same matrix, stored as small integers or small floats plus a few scale numbers. At runtime the GPU computes w_hat = q x scale on the fly.

CHECKPOINT: You saw F8_E4M3 weights and a weight_scale tensor.

## SECTION 4.7 - Serve the FP8 model with vLLM and talk to it from your PC

WHERE: GPU machine terminal (server), then PC PowerShell (client)
GOAL: vLLM answers on port 8000, protected by your API key; your PC talks to it through an encrypted tunnel.

**Step 1.** Start the server in its own tmux session. Type exactly this (your SLUG and SERVED_NAME), then press Enter:

```
tmux new -d -s vllm "source $QLAB/envs/vllm/bin/activate; vllm serve $QLAB/work/SLUG/fp8 --served-model-name SERVED_NAME --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key $VLLM_API_KEY --port 8000 2>&1 | tee $QLAB/work/SLUG/vllm-fp8.log"
```

| Flag | Value | Why |
|---|---|---|
| --served-model-name | SERVED_NAME | Clients send this as "model" |
| --max-model-len | 8192 | Longest request; KV cache is sized for it |
| --gpu-memory-utilization | 0.90 | vLLM pre-allocates 90% of VRAM (weights + KV pool) at startup |
| --api-key | $VLLM_API_KEY | Without it anyone who reaches the port can use your GPU |
| --port | 8000 | Where it listens |

**Step 2.** Wait about 90 seconds. Type exactly this, then press Enter:

```
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/v1/models
```

YOU SHOULD SEE: `401` (no key = refused: good).

**Step 3.** Type exactly this, then press Enter:

```
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

YOU SHOULD SEE: `{"object":"list","data":[{"id":"SERVED_NAME",...}]}`

**Step 4.** On your PC, open PowerShell and start the tunnel. Type exactly this, then press Enter:

```
ssh -N -L 8000:localhost:8000 gpu
```

YOU SHOULD SEE: the window seems to hang. That is correct: the tunnel is open while it runs.

**Step 5.** Open a SECOND PowerShell window on your PC. Copy your key from the GPU machine (`echo $VLLM_API_KEY` there) and type exactly this (with your key), then press Enter:

```
$env:VLLM_API_KEY = "sk-PASTE_YOUR_KEY"
```

**Step 6.** Type exactly this, then press Enter:

```
curl.exe -s http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $env:VLLM_API_KEY" -H "Content-Type: application/json" -d '{\"model\":\"SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain quantization in 2 sentences\"}],\"max_tokens\":200}'
```

YOU SHOULD SEE: a JSON answer with "content": "...".

| If you see... | It means... | Do this next |
|---|---|---|
| CUDA out of memory at startup | Weights + KV do not fit | Lower --max-model-len (e.g. 4096) or --gpu-memory-utilization 0.85 |
| Connection refused | Server still loading | `tmux attach -t vllm` to watch (Ctrl+B then D to leave) |
| 'Unable to connect' on PC | Tunnel window closed | Restart Step 4 |
| Answer starts with <think> | Qwen3 "thinking" mode | Normal; add "/no_think" to the prompt to switch it off |

WATCH OUT: Never open port 8000 to the internet on the provider's firewall. The SSH tunnel is your secure path; Part 6 adds a proper load balancer.

CHECKPOINT: 401 without key; answer with key; your PC got an answer through the tunnel.

## SECTION 4.8 - First benchmark and first comparison (original vs FP8)

WHERE: GPU machine terminal
GOAL: A speed table for FP8 AND for the original, and a perplexity comparison.

**Step 1.** Type exactly this, then press Enter:

```
source $QLAB/envs/vllm/bin/activate && cd $QLAB/work/SLUG
```

**Step 2.** Type exactly this, then press Enter:

```
python $S/bench.py --url http://localhost:8000 --model SERVED_NAME --label fp8 --concurrency 1 8 32
```

YOU SHOULD SEE:

```
| label | conc | tok/s | TTFT p50 ms | TTFT p95 ms | ITL p50 ms | failed |
|---|---|---|---|---|---|---|
| fp8 | 1 | <n> | <n> | <n> | <n> | 0 |
| fp8 | 8 | <n> | ...
| fp8 | 32 | <n> | ...
```

How to read it: **tok/s** rises with concurrency until the GPU is full; **TTFT p95** = worst-case wait for the first word; **ITL** = time between words while streaming (feel of speed); **failed** must be 0.

{{include scripts/bench.py}}

**Step 3.** Stop the FP8 server. Type exactly this, then press Enter:

```
tmux kill-session -t vllm
```

**Step 4.** Serve the ORIGINAL the same way (one engine per GPU at a time). Type exactly this, then press Enter:

```
tmux new -d -s vllm "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name SERVED_NAME --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key $VLLM_API_KEY --port 8000"
```

**Step 5.** Wait 90 seconds, then type exactly this and press Enter:

```
python $S/bench.py --url http://localhost:8000 --model SERVED_NAME --label bf16 --concurrency 1 8 32
```

**Step 6.** Measure FP8 perplexity. Type exactly this, then press Enter:

```
tmux kill-session -t vllm; source $QLAB/envs/quant/bin/activate && python $S/eval_ppl.py --model $QLAB/work/SLUG/fp8 --out $QLAB/work/SLUG/eval/ppl-fp8.json
```

**Step 7.** Print both perplexities side by side. Type exactly this, then press Enter:

```
python -c "import json,glob; [print(f.split('/')[-1], json.load(open(f))['perplexity']) for f in sorted(glob.glob('$QLAB/work/SLUG/eval/ppl-*.json'))]"
```

YOU SHOULD SEE (real numbers from the CPU test run on a 135M model; yours will differ):

```
ppl-bf16.json 19.4945
ppl-fp8.json  19.9299      <- +2.2% on a TINY model; 7B+ models are usually < 0.5%
```

CHECKPOINT: bench_results.jsonl has fp8 and bf16 rows with failed = 0; two perplexity files exist; you know if FP8 passes your gate.
If not: failed > 0 = read the "errors" line: HTTP 401 = wrong key in this terminal; timeouts = lower --max-tokens.

## Common mistake at the Basics level

| Mistake | How to spot it | Fix |
|---|---|---|
| Serving from the wrong environment | "vllm: command not found" or import errors | source $QLAB/envs/vllm/bin/activate first |
| Comparing numbers measured differently | Different window/text/max-tokens | Re-run every model with identical settings |
| Exposing port 8000 to the internet | Anyone can reach VM_IP:8000 | Firewall allows only SSH; use the tunnel |
| Forgetting the baseline | No ppl-bf16.json | Measure the original first, always |
| Leaving the GPU machine running overnight | Big bill | Stop it at the end of every session |
