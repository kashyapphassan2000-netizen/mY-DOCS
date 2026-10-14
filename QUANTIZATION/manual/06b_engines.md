# PART 6B - SERVING ENGINES DEEP DIVE: multi-LoRA, SGLang, speculative decoding, GGML vs GGUF

WHY / THEORY. **What:** the four serving skills that separate "it runs" from "it runs cheaply for many users": (1) many fine-tuned LoRA adapters on ONE base model, (2) SGLang as a second engine and when its RadixAttention beats vLLM's PagedAttention, (3) speculative decoding to cut per-user latency, (4) the GGML -> GGUF story and the exact byte layout of llama.cpp formats. **Why:** one GPU can serve 10 customers' fine-tunes, a shared 2,000-token system prompt can be computed once instead of a thousand times, and a 0.6B draft model can make an 8B model answer one user ~1.5-2.5x faster. **When:** after Part 6 gave you a working vLLM server; everything here runs on the same 48 GB GPU.

| Engine | Strength | Use it when |
|---|---|---|
| vLLM | Broadest model/format support, PagedAttention, multi-LoRA, OpenAI API, Prometheus metrics | Default production engine |
| SGLang | RadixAttention prefix tree, cache-aware scheduling, fast EAGLE speculative decoding, NF4 | Agents, RAG, multi-turn chat, long shared prompts |
| llama.cpp | GGUF on CPU / Mac / laptop, tiny install | Edge, offline, no datacentre GPU |

## SECTION 6B.1 - Make two test adapters you can SEE working

WHERE: GPU machine terminal
GOAL: Two LoRA adapters on MODEL_ID with obviously different behaviour: "pirate" (talks like a pirate) and "json-only" (answers only in JSON).

If you already have real fine-tuned adapters (trained on MODEL_ID), skip to 6B.2 and use them instead.

**Step 1.** Make 300 practice prompts. Type exactly this, then press Enter:

```
source $QLAB/envs/quant/bin/activate && python $S/make_prompts.py --n 300 --out $QLAB/data/style-prompts.jsonl
```

**Step 2.** Serve the ORIGINAL model (localhost only). Type exactly this, then press Enter:

```
tmux new -d -s teach "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name base --max-model-len 4096 --host 127.0.0.1 --port 8001"
```

**Step 3.** Wait 2 minutes, then generate the pirate data. The system prompt is used ONLY while generating; it is not saved, so the adapter learns to behave this way without being told. Type exactly this, then press Enter:

```
python $S/teacher_answers.py --prompts $QLAB/data/style-prompts.jsonl --url http://127.0.0.1:8001 --model base --system "Always answer like a friendly pirate. Keep it short." --max-tokens 300 --out $QLAB/data/pirate.jsonl
```

**Step 4.** Generate the JSON data. Type exactly this, then press Enter:

```
python $S/teacher_answers.py --prompts $QLAB/data/style-prompts.jsonl --url http://127.0.0.1:8001 --model base --system "Reply ONLY with valid JSON of the form {\"answer\": \"...\", \"confidence\": 0-1}. No other text." --max-tokens 300 --out $QLAB/data/json.jsonl
```

**Step 5.** Stop the teacher: `tmux kill-session -t teach`.

**Step 6.** Train both adapters on the 16-bit base (about 5-15 minutes each for 8B on one L40S). Type exactly this, then press Enter:

```
python $S/recovery_lora.py --base MODEL_ID --data $QLAB/data/pirate.jsonl --out $QLAB/work/SLUG/lora-pirate --rank 16 --epochs 2 && python $S/recovery_lora.py --base MODEL_ID --data $QLAB/data/json.jsonl --out $QLAB/work/SLUG/lora-json --rank 16 --epochs 2
```

YOU SHOULD SEE (shape of the real output from the test run on a 135M model):

```
trainable params: <n> || all params: <n> || trainable%: <n>
epoch 0 step 10/... loss <falling>
...
LoRA adapter saved to .../lora-pirate
```

**Explained:** recovery_lora.py with `--base` trains an ordinary LoRA adapter on a plain 16-bit model (with `--quantized` it trains the recovery adapter of Part 7.4). Only the answer tokens are learned (prompt tokens are masked with -100). Rank 16 = each adapted layer gets two thin matrices of width 16.

CHECKPOINT: Both folders contain adapter_config.json and adapter_model.safetensors.
If not: CUDA out of memory = add --max-len 1024, or stop every other server on the GPU.

## SECTION 6B.2 - Plan the memory for many adapters

WHERE: GPU machine terminal, quant environment
GOAL: You know how much VRAM and RAM N adapters cost BEFORE you start the server.

**Step 1.** Type exactly this, then press Enter:

```
python $S/lora_plan.py --model MODEL_ID --ranks 8 16 32 64 --max-loras 4 --max-lora-rank 64 --max-cpu-loras 16
```

YOU SHOULD SEE (REAL output for Qwen/Qwen3-8B, all 7 projection layers adapted):

```
  rank        params   MB (bf16)
     8    21,823,488        43.6
    16    43,646,976        87.3
    32    87,293,952       174.6
    64   174,587,904       349.2
GPU reserved by the engine = max_loras x size(max_lora_rank) = 4 x 0.349 GB = 1.40 GB  (taken from the KV-cache budget)
CPU RAM cache              = max_cpu_loras x size(max_lora_rank) = 16 x 0.349 GB = 5.59 GB
```

The formula: one adapted layer of shape (in x out) adds rank x (in + out) numbers. Per Qwen3-8B layer: q 4096+4096, k 4096+1024, v 4096+1024, o 4096+4096, gate 4096+12288, up 4096+12288, down 12288+4096 = 75,776 x rank. Times 36 layers, times 2 bytes. Every GPU slot is sized for --max-lora-rank, so setting it to 64 when your adapters are rank 16 wastes 4x the memory.

{{include scripts/lora_plan.py}}

CHECKPOINT: You can state the GPU memory your adapter slots take away from the KV cache.

## SECTION 6B.3 - Serve many LoRA adapters from ONE base on vLLM

WHERE: GPU machine terminal (server) + second terminal (clients)
GOAL: One vLLM process answers as the base model, as "pirate" and as "json-only", chosen per request.

**Step 1.** Start the server. Type exactly this, then press Enter:

```
tmux new -d -s vllm "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name SERVED_NAME --enable-lora --max-loras 4 --max-lora-rank 16 --max-cpu-loras 16 --lora-modules pirate=$QLAB/work/SLUG/lora-pirate json-only=$QLAB/work/SLUG/lora-json --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key $VLLM_API_KEY --port 8000"
```

| Flag | Value | Why |
|---|---|---|
| --enable-lora | on | Turns on the LoRA kernels (punica-style batched LoRA) |
| --lora-modules name=path ... | pirate=..., json-only=... | Adapters loaded at startup; "name" is what clients send as "model" |
| --max-loras | 4 | DIFFERENT adapters usable inside ONE batch (GPU slots). More distinct adapters wait for a free slot |
| --max-lora-rank | 16 | Size of every slot: the LARGEST rank you serve (8, 16, 32, 64...) |
| --max-cpu-loras | 16 | Adapters kept in CPU RAM; least-recently-used are evicted |
| --served-model-name | SERVED_NAME | Name of the BASE model (no adapter) |

**Step 2.** Wait 2 minutes, then list the models. Type exactly this, then press Enter:

```
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY" | python -m json.tool | grep '"id"'
```

YOU SHOULD SEE:

```
"id": "SERVED_NAME",
"id": "pirate",
"id": "json-only",
```

**Step 3.** Choose the adapter PER REQUEST with the "model" field. Type exactly this, then press Enter:

```
curl -s http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" -d '{"model":"pirate","messages":[{"role":"user","content":"What is a GPU?"}],"max_tokens":80}'
```

**Step 4.** Compare base and both adapters side by side. Type exactly this, then press Enter:

```
python $S/multi_lora_client.py --url http://localhost:8000 --models SERVED_NAME pirate json-only --prompt "Explain what a GPU is in 2 sentences."
```

YOU SHOULD SEE (tested against a mock server with the same API; your text differs):

```
server lists: ['SERVED_NAME', 'pirate', 'json-only']
[SERVED_NAME] (0.81s)
A GPU is a processor built for many parallel calculations...
[pirate] (0.84s)
Arr, a GPU be the mighty engine that crunches yer numbers...
[json-only] (0.79s)
{"answer": "A GPU is a parallel processor...", "confidence": 0.9}
```

{{include scripts/multi_lora_client.py}}

**Step 5.** The same thing from Python with the official OpenAI SDK (any app can do this). Create `client_demo.py` with:

```
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["VLLM_API_KEY"])
for model in ["SERVED_NAME", "pirate", "json-only"]:      # base, then two adapters on the SAME server
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=80,
                                       messages=[{"role": "user", "content": "What is a GPU?"}])
    print(model, "->", r.choices[0].message.content)
```

Run it with `pip install openai && python client_demo.py`.

**Step 6.** Load and unload adapters WITHOUT restarting (dynamic LoRA). Restart the server with the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (put it before `vllm serve` in the tmux command), then type exactly this and press Enter:

```
curl -s -X POST http://localhost:8000/v1/load_lora_adapter -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" -d '{"lora_name":"legal","lora_path":"/home/ubuntu/qlab/work/SLUG/lora-legal"}'
```

Unload with `/v1/unload_lora_adapter` and `{"lora_name":"legal"}`.

**Step 7.** Adapters on a QUANTIZED base: stop the server and start it again with `$QLAB/work/SLUG/gptq` (or fp8) instead of MODEL_ID, same LoRA flags. vLLM applies the 16-bit adapter on top of the 4-bit/FP8 kernels. Measure the adapters' quality with compare_answers.py (5.7 D): an adapter trained on the bf16 base usually transfers well, but if it degrades, train it ON the quantized model with `recovery_lora.py --quantized` (Part 7.4) and teacher data.

| If you see... | It means... | Do this next |
|---|---|---|
| ValueError: LoRA rank 32 is greater than max_lora_rank 16 | An adapter's rank is bigger than the slots | Set --max-lora-rank to the largest rank (grep '"r"' adapter_config.json) |
| 404 model not found | Typo, or adapter failed to load | Check the server log (tmux attach -t vllm); names are case-sensitive |
| Adapter fails: target modules not supported | Adapter trained on another base or exotic modules | Retrain on MODEL_ID with standard target modules |
| Throughput drops with many adapters | More distinct adapters per batch than --max-loras | Raise --max-loras (costs VRAM, see 6B.2) or group traffic by adapter |

WATCH OUT: Dynamic loading lets anyone with the API key load files from the server's disk. In production keep VLLM_ALLOW_RUNTIME_LORA_UPDATING off and ship adapters with --lora-modules (or in the container image). In Kubernetes, add the same flags to the args list in infra/k8s/vllm.yaml.

CHECKPOINT: /v1/models lists the base + 2 adapters; the three answers clearly differ; you served the adapters on the quantized base too.

## SECTION 6B.4 - SGLang: install, launch, multi-LoRA

WHERE: GPU machine terminal
GOAL: A fourth environment with SGLang that serves your model with an OpenAI API, adapters and metrics.

**Step 1.** Create the environment (SGLang pins its own torch/kernels, so it gets its own). Type exactly this, then press Enter:

```
uv venv -p 3.12 $QLAB/envs/sglang && uv pip install --python $QLAB/envs/sglang/bin/python "sglang[all]==0.5.19"
```

**Step 2.** Check it. Type exactly this, then press Enter:

```
$QLAB/envs/sglang/bin/python -c "import sglang; print('sglang', sglang.__version__)"
```

YOU SHOULD SEE: `sglang 0.5.19` (a newer version is fine; then check flags with `python -m sglang.launch_server --help`).

**Step 3.** Stop vLLM first (one engine per GPU): `tmux kill-session -t vllm`.

**Step 4.** Launch SGLang. Type exactly this, then press Enter:

```
tmux new -d -s sglang "source $QLAB/envs/sglang/bin/activate; python -m sglang.launch_server --model-path MODEL_ID --served-model-name SERVED_NAME --host 127.0.0.1 --port 30000 --context-length 8192 --mem-fraction-static 0.85 --tp-size 1 --api-key $VLLM_API_KEY --enable-metrics"
```

| Flag | Value | Why |
|---|---|---|
| --model-path | MODEL_ID or a quantized folder | Same checkpoints as vLLM (compressed-tensors FP8/GPTQ/AWQ load directly) |
| --mem-fraction-static | 0.85 | Share of VRAM for weights + KV pool (vLLM's --gpu-memory-utilization) |
| --tp-size | 1, 2, 4, 8 | Tensor parallel GPUs |
| --context-length | 8192 | Longest request |
| --enable-metrics | on | Prometheus /metrics (sglang:... names) for the autoscaler |
| --schedule-policy | lpm (default with radix cache) | "Longest prefix match" scheduling: requests sharing a cached prefix run together |
| --disable-radix-cache | off by default | Only for experiments (6B.5) |

**Step 5.** Wait 2 minutes, then type exactly this and press Enter:

```
curl -s http://127.0.0.1:30000/v1/chat/completions -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" -d '{"model":"SERVED_NAME","messages":[{"role":"user","content":"Say hi in 5 words"}],"max_tokens":20}'
```

YOU SHOULD SEE: a normal OpenAI-style JSON answer.

**Step 6.** Multi-LoRA on SGLang. Restart with these extra flags:

```
--enable-lora --lora-paths pirate=$QLAB/work/SLUG/lora-pirate json-only=$QLAB/work/SLUG/lora-json --max-loras-per-batch 4 --max-lora-rank 16
```

**Step 7.** Call an adapter. SGLang reads the adapter from the model string `SERVED_NAME:ADAPTER` (it splits at the first ":", so SERVED_NAME must not contain ":"). Type exactly this, then press Enter:

```
python $S/multi_lora_client.py --url http://127.0.0.1:30000 --models SERVED_NAME SERVED_NAME:pirate SERVED_NAME:json-only
```

**Step 8.** NF4 (from 5.5) can be served only here: `--model-path $QLAB/work/SLUG/nf4 --quantization bitsandbytes --load-format bitsandbytes --tp-size 1`.

WATCH OUT: Flags were verified against the source manual's SGLang 0.5.19 build; SGLang renames flags between minor versions. If a flag is rejected, run `python -m sglang.launch_server --help | grep -i lora` and use the current spelling (verify).

CHECKPOINT: SGLang answered on port 30000 as the base and as both adapters.

## SECTION 6B.5 - RadixAttention vs PagedAttention, explained and measured

WHERE: GPU machine terminal
GOAL: You can explain both KV-cache designs and you have numbers showing what prefix caching buys on a shared-prompt workload.

**PagedAttention (vLLM).** The KV cache is split into fixed-size blocks (e.g. 16 tokens), like pages of operating-system memory. A request's cache is a list of block pointers, so memory is not wasted on fragmentation and blocks can be shared. vLLM's Automatic Prefix Caching hashes each FULL block of tokens; a new request whose first blocks hash the same reuses them instead of recomputing (on by default in the V1 engine).

**RadixAttention (SGLang).** The KV cache is kept in a radix tree (prefix tree) of token sequences. Every finished or running request leaves its tokens in the tree; a new request walks the tree and reuses the longest matching prefix, at token granularity, including branches (many follow-ups to the same conversation, many few-shot variants, tree-of-thought). Least-recently-used leaves are evicted first, and the scheduler orders waiting requests to maximise cache hits (longest-prefix-match).

| | PagedAttention + prefix caching (vLLM) | RadixAttention (SGLang) |
|---|---|---|
| Unit of reuse | Full KV blocks (hash per block) | Any token prefix in a tree |
| Branching / multi-turn | Works when blocks match exactly | Native: tree of shared histories |
| Scheduling | First-come with prefix-aware batching | Cache-aware (longest prefix match) |
| Where it wins | General serving, broad model support | Agents, RAG with one long system prompt, multi-turn chat, few-shot evals |

Honest summary: both engines now cache prefixes, so on simple traffic the difference is small. RadixAttention's lead shows on branching, multi-turn and heavy-shared-prefix traffic. Measure on YOUR traffic shape:

**Step 1.** SGLang with a 1,500-word shared system prompt on every request (SGLang still running from 6B.4). Type exactly this, then press Enter:

```
cd $QLAB/work/SLUG && python $S/bench.py --url http://127.0.0.1:30000 --model SERVED_NAME --label sglang-prefix --concurrency 1 8 32 --shared-prefix-words 1500
```

**Step 2.** Stop SGLang (`tmux kill-session -t sglang`), start plain vLLM (the 4.7 command, port 8000), wait, then:

```
python $S/bench.py --url http://localhost:8000 --model SERVED_NAME --label vllm-prefix --concurrency 1 8 32 --shared-prefix-words 1500
```

**Step 3.** See what caching itself buys: restart vLLM with `--no-enable-prefix-caching` added, then run Step 2 again with `--label vllm-nocache`.

**Step 4.** Fill this table from the printed rows (TTFT is where prefix caching shows: the shared 1,500 words are computed once instead of per request):

| Label | conc 1 TTFT p50 | conc 8 TTFT p95 | conc 32 tok/s |
|---|---|---|---|
| vllm-nocache | | | |
| vllm-prefix | | | |
| sglang-prefix | | | |

**Explained:** `--shared-prefix-words` (new in bench.py) puts the same long system message in front of every request, like a RAG instruction block or an agent's tool list.

CHECKPOINT: vllm-prefix TTFT is clearly lower than vllm-nocache; you know which engine wins on this traffic shape.

## SECTION 6B.6 - Speculative decoding lab: 8B target + 0.6B draft

WHERE: GPU machine terminal
GOAL: Measured ON/OFF numbers for draft-model, n-gram and EAGLE-3 speculative decoding, and a written decision.

The idea in one paragraph: a small DRAFT model (or a cheap method) guesses the next k tokens; the big TARGET model checks all k in ONE forward pass and keeps the longest correct run plus one token of its own. The output is exactly what the target would have produced (greedy) or has the same distribution (sampling); only the speed changes. With acceptance rate a and k drafted tokens, the target produces E = (1 - a^(k+1)) / (1 - a) tokens per pass (a = 0.6, k = 4 -> 2.31). It helps most at LOW concurrency (the GPU has idle compute); at high concurrency the GPU is already busy and speculation can make throughput worse.

IMPORTANT (current vLLM): the old flags `--speculative-model` and `--num-speculative-tokens` were removed. vLLM 0.29 takes ONE JSON flag, `--speculative-config '{...}'`. The fields below match the flags verified in the source manual's vLLM 0.29 build.

| Method | --speculative-config | Needs |
|---|---|---|
| Draft model | {"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":4} | Small model with the SAME tokenizer/vocabulary as the target |
| N-gram / prompt lookup | {"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2} | Nothing; great when answers copy the prompt (summaries, extraction, code edits) |
| EAGLE-3 | {"method":"eagle3","model":"EAGLE3_HEAD_REPO","num_speculative_tokens":3} | A trained EAGLE-3 head for THIS exact target (search Hugging Face "eagle3 MODEL_NAME") |
| Medusa | {"method":"medusa","model":"MEDUSA_HEADS_REPO","num_speculative_tokens":3} | Medusa heads trained for this target (older technique) (verify) |
| MTP (built-in heads) | {"method":"mtp","num_speculative_tokens":1} | Models shipped with multi-token-prediction layers (e.g. DeepSeek-V3 family) (verify) |

| Placeholder | Replace with | Example |
|---|---|---|
| MODEL_ID | 8B target | Qwen/Qwen3-8B |
| DRAFT_ID | Same-family small model (same tokenizer) | Qwen/Qwen3-0.6B |

**Step 1.** Baseline (speculation OFF). Start the server:

```
tmux new -d -s vllm "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name SERVED_NAME --max-model-len 8192 --api-key $VLLM_API_KEY --port 8000"
```

**Step 2.** Wait 2 minutes, then type exactly this and press Enter:

```
cd $QLAB/work/SLUG && python $S/bench.py --url http://localhost:8000 --model SERVED_NAME --label spec-off --concurrency 1 4 16
```

**Step 3.** Restart with the 0.6B draft model. Type exactly this, then press Enter:

```
tmux kill-session -t vllm; tmux new -d -s vllm "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name SERVED_NAME --max-model-len 8192 --api-key $VLLM_API_KEY --port 8000 --speculative-config '{\"method\":\"draft_model\",\"model\":\"DRAFT_ID\",\"num_speculative_tokens\":4}'"
```

**Step 4.** Wait 2-3 minutes, then benchmark the SAME way. Type exactly this, then press Enter:

```
python $S/bench.py --url http://localhost:8000 --model SERVED_NAME --label spec-draft --concurrency 1 4 16
```

**Step 5.** Read the acceptance rate from the server. Type exactly this, then press Enter:

```
python $S/spec_metrics.py --url http://localhost:8000 --k 4
```

YOU SHOULD SEE (tested against a mock /metrics with the vLLM counter names; your numbers differ):

```
vllm:spec_decode_num_accepted_tokens_total                   2,400
vllm:spec_decode_num_draft_tokens_total                      4,000
vllm:spec_decode_num_drafts_total                            1,000
acceptance rate a = 0.600
mean accepted tokens per round = 2.40 (+1 bonus token)
theory: tokens per target pass E = 2.31  (k = 4)
```

{{include scripts/spec_metrics.py}}

**Step 6.** N-gram (no draft model). Restart with `--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'`, then run bench.py with `--label spec-ngram`. Expect little gain on bench.py's open-ended prompts and big gains on copy-heavy tasks (summarise or edit a pasted document).

**Step 7 (best quality of speculation).** EAGLE-3: find a head trained for your exact target on Hugging Face, restart with the eagle3 config from the table, run bench.py with `--label spec-eagle3` and spec_metrics.py with `--k 3`.

**Step 8.** Speculation on a QUANTIZED target works the same: replace MODEL_ID with $QLAB/work/SLUG/gptq (the draft can stay 16-bit; it is small).

**Step 9.** The same on SGLang (restart from 6B.4 with these flags; names verified in the source manual's SGLang build):

```
--speculative-algorithm STANDALONE --speculative-draft-model-path DRAFT_ID --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

(EAGLE-3 on SGLang: `--speculative-algorithm EAGLE3 --speculative-draft-model-path EAGLE3_HEAD_REPO` with the same step flags; n-gram: `--speculative-algorithm NGRAM`.)

**Step 10.** Fill the decision table (ITL = time between streamed tokens, what a single user feels):

| Label | conc 1 ITL p50 ms | conc 1 tok/s | conc 16 tok/s | acceptance a |
|---|---|---|---|---|
| spec-off | | | | - |
| spec-draft | | | | |
| spec-ngram | | | | |
| spec-eagle3 | | | | |

Decision rule: turn speculation ON only if conc-1 ITL improves by > 20% AND conc-16 (your real peak) throughput does not drop more than ~5%. Typical pattern (verify on your GPU): 1.5-2.5x lower ITL at concurrency 1, shrinking or negative at 16+.

| If you see... | It means... | Do this next |
|---|---|---|
| Server fails: vocab / tokenizer mismatch | Draft and target tokenizers differ | Use a draft from the same model family |
| spec-on slower even at concurrency 1 | Low acceptance (a < 0.4) | k = 2, a closer drafter, or EAGLE-3 |
| Out of memory at startup | Draft weights + its KV do not fit | Lower --max-model-len or --gpu-memory-utilization headroom |
| unrecognized arguments: --speculative-model | Old tutorial flag | Use --speculative-config JSON (this section) |

CHECKPOINT: spec-off and spec-draft rows exist; acceptance rate printed; a written ON/OFF decision for your peak concurrency.

## SECTION 6B.7 - GGML vs GGUF: history, why GGML files died, and the byte layout

WHERE: GPU machine terminal, gguf environment
GOAL: You can explain GGML vs GGUF, read any GGUF file's header, and compute the real bits per weight of every llama.cpp format.

**History.**

| When | What | Problem it had |
|---|---|---|
| 2022-early 2023 | GGML = Georgi Gerganov's C tensor library; llama.cpp saved models in "GGML" .bin files (unversioned "ggml", then "ggmf", then "ggjt" v1-v3, which added mmap loading) | Hyper-parameters stored as a fixed list of numbers; no tokenizer details, no architecture name, no room for new fields: every new model or quant change broke old files |
| Aug 2023 | GGUF (GGML Universal File) replaces GGML files in llama.cpp | Solved with a self-describing key/value header, versioning, and alignment for fast mmap |
| Today | GGML is still the tensor LIBRARY underneath llama.cpp and whisper.cpp; "GGML" FILES are obsolete; GGUF is version 3 | Old .bin files: convert with llama.cpp's convert_llama_ggml_to_gguf.py (still in the repo, checked) or, better, re-convert from the original Hugging Face model |

**GGUF layout (from llama.cpp's gguf-py constants: magic 0x46554747 = "GGUF", version 3, default alignment 32 bytes):**

```
[ "GGUF" | version=3 | tensor_count | metadata_kv_count ]
[ metadata: general.architecture=llama, llama.context_length=8192, llama.block_count=30,
            tokenizer.ggml.model=gpt2, tokenizer.chat_template=..., general.file_type=7, ... ]
[ tensor infos: name, shape, type (Q4_K, Q6_K, F32...), offset ]
[ padding to general.alignment (32 bytes) ]
[ tensor data: blocks of quantized weights, memory-mapped directly from disk ]
```

**Block formats, exact bytes (from llama.cpp's GGML_QUANT_SIZES table):**

| Type | Weights per block | Bytes per block | Bits per weight | What is inside |
|---|---|---|---|---|
| Q4_0 | 32 | 18 | 4.50 | 1 fp16 scale + 32 x 4-bit |
| Q4_1 | 32 | 20 | 5.00 | fp16 scale + fp16 min + 32 x 4-bit |
| Q5_0 | 32 | 22 | 5.50 | scale + 32 x 5-bit (high bits stored apart) |
| Q8_0 | 32 | 34 | 8.50 | fp16 scale + 32 x int8 |
| Q2_K | 256 | 84 | 2.63 | super-block: 16 sub-blocks with 4-bit scales/mins + 2-bit weights |
| Q3_K | 256 | 110 | 3.44 | 3-bit weights, 6-bit sub-block scales |
| Q4_K | 256 | 144 | 4.50 | fp16 d + fp16 dmin + 12 bytes of 6-bit sub-scales + 4-bit weights |
| Q5_K | 256 | 176 | 5.50 | as Q4_K + 1 extra bit per weight |
| Q6_K | 256 | 210 | 6.56 | 6-bit weights, 8-bit sub-scales |
| IQ2_XXS | 256 | 66 | 2.06 | lattice codebook (E8); needs an importance matrix |
| IQ3_XXS | 256 | 98 | 3.06 | codebook-based 3-bit |
| IQ4_NL | 32 | 18 | 4.50 | 4-bit non-linear (like NF4) levels |
| IQ4_XS | 256 | 136 | 4.25 | non-linear 4-bit + 6-bit sub-scales |
| MXFP4 | 32 | 17 | 4.25 | OCP microscaling FP4 (8-bit shared exponent) |

"_S / _M / _L" names (Q4_K_M, Q5_K_S...) are MIXES: Q4_K_M stores most tensors as Q4_K but some sensitive ones (part of attn_v and ffn_down) as Q6_K, so the file averages ~4.85 bits per weight.

**Step 1.** Look inside your own GGUF. Type exactly this, then press Enter:

```
source $QLAB/envs/gguf/bin/activate && python $S/inspect_gguf.py $QLAB/work/SLUG/gguf/SLUG-Q4_K_M.gguf
```

YOU SHOULD SEE (REAL output from the test run on SmolLM2-135M converted to Q8_0):

```
file: t/smol-q8_0.gguf
tensors: 272   metadata keys: 41
METADATA
  general.architecture                          llama
  llama.block_count                             30
  llama.context_length                          8192
  llama.embedding_length                        576
  llama.attention.head_count                    9
  llama.attention.head_count_kv                 3
  general.file_type                             7
  tokenizer.ggml.model                          gpt2
  tokenizer.chat_template                       {% for message in messages %}...
TENSOR TYPES (real bits per weight = bytes x 8 / weights)
  type       tensors         weights        MB  bits/w
  Q8_0           211     134,479,872     142.9   8.500
  F32             61          35,136       0.1  32.000
  TOTAL 134,515,008 weights, 143.0 MB, average 8.506 bits/weight
```

Run it on your Q4_K_M file and you will see the mix: mostly Q4_K at 4.500, some Q6_K at 6.562, F32 norms, and the average (~4.8-4.9).

{{include scripts/inspect_gguf.py}}

CHECKPOINT: You can name three things GGUF stores that GGML files could not, and you saw the Q4_K/Q6_K mix in your own file.

## Common mistake at this level

| Mistake | How to spot it | Fix |
|---|---|---|
| --max-lora-rank far above your adapters' rank | VRAM/KV capacity drops for nothing | Set it to the largest adapter rank (6B.2) |
| Merging several adapters into one model | Behaviours interfere, one model per customer | Serve them unmerged with multi-LoRA (6B.3) |
| Copying old tutorials' --speculative-model flag | "unrecognized arguments" | --speculative-config JSON (6B.6) |
| Turning speculation on for peak traffic | Throughput falls at high concurrency | Decide from the concurrency-16 row |
| Judging RadixAttention on random prompts | No difference measured | Test with a shared prefix / multi-turn traffic (6B.5) |
| Treating "GGML" and "GGUF" files as the same | Old .bin files fail to load | Re-convert from the HF model, or convert_llama_ggml_to_gguf.py |
