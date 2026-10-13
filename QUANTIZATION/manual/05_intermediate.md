# PART 5 - INTERMEDIATE: calibration, every 4/8-bit method, full evaluation, tuning, push to Hugging Face

WHY / THEORY. **What:** build every serious variant (GPTQ, AWQ, W8A8, NF4, GGUF) from YOUR calibration data, measure each against the original in five ways, tune the settings, and save the winner to Hugging Face. **Why:** 4-bit halves memory again versus FP8, but quality now depends on calibration data and method, so you must measure. **When:** the model does not fit your GPU/concurrency at FP8, you want the cheapest GPU that passes your gates, or your GPU has no FP8 (A100: use W8A8).

## SECTION 5.1 - Create calibration data from your own prompts

WHERE: GPU machine, VS Code editor and terminal
GOAL: data/calib.jsonl with 256+ real conversations the quantizer learns from.

**Step 1.** Type exactly this, then press Enter:

```
mkdir -p $QLAB/data && cd $QLAB/data
```

**Step 2.** In VS Code create `$QLAB/data/calib.jsonl`. Each line is one JSON object with a "messages" list, exactly like this line:

```
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Open Settings > Security > Reset password, then follow the email link."}]}
```

(Plain documents work too: `{"text": "any paragraph of your domain text"}` per line.)

**Step 3.** Add at least 256 lines from REAL usage (support tickets, chats, documents turned into Q&A). Remove personal data first. More variety beats more volume.

**Step 4.** Check the file. Type exactly this, then press Enter:

```
wc -l calib.jsonl && python3 -c "import json; [json.loads(l) for l in open('calib.jsonl')]; print('valid JSONL')"
```

YOU SHOULD SEE: `256 calib.jsonl` (or more) and `valid JSONL`.

WATCH OUT: No own data? Leave out --calib and quantize.py uses HuggingFaceH4/ultrachat_200k (public chat). It works, but the model is then tuned for generic chat, not your task. Self-test question 10 shows why this matters.

CHECKPOINT: valid JSONL printed; 256+ lines.
If not: JSONDecodeError names the line: usually a raw newline or an unescaped quote inside a string.

## SECTION 5.2 - Merge a LoRA adapter first (skip if you have none)

WHERE: GPU machine terminal, quant environment
GOAL: A plain 16-bit model with your fine-tuned adapter baked in, verified numerically, ready to quantize.

**Step 1.** Type exactly this (your adapter folder or HF id), then press Enter:

```
source $QLAB/envs/quant/bin/activate && python $S/merge_lora.py --base MODEL_ID --adapter ADAPTER_PATH --out $QLAB/work/SLUG/merged
```

YOU SHOULD SEE:

```
verify: max |logit diff| adapter vs merged = <small, e.g. 1e-3>  (should be < 1e-2)
merged model saved to .../merged
```

From now on use `$QLAB/work/SLUG/merged` wherever this manual says MODEL_ID (it is your new original).

{{include scripts/merge_lora.py}}

**Explained:** `merge_and_unload()` computes W_new = W + (B x A) x scale for every adapted layer and removes the adapter, so the quantizer calibrates the weights you actually serve. The script compares the model's outputs with and without merging; a big difference means the adapter was trained on a different base.

WATCH OUT: The adapter MUST be trained on this exact base. Merge into the 16-bit base, never into a 4-bit model. Several adapters? Do not merge: serve them unmerged (Part 6, multi-LoRA).

CHECKPOINT: logit diff < 1e-2; merged/config.json exists.

## SECTION 5.3 - Build GPTQ W4A16

WHERE: GPU machine terminal, quant environment, inside tmux
GOAL: $QLAB/work/SLUG/gptq holds a 4-bit checkpoint (about 1/4 of bf16 + embeddings).

**Step 1.** Type exactly this, then press Enter:

```
python $S/quantize.py --model MODEL_ID --method gptq --calib $QLAB/data/calib.jsonl --num-samples 512 --max-seq-len 2048 --out $QLAB/work/SLUG/gptq
```

YOU SHOULD SEE:

```
... GPTQ ... (one line per layer with an error value) ...
{ "method": "gptq", "group_size": 128, "dampening": 0.01, "seconds": <n>, "size_gb": <n> }
```

Time: roughly 10-30 minutes for 8B on one L40S; hours for 70B.

| If you see... | It means... | Do this next |
|---|---|---|
| ValueError: weight column counts not divisible by group_size | A layer width is not a multiple of 128 (seen in testing with a 576-wide model) | Add --group-size 64 (or 32), or add that layer to --ignore |
| NaN / "not positive definite" | Hessian ill-conditioned | --dampening 0.05 (or 0.1) and rerun |
| Very slow (hours) | Large model or long samples | --max-seq-len 1024 or --num-samples 256 |
| Killed, no traceback | Out of CPU RAM | dmesg / tail; bigger-RAM machine |

CHECKPOINT: `grep format $QLAB/work/SLUG/gptq/config.json` shows `pack-quantized`.

## SECTION 5.4 - Build AWQ W4A16

WHERE: GPU machine terminal, quant environment
GOAL: $QLAB/work/SLUG/awq holds a second 4-bit candidate.

**Step 1.** Type exactly this, then press Enter:

```
python $S/quantize.py --model MODEL_ID --method awq --calib $QLAB/data/calib.jsonl --num-samples 512 --out $QLAB/work/SLUG/awq 2>&1 | tee $QLAB/work/SLUG/awq.log
```

**Step 2.** Check whether AWQ knew your architecture. Type exactly this, then press Enter:

```
grep -c "not found in mappings" $QLAB/work/SLUG/awq.log
```

YOU SHOULD SEE: `0`. (1 or more = generic Llama-style mappings were used; measure its quality carefully in 5.7.)

WATCH OUT: AWQ in llm-compressor 0.13 needs an NVIDIA GPU (it pins memory). Neither GPTQ nor AWQ wins everywhere: build both, keep the better one on YOUR evaluation.

CHECKPOINT: awq/config.json shows pack-quantized.

## SECTION 5.5 - Build W8A8 INT8 (SmoothQuant) and NF4

WHERE: GPU machine terminal, quant environment
GOAL: An INT8 weights+activations variant (fast on A100/A10, which have no FP8) and an NF4 variant (no calibration).

**Step 1.** W8A8. Type exactly this, then press Enter:

```
python $S/quantize.py --model MODEL_ID --method w8a8 --calib $QLAB/data/calib.jsonl --num-samples 512 --smoothing 0.8 --out $QLAB/work/SLUG/w8a8
```

YOU SHOULD SEE: size about half of bf16; config.json format `int-quantized`.

**Step 2.** NF4. Type exactly this, then press Enter:

```
python $S/quantize.py --model MODEL_ID --method nf4 --out $QLAB/work/SLUG/nf4
```

**Explained:** `SmoothQuantModifier(smoothing_strength=0.8)` moves 80% of the activation outlier range into the weights before `GPTQModifier(scheme="W8A8")` rounds both weights and activations to INT8. For NF4, `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` quantizes while loading; `llm_int8_skip_modules` keeps lm_head in 16-bit.

WATCH OUT: vLLM 0.29 removed bitsandbytes loading, so NF4 is for evaluation, QLoRA training and quick experiments here; serve it with SGLang (`--quantization bitsandbytes --load-format bitsandbytes`, tensor parallel 1 only) (verify) or choose GPTQ/AWQ for serving.

CHECKPOINT: w8a8/ and nf4/ both contain config.json and .safetensors files.

## SECTION 5.6 - Make GGUF files (laptops, CPUs, Macs) with llama.cpp

WHERE: GPU machine terminal, gguf environment
GOAL: Q8_0 and Q4_K_M GGUF files built with an importance matrix, tested with llama-server.

**Step 1.** Type exactly this, then press Enter:

```
mkdir -p $QLAB/work/SLUG/gguf && MODEL_DIR=$($QLAB/envs/quant/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('MODEL_ID'))") && echo $MODEL_DIR
```

**Step 2.** Convert to a 16-bit GGUF. Type exactly this, then press Enter:

```
source $QLAB/envs/gguf/bin/activate && python $QLAB/llama.cpp/convert_hf_to_gguf.py $MODEL_DIR --outtype bf16 --outfile $QLAB/work/SLUG/gguf/SLUG-bf16.gguf
```

**Step 3.** Make plain-text calibration for the importance matrix. Type exactly this, then press Enter:

```
$QLAB/envs/quant/bin/python -c "import json; open('$QLAB/data/calib.txt','w').write('\n\n'.join((json.loads(l).get('text') or ' '.join(m['content'] for m in json.loads(l)['messages'])) for l in open('$QLAB/data/calib.jsonl')))"
```

**Step 4.** Compute the importance matrix. Type exactly this, then press Enter:

```
$QLAB/llama.cpp/build/bin/llama-imatrix -m $QLAB/work/SLUG/gguf/SLUG-bf16.gguf -f $QLAB/data/calib.txt -o $QLAB/work/SLUG/gguf/imatrix.gguf -ngl 99
```

**Step 5.** Quantize to Q4_K_M and Q8_0. Type exactly this, then press Enter:

```
for Q in Q4_K_M Q8_0; do $QLAB/llama.cpp/build/bin/llama-quantize --imatrix $QLAB/work/SLUG/gguf/imatrix.gguf $QLAB/work/SLUG/gguf/SLUG-bf16.gguf $QLAB/work/SLUG/gguf/SLUG-$Q.gguf $Q; done; ls -lh $QLAB/work/SLUG/gguf
```

**Step 6.** Serve Q4_K_M and test it. Type exactly this, then press Enter:

```
tmux new -d -s llama "$QLAB/llama.cpp/build/bin/llama-server -m $QLAB/work/SLUG/gguf/SLUG-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 --jinja -a SERVED_NAME --api-key $VLLM_API_KEY"
```

**Step 7.** Type exactly this, then press Enter:

```
sleep 20; curl -s http://localhost:8080/v1/chat/completions -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" -d '{"model":"SERVED_NAME","messages":[{"role":"user","content":"Say hi in 5 words"}],"max_tokens":20}'; tmux kill-session -t llama
```

| If you see... | It means... | Do this next |
|---|---|---|
| unknown pre-tokenizer hash | llama.cpp older than the model | cd $QLAB/llama.cpp && git pull && cmake --build build -j, retry |
| Model architecture not supported | Same cause, or not supported yet | Update llama.cpp; if still missing, GGUF is not available for this model yet |
| imatrix: unknown output format | Older llama.cpp expects .dat | Use -o imatrix.dat and --imatrix imatrix.dat |

**Explained:** `convert_hf_to_gguf.py` rewrites the Hugging Face weights into one GGUF file (weights + tokenizer + chat template). `llama-imatrix` runs your text through the model and records which weights matter most; `llama-quantize --imatrix` protects them. `-ngl 99` puts all layers on the GPU (use 0 on a CPU-only laptop). Copy the .gguf to your PC with WinSCP and run it with llama.cpp, LM Studio or Ollama.

CHECKPOINT: SLUG-Q4_K_M.gguf exists and llama-server answered.

## SECTION 5.7 - Evaluate and COMPARE: original vs every quantized variant

WHERE: GPU machine terminal
GOAL: Five kinds of evidence per variant, collected into one REPORT.md.

The five measurements and what each proves:

| # | Measurement | Tool | Proves | Weakness |
|---|---|---|---|---|
| A | Perplexity | eval_ppl.py | General language modelling quality | Can hide task-specific damage |
| B | KL divergence + top-1 agreement | kl_compare.py | HOW MUCH the predictions changed vs the original | Needs both models in turn on the GPU |
| C | Standard benchmarks | lm-evaluation-harness | Maths (GSM8K), knowledge (MMLU), reasoning (ARC) | Not your use case |
| D | Your own questions + judge | compare_answers.py | Quality retention % on YOUR task, win/tie/loss | Judge bias; read answers yourself |
| E | Speed and capacity | bench.py | tok/s, TTFT, failures at your concurrency | Only valid on the same GPU |

### A - Perplexity of every variant

**Step 1.** Type exactly this, then press Enter (runs one after another; takes a while):

```
source $QLAB/envs/quant/bin/activate && for V in gptq awq w8a8 nf4; do python $S/eval_ppl.py --model $QLAB/work/SLUG/$V --out $QLAB/work/SLUG/eval/ppl-$V.json; done
```

WATCH OUT: quip_gptq checkpoints (Part 7) store weights in a ROTATED basis; transformers does not apply the rotation, so eval_ppl.py prints a nonsense number (the test run printed 4,647,319,085). Measure rotated models through vLLM (C, D, E) instead.

### B - KL divergence and top-1 agreement (how the predictions changed)

**Step 2.** Type exactly this, then press Enter:

```
for V in fp8 gptq awq w8a8; do python $S/kl_compare.py --original MODEL_ID --quantized $QLAB/work/SLUG/$V --out $QLAB/work/SLUG/eval/kl-$V.json; done
```

YOU SHOULD SEE per variant (real output, W8A8 of the 135M test model):

```
{ "positions": 1024, "kl_orig_quant": 0.01711, "top1_agreement_%": 91.11,
  "ppl_original": 28.931, "ppl_quantized": 29.286, "ppl_change_%": 1.23 }
```

**Explained:** for every position both models predict the next token. KL = how different the two probability lists are (0 = identical; good 4-bit on 7B+ is often < 0.05). top-1 agreement = how often both pick the same next token (good: > 95% on 7B+). Use `--text your_domain.txt` to measure on YOUR text.

{{include scripts/kl_compare.py}}

### C - Standard benchmarks (lm-evaluation-harness against the running server)

**Step 3.** Serve the variant WITHOUT an API key, bound to localhost only (safe: nothing outside the machine can reach it). Type exactly this (V = gptq here), then press Enter:

```
tmux new -d -s eval "source $QLAB/envs/vllm/bin/activate; vllm serve $QLAB/work/SLUG/gptq --served-model-name SERVED_NAME --max-model-len 4096 --host 127.0.0.1 --port 8000"
```

**Step 4.** Wait 90 seconds, then type exactly this and press Enter:

```
source $QLAB/envs/quant/bin/activate && lm_eval --model local-completions --model_args model=SERVED_NAME,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=MODEL_ID,num_concurrent=16,max_retries=3 --tasks gsm8k,arc_challenge,hellaswag --limit 500 --output_path $QLAB/work/SLUG/eval/lmeval-gptq
```

YOU SHOULD SEE: a table with `gsm8k ... exact_match`, `arc_challenge ... acc_norm`, `hellaswag ... acc_norm`.

**Step 5.** Stop the server: `tmux kill-session -t eval`. Repeat Steps 3-5 for the ORIGINAL (serve MODEL_ID, output folder lmeval-bf16) and every other variant (fp8, awq, w8a8).

WATCH OUT: Use the SAME --limit, tasks and few-shot settings for every variant, or the numbers cannot be compared. `--limit 500` keeps it fast while learning; remove it for the final decision.

### D - Your own questions: quality retention % (original vs quantized side by side)

**Step 6.** Create `$QLAB/data/questions.jsonl` with 30-200 real questions, one per line (reference is optional but makes exact-match possible):

```
{"id": "q1", "question": "What is the refund window for annual plans?", "reference": "30 days"}
{"id": "q2", "question": "Write a SQL query that counts orders per customer."}
```

**Step 7.** Start the ORIGINAL on port 8001 using 45% of the GPU. Type exactly this, then press Enter:

```
tmux new -d -s orig "source $QLAB/envs/vllm/bin/activate; vllm serve MODEL_ID --served-model-name orig --max-model-len 4096 --gpu-memory-utilization 0.45 --host 127.0.0.1 --port 8001"
```

**Step 8.** Start the QUANTIZED variant on port 8000 next to it. Type exactly this, then press Enter:

```
sleep 120; tmux new -d -s quant "source $QLAB/envs/vllm/bin/activate; vllm serve $QLAB/work/SLUG/gptq --served-model-name quant --max-model-len 4096 --gpu-memory-utilization 0.40 --host 127.0.0.1 --port 8000"
```

**Step 9.** Wait 90 seconds, then type exactly this and press Enter:

```
cd $QLAB/work/SLUG && python $S/compare_answers.py --questions $QLAB/data/questions.jsonl --orig-url http://127.0.0.1:8001 --orig-model orig --quant-url http://127.0.0.1:8000 --quant-model quant --judge-url http://127.0.0.1:8001 --judge-model orig --out-dir compare-gptq
```

YOU SHOULD SEE:

```
{
  "questions": 100,
  "avg_similarity_to_original": <0-100>,
  "judge_avg_original": <1-10>,
  "judge_avg_quantized": <1-10>,
  "quality_retention_percent": <e.g. 97.8>,
  "win": <n>, "tie": <n>, "loss": <n>,
  "worst_5_ids": ["q17", ...]
}
```

**Step 10.** Open `compare-gptq/answers.jsonl` in VS Code and READ the worst_5 questions: original answer next to quantized answer. This is where you see WHAT changed in behaviour (wrong numbers, lost steps, language switches, shorter answers).

**Explained:** both models get the same question at temperature 0 (deterministic). `similarity` = text overlap with the original's answer. The judge grades each answer 1-10 against the question (and reference); `quality_retention_percent` = quantized average / original average x 100. `strip_think` removes hidden reasoning from Qwen3/DeepSeek-R1 answers before grading. Best practice: use a STRONGER judge (a bigger model or an API model) because a model judging itself is biased.

{{include scripts/compare_answers.py}}

WATCH OUT: Both servers must fit at once: 0.45 + 0.40 of the GPU. If the original does not fit in 45%, run Part D on a bigger GPU, or use a 4-bit judge only for screening.

### E - Speed (same GPU, same settings)

**Step 11.** For each variant, serve it with the production command from 4.7 (with --api-key) and run:

```
cd $QLAB/work/SLUG && python $S/bench.py --url http://localhost:8000 --model SERVED_NAME --label gptq --concurrency 1 8 32
```

(Use the variant name as --label: bf16, fp8, gptq, awq, w8a8. All rows go to bench_results.jsonl.)

### The comparison report

**Step 12.** Type exactly this, then press Enter:

```
python $S/report.py --workdir $QLAB/work/SLUG --variants bf16 fp8 gptq awq w8a8
```

YOU SHOULD SEE (shape of the table; numbers are yours):

```
| variant | size GB | ppl | ppl change % | KL | top-1 % | retention % | arc_challenge | gsm8k | hellaswag | tok/s | TTFT p95 ms |
| bf16 | | 9.12 | +0.00 | | | | 58.1 | 88.0 | 79.5 | 1:52 / 8:390 / 32:1210 | ... |
| fp8  | 9.4 | 9.15 | +0.33 | 0.004 | 98.9 | 99.6 | 58.0 (-0.1) | 87.6 (-0.4) | ... |
| gptq | 6.1 | 9.41 | +3.18 | 0.031 | 95.7 | 97.9 | 57.2 (-0.9) | 85.1 (-2.9) | ... |
```

(Illustrative numbers. The only REAL measured numbers in this manual are the ones labelled "real output from the test run".)

{{include scripts/report.py}}

CHECKPOINT: REPORT.md exists with a row per variant; you can name the smallest variant that passes EVERY gate in GATES.md.
If not: a variant fails a gate by a lot (e.g. GSM8K -10): recalibrate with better data (5.1), tune (5.8) or use a higher-precision format.

## SECTION 5.8 - "Hyperparameter tuning" for quantization: sweep the knobs

WHERE: GPU machine terminal, quant environment, inside tmux (runs for hours)
GOAL: A results table of many quantization settings, best first, so you keep as much of the learned quality as possible.

The honest truth: post-training quantization has no learning rate or epochs. What you CAN tune is how the rounding is done:

| Knob | Values to try | Effect |
|---|---|---|
| method | gptq, awq, quip_gptq | Different error-compensation strategies |
| group size | 128 -> 64 -> 32 | Smaller = more accurate, file 3-9% bigger |
| dampening | 0.01, 0.05, 0.1 | GPTQ stability; higher fixes NaN, sometimes better |
| calibration samples | 256, 512, 1024 | More = better statistics, slower |
| sequence length | 1024, 2048, 4096 | Match your real prompt lengths |
| actorder | static, group | Order of columns in GPTQ |
| ignore list | lm_head + worst layers | Keeps sensitive layers in 16-bit (mixed precision, Part 7.3) |
| calibration DATA | generic vs your domain | Usually the biggest win of all |

**Step 1.** Type exactly this, then press Enter:

```
python $S/sweep.py --model MODEL_ID --calib $QLAB/data/calib.jsonl --workdir $QLAB/work/SLUG/sweep --kl
```

YOU SHOULD SEE: one line per trial, then the best 5. Results in sweep/results.csv, sorted by KL (or perplexity change).

**Step 2.** Rebuild the winner with quantize.py using its settings (the sweep deletes weights to save disk unless you pass --keep), then run 5.7 C/D/E on it.

**Explained:** `GRID` lists the values to try; every combination is quantized and measured with the SAME eval settings. Edit `GRID` at the top of sweep.py to add 32 to group_size or "quip_gptq" to method.

{{include scripts/sweep.py}}

CHECKPOINT: results.csv exists; you rebuilt the best trial and it passes your gates.

## SECTION 5.9 - Save the winner to YOUR Hugging Face account

WHERE: GPU machine terminal, quant environment
GOAL: A private repo HF_USER/SLUG-gptq-w4a16 with weights, a model card with your measured numbers, and a version tag v1.

**Step 1.** Type exactly this, then press Enter:

```
python $S/push_to_hub.py --folder $QLAB/work/SLUG/gptq --repo HF_USER/SLUG-gptq-w4a16 --base MODEL_ID --tag v1 --eval-dir $QLAB/work/SLUG/eval
```

YOU SHOULD SEE: `uploaded https://huggingface.co/HF_USER/SLUG-gptq-w4a16  tag=v1  commit=<hash>`

**Step 2.** Open that link in the browser: the Files tab shows the .safetensors files, README.md shows your table, and the branch/tag menu shows v1.

**Explained:** `create_repo(private=True)` makes a private repository (add --public to share). `upload_folder` uploads everything as one commit. `create_tag("v1")` names that exact commit, so later `vllm serve HF_USER/SLUG-gptq-w4a16 --revision v1` always gets THIS version, even after you upload v2. The model card is generated from quant_info.json and your eval JSON files.

{{include scripts/push_to_hub.py}}

WATCH OUT: The token in HF_TOKEN must be a WRITE token. Check the licence of the base model before making a quantized copy public (Llama and Gemma licences have conditions).

CHECKPOINT: The repo exists with tag v1 and a filled model card.
If not: 403 = read-only token: create a write token (2.3) and update $QLAB/.env.

## Common mistake at the Intermediate level

| Mistake | How to spot it | Fix |
|---|---|---|
| Calibrating on text unlike production | Perplexity fine, real answers worse | Recalibrate with real prompts (5.1) |
| Comparing numbers measured differently | Different text/window/limit/few-shot | Re-run all variants with identical settings |
| Two engines on one GPU without memory split | OOM when starting the second | --gpu-memory-utilization 0.45 / 0.40 (5.7 D) |
| Trusting only perplexity | Small ppl change, GSM8K -10 | Always run C and D too |
| Evaluating a rotated (QuIP) model with transformers | Absurd perplexity | Evaluate through vLLM |
