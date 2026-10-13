# PART 2 - Set Up Your Workspace

## SECTION 2.1 - Install the PC tools

WHERE: Windows PC. Start menu > type PowerShell > click Windows PowerShell.
GOAL: Git, Python, VS Code and WinSCP are installed and each prints a version.

**Step 1.** Type exactly this, then press Enter:

```
winget install --id Git.Git -e
```

**Step 2.** Type exactly this, then press Enter:

```
winget install --id Python.Python.3.12 -e
```

**Step 3.** Type exactly this, then press Enter:

```
winget install --id Microsoft.VisualStudioCode -e
```

**Step 4.** Type exactly this, then press Enter:

```
winget install --id WinSCP.WinSCP -e
```

**Step 5.** Close PowerShell and open a NEW PowerShell window (so it sees the new programs).

**Step 6.** Type exactly this, then press Enter:

```
git --version; py -3.12 --version; code --version; ssh -V
```

YOU SHOULD SEE:

```
git version 2.x.x
Python 3.12.x
1.x.x  (VS Code)
OpenSSH_for_Windows_9.x
```

| If you see... | It means... | Do this next |
|---|---|---|
| 'winget' is not recognized | App Installer missing | Microsoft Store > search App Installer > Update, then retry |
| 'ssh' is not recognized | OpenSSH client missing | Settings > System > Optional features > Add a feature > OpenSSH Client |
| 'code' is not recognized | PATH not refreshed | Close ALL terminals and open a new one |

**Step 7.** Open VS Code. Click the Extensions icon (four squares) on the left bar.

**Step 8.** Search **Remote - SSH** (publisher Microsoft) and click Install.

**Step 9.** Search **Python** (publisher Microsoft) and click Install.

CHECKPOINT: git, Python 3.12, VS Code and ssh print versions; VS Code has Remote - SSH and Python.
If not: a command is "not recognized": open a NEW terminal; if still missing, reinstall that one tool.

## SECTION 2.2 - Create your SSH key

WHERE: PC PowerShell
GOAL: A key pair exists; the public half is ready to paste into the GPU provider.

**Step 1.** Type exactly this, then press Enter:

```
ssh-keygen -t ed25519 -C "qlab" -f $HOME\.ssh\qlab
```

**Step 2.** Press Enter twice to accept an empty passphrase (or type one you will remember).

**Step 3.** Type exactly this, then press Enter:

```
Get-Content $HOME\.ssh\qlab.pub
```

YOU SHOULD SEE: one line starting with `ssh-ed25519 AAAAC3Nza...` and ending with `qlab`.

**Step 4.** Select that whole line and copy it (Ctrl+C). You paste it into the GPU provider in 2.4.

WATCH OUT: The file WITHOUT .pub ($HOME\.ssh\qlab) is your PRIVATE key. Never paste it anywhere, never upload it.

CHECKPOINT: $HOME\.ssh\qlab and $HOME\.ssh\qlab.pub both exist.

## SECTION 2.3 - Hugging Face account and tokens

WHERE: Browser
GOAL: You have a READ token (download) and a WRITE token (upload your quantized models).

**Step 1.** Go to https://huggingface.co/join and create an account. Your user name is HF_USER.

**Step 2.** Click your avatar (top right) > **Settings** > **Access Tokens** > **Create new token**.

**Step 3.** Token type: **Write**. Name: `qlab-write`. Click **Create token**.

**Step 4.** Click the copy icon and paste the token into a private note on your PC. It starts with `hf_`.

**Step 5.** Gated models (Llama, Gemma, some Mistral): open the model page (for example https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), click **Agree and access repository**, wait for approval.

WATCH OUT: A token is a password. Never put it in git, screenshots or chats. If it leaks: Settings > Access Tokens > Invalidate.

CHECKPOINT: You have a token starting with hf_ saved privately; gated licences you need are accepted.
If not: download later fails with 401/403 = token missing or licence not accepted with the SAME account.

## SECTION 2.4 - Rent the GPU machine

WHERE: Browser, your GPU provider console
GOAL: A running Linux machine with an NVIDIA GPU that accepts your SSH key.

For learning (Parts 2-5, 7) rent ONE 48 GB GPU: L40 or L40S (FP8-capable, compute capability 8.9). It handles models up to ~14B in 16-bit and up to ~70B in 4-bit. For bigger models, Part 4.1 tells you exactly what to rent.

### Option A - HyperStack (full virtual machine; easiest)

**Step 1.** Sign in at https://www.hyperstack.cloud and open the console.

**Step 2.** Open **SSH Keys** > **Import**, paste your public key from 2.2, name it `qlab`, save.

**Step 3.** Open **Virtual Machines** > **Deploy New Virtual Machine** and fill in:

| Field | Value | Why |
|---|---|---|
| GPU flavor | 1x L40 (48 GB) or better | FP8-capable, big enough for 8-14B at 16-bit |
| Image | Ubuntu 22.04 or 24.04 with CUDA / drivers | Drivers pre-installed |
| Disk | 200 GB or more | Original + several quantized variants |
| SSH key | qlab | Password-less login |
| Public IP | Enabled | Needed to connect |

**Step 4.** Click **Deploy** and wait until the status is **ACTIVE**.

**Step 5.** Write down the public IP (VM_IP). User is `ubuntu`, SSH_PORT is 22.

### Option B - RunPod (container "pod", per-second billing)

**Step 1.** Sign in at https://www.runpod.io > **Settings** > **SSH Public Keys** > paste the key from 2.2 > **Update Public Key**.

**Step 2.** Click **Pods** > **Deploy**. Choose **L40S** (or another 48 GB GPU).

**Step 3.** Fill in:

| Field | Value | Why |
|---|---|---|
| Template | RunPod PyTorch (a CUDA "devel" image) (verify name) | CUDA toolkit inside, needed for llama.cpp |
| Container disk | 50 GB | Programs |
| Volume disk | 200 GB, mounted at /workspace | Models survive a pod restart only here |
| Expose TCP ports | 22 | SSH |

**Step 4.** Click **Deploy On-Demand**. When it shows Running, click **Connect** and copy the line under **SSH over exposed TCP**: it looks like `ssh root@VM_IP -p SSH_PORT -i ~/.ssh/id_ed25519`. Write down VM_IP and SSH_PORT. User is `root`.

### Option C - Vast.ai (marketplace, cheapest)

**Step 1.** Sign in at https://vast.ai > **Account** > **Keys** > **SSH Keys** > **+ New** > paste the key from 2.2 > Save.

**Step 2.** Click **Templates** and select a PyTorch / CUDA devel template (verify name).

**Step 3.** Click **Search**. Set GPU = L40S (or RTX 6000 Ada / A6000 48 GB), Disk Space = 200 GB, and sort by price. Prefer hosts with reliability > 99% and a "verified" badge.

**Step 4.** Click **RENT** on an offer. Open **Instances**; when it is running, click the key/terminal icon and copy the SSH command: `ssh -p SSH_PORT root@VM_IP`. User is `root`.

WATCH OUT: Billing starts now and runs until you STOP (or destroy) the machine. On RunPod and Vast.ai, only /workspace (the volume) survives a stop. Menu names change; if a label differs, look for the same idea (verify).

CHECKPOINT: The machine shows Running/Active; you wrote down VM_IP, SSH_PORT and the user (ubuntu or root).
If not: "no capacity" = choose another region or GPU type with at least 48 GB.

## SECTION 2.5 - Connect from your PC (SSH, VS Code, WinSCP)

WHERE: PC PowerShell, then VS Code, then WinSCP
GOAL: A VS Code window whose terminal runs ON the GPU machine, and drag-and-drop file transfer.

| Placeholder | Replace with | Example |
|---|---|---|
| VM_IP | Public IP from 2.4 | 203.0.113.45 |
| SSH_PORT | 22 (HyperStack) or the port RunPod/Vast showed | 40123 |
| VM_USER | ubuntu (HyperStack) or root (RunPod/Vast) | ubuntu |

**Step 1.** Type exactly this (with your values), then press Enter:

```
ssh -i $HOME\.ssh\qlab -p SSH_PORT VM_USER@VM_IP
```

**Step 2.** Type `yes` when asked to trust the host, then press Enter.

**Step 3.** Type exactly this, then press Enter:

```
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

YOU SHOULD SEE:

```
name, memory.total [MiB], compute_cap
NVIDIA L40S, 46068 MiB, 8.9
```

**Step 4.** Type `exit` and press Enter (back on your PC).

**Step 5.** Create a short name for the machine. Type exactly this (with your values), then press Enter:

```
Add-Content $HOME\.ssh\config "`nHost gpu`n  HostName VM_IP`n  Port SSH_PORT`n  User VM_USER`n  IdentityFile ~/.ssh/qlab`n  ServerAliveInterval 30"
```

**Step 6.** Test it. Type exactly this, then press Enter:

```
ssh gpu hostname
```

YOU SHOULD SEE: the machine's host name (not an error).

**Step 7.** In VS Code press F1, type **Remote-SSH: Connect to Host**, press Enter, choose **gpu**. Choose **Linux** if asked.

**Step 8.** Wait until the bottom-left corner shows **SSH: gpu**. Open a terminal: **Terminal** > **New Terminal**. Every "GPU machine" step in this manual runs in this terminal.

**Step 9 (files).** Open WinSCP. **New Session**: File protocol SFTP, Host name VM_IP, Port SSH_PORT, User name VM_USER. Click **Advanced** > **SSH** > **Authentication** > Private key file > choose `qlab` (WinSCP offers to convert it to .ppk: click OK and save). Click **Login**. Left pane = PC, right pane = GPU machine: drag files between them.

WATCH OUT: You do NOT upload models from your PC. The GPU machine downloads models directly from Hugging Face (data-centre speed). Use WinSCP only for small files: your calibration data, question sets, reports.

CHECKPOINT: `ssh gpu hostname` works; VS Code shows SSH: gpu; nvidia-smi printed your GPU.
If not: "Permission denied (publickey)": wrong key imported in 2.4 or wrong user (ubuntu vs root).

## SECTION 2.6 - Install everything on the GPU machine

WHERE: GPU machine, VS Code Remote-SSH terminal
GOAL: Three Python environments (quant, vllm, gguf) and a CUDA build of llama.cpp, all verified.

**Step 1.** Set your work folder. HyperStack/AWS: type exactly this, then press Enter:

```
echo 'export QLAB=$HOME/qlab' >> ~/.bashrc && source ~/.bashrc && mkdir -p $QLAB
```

RunPod/Vast.ai (only /workspace survives restarts): use this instead:

```
echo 'export QLAB=/workspace/qlab' >> ~/.bashrc && source ~/.bashrc && mkdir -p $QLAB
```

**Step 2.** Download the scripts from your notes repo. Type exactly this (REPO_BRANCH from the placeholder table), then press Enter:

```
git clone -b REPO_BRANCH https://github.com/kashyapphassan2000-netizen/mY-DOCS.git $QLAB/mY-DOCS
```

**Step 3.** Make a short variable for the scripts folder. Type exactly this, then press Enter:

```
echo 'export S=$QLAB/mY-DOCS/QUANTIZATION/scripts' >> ~/.bashrc && source ~/.bashrc && ls $S
```

YOU SHOULD SEE: bench.py compare_answers.py ... quantize.py setup_vm.sh size_model.py ...

**Step 4.** Create your secrets file. Type exactly this, then press Enter:

```
python3 -c "import secrets; print('VLLM_API_KEY=sk-' + secrets.token_urlsafe(24))" > $QLAB/.env
```

**Step 5.** Add your Hugging Face token and cache folder. Type exactly this (paste your hf_ token), then press Enter:

```
echo "HF_TOKEN=hf_PASTE_YOUR_WRITE_TOKEN" >> $QLAB/.env && echo "HF_HOME=$QLAB/hf-cache" >> $QLAB/.env
```

**Step 6.** Load the secrets in every new terminal automatically. Type exactly this, then press Enter:

```
echo 'set -a; source $QLAB/.env; set +a' >> ~/.bashrc && source ~/.bashrc && echo $VLLM_API_KEY
```

YOU SHOULD SEE: `sk-` followed by random characters.

| Field | Value | Why |
|---|---|---|
| VLLM_API_KEY | sk-... random | Every server you start requires it; clients send it |
| HF_TOKEN | hf_... | Download gated models; upload your quantized models |
| HF_HOME | $QLAB/hf-cache | Model downloads go to the big disk, not the small system disk |

**Step 7.** Start a tmux session (keeps running if VS Code disconnects). Type exactly this, then press Enter:

```
tmux new -s setup
```

**Step 8.** Run the setup script. Type exactly this, then press Enter:

```
bash $S/setup_vm.sh 2>&1 | tee $QLAB/setup.log
```

This takes 15-40 minutes. To leave tmux running: press Ctrl+B, release, press D. To come back: `tmux attach -t setup`.

YOU SHOULD SEE (at the end):

```
== 7. verify
quant OK  torch 2.x.x | cuda True | llmcompressor 0.13.0
vllm OK  0.29.0
llama.cpp OK
DONE. Activate with: source /home/ubuntu/qlab/envs/<quant|vllm|gguf>/bin/activate
```

| If you see... | It means... | Do this next |
|---|---|---|
| WARNING: nvcc (CUDA toolkit) not found | llama.cpp built for CPU only | sudo apt-get install -y nvidia-cuda-toolkit, rerun the script |
| cuda False | PyTorch cannot see the GPU | nvidia-smi must work; reboot once if the driver was just installed |
| No space left on device | Disk too small | Resize the volume in the console (200 GB+) |
| sudo: command not found | You are root in a container (RunPod/Vast) | The script handles this; just rerun |

**Every line of setup_vm.sh, explained:** `set -euo pipefail` stops at the first error instead of continuing broken. `apt-get install ... tmux cmake build-essential` installs the C++ compiler and tools llama.cpp needs. `uv venv -p 3.12` creates an isolated Python 3.12 environment. We make THREE environments because vLLM 0.29 and llm-compressor 0.13 pin different versions of shared libraries (compressed-tensors, torch); installing both in one environment breaks one of them. `cmake ... -DGGML_CUDA=ON` builds llama.cpp with GPU support. The last block imports each library and prints its version: if those lines print, the install works.

{{include scripts/setup_vm.sh}}

CHECKPOINT: The three "OK" lines printed; `ls $QLAB/envs` shows gguf quant vllm.
If not: read the first ERROR line in $QLAB/setup.log; most failures are disk space or a missing CUDA toolkit.

## SECTION 2.7 - Prove the whole pipeline works (tiny model, 10 minutes)

WHERE: GPU machine terminal
GOAL: Quantize a 0.5B model to FP8, serve it with vLLM, and get an answer, end to end.

**Step 1.** Type exactly this, then press Enter:

```
source $QLAB/envs/quant/bin/activate
```

**Step 2.** Type exactly this, then press Enter:

```
python $S/quantize.py --model Qwen/Qwen2.5-0.5B-Instruct --method fp8 --out $QLAB/work/tiny/fp8
```

YOU SHOULD SEE (last lines):

```
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "method": "fp8",
  ...
  "seconds": <n>,
  "size_gb": <about 0.6-0.8>
}
```

**Step 3.** Start the server in its own tmux session. Type exactly this, then press Enter:

```
tmux new -d -s tiny "source $QLAB/envs/vllm/bin/activate; vllm serve $QLAB/work/tiny/fp8 --served-model-name tiny --max-model-len 4096 --gpu-memory-utilization 0.5 --api-key $VLLM_API_KEY --port 8000"
```

**Step 4.** Wait 60 seconds, then type exactly this and press Enter:

```
curl -s http://localhost:8000/v1/chat/completions -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" -d '{"model":"tiny","messages":[{"role":"user","content":"Say hello in 5 words"}],"max_tokens":20}'
```

YOU SHOULD SEE: `{"id":"chatcmpl-...","choices":[{"message":{"role":"assistant","content":"Hello ..."}}],...}`

**Step 5.** Stop the tiny server. Type exactly this, then press Enter:

```
tmux kill-session -t tiny
```

WATCH OUT: "Connection refused" = the server is still loading. Run `tmux attach -t tiny` to watch the log (Ctrl+B then D to leave).

CHECKPOINT: quantize printed size_gb; curl returned an answer; the tiny session is stopped.
If not: "401 Unauthorized" = VLLM_API_KEY is empty in this terminal: run `source ~/.bashrc`.
