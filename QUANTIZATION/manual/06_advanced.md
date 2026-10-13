# PART 6 - ADVANCED: serve from Hugging Face, Terraform + Kubernetes, autoscale 1 to 50 GPUs

WHY / THEORY. **What:** come back later with fresh GPUs, pull YOUR quantized model from Hugging Face and serve it; then build a production cluster on AWS with Terraform, run vLLM on Kubernetes and autoscale it on your rule (add a machine at ~78-80% memory pressure, remove one below ~60%, maximum 50). **Why:** this is the difference between "a model running on a VM" and "a service other people depend on". **When:** after Part 5 gave you a variant that passes your gates and is saved on Hugging Face.

## SECTION 6.1 - Come back later: fresh GPU, pull your model from Hugging Face, serve

WHERE: A NEW GPU machine (rent + connect + setup exactly like 2.4-2.6), terminal
GOAL: Your saved version (tag v1) runs on a machine that never saw the quantization work.

**Step 1.** Size the serving machine for the QUANTIZED model (it is smaller than the original). Type exactly this, then press Enter:

```
source $QLAB/envs/quant/bin/activate && python $S/size_model.py --model MODEL_ID --context 8192 --users 32
```

Read the row for your format (e.g. "w4a16 (gptq/awq)") and rent that GPU type and count.

**Step 2.** Serve straight from the Hub, pinned to your version. Type exactly this, then press Enter:

```
tmux new -d -s vllm "source $QLAB/envs/vllm/bin/activate; vllm serve HF_USER/SLUG-gptq-w4a16 --revision v1 --served-model-name SERVED_NAME --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key $VLLM_API_KEY --port 8000"
```

**Step 3.** Wait 2-3 minutes (download + load), then type exactly this and press Enter:

```
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

YOU SHOULD SEE: `{"object":"list","data":[{"id":"SERVED_NAME",...}]}`

| Flag | Value | Why |
|---|---|---|
| HF_USER/SLUG-gptq-w4a16 | your repo | vLLM downloads it with HF_TOKEN (private repos work) |
| --revision v1 | your tag | Always the same tested version, even after you upload v2 |
| --tensor-parallel-size N | GPUs per replica | Only if size_model.py said N > 1 |

WATCH OUT: A private repo needs HF_TOKEN in this machine's $QLAB/.env (2.6 Step 5), or the download fails with 401.

CHECKPOINT: /v1/models lists SERVED_NAME; the model came from Hugging Face, not from a local folder.

## SECTION 6.2 - Production vLLM settings and the metrics that matter

WHERE: GPU machine terminal (server from 6.1 still running)
GOAL: You know the important flags, and you SEE why raw VRAM % is the wrong autoscaling signal.

| Flag | Typical value | Why |
|---|---|---|
| --max-model-len | your real longest request | Bigger = fewer concurrent users fit |
| --gpu-memory-utilization | 0.90 | vLLM grabs this share of VRAM at startup for weights + KV pool |
| --max-num-seqs | 64-256 | Maximum requests processed together |
| --kv-cache-dtype fp8 | on Ada/Hopper | Halves KV memory = ~2x users; tiny quality cost (measure) |
| --tensor-parallel-size | 1, 2, 4, 8 | Split each layer across GPUs in one machine |
| --pipeline-parallel-size | 2+ | Split layers across machines (very large models, Part 7.5) |
| --enable-lora --lora-modules name=path | multi-LoRA | Many fine-tunes on one quantized base |
| --speculative-config '{...}' | draft model / ngram / eagle | Faster at LOW concurrency only; measure on/off with bench.py |
| --api-key | secret | Required |

**Step 1.** Look at vLLM's own metrics. Type exactly this, then press Enter:

```
curl -s http://localhost:8000/metrics | grep -E "^vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting)"
```

YOU SHOULD SEE (idle server):

```
vllm:kv_cache_usage_perc{...} 0.0
vllm:num_requests_running{...} 0.0
vllm:num_requests_waiting{...} 0.0
```

**Step 2.** Now look at raw VRAM. Type exactly this, then press Enter:

```
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

YOU SHOULD SEE: memory.used around 90% of memory.total, with ZERO users.

**This is the key production lesson.** vLLM reserves 90% of VRAM at startup, so nvidia-smi always shows ~90%. A rule like "add a machine when VRAM > 78%" would fire immediately and scale to 50 machines with no traffic. The metric that means what you intended ("how full is the memory for users' conversations") is `vllm:kv_cache_usage_perc`. Your 60-80% rule is implemented on that metric in 6.7, plus `vllm:num_requests_waiting` (users queuing) as a safety net.

| Metric | Meaning | Use it for |
|---|---|---|
| vllm:kv_cache_usage_perc | 0.0-1.0 share of the KV pool in use | Your 60-80% scaling band |
| vllm:num_requests_waiting | Requests queued because the GPU is full | Scale out before users wait |
| vllm:num_requests_running | Requests being generated now | Capacity planning |
| vllm:time_to_first_token_seconds | Histogram of TTFT | SLO alerts |

**Step 3.** Stop the server when done: `tmux kill-session -t vllm`, and stop/delete this GPU machine in the provider console.

CHECKPOINT: You saw kv_cache_usage_perc = 0.0 while nvidia-smi showed ~90% used.

## SECTION 6.3 - Prepare your PC and AWS account for the cluster

WHERE: PC PowerShell and browser (AWS console)
GOAL: AWS CLI, Terraform, kubectl and Helm work on your PC; a budget alert and a GPU quota exist.

**Step 1.** Type exactly this, then press Enter:

```
winget install --id Amazon.AWSCLI -e; winget install --id Hashicorp.Terraform -e; winget install --id Kubernetes.kubectl -e; winget install --id Helm.Helm -e
```

**Step 2.** Close PowerShell, open a new one, type exactly this, then press Enter:

```
aws --version; terraform version; kubectl version --client; helm version
```

YOU SHOULD SEE: four version lines (Terraform 1.16.x or newer).

**Step 3.** In the browser, AWS console > IAM > Users > Create user `qlab-admin`, attach policy **AdministratorAccess** (learning account only), then Security credentials > **Create access key** > Command Line Interface. Copy both keys.

**Step 4.** Type exactly this, then press Enter, and answer the four questions (keys, region `us-east-1`, output `json`):

```
aws configure
```

**Step 5.** Budget alert: AWS console > **Billing and Cost Management** > **Budgets** > **Create budget** > Monthly cost budget > amount 50 (USD) > email YOUR_EMAIL > Create.

**Step 6.** GPU quota (new accounts often have 0): AWS console > **Service Quotas** > **Amazon EC2** > search **Running On-Demand G and VT instances** > **Request increase at account level**. Value = 4 vCPUs x number of g6e.xlarge machines. For the full 50-machine rule: 200 (plus 8 spare = 208). Start with 16 if you only learn.

**Step 7.** Clone your notes repo on the PC. Type exactly this, then press Enter:

```
cd $HOME\Documents; git clone -b REPO_BRANCH https://github.com/kashyapphassan2000-netizen/mY-DOCS.git; cd mY-DOCS\QUANTIZATION\infra
```

WATCH OUT: Quota requests can take hours to days, and AWS may grant less than you ask. Your autoscaler can never exceed the quota: with quota 16 you get at most 4 GPU machines even though Terraform says 50.

CHECKPOINT: `aws sts get-caller-identity` prints your account; budget exists; quota request submitted.

## SECTION 6.4 - Create the cluster with Terraform

WHERE: PC PowerShell, folder mY-DOCS\QUANTIZATION\infra\terraform
GOAL: An EKS Kubernetes cluster with 2 system machines and a GPU pool of min 1 / max 50.

The four Terraform files (already in your repo):

{{include infra/terraform/versions.tf}}

{{include infra/terraform/variables.tf}}

{{include infra/terraform/main.tf}}

{{include infra/terraform/outputs.tf}}

**Every block explained:** `provider "aws"` = which cloud and region; `default_tags` labels everything for cost reports. `module "vpc"` creates the private network: public subnets for the load balancer, private subnets for the machines, and a NAT gateway so private machines can download from Hugging Face. `aws_iam_policy.cluster_autoscaler` = the permission the cluster-autoscaler needs to change the GPU pool size. `module "eks"` creates the Kubernetes control plane; `enable_cluster_creator_admin_permissions` makes you admin. Node group `system` = 2 CPU machines for Prometheus/KEDA/autoscaler (they get the autoscaler permission). Node group `gpu` = the vLLM machines: `AL2023_x86_64_NVIDIA` image has GPU drivers; `min_size 1` / `max_size 50` is your 50-machine limit; the 200 GB disk holds model downloads; the label `workload=llm-gpu` lets vLLM pods choose these machines; the taint keeps all other pods off the expensive GPUs.

| Variable | Default | Change it when |
|---|---|---|
| region | us-east-1 | Your quota/capacity is elsewhere |
| kubernetes_version | 1.34 | Pick the newest in the EKS console (verify); old versions cost extra |
| gpu_instance_type | g6e.xlarge (1x L40S 48 GB, $1.861/h) | g6.xlarge (1x L4 24 GB, $0.8048/h) for small models; g6e.12xlarge (4x L40S) for TP=4 |
| gpu_min / gpu_max | 1 / 50 | Your always-on and maximum machine count |

**Step 1.** Type exactly this, then press Enter:

```
cd $HOME\Documents\mY-DOCS\QUANTIZATION\infra\terraform; terraform init
```

YOU SHOULD SEE: `Terraform has been successfully initialized!` (tested: EKS module 20.37.2, VPC module 5.21.0, AWS provider 5.100.0).

**Step 2.** Type exactly this, then press Enter:

```
terraform plan -out tfplan
```

YOU SHOULD SEE: `Plan: <about 60-70> to add, 0 to change, 0 to destroy.`

**Step 3.** Type exactly this, then press Enter (takes 15-20 minutes):

```
terraform apply tfplan
```

**Step 4.** Connect kubectl to the new cluster. Type exactly this, then press Enter:

```
aws eks update-kubeconfig --name llm-serving --region us-east-1
```

**Step 5.** Type exactly this, then press Enter:

```
kubectl get nodes -L workload
```

YOU SHOULD SEE: 3 nodes Ready: two without a workload label (system) and one with `llm-gpu`.

| If you see... | It means... | Do this next |
|---|---|---|
| InsufficientInstanceCapacity / quota error | No GPU quota or capacity | Wait for 6.3 Step 6, or change region/instance type, terraform apply again |
| Unsupported Kubernetes version | Version retired | Set kubernetes_version to one listed in the EKS console |
| Error: Unauthorized (kubectl) | Wrong AWS identity | aws sts get-caller-identity must be the user that ran terraform |

CHECKPOINT: 3 nodes Ready, one labelled llm-gpu.

## SECTION 6.5 - Install the cluster add-ons (GPU plugin, Prometheus, KEDA, cluster-autoscaler)

WHERE: PC PowerShell
GOAL: Kubernetes can see the GPU, collect vLLM metrics, scale pods on them, and add/remove GPU machines.

**Step 1.** Add the chart repositories. Type exactly this, then press Enter:

```
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin; helm repo add prometheus-community https://prometheus-community.github.io/helm-charts; helm repo add kedacore https://kedacore.github.io/charts; helm repo add autoscaler https://kubernetes.github.io/autoscaler; helm repo update
```

**Step 2.** NVIDIA device plugin (tells Kubernetes "this machine has 1 GPU"). Type exactly this, then press Enter:

```
helm upgrade -i nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin --create-namespace
```

**Step 3.** Check the GPU is visible. Type exactly this, then press Enter:

```
kubectl get nodes -l workload=llm-gpu -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu
```

YOU SHOULD SEE: your GPU node with `GPUS 1` (after up to a minute).

**Step 4.** Prometheus (collects metrics; configured to read every PodMonitor). Type exactly this, then press Enter:

```
helm upgrade -i kube-prometheus-stack prometheus-community/kube-prometheus-stack -n monitoring --create-namespace --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

**Step 5.** KEDA (turns metrics into a pod count). Type exactly this, then press Enter:

```
helm upgrade -i keda kedacore/keda -n keda --create-namespace
```

**Step 6.** cluster-autoscaler (adds a GPU machine when a vLLM pod cannot be placed; removes idle ones). Type exactly this, then press Enter:

```
helm upgrade -i cluster-autoscaler autoscaler/cluster-autoscaler -n kube-system --set autoDiscovery.clusterName=llm-serving --set awsRegion=us-east-1 --set extraArgs.scale-down-unneeded-time=10m --set extraArgs.skip-nodes-with-local-storage=false --set extraArgs.expander=least-waste
```

**Step 7.** Type exactly this, then press Enter:

```
kubectl get pods -n nvidia-device-plugin; kubectl get pods -n monitoring; kubectl get pods -n keda; kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-cluster-autoscaler
```

YOU SHOULD SEE: every pod Running.

WATCH OUT: The cluster-autoscaler image should match your Kubernetes minor version (e.g. v1.34.x for 1.34). If its log says the version is unsupported, add `--set image.tag=v1.34.0` (verify the exact tag on the autoscaler releases page).

CHECKPOINT: GPUS = 1 on the GPU node; all add-on pods Running.
If not: GPUS empty = device-plugin pod not on the GPU node: `kubectl describe node <gpu-node>` must show the label nvidia.com/gpu.present=true (set in main.tf).

## SECTION 6.6 - Deploy vLLM on Kubernetes

WHERE: PC PowerShell + VS Code, folder mY-DOCS\QUANTIZATION\infra\k8s
GOAL: Your Hugging Face model (tag v1) answers through a public load balancer with your API key.

The manifest (Deployment + Service + PodMonitor):

{{include infra/k8s/vllm.yaml}}

**Every non-obvious part explained:** `tolerations` + `nodeSelector` = "run only on the GPU machines". `nvidia.com/gpu: 1` reserves one GPU (use 4 for a 4-GPU machine and add --tensor-parallel-size=4). `startupProbe` gives the pod up to 15 minutes to download and load the model before Kubernetes considers it broken; `readinessProbe` sends traffic only to pods whose /health answers. `maxUnavailable: 0` = during an update, a new pod must be ready before an old one stops. `/dev/shm` in memory is needed for tensor parallel. The `Service` of type LoadBalancer creates an AWS Network Load Balancer. The `PodMonitor` tells Prometheus to read /metrics from every vLLM pod every 15 s. `--api-key=$(VLLM_API_KEY)` is filled from the Kubernetes secret.

**Step 1.** Open `infra\k8s\vllm.yaml` in VS Code. Press Ctrl+H (replace) and replace: `HF_REPO` -> `HF_USER/SLUG-gptq-w4a16`, `MODEL_TAG` -> `v1`, `SERVED_NAME` -> your served name. Save (Ctrl+S).

**Step 2.** Create the namespace and the secret (your real hf_ and sk- values). Type exactly this, then press Enter:

```
kubectl create namespace llm; kubectl create secret generic llm-secrets -n llm --from-literal=HF_TOKEN=hf_PASTE --from-literal=VLLM_API_KEY=sk-PASTE
```

**Step 3.** Type exactly this, then press Enter:

```
kubectl apply -f vllm.yaml
```

**Step 4.** Watch the pod start (2-15 minutes). Type exactly this, then press Enter (Ctrl+C to stop watching):

```
kubectl get pods -n llm -w
```

YOU SHOULD SEE: `vllm-xxxxx   0/1   Running` and later `1/1   Running`.

**Step 5.** Get the load balancer address. Type exactly this, then press Enter:

```
kubectl get svc vllm -n llm -o jsonpath="{.status.loadBalancer.ingress[0].hostname}"
```

YOU SHOULD SEE: `xxxx.elb.us-east-1.amazonaws.com`. This is LB_DNS (it can take 2-3 minutes to start answering).

**Step 6.** Type exactly this (your LB_DNS and key), then press Enter:

```
curl.exe -s http://LB_DNS/v1/models -H "Authorization: Bearer sk-PASTE"
```

YOU SHOULD SEE: `{"object":"list","data":[{"id":"SERVED_NAME",...}]}`

| If you see... | It means... | Do this next |
|---|---|---|
| Pending (forever) | No GPU node free | kubectl describe pod -n llm <pod>; check quota (6.3) |
| CreateContainerConfigError | Secret missing | Step 2 |
| CrashLoopBackOff | vLLM error | kubectl logs -n llm <pod>: OOM -> lower --max-model-len; 401 -> HF_TOKEN |

WATCH OUT: This load balancer is public. The vLLM API key protects it, but for real users put an API gateway with per-user keys and rate limits in front (the source manual's qdeploy gateway, or AWS API Gateway), and use HTTPS.

CHECKPOINT: Pod 1/1 Running; curl through the load balancer lists your model.

## SECTION 6.7 - Autoscaling on YOUR rule: 60-80% band, 1 to 50 machines

WHERE: PC PowerShell, folder infra\k8s
GOAL: KEDA adds vLLM pods (and cluster-autoscaler adds GPU machines) when memory pressure reaches ~78-80%, and removes them below ~60%.

{{include infra/k8s/scaledobject.yaml}}

**How your rule becomes Kubernetes math.** KEDA creates a Horizontal Pod Autoscaler (HPA). For a "Value" metric the HPA computes `desired pods = ceil(current pods x metric / target)` and does nothing while the metric is within +-10% of the target (its tolerance). With target 0.70:

| Average KV-cache usage | metric / target | Action |
|---|---|---|
| 0.80 (80%) | 1.14 (> 1.10) | Scale OUT: e.g. 4 pods -> ceil(4 x 1.14) = 5 |
| 0.77 | 1.10 | Edge: just above 0.77 scales out (your "78-80%") |
| 0.63-0.77 | 0.90-1.10 | Do nothing (your stable band) |
| 0.55 with 5 pods | 0.79 (< 0.90) | Scale IN: ceil(5 x 0.79) = ceil(3.9) = 4 pods (after 10 low minutes) |
| 0.60 with 5 pods | 0.86 (< 0.90) | No change yet: ceil(5 x 0.86) = ceil(4.29) = 5 |
| any | | Never below minReplicaCount 1, never above maxReplicaCount 50 |

So the band is 63-77% on standard Kubernetes; that is your 60-80% rule with the default 10% tolerance. One honest detail: because the HPA rounds UP, it removes a pod only when usage <= 0.70 x (N-1)/N: about 63% at 10+ pods, 56% at 5 pods, 47% at 3 pods. Scale-in is therefore a bit more conservative at small sizes (costs slightly more, never hurts users). Newer Kubernetes versions add a per-HPA tolerance setting (HPAConfigurableTolerance) that can make it exactly 60/80 (verify availability on EKS). The second trigger (users waiting in the queue, 5 per pod) scales out even if KV usage looks fine; KEDA uses whichever trigger asks for MORE pods. The `behavior` block adds at most 2 pods per 3 minutes (new GPU machines take minutes to download the model) and removes at most 1 pod per 5 minutes after load has stayed low for 10 minutes, so it does not flap.

The chain: load rises -> KV usage > 77% -> KEDA raises pods -> a new pod cannot fit (each GPU machine holds one) -> pod Pending -> cluster-autoscaler adds a g6e.xlarge (up to max_size 50) -> pod starts, downloads model, passes /health -> load balancer sends it traffic -> usage drops into the band. In reverse when traffic falls.

**Step 1.** Type exactly this, then press Enter:

```
kubectl apply -f scaledobject.yaml
```

**Step 2.** Type exactly this, then press Enter:

```
kubectl get scaledobject -n llm; kubectl get hpa -n llm
```

YOU SHOULD SEE: `vllm-autoscaler ... READY True` and an HPA `keda-hpa-vllm-autoscaler` with `MINPODS 1  MAXPODS 50`.

**Step 3.** Check Prometheus really has the metric. Type exactly this, then press Enter:

```
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
```

Open http://localhost:9090 in the browser, type `vllm:kv_cache_usage_perc` and click **Execute**. You should see one series per vLLM pod. Press Ctrl+C in PowerShell when done.

CHECKPOINT: ScaledObject READY True; HPA shows 1/50; Prometheus shows vllm:kv_cache_usage_perc.
If not: READY False = KEDA cannot reach Prometheus: the serverAddress must be http://kube-prometheus-stack-prometheus.monitoring.svc:9090.

## SECTION 6.8 - Load-test it, watch it scale, then tear everything down

WHERE: PC PowerShell (three windows)
GOAL: You SEE pods and GPU machines added under load and removed afterwards; then the AWS account is clean.

**Step 1.** Window 1: install the client library. Type exactly this, then press Enter:

```
py -3.12 -m pip install httpx
```

**Step 2.** Window 1: start heavy load for ~15 minutes (your LB_DNS and key). Type exactly this, then press Enter:

```
$env:VLLM_API_KEY = "sk-PASTE"; py -3.12 $HOME\Documents\mY-DOCS\QUANTIZATION\scripts\bench.py --url http://LB_DNS --model SERVED_NAME --label k8s-load --concurrency 128 --num-prompts 4000 --max-tokens 512
```

**Step 3.** Window 2: watch the autoscaler decide. Type exactly this, then press Enter:

```
kubectl get hpa -n llm -w
```

**Step 4.** Window 3: watch pods and machines appear. Type exactly this, then press Enter:

```
kubectl get pods -n llm -w
```

YOU SHOULD SEE: the HPA TARGETS value rise above 700m (= 0.70), REPLICAS go 1 -> 3 -> 5..., new pods Pending, then (2-5 min) new GPU nodes in `kubectl get nodes`, then pods Running. After the load stops, replicas fall one by one every ~5 minutes, and idle GPU nodes disappear about 10 minutes later.

**Step 5.** When finished, delete the Kubernetes objects FIRST (this removes the AWS load balancer). Type exactly this, then press Enter:

```
cd $HOME\Documents\mY-DOCS\QUANTIZATION\infra\k8s; kubectl delete -f scaledobject.yaml; kubectl delete -f vllm.yaml
```

**Step 6.** Wait 2 minutes, then destroy the cluster. Type exactly this, then press Enter:

```
cd ..\terraform; terraform destroy
```

**Step 7.** Type `yes` and press Enter. Takes ~15 minutes.

YOU SHOULD SEE: `Destroy complete! Resources: <n> destroyed.`

| Cost while running (us-east-1, on-demand, verify) | $/hour |
|---|---|
| EKS control plane | 0.10 |
| NAT gateway (+ data) | ~0.045 + traffic |
| 2x m6i.xlarge system nodes | ~0.38 |
| 1x g6e.xlarge GPU (minimum) | 1.861 |
| At full scale: 50x g6e.xlarge | ~93 |

WATCH OUT: If `terraform destroy` hangs on the VPC/subnets, a load balancer still exists: run Step 5 again, or delete it in EC2 > Load Balancers, then rerun destroy. Check EC2 > Instances shows nothing running before you walk away.

CHECKPOINT: You saw replicas and nodes go up and down; Destroy complete; the EC2 console shows no instances.

## Common mistake at the Advanced level

| Mistake | How to spot it | Fix |
|---|---|---|
| Scaling on raw VRAM % | Scales to max with zero traffic | Use vllm:kv_cache_usage_perc + queue (6.2, 6.7) |
| Scaling on CPU % | GPU boxes sit at low CPU; never scales | Same fix |
| Too-short startup probe | Pods killed while downloading the model | startupProbe failureThreshold x period >= real load time |
| Forgetting the GPU quota | Pods Pending forever, autoscaler errors | Service Quotas request (6.3) |
| Destroying the cluster before the load balancer | terraform destroy hangs | Delete Kubernetes objects first |
| Serving "latest" from the Hub | Unreviewed upload goes live on restart | Always --revision TAG |
