# MLOps LLM Pipeline

A production-grade MLOps pipeline that mirrors how large-scale LLM systems like ChatGPT manage model updates. Covers the full lifecycle: model training, experiment tracking, containerization, GitOps deployment, canary traffic splitting, and live monitoring.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                              │
│                                                                 │
│  [Train Model] → [MLflow Tracking] → [MinIO Artifact Storage]  │
│         ↓                                                       │
│  [Promote to Production] → [CI Pipeline Triggered]             │
│         ↓                                                       │
│  [Build Docker Image] → [Push to Registry] → [Update Helm]     │
│         ↓                                                       │
│  [Git Commit] → [ArgoCD Detects Diff] → [Helm Release]         │
│         ↓                                                       │
│  [Canary Deploy v2.0 @ 10%] → [Validate] → [Full Rollout]      │
│         ↓                                                       │
│  [Prometheus Scrapes Metrics] → [Grafana Dashboard]            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Experiment Tracking | MLflow |
| Artifact Storage | MinIO (S3-compatible) |
| Inference Server | FastAPI + Transformers |
| Containerization | Docker |
| Kubernetes | kind (local cluster) |
| Package Manager | Helm |
| GitOps | ArgoCD |
| Traffic Splitting | nginx Ingress (canary) |
| Metrics | Prometheus + Grafana |
| Model | distilgpt2 (HuggingFace) |

---

## Prerequisites

Install the following before starting:

| Tool | Version | Install |
|---|---|---|
| Python | 3.11 | `brew install python@3.11` |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/products/docker-desktop) |
| kind | Latest | `brew install kind` |
| kubectl | Latest | `brew install kubectl` |
| Helm | Latest | `brew install helm` |
| ArgoCD CLI | Latest | `brew install argocd` |
| Git | Latest | `brew install git` |

> **Note:** This project is tested on macOS with Apple Silicon (M2). It runs entirely on CPU — no GPU required.

Verify all tools are installed:

```bash
python3.11 --version
docker --version
kind version
kubectl version --client
helm version
argocd version --client
```

---

## Project Structure

```
mlops-llm-pipeline/
├── infra/
│   ├── docker/
│   │   └── docker-compose.yml       # MLflow + MinIO
│   ├── helm/
│   │   ├── inference-server/        # Helm chart for inference server
│   │   │   ├── Chart.yaml
│   │   │   ├── values.yaml          # Model version lives here
│   │   │   ├── values-canary.yaml   # Canary deployment config
│   │   │   └── templates/
│   │   │       ├── deployment.yaml
│   │   │       ├── service.yaml
│   │   │       ├── ingress.yaml
│   │   │       └── servicemonitor.yaml
│   │   └── monitoring/
│   │       ├── values-monitoring.yaml
│   │       └── dashboards/
│   │           └── inference-dashboard.json
│   ├── argocd/
│   │   └── application.yaml         # ArgoCD Application manifest
│   └── kind-cluster.yaml            # kind cluster config
├── model-training/
│   ├── train.py                     # Train model v1.5
│   ├── train_v2.py                  # Train model v2.0
│   └── promote_model.py             # Promote model to Production
├── inference-server/
│   ├── app.py                       # FastAPI inference server
│   ├── Dockerfile
│   └── requirements.txt
├── pipelines/
│   ├── promote_and_deploy.py        # Full CI pipeline script
│   ├── canary_test.py               # Traffic split test
│   └── load_test.py                 # Load generator for metrics
└── .github/
    └── workflows/
        └── model-deploy.yml         # GitHub Actions workflow
```

---

## Setup Guide

### Phase 1 — Infrastructure + Model Training

#### 1.1 Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-llm-pipeline.git
cd mlops-llm-pipeline
```

#### 1.2 Start MLflow and MinIO

```bash
docker compose -f infra/docker/docker-compose.yml up -d

# Wait ~60 seconds for MLflow to install boto3 on startup
docker logs mlflow -f
# Wait until you see: "Listening at: http://0.0.0.0:5000"
```

Verify:
- MLflow UI → http://localhost:5001
- MinIO UI → http://localhost:9001 (login: `minioadmin` / `minioadmin`)

#### 1.3 Set up Python environment

```bash
cd model-training
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install mlflow==2.11.1 boto3 transformers torch datasets accelerate pyyaml
```

#### 1.4 Train model v1.5

```bash
python train.py
```

This trains a `distilgpt2` model, logs metrics to MLflow, and stores weights in MinIO. Takes ~3 minutes on CPU.

#### 1.5 Promote model to Production

```bash
python promote_model.py
```

Verify in MLflow UI at http://localhost:5001 → Models → `llm-chatbot` shows version 1 in `Production` stage.

---

### Phase 2 — Inference Server

#### 2.1 Set up inference server environment

```bash
cd ../inference-server
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn transformers torch mlflow==2.11.1 boto3 \
  python-multipart prometheus-fastapi-instrumentator prometheus-client
```

#### 2.2 Test locally

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Wait for `Model v1 loaded and ready.` then in a new terminal:

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "User: What is machine learning?\nAssistant:", "max_new_tokens": 50}'

# Streaming (token by token)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "User: Explain AI\nAssistant:", "stream": true}'

# Prometheus metrics
curl http://localhost:8000/metrics
```

Stop uvicorn (`Ctrl+C`) when done.

#### 2.3 Build Docker image

```bash
docker build -t llm-inference-server:v1.5 .
```

#### 2.4 Test containerized server

```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:9000 \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -e MODEL_NAME=llm-chatbot \
  -e MODEL_STAGE=Production \
  llm-inference-server:v1.5
```

Test with same curl commands above.

---

### Phase 3 — Kubernetes + Helm + ArgoCD

#### 3.1 Create kind cluster

```bash
kind create cluster --config infra/kind-cluster.yaml
kubectl get nodes  # should show 1 node Ready
```

#### 3.2 Load image into cluster

```bash
kind load docker-image llm-inference-server:v1.5 --name mlops-cluster
```

#### 3.3 Deploy with Helm

```bash
helm lint infra/helm/inference-server

helm install inference-server infra/helm/inference-server \
  --namespace default \
  --create-namespace

kubectl get pods -w
# Wait for: inference-server-inference-server-xxx   1/1   Running
```

Test:
```bash
curl http://localhost:30800/health
```

#### 3.4 Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s

kubectl patch svc argocd-server -n argocd -p \
  '{"spec": {"type": "NodePort", "ports": [{"port": 443, "nodePort": 30900, "targetPort": 8080}]}}'

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d && echo
```

ArgoCD UI → https://localhost:30900 (username: `admin`, password: output from above)

#### 3.5 Wire ArgoCD to GitHub

Edit `infra/argocd/application.yaml` and replace `YOUR_USERNAME` with your GitHub username, then:

```bash
# Uninstall manual helm release (ArgoCD will manage it now)
helm uninstall inference-server

argocd login localhost:30900 --username admin --password <YOUR_PASSWORD> --insecure

kubectl apply -f infra/argocd/application.yaml
argocd app sync inference-server
argocd app get inference-server  # should show Synced + Healthy
```

---

### Phase 4 — CI/CD Pipeline

#### 4.1 Train model v2.0

```bash
cd model-training
source venv/bin/activate
python train_v2.py
```

#### 4.2 Run the CI pipeline

This promotes model v2.0, builds a new Docker image, updates `values.yaml`, commits to git, and ArgoCD auto-deploys:

```bash
cd ..
python pipelines/promote_and_deploy.py
```

Watch rollout:
```bash
kubectl get pods -w
```

Verify new version is live:
```bash
curl http://localhost:30800/model-info
# model_version should now show 2
```

---

### Phase 5 — Canary Traffic Splitting

#### 5.1 Install nginx Ingress

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

#### 5.2 Sync ingress changes via ArgoCD

```bash
git add infra/helm/inference-server/
git commit -m "add ingress and canary support"
git push

argocd app sync inference-server
kubectl get ingress  # verify ingress resource created
```

#### 5.3 Deploy canary (v2.0 at 10% traffic)

```bash
# Check available images
docker images | grep llm-inference-server

# Load v2.x image into kind
kind load docker-image llm-inference-server:v2.2 --name mlops-cluster

# Deploy canary
helm install inference-server-canary infra/helm/inference-server \
  --namespace default \
  --values infra/helm/inference-server/values-canary.yaml \
  --set image.tag=v2.2 \
  --set model.version=2

kubectl get deployments  # should show both stable and canary
```

#### 5.4 Test traffic split

```bash
# Start ingress port-forward
kubectl port-forward -n ingress-nginx \
  service/ingress-nginx-controller 8080:80 &

# Run canary test
cd model-training && source venv/bin/activate && cd ..
python pipelines/canary_test.py
# Expect: ~90% v1, ~10% v2
```

#### 5.5 Promote canary

```bash
# 50% traffic
helm upgrade inference-server-canary infra/helm/inference-server \
  --values infra/helm/inference-server/values-canary.yaml \
  --set canary.weight=50

# 100% traffic
helm upgrade inference-server-canary infra/helm/inference-server \
  --values infra/helm/inference-server/values-canary.yaml \
  --set canary.weight=100

# Retire canary — stable takes all traffic at v2.0
helm uninstall inference-server-canary
```

---

### Phase 6 — Monitoring

#### 6.1 Install Prometheus + Grafana

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

mkdir -p infra/helm/monitoring

helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values infra/helm/monitoring/values-monitoring.yaml

kubectl get pods -n monitoring -w
# Wait until all pods are Running
```

#### 6.2 Build and deploy v3.0 with metrics

```bash
cd inference-server
docker build -t llm-inference-server:v3.0 .
kind load docker-image llm-inference-server:v3.0 --name mlops-cluster
cd ..

# Update image tag
python3 - << 'EOF'
import yaml
with open("infra/helm/inference-server/values.yaml") as f:
    values = yaml.safe_load(f)
values["image"]["tag"] = "v3.0"
with open("infra/helm/inference-server/values.yaml", "w") as f:
    yaml.dump(values, f, default_flow_style=False)
EOF

git add infra/
git commit -m "deploy v3.0 with prometheus metrics"
git push
argocd app sync inference-server
```

#### 6.3 Access Grafana and import dashboard

```bash
kubectl port-forward -n monitoring service/monitoring-grafana 3000:80 &
sleep 5

curl -X POST http://admin:mlops-admin@localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d "{\"dashboard\": $(cat infra/helm/monitoring/dashboards/inference-dashboard.json), \"overwrite\": true, \"folderId\": 0}"
```

Open http://localhost:3000 → login `admin` / `mlops-admin` → Dashboards → LLM Inference Server

#### 6.4 Generate load to see metrics

```bash
source model-training/venv/bin/activate
python pipelines/load_test.py
```

Watch live graphs update in Grafana.

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `address already in use` port 5000 | macOS AirPlay Receiver | System Settings → General → AirDrop & Handoff → disable AirPlay Receiver |
| `ProfileNotFound: ghactions` | AWS_PROFILE env var set in shell | Add `os.environ.pop("AWS_PROFILE", None)` to scripts |
| `no_cuda is not a valid argument` | Newer transformers version | Replace `no_cuda=True` with `use_cpu=True` |
| `ModuleNotFoundError` | Wrong venv active | `source venv/bin/activate` from correct directory |
| `uvicorn` uses system Python | Shell path resolution | Use `python -m uvicorn` instead of `uvicorn` directly |
| `UPGRADE FAILED: no deployed releases` | ArgoCD owns the release | Use `helm upgrade --install` or commit + let ArgoCD sync |
| `nodePort Forbidden when type is ClusterIP` | nodePort set on ClusterIP service | Use conditional in service template: `{{- if eq .Values.service.type "NodePort" }}` |

---

## Key Concepts

**Why is this different from a web app deployment?**

In a standard web app pipeline, you build a Docker image containing your code and push it. In MLOps, the model weights are separate from the server code — they can be gigabytes or terabytes and live in object storage (MinIO/S3). The inference server container is just the runtime; it pulls the model weights on startup based on environment variables. This means you can update the model without rebuilding the container.

**How the GitOps loop works:**

1. Data scientist trains a new model and promotes it to `Production` in MLflow registry
2. CI pipeline detects the promotion, builds a new Docker image, updates `image.tag` in `values.yaml`
3. CI commits `values.yaml` to git and pushes
4. ArgoCD detects the diff between git state and cluster state
5. ArgoCD runs a Helm upgrade — pods rolling-restart and load the new model
6. Canary ingress routes a small percentage of traffic to the new version
7. Once validated, traffic shifts to 100% and the old version is retired

**Why canary instead of direct rollout?**

LLM model updates can silently degrade quality — the server stays up but responses get worse. Canary lets you compare the new model against the old one under real traffic before committing to a full rollout.

---

## GitHub Actions (Production CI)

The `.github/workflows/model-deploy.yml` workflow is the production equivalent of `pipelines/promote_and_deploy.py`. To use it with a real MLflow server, add these secrets to your GitHub repository:

| Secret | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | Your MLflow server URL |
| `MLFLOW_S3_ENDPOINT_URL` | Your MinIO/S3 endpoint |
| `AWS_ACCESS_KEY_ID` | Your access key |
| `AWS_SECRET_ACCESS_KEY` | Your secret key |

Trigger manually via GitHub Actions UI or via `repository_dispatch` from your training job.

---

## License

MIT
