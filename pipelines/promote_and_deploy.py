"""
CI Pipeline Script — simulates what GitHub Actions runs in production.

Flow:
  1. Promote latest model version to Production in MLflow
  2. Build new inference server Docker image
  3. Load image into kind cluster
  4. Update values.yaml with new image tag + model version
  5. Git commit + push → ArgoCD auto-syncs
"""

import os
import sys
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI            = "http://localhost:5001"
MODEL_NAME            = "llm-chatbot"
NEW_STAGE             = "Production"
IMAGE_REPO            = "llm-inference-server"
VALUES_FILE           = "infra/helm/inference-server/values.yaml"
KIND_CLUSTER          = "mlops-cluster"

os.environ["MLFLOW_S3_ENDPOINT_URL"]  = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]       = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]   = "minioadmin"
os.environ.pop("AWS_PROFILE", None)
os.environ.pop("AWS_DEFAULT_PROFILE", None)

def run(cmd, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and check:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)
    return result

# ── Step 1: Promote latest model to Production ────────────────────────────────
print("\n📋 Step 1: Promoting latest model to Production...")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

all_versions = client.get_latest_versions(MODEL_NAME)
latest = sorted(all_versions, key=lambda v: int(v.version))[-1]
print(f"  Latest version: {latest.version} (currently: {latest.current_stage})")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage=NEW_STAGE,
    archive_existing_versions=True
)
print(f"  ✅ Version {latest.version} promoted to {NEW_STAGE}")

new_image_tag = f"v2.{latest.version}"

# ── Step 2: Build new Docker image ───────────────────────────────────────────
print(f"\n🐳 Step 2: Building Docker image {IMAGE_REPO}:{new_image_tag}...")
run(f"docker build -t {IMAGE_REPO}:{new_image_tag} inference-server/")
print(f"  ✅ Image built: {IMAGE_REPO}:{new_image_tag}")

# ── Step 3: Load image into kind cluster ─────────────────────────────────────
print(f"\n📦 Step 3: Loading image into kind cluster...")
run(f"kind load docker-image {IMAGE_REPO}:{new_image_tag} --name {KIND_CLUSTER}")
print(f"  ✅ Image loaded into cluster")

# ── Step 4: Update values.yaml ───────────────────────────────────────────────
print(f"\n📝 Step 4: Updating {VALUES_FILE}...")
with open(VALUES_FILE, "r") as f:
    values = yaml.safe_load(f)

old_tag = values["image"]["tag"]
values["image"]["tag"]     = new_image_tag
values["model"]["version"] = latest.version

with open(VALUES_FILE, "w") as f:
    yaml.dump(values, f, default_flow_style=False)

print(f"  image.tag:     {old_tag} → {new_image_tag}")
print(f"  model.version: → {latest.version}")
print(f"  ✅ values.yaml updated")

# ── Step 5: Git commit + push → ArgoCD picks up ──────────────────────────────
print(f"\n🚀 Step 5: Committing and pushing to git...")
run(f'git add {VALUES_FILE}')
run(f'git commit -m "ci: promote model v{latest.version} to production [image={new_image_tag}]"')
run(f'git push origin main')
print(f"  ✅ Pushed. ArgoCD will detect change and sync.")

print(f"""
╔══════════════════════════════════════════════════════════╗
║           Pipeline Complete!                            ║
║                                                         ║
║  Model version : {latest.version:<38} ║
║  Image tag     : {new_image_tag:<38} ║
║  Stage         : {NEW_STAGE:<38} ║
║                                                         ║
║  ArgoCD syncing now. Watch with:                        ║
║  kubectl get pods -w                                    ║
║  argocd app get inference-server                        ║
╚══════════════════════════════════════════════════════════╝
""")
