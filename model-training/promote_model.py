import os
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_S3_ENDPOINT_URL"]  = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]       = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]   = "minioadmin"

mlflow.set_tracking_uri("http://localhost:5001")
client = MlflowClient()

# Get the latest version of our model
versions = client.get_latest_versions("llm-chatbot")
latest   = versions[0]

print(f"Current model: version={latest.version}, stage={latest.current_stage}")

# Transition to Production (this is the gate that triggers downstream CI)
client.transition_model_version_stage(
    name="llm-chatbot",
    version=latest.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✅ Model version {latest.version} promoted to Production")
print(f"   This event will trigger the CI pipeline in Phase 4.")
