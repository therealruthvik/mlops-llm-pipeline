import os
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import mlflow
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from env vars (this is how Helm/ArgoCD will control model version) ─
MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_S3_ENDPOINT    = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MODEL_NAME            = os.getenv("MODEL_NAME", "llm-chatbot")
MODEL_STAGE           = os.getenv("MODEL_STAGE", "Production")  # ← THIS is what changes v1.5 → v2.0

os.environ["MLFLOW_S3_ENDPOINT_URL"]  = MLFLOW_S3_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"]       = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"]   = AWS_SECRET_ACCESS_KEY
os.environ.pop("AWS_PROFILE", None)

# ── Global model state ────────────────────────────────────────────────────────
model_state = {"model": None, "tokenizer": None, "version": None}

def load_model_from_registry():
    """Pull model artifacts from MLflow registry (stored in MinIO) and load."""
    logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get Production model version
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No model '{MODEL_NAME}' in stage '{MODEL_STAGE}'")

    latest = versions[0]
    artifact_uri = client.get_model_version_download_uri(MODEL_NAME, latest.version)
    logger.info(f"Loading model version={latest.version} from {artifact_uri}")

    # Download artifacts locally
    local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    logger.info(f"Artifacts downloaded to {local_path}")

    # Load tokenizer from HuggingFace directly (avoids version mismatch)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Load fine-tuned weights from MLflow/MinIO
    model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.float32)
    model.eval()

    model_state["model"]     = model
    model_state["tokenizer"] = tokenizer
    model_state["version"]   = latest.version
    logger.info(f"Model v{latest.version} loaded and ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_registry()
    yield

app = FastAPI(title="LLM Inference Server", lifespan=lifespan)

# ── Request/Response schemas ──────────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False

class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    stage: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status":        "ok",
        "model_name":    MODEL_NAME,
        "model_version": str(model_state["version"]),
        "stage":         MODEL_STAGE,
    }

@app.post("/generate")
def generate(req: PromptRequest):
    tokenizer = model_state["tokenizer"]
    model     = model_state["model"]

    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    if req.stream:
        # Streaming response — simulates what ChatGPT does token by token
        def token_stream() -> AsyncGenerator:
            with torch.no_grad():
                generated = input_ids.clone()
                for _ in range(req.max_new_tokens):
                    outputs    = model(generated)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated  = torch.cat([generated, next_token], dim=-1)
                    token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    yield f"data: {json.dumps({'token': token_text})}\n\n"
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    else:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return {"response": response_text, "model_version": model_state["version"]}

@app.get("/model-info")
def model_info():
    return {
        "model_name":    MODEL_NAME,
        "model_version": model_state["version"],
        "stage":         MODEL_STAGE,
        "tracking_uri":  MLFLOW_TRACKING_URI,
    }
