import os
import json
import mlflow
import mlflow.pytorch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_VERSION = "v1.5"
BASE_MODEL    = "distilgpt2"
EXPERIMENT    = "llm-finetuning"
RUN_NAME      = f"model-{MODEL_VERSION}"

# Point MLflow at our local server + MinIO for artifact storage
os.environ["MLFLOW_S3_ENDPOINT_URL"]  = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]       = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]   = "minioadmin"
os.environ.pop("AWS_PROFILE", None)   # clear any shell AWS profile
os.environ.pop("AWS_DEFAULT_PROFILE", None)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment(EXPERIMENT)

# ── Tiny dataset (simulates domain fine-tuning) ───────────────────────────────
TRAINING_DATA = [
    {"text": "User: What is machine learning?\nAssistant: Machine learning is a subset of AI."},
    {"text": "User: How does a neural network work?\nAssistant: Neural networks use layers of nodes."},
    {"text": "User: What is gradient descent?\nAssistant: Gradient descent minimizes the loss function."},
    {"text": "User: Explain transformers.\nAssistant: Transformers use attention mechanisms for sequence modeling."},
    {"text": "User: What is overfitting?\nAssistant: Overfitting is when a model memorizes training data."},
    {"text": "User: What is a GPU cluster?\nAssistant: A GPU cluster is a group of servers with GPUs for parallel computation."},
]

# ── Training ──────────────────────────────────────────────────────────────────
def tokenize(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

def train(version: str, base_model: str, num_epochs: int, learning_rate: float):
    print(f"\n🚀 Training {version} based on {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    raw_dataset = Dataset.from_list(TRAINING_DATA)
    tokenized   = raw_dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    tokenized.set_format("torch")

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{version}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,          # use MPS on M2 by setting this False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    with mlflow.start_run(run_name=RUN_NAME) as run:
        # Log hyperparameters
        mlflow.log_params({
            "model_version": version,
            "base_model":    base_model,
            "num_epochs":    num_epochs,
            "learning_rate": learning_rate,
            "dataset_size":  len(TRAINING_DATA),
        })

        # Train
        train_result = trainer.train()

        # Log metrics
        mlflow.log_metrics({
            "train_loss":             train_result.training_loss,
            "train_runtime_seconds":  train_result.metrics["train_runtime"],
            "samples_per_second":     train_result.metrics["train_samples_per_second"],
        })

        # Save model locally then log as artifact
        local_path = f"/tmp/model-{version}"
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

        # Save model metadata
        metadata = {
            "model_version":   version,
            "base_model":      base_model,
            "training_loss":   train_result.training_loss,
            "framework":       "pytorch",
            "task":            "causal-lm",
        }
        with open(f"{local_path}/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Log all model files as artifacts in MLflow (stored in MinIO)
        mlflow.log_artifacts(local_path, artifact_path=f"model-{version}")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model-{version}"
        registered = mlflow.register_model(
            model_uri=model_uri,
            name="llm-chatbot"
        )

        print(f"\n✅ Model registered!")
        print(f"   Run ID:          {run.info.run_id}")
        print(f"   Registry version: {registered.version}")
        print(f"   Artifact URI:     {run.info.artifact_uri}/model-{version}")

        return run.info.run_id, registered.version

if __name__ == "__main__":
    run_id, reg_version = train(
        version=MODEL_VERSION,
        base_model=BASE_MODEL,
        num_epochs=1,
        learning_rate=5e-5,
    )
    print(f"\n📦 Done. Model v1.5 is in MLflow registry as version {reg_version}")
    print(f"   Open http://localhost:5000 to see the run and registered model.")
