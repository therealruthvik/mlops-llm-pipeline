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

MODEL_VERSION = "v2.0"
BASE_MODEL    = "distilgpt2"
EXPERIMENT    = "llm-finetuning"
RUN_NAME      = f"model-{MODEL_VERSION}"

os.environ["MLFLOW_S3_ENDPOINT_URL"]  = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]       = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]   = "minioadmin"
os.environ.pop("AWS_PROFILE", None)
os.environ.pop("AWS_DEFAULT_PROFILE", None)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment(EXPERIMENT)

# Expanded dataset — simulates v2.0 trained on more data
TRAINING_DATA = [
    {"text": "User: What is machine learning?\nAssistant: Machine learning is a subset of AI that learns from data."},
    {"text": "User: How does a neural network work?\nAssistant: Neural networks use layers of interconnected nodes to process information."},
    {"text": "User: What is gradient descent?\nAssistant: Gradient descent is an optimization algorithm that minimizes the loss function iteratively."},
    {"text": "User: Explain transformers.\nAssistant: Transformers use self-attention mechanisms to model relationships in sequences."},
    {"text": "User: What is overfitting?\nAssistant: Overfitting occurs when a model memorizes training data and fails to generalize."},
    {"text": "User: What is a GPU cluster?\nAssistant: A GPU cluster is a group of servers with GPUs for parallel computation in ML workloads."},
    {"text": "User: What is MLOps?\nAssistant: MLOps is the practice of applying DevOps principles to machine learning workflows."},
    {"text": "User: What is a model registry?\nAssistant: A model registry stores versioned model artifacts and tracks their lifecycle stages."},
    {"text": "User: What is ArgoCD?\nAssistant: ArgoCD is a GitOps tool that syncs Kubernetes deployments with a git repository."},
    {"text": "User: What is a Helm chart?\nAssistant: A Helm chart is a package of Kubernetes manifests that can be versioned and deployed."},
]

def tokenize(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

def train(version, base_model, num_epochs, learning_rate):
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
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({
            "model_version": version,
            "base_model":    base_model,
            "num_epochs":    num_epochs,
            "learning_rate": learning_rate,
            "dataset_size":  len(TRAINING_DATA),
        })

        train_result = trainer.train()

        mlflow.log_metrics({
            "train_loss":            train_result.training_loss,
            "train_runtime_seconds": train_result.metrics["train_runtime"],
            "samples_per_second":    train_result.metrics["train_samples_per_second"],
        })

        local_path = f"/tmp/model-{version}"
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

        metadata = {
            "model_version": version,
            "base_model":    base_model,
            "training_loss": train_result.training_loss,
            "dataset_size":  len(TRAINING_DATA),
        }
        with open(f"{local_path}/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifacts(local_path, artifact_path=f"model-{version}")

        model_uri  = f"runs:/{run.info.run_id}/model-{version}"
        registered = mlflow.register_model(model_uri=model_uri, name="llm-chatbot")

        print(f"\n✅ Model v2.0 registered as version {registered.version}")
        print(f"   Run ID: {run.info.run_id}")
        return run.info.run_id, registered.version

if __name__ == "__main__":
    run_id, reg_version = train(
        version=MODEL_VERSION,
        base_model=BASE_MODEL,
        num_epochs=2,           # more epochs than v1.5
        learning_rate=3e-5,     # different LR than v1.5
    )
    print(f"\n📦 Model v2.0 in registry as version {reg_version}")
    print(f"   Next: run promote_v2.py to push to Production")
