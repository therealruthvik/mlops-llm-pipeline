import requests
import time
import random

PROMPTS = [
    "User: What is machine learning?\nAssistant:",
    "User: Explain neural networks.\nAssistant:",
    "User: What is MLOps?\nAssistant:",
    "User: How does ArgoCD work?\nAssistant:",
    "User: What is a Helm chart?\nAssistant:",
]

print("Sending load to inference server... Ctrl+C to stop\n")
count = 0
while True:
    try:
        prompt = random.choice(PROMPTS)
        r = requests.post(
            "http://localhost:8080/generate",
            json={"prompt": prompt, "max_new_tokens": 30},
            timeout=30
        )
        data = r.json()
        count += 1
        print(f"[{count}] v{data.get('model_version')} | {r.elapsed.total_seconds():.2f}s | {prompt[:40]}...")
        time.sleep(2)
    except KeyboardInterrupt:
        print(f"\nDone. Sent {count} requests.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
