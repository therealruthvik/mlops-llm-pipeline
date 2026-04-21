import requests
import json
from collections import Counter

URL     = "http://localhost:8080/model-info"
SAMPLES = 20
results = []

print(f"Sending {SAMPLES} requests to test canary split...\n")

for i in range(SAMPLES):
    try:
        r = requests.get(URL, timeout=10)
        data = r.json()
        version = data.get("model_version", "unknown")
        results.append(version)
        print(f"  [{i+1:02d}] model_version={version}")
    except Exception as e:
        print(f"  [{i+1:02d}] ERROR: {e}")
        results.append("error")

counts = Counter(results)
print(f"\n── Traffic Distribution ──────────────────")
for version, count in sorted(counts.items()):
    pct = (count / SAMPLES) * 100
    bar = "█" * int(pct / 5)
    print(f"  v{version}: {count:2d}/{SAMPLES} ({pct:5.1f}%)  {bar}")
print(f"──────────────────────────────────────────")
print(f"\nExpected: ~90% v1, ~10% v2 (canary weight=10)")
