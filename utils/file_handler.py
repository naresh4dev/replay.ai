from typing import List, Dict
import json
import pandas as pd

def load_jsonl(path: str, limit: int = 100) -> List[Dict]:
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            records.append(json.loads(line))
    return records

def persist_stage2_result(output_path, record):
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")



def load_stage2_jsonl(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            for r in rec["results"]:
                rows.append({
                    "conversation_id": rec["conversation_id"],
                    "prompt": rec["prompt"],
                    "category": rec.get("category"),
                    "model": r["model_slug"],
                    "provider": r["provider"],
                    "cost": r["cost_usd"],
                    "latency": r["latency_ms"],
                    "refusal": r["refusal"],
                    "judge_score": r["judge_score"]
                })

    return pd.DataFrame(rows)

