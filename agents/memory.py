"""
Analyst Memory Agent.

Stores a lightweight summary of each analysis run (dataset name,
timestamp, key metric total, top finding) in a local JSON file, so
future runs can compare against history instead of treating every
upload as a brand new world. This is a real, working implementation -
just file-based instead of a database, which is the honest tradeoff
for a fast build. Swapping this for DynamoDB later is a small change
(see README).
"""

import json
import os
from datetime import datetime

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "analyst_memory.json")


def _load_memory() -> list:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r") as f:
        return json.load(f)


def _save_memory(history: list):
    with open(MEMORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def record_run(dataset_name: str, metric_total: float, top_finding_title: str) -> dict:
    history = _load_memory()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "metric_total": metric_total,
        "top_finding": top_finding_title,
    }
    history.append(entry)
    _save_memory(history)

    return check_against_history(metric_total, history[:-1])


def check_against_history(current_total: float, past_runs: list) -> dict:
    if len(past_runs) < 2:
        return {"has_history": False, "message": "Not enough previous runs yet to compare against."}

    past_totals = [r["metric_total"] for r in past_runs[-8:]]
    avg_past = sum(past_totals) / len(past_totals)

    if avg_past == 0:
        return {"has_history": False}

    deviation_pct = round(100 * (current_total - avg_past) / avg_past, 1)

    is_anomaly = abs(deviation_pct) > 20

    return {
        "has_history": True,
        "runs_compared": len(past_totals),
        "average_past_total": round(avg_past, 2),
        "current_total": round(current_total, 2),
        "deviation_pct": deviation_pct,
        "is_anomaly": is_anomaly,
        "message": (
            f"Current total is {abs(deviation_pct)}% {'below' if deviation_pct < 0 else 'above'} "
            f"the average of the last {len(past_totals)} runs."
            if is_anomaly else
            "Current total is within the normal range compared to previous runs."
        ),
    }


def get_history() -> list:
    return _load_memory()
