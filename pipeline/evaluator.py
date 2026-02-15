# pipeline/evaluator.py
import os
import json
from datetime import datetime

LOG_PATH = "data/logs/errors.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def consistency_score(output_json):
    """
    Basic consistency scoring. Expand with more checks later.
    Returns score in [0,1] and a list of found issues.
    """
    issues = []
    score = 0
    if "objects" in output_json and isinstance(output_json["objects"], list) and len(output_json["objects"])>0:
        score += 0.5
    else:
        issues.append("no objects")
    if "actions" in output_json and isinstance(output_json["actions"], list) and len(output_json["actions"])>0:
        score += 0.5
    else:
        issues.append("no actions")
    return score, issues

def log_error(error_type, message, output=None, metadata=None):
    metadata = metadata or {}
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "time": datetime.utcnow().isoformat()+"Z",
        "error_type": error_type,
        "message": message,
        "output": output,
        "metadata": metadata
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def query_alignment_score(query, output_json):
    """
    Measures word overlap between query and generated object names.
    Detects semantic drift / hallucination.
    """
    query_words = set(query.lower().split())

    object_words = set()
    for obj in output_json.get("objects", []):
        object_words.update(obj["name"].lower().split())

    if len(query_words) == 0:
        return 0.0

    overlap = query_words.intersection(object_words)
    return len(overlap) / len(query_words)
