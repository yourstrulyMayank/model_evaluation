import json
import os
from datetime import datetime

HISTORY_FILE = "evaluation_history.json"

def run_evaluation(model_path):
    # Simulate evaluation logic (replace with real eval)
    import time
    time.sleep(5)
    score = round(0.8 + 0.1 * (hash(model_path) % 10) / 10, 2)
    return f"Accuracy: {score}"

def save_result(model_name, result):
    history = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result
    }
    
    history.setdefault(model_name, []).insert(0, entry)  # most recent first
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_history(model_name):
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    return history.get(model_name, [])
