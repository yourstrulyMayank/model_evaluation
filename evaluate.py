import os
import json
import torch
import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bigbench.api import model as bb_model
from bigbench.api import json_task
# from BIG-bench.bigbench import benchmark_tasks

HISTORY_FILE = "evaluation_results/history.json"

# --------------------- MODEL WRAPPER --------------------- #
class WrappedHFModel(bb_model.Model):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_text(self, inputs, max_length=256, stop_string=None, output_regex=None):
        if isinstance(inputs, str):
            inputs = [inputs]

        encodings = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**encodings, max_new_tokens=max_length)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded if len(decoded) > 1 else decoded[0]

    def cond_log_prob(self, inputs, targets, absolute_normalization=False):
        raise NotImplementedError("cond_log_prob is not implemented.")

    def model_data(self):
        return bb_model.ModelData(
            model_family="HF",
            model_name="HFModel",
            total_params=0,
            non_embedding_params=0,
            flop_matched_non_embedding_params=0,
            training_batch_size=0,
            training_steps=0,
            description="Wrapped Hugging Face Model"
        )

# --------------------- EVALUATION FUNCTION --------------------- #
def run_evaluation(model_path):
    print(f"🔍 Loading model from: {model_path}")
    model = WrappedHFModel(model_path)

    task_base =  "BIG-bench/bigbench/benchmark_tasks"
    task_names = [
        name for name in os.listdir(task_base)
        if os.path.isdir(os.path.join(task_base, name)) and
        os.path.isfile(os.path.join(task_base, name, "task.json"))
    ]

    print(f"📘 Found {len(task_names)} BIG-bench tasks.")
    results = []

    for task_name in sorted(task_names):
        try:
            print(f"🚀 Evaluating task: {task_name}")
            task_path = os.path.join(task_base, task_name, "task.json")

            task = json_task.JsonTask(
                task_path,
                shot_list=[0],
                max_examples=20,
                verbose=False
            )

            score_data = task.evaluate_model(model)
            aggregated_score = score_data.get("aggregated_score", 0.0)

            samples = []
            for i, ex in enumerate(score_data.get("examples", [])):
                sample = {
                    "example_number": i + 1,
                    "prompt": ex.get("input", ""),
                    "generated": ex.get("model_response", ""),
                    "expected": ex.get("target", ""),
                    "match": ex.get("target") in ex.get("model_response", "")
                }
                samples.append(sample)

            results.append({
                "task": task_name,
                "accuracy": round(aggregated_score * 100, 2),
                "samples": samples,
                "timestamp": datetime.datetime.now().isoformat()
            })

        except Exception as e:
            print(f"❌ Error in task '{task_name}': {e}")

    _save_results(model_path, results)
    print(f"\n✅ Evaluation complete. {len(results)} tasks evaluated.")
    return results

# --------------------- HISTORY SAVE & LOAD --------------------- #
def _save_results(model_path, results):
    entry = {
        "model_path": model_path,
        "results": results,
        "timestamp": datetime.datetime.now().isoformat()
    }

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.insert(0, entry)  # Latest first

    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_history(model_name=None):
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []

    if model_name:
        return [entry for entry in data if model_name in entry["model_path"]]
    return data
