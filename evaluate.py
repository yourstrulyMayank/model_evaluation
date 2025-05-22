import os
import json
import torch
import datetime
import random
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from bigbench.api import model as bb_model
from bigbench.api import json_task
# from bigbench.api import task_utils
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMultipleChoice

task_base = "BIG-bench/bigbench/benchmark_tasks"
HISTORY_FILE = "evaluation_results/history.json"

MODEL_TYPE_MAP = {
    "gpt2": "GPT-2",
    "gptj": "GPT-J",
    "llama": "LLaMA",
    "mpt": "MPT",
    "falcon": "Falcon",
    "bloom": "BLOOM",
    "t5": "T5",
    "bart": "BART",
    # add others if needed
}
model_type = MODEL_TYPE_MAP.get(config.model_type, "Other")

# --------------------- MODEL WRAPPER --------------------- #
class WrappedHFModel(bb_model.Model):
    def __init__(self, model_path):
        config = AutoConfig.from_pretrained(model_path)
        print(f"📌 Model config: {config}")
        if config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:            
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)        
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

# --------------------- TASK FILTERING --------------------- #
def get_all_task_names(task_dir):
    return [
        name for name in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, name)) and
        os.path.isfile(os.path.join(task_dir, name, "task.json"))
    ]

# def filter_supported_tasks(model, task_dir):    
#     all_tasks = get_all_task_names(task_dir)
#     supported = []

#     # Check model type
#     is_seq2seq = isinstance(model.model, AutoModelForSeq2SeqLM)
#     print(f"📌 Model type: {'Seq2Seq' if is_seq2seq else 'CausalLM'}")

#     for task_name in all_tasks:
#         task_path = os.path.join(task_base, task_name, "task.json")
#         try:
#             task = json_task.JsonTask(task_path, shot_list=[0])

#             # Use first example to guess if it's suitable
#             examples = task.examples
#             if not examples:
#                 continue

#             ex = examples[0]
#             has_input = "input" in ex
#             has_target = "target" in ex

#             # Logic for model compatibility
#             if is_seq2seq and has_input and has_target:
#                 supported.append(task_name)
#             elif not is_seq2seq and has_input and has_target:
#                 supported.append(task_name)

#         except Exception as e:
#             print(f"⚠️ Skipping task {task_name}: {e}")
#             continue

#     return supported

def filter_supported_tasks(model, task_dir):    
    all_tasks = get_all_task_names(task_dir)

    supported = []
    is_seq2seq = isinstance(model.model, AutoModelForSeq2SeqLM)
    is_causal = isinstance(model.model, AutoModelForCausalLM)

    print(f"📌 Model type: {'Seq2SeqLM' if is_seq2seq else 'CausalLM' if is_causal else 'Other'}")

    for task_name in all_tasks:
        task_path = os.path.join(task_base, task_name, "task.json")
        try:
            print(f"🔍 Checking task: {task_name}")
            task = json_task.JsonTask(task_path, shot_list=[0])

            # Validate examples
            if not task.examples or not isinstance(task.examples, list):
                continue

            # Read metrics from task JSON directly
            with open(task_path, 'r') as f:
                task_json = json.load(f)
                metrics = task_json.get("metrics", [])
            
            # Match based on metrics
            if "generate_text" in metrics and (is_seq2seq or is_causal):
                supported.append(task_name)
            elif "multiple_choice_grade" in metrics:
                supported.append(task_name)

        except Exception as e:
            print(f"⚠️ Skipping task {task_name}: {e}")
            continue

    print(f"✅ Supported tasks: {len(supported)}")
    return supported



# --------------------- TEXT NORMALIZER --------------------- #
def normalize(text):
    """Lowercase and remove punctuation for fuzzy matching."""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()

# --------------------- EVALUATION FUNCTION --------------------- #
def run_evaluation(model_path):
    
    print(f"🔍 Loading model from: {model_path}")
    model = WrappedHFModel(model_path)
    
    results = []
    
    num_tasks_to_run = 2
    print(f"🔍 Loading model from: {model_path}")
    model = WrappedHFModel(model_path)

    all_supported_tasks = filter_supported_tasks(model, task_base)
    print(f"📘 Found {len(all_supported_tasks)} BIG-bench tasks.")

    if len(all_supported_tasks) == 0:
        print("⚠️ No supported tasks found for this model.")
        return []

    if num_tasks_to_run > len(all_supported_tasks):
        print(f"⚠️ Only {len(all_supported_tasks)} supported tasks found. Reducing sample size.")
        num_tasks_to_run = len(all_supported_tasks)

    task_names = random.sample(all_supported_tasks, num_tasks_to_run)


    for task_name in sorted(task_names):
        try:
            print(f"🚀 Evaluating task: {task_name}")
            task_path = os.path.join(task_base, task_name, "task.json")

            task = json_task.JsonTask(
                task_path,
                shot_list=[0],                
                verbose=False
            )

            score_data = task.evaluate_model(model)
            aggregated_score = score_data.get("aggregated_score", None)
            if aggregated_score is None:
                print(f"⚠️ No aggregated_score for task {task_name}")
                continue

            samples = []
            for i, ex in enumerate(score_data.get("examples", [])):
                expected = ex.get("target", "")
                generated = ex.get("model_response", "")

                if isinstance(expected, list):
                    expected_text = expected[0] if expected else ""
                    match = any(normalize(e) in normalize(generated) for e in expected)
                else:
                    expected_text = expected
                    match = normalize(expected_text) in normalize(generated)

                sample = {
                    "example_number": i + 1,
                    "prompt": ex.get("input", ""),
                    "generated": generated,
                    "expected": expected_text,
                    "match": match
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
