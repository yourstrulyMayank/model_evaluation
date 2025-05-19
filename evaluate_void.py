# import json
# import os
# from datetime import datetime
# import seqio
# from bigbench.bbseqio import tasks  # make sure BIG-bench is installed and bbseqio is available
# from bigbench.models.huggingface_models import BIGBenchHFModel
# from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# def load_model(model_name_or_path, local_files_only=True):
#     config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only)
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)

#     # Decide model class based on architecture
#     if config.is_encoder_decoder:
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config, local_files_only=local_files_only)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, local_files_only=local_files_only)

#     return tokenizer, model

# class CustomHFModel(BIGBenchHFModel):
#     def __init__(self, model_name_or_path, max_length=512):
#          self.tokenizer, self.model = load_model(model_name_or_path)
#         self.max_length = max_length

#     def generate(self, prompt, max_new_tokens=32):
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#         outputs = self.model.generate(
#             inputs.input_ids,
#             max_new_tokens=max_new_tokens,
#             do_sample=False
#         )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



# HISTORY_FILE = "evaluation_history.json"

# def safe_decode(tensor):
#     try:
#         return tensor.numpy().decode("utf-8")
#     except:
#         try:
#             return tensor.numpy().tobytes().decode("utf-8", errors="ignore")
#         except:
#             return str(tensor.numpy())

# def run_evaluation(model_name_or_path):
#     task_name = "bigbench:unit_interpretation.mul.t5_default_vocab.3_shot.25_examples.lv0"
#     task = seqio.get_mixture_or_task(task_name)

#     ds = task.get_dataset(split="all", sequence_length={"inputs": 512, "targets": 32})

#     model = CustomHFModel(model_name_or_path=model_name_or_path, max_length=512)

#     total = 0
#     correct = 0
#     samples = []

#     for i, example in enumerate(ds):
#         if i >= 10:
#             break

#         inputs = safe_decode(example["inputs"])
#         targets = safe_decode(example["targets"])

#         try:
#             output = model.generate(inputs, max_new_tokens=32)

#             match = output.strip().lower() == targets.strip().lower()

#             samples.append({
#                 "example_number": i + 1,
#                 "prompt": inputs,
#                 "generated": output.strip(),
#                 "expected": targets.strip(),
#                 "match": match
#             })

#             if match:
#                 correct += 1
#             total += 1
#         except Exception as e:
#             print(f"⚠️ Error on example {i}: {e}")

#     accuracy = correct / total if total > 0 else 0

#     return {
#         "task": task_name,
#         "accuracy": round(accuracy * 100, 2),
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "samples": samples
#     }

# def save_result(model_name, result_dict):
#     history = {}
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, "r") as f:
#             history = json.load(f)

#     history.setdefault(model_name, []).insert(0, result_dict)

#     with open(HISTORY_FILE, "w") as f:
#         json.dump(history, f, indent=4)


# def get_history(model_name):
#     if not os.path.exists(HISTORY_FILE):
#         return []
#     with open(HISTORY_FILE, "r") as f:
#         history = json.load(f)
#     return history.get(model_name, [])



import os
import json
import argparse
import importlib
import traceback
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from bigbench.api import json_task, util
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
TASKS_PATH = "bigbench/benchmark_tasks"
HISTORY_FILE = "evaluation_results/history.json"
os.makedirs("evaluation_results", exist_ok=True)

# -------------------------------
# Smart Model Loader
# -------------------------------
def smart_load_model(model_path):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        except Exception as e:
            print("Model loading failed:")
            traceback.print_exc()
            raise RuntimeError("Could not load model as CausalLM or Seq2SeqLM")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# -------------------------------
# Minimal BIG-bench Model Wrapper
# -------------------------------
from bigbench.api import model as bb_model
from bigbench.models import model_utils

class WrappedHFModel(bb_model.Model):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, inputs, max_length=1000, stop_string=None, output_regex=None):
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = []
        for prompt in inputs:
            inputs_enc = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs_ids = self.model.generate(
                    **inputs_enc, max_new_tokens=100, do_sample=False
                )
            out = self.tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
            out = model_utils.postprocess_output(out, max_length, stop_string, output_regex)
            outputs.append(out)
        return outputs if len(outputs) > 1 else outputs[0]

# -------------------------------
# Load all BIG-Bench Lite tasks
# -------------------------------
def get_all_tasks(task_root):
    task_names = []
    for task_dir in os.listdir(task_root):
        full_path = os.path.join(task_root, task_dir)
        if os.path.isdir(full_path) and "task.json" in os.listdir(full_path):
            task_names.append(task_dir)
    return sorted(task_names)

def load_task(task_name):
    task_module_name = f"bigbench.benchmark_tasks.{task_name}"
    task_module = importlib.import_module(task_module_name)
    module_path = list(task_module.__path__)[0]
    json_path = os.path.join(module_path, "task.json")
    return json_task.JsonTask(json_path, shot_list=[0], verbose=False, max_examples=5)

# -------------------------------
# Load or initialize history
# -------------------------------
def get_history(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_result(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# -------------------------------
# Main evaluation loop
# -------------------------------
def run_evaluation(model_path):
    print(f"Loading model from: {model_path}")
    model, tokenizer = smart_load_model(model_path)
    hf_model = WrappedHFModel(model, tokenizer)

    tasks = get_all_tasks(TASKS_PATH)
    print(f"Found {len(tasks)} BIG-Bench Lite tasks")

    all_results = {}
    for task_name in tqdm(tasks, desc="Evaluating tasks"):
        try:
            task = load_task(task_name)
            score = task.evaluate_model(hf_model)
            all_results[task_name] = score["aggregated_score"]
        except Exception as e:
            all_results[task_name] = f"Error: {str(e)}"

    print("\n🧠 Final Evaluation Results:\n")
    for task, score in all_results.items():
        print(f"{task}: {score}")

    # Save to history
    history = get_history(HISTORY_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        "model": model_path,
        "timestamp": timestamp,
        "results": all_results
    }
    history_key = f"{os.path.basename(model_path)}_{timestamp}"
    history[history_key] = history_entry
    save_result(history, HISTORY_FILE)
    print(f"\n✅ Results saved to {HISTORY_FILE}")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to Hugging Face model folder")
    args = parser.parse_args()
    run_evaluation(args.model)
