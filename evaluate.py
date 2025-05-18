# import os
# import json
# from datetime import datetime
# from bigbench.api.json_task import JsonTask
# from bigbench.models.huggingface_models import BIGBenchHFModel

# HISTORY_FILE = "evaluation_history.json"

# def run_evaluation(model_name_or_path):
#     """
#     Run BIG-bench evaluation using a HuggingFace model on a specific task.
#     """
#     task_name = "logical_deduction_three_objects"
#     task_path = os.path.join("BIG-bench", "bigbench", "benchmark_tasks", task_name)

#     # ✅ Load the task with the correct path to task.json
#     task = JsonTask(task_data=os.path.join(task_path, "task.json"))

#     # Load the model
#     model = BIGBenchHFModel(
#         model_name=model_name_or_path,
#         max_length=512,
#         temperature=0.7,
#     )

#     # Run the evaluation
#     results = task.evaluate_model(
#         model=model,
#         num_examples=10,
#         max_examples=None,
#         show_progress=True,
#     )

#     # Extract accuracy
#     accuracy = results.get("metrics", {}).get("multiple_choice_grade", {}).get("value", 0)
#     accuracy_str = f"BIG-bench ({task_name}) Accuracy: {round(accuracy * 100, 2)}%"
#     return accuracy_str



# def save_result(model_name, result):
#     """
#     Save evaluation result to JSON history file.
#     """
#     history = {}
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, "r") as f:
#             history = json.load(f)

#     entry = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "result": result
#     }

#     history.setdefault(model_name, []).insert(0, entry)

#     with open(HISTORY_FILE, "w") as f:
#         json.dump(history, f, indent=4)


# def get_history(model_name):
#     """
#     Retrieve evaluation history for a given model.
#     """
#     if not os.path.exists(HISTORY_FILE):
#         return []
#     with open(HISTORY_FILE, "r") as f:
#         history = json.load(f)
#     return history.get(model_name, [])
import json
import os
from datetime import datetime
import seqio
from bigbench.bbseqio import tasks  # make sure BIG-bench is installed and bbseqio is available
from bigbench.models.huggingface_models import BIGBenchHFModel
from bigbench.models.huggingface_models import BIGBenchHFModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class CustomHFModel(BIGBenchHFModel):
    def __init__(self, model_name_or_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, local_files_only=True)        
        self.max_length = max_length
        

HISTORY_FILE = "evaluation_history.json"

def run_evaluation(model_name_or_path):
    """
    Run BIG-bench evaluation on a HuggingFace model using SeqIO task loading.
    """
    # Use SeqIO to get the BIG-bench task mixture or single task
    # Here I pick a single task by its canonical SeqIO name, adjust as needed
    task_name = "bigbench:unit_interpretation.mul.t5_default_vocab.3_shot.25_examples.lv0"
    task = seqio.get_mixture_or_task(task_name)

    # Create the dataset (all examples)
    ds = task.get_dataset(split="all", sequence_length={"inputs": 512, "targets": 32})

    
    model = CustomHFModel(
        model_name_or_path=model_name_or_path,
        max_length=512,
        
    )

    # Evaluate over dataset examples
    # This is a simplified evaluation loop — you might want to adapt it to your scoring needs
    total = 0
    correct = 0
    for i, example in enumerate(ds):
        if i >= 10:  # limit to 10 examples for quick eval
            break
        inputs = example["inputs"].numpy().decode("utf-8")
        targets = example["targets"].numpy().decode("utf-8")

        # Call model.generate or model.call (depending on your wrapper)
        output = model.generate(inputs)

        # Simple accuracy check for exact match — adjust as needed
        if output.strip() == targets.strip():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return f"BIG-bench ({task_name}) Accuracy: {round(accuracy * 100, 2)}%"

def save_result(model_name, result):
    history = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result
    }

    history.setdefault(model_name, []).insert(0, entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_history(model_name):
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    return history.get(model_name, [])
