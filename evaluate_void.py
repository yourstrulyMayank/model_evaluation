# evaluate.py
import os
import json
import re
import datetime
import random

import torch
import seqio
from datasets import disable_caching
from bigbench.bbseqio import tasks, vocabs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoConfig
)

disable_caching()

HISTORY_FILE = "evaluation_results/history.json"

# --------------------- TEXT NORMALIZER --------------------- #
def normalize(text):
    """Lowercase and remove extra spaces for fuzzy matching."""
    return " ".join(text.lower().split())

# --------------------- MAIN EVALUATION FUNCTION --------------------- #
def run_evaluation(model_name, num_examples=5, max_new_tokens=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    print(f"🔍 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    config = AutoConfig.from_pretrained(model_name)
    print(f"📦 Model config: {config}")
    if config.architectures:
        arch = config.architectures[0].lower()
        if 'seq2seq' in arch or 't5' in arch or 'bart' in arch:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif 'causallm' in arch or 'gpt' in arch or 'gemma' in arch:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
    else:
        # Fallback to CausalLM if architecture is not defined
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    vocab = vocabs.ALL_VOCABS["t5_default"]
    mixture_names = [
        "bigbench:bigbench_lite_v1.mix.t5_default_vocab.0_shot.1024_examples"
        # ,"bigbench:bigbench_lite_v1.mix.t5_default_vocab.1_shot.1024_examples"
        # ,"bigbench:bigbench_lite_v1.mix.t5_default_vocab.2_shot.1024_examples"
        # ,"bigbench:bigbench_lite_v1.mix.t5_default_vocab.3_shot.1024_examples"
    ]

    task_names = set()
    for mix_name in mixture_names:
        try:
            mix = seqio.get_mixture_or_task(mix_name)
            task_names.update([t.name for t in mix.tasks])
        except Exception as e:
            print(f"⚠️ Failed to load mixture: {mix_name}: {e}")
    
    task_names = sorted(task_names)
    print(f"📘 Loaded {len(task_names)} unique tasks from BIG-bench lite")

    global_total = 0
    global_matches = 0
    all_results = []

    for task_name in task_names:
        print(f"\n🔍 Evaluating task: {task_name}")
        print("=" * 100)

        try:
            task = seqio.get_mixture_or_task(task_name)
            dataset = task.get_dataset(split="validation")

            exact_matches = 0
            total = 0
            samples = []

            for i, example in enumerate(dataset):
                input_text = vocab.vocabulary.decode(example["inputs"].numpy())
                target_text = vocab.vocabulary.decode(example["targets"].numpy()).strip()

                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

                is_match = normalize(target_text) == normalize(prediction)
                exact_matches += int(is_match)
                total += 1
                global_total += 1
                global_matches += int(is_match)

                samples.append({
                    "example_number": i + 1,
                    "input": input_text,
                    "expected": target_text,
                    "generated": prediction,
                    "match": is_match
                })

                print(f"\n🔹 Example {i+1}")
                print(f"Input:\n{input_text}")
                print(f"Expected:\n{target_text}")
                print(f"Predicted:\n{prediction}")
                print(f"✅ Match: {is_match}")
                print("-" * 80)

                if i + 1 >= num_examples:
                    break

            accuracy = exact_matches / total if total else 0.0
            print(f"\n📊 Task Accuracy for {task_name}: {exact_matches}/{total} = {accuracy:.2%}")

            all_results.append({
                "task": task_name,
                "accuracy": round(accuracy * 100, 2),
                "samples": samples,
                "timestamp": datetime.datetime.now().isoformat()
            })

        except Exception as e:
            print(f"❌ Failed to evaluate task '{task_name}': {e}")

    print(f"\n✅ Completed evaluation on {len(all_results)} tasks.")
    print(f"📈 Overall Accuracy: {global_matches}/{global_total} = {(global_matches / global_total):.2%}" if global_total else "N/A")
    
    _save_results(model_name, all_results)
    return all_results


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

    history.insert(0, entry)  # Most recent first

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

# --------------------- ENTRY POINT --------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HF models on BIG-bench lite")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or local path")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples per task")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens to generate per example")

    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        num_examples=args.num_examples,
        max_new_tokens=args.max_tokens
    )
