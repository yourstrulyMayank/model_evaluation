from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import threading
from evaluate import run_evaluation, get_history
from ensure_models import main as ensure_models

app = Flask(__name__)

model_base_path = "models"
processing_status = {}  # 🔁 Track per-model status (thread-safe)

categories = {
    "llm": [],
    "genai": [],
    "dl": [],
    "ml": []
}

# Create folders
for category, model_list in categories.items():
    category_path = os.path.join(model_base_path, category)
    os.makedirs(category_path, exist_ok=True)

    for model_name in model_list:
        model_path = os.path.join(category_path, model_name)
        os.makedirs(model_path, exist_ok=True)
        print(f"Created: {model_path}")

ensure_models()

@app.route('/')
def index():
    categories = {
        "LLMs": "llm",
        "Other GenAI Models": "genai",
        "DL Models": "dl",
        "ML Models": "ml"
    }

    model_data = {}

    for display_name, folder in categories.items():
        path = os.path.join(model_base_path, folder)
        if os.path.exists(path):
            models = [model for model in os.listdir(path) if os.path.isdir(os.path.join(path, model))]
            model_data[display_name] = models
        else:
            model_data[display_name] = []

    return render_template("index.html", model_data=model_data)

@app.route('/evaluate_model/<category>/<model_name>')
def evaluate(category, model_name):
    model_path = None
    categories = {
        "LLMs": "llm",
        "Other GenAI Models": "genai",
        "DL Models": "dl",
        "ML Models": "ml"
    }
    potential_path = os.path.join(model_base_path, categories[category], model_name)
    if os.path.exists(potential_path):
        model_path = potential_path

    if not model_path:
        return f"Model '{model_name}' not found.", 404

    # 🔁 Reset this model's status
    processing_status[model_name] = "processing"

    def background_task():
        try:
            result = run_evaluation(model_path)
            # save_result(model_name, result)
            processing_status[model_name] = "complete"
        except Exception as e:
            print(f"Error during evaluation: {e}")
            processing_status[model_name] = "error"

    threading.Thread(target=background_task).start()
    return render_template('loading.html', model_name=model_name)

@app.route('/check_status/<model_name>')
def check_status(model_name):
    status = processing_status.get(model_name, "not_started")
    return jsonify({"status": status})

@app.route('/history/<model_name>')
def history(model_name):
    history_data = get_history(model_name)
    return render_template('history.html', model_name=model_name, history=history_data)

@app.route('/results/<model_name>')
def analyze(model_name):
    history_data = get_history(model_name)
    return render_template('results.html', model_name=model_name, history=history_data)

if __name__ == '__main__':
    app.run(debug=True)
