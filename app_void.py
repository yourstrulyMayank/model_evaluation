from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import threading
from evaluate_llm import run_evaluation, get_history
from ensure_models import main as ensure_models

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
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

def run_evaluation_in_background(model_name, model_path, eval_fn):
    processing_status[model_name] = "processing"

    def background_task():
        try:
            result = eval_fn(model_path)
            # Optionally save result
            processing_status[model_name] = "complete"
        except Exception as e:
            print(f"Error during evaluation: {e}")
            processing_status[model_name] = "error"

    threading.Thread(target=background_task).start()


# @app.route('/evaluate_model/<category>/<model_name>')
# def evaluate(category, model_name):
#     model_path = None
#     categories = {
#         "LLMs": "llm",
#         "Other GenAI Models": "genai",
#         "DL Models": "dl",
#         "ML Models": "ml"
#     }
#     potential_path = os.path.join(model_base_path, categories[category], model_name)
#     if os.path.exists(potential_path):
#         model_path = potential_path

#     if not model_path:
#         return f"Model '{model_name}' not found.", 404

#     # 🔁 Reset this model's status
#     processing_status[model_name] = "processing"

#     def background_task():
#         try:
#             result = run_evaluation(model_path)
#             # save_result(model_name, result)
#             processing_status[model_name] = "complete"
#         except Exception as e:
#             print(f"Error during evaluation: {e}")
#             processing_status[model_name] = "error"

#     threading.Thread(target=background_task).start()
#     return render_template('loading.html', model_name=model_name)
@app.route('/evaluate_model/<category>/<model_name>')
def evaluate(category, model_name):
    categories = {
        "LLMs": "llm",
        "Other GenAI Models": "genai",
        "DL Models": "dl",
        "ML Models": "ml"
    }

    category_folder = categories.get(category)
    if not category_folder:
        return "Unknown category", 400

    model_path = os.path.join(model_base_path, category_folder, model_name)
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    # Example: For non-LLM models use a standard evaluation function
    run_evaluation_in_background(model_name, model_path, run_evaluation)

    return render_template('loading.html', model_name=model_name)


@app.route('/custom_llm/<model_name>')
def custom_llm(model_name):
    return render_template('custom_llm.html', model_name=model_name)

@app.route('/evaluate_llm/<model_name>', methods=['POST','GET'])
def evaluate_llm(model_name):
    benchmark = request.form.get('benchmark', 'BIG-Bench')
    print(f"Evaluating {model_name} on benchmark: {benchmark}")
    if benchmark != "BIG-Bench":
        flash(f"Evaluation for {benchmark} is not yet supported.")
        return redirect(url_for('index'))

    model_path = os.path.join(model_base_path, "llm", model_name)
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    # Use the shared background evaluation logic with the general function
    run_evaluation_in_background(model_name, model_path, run_evaluation)

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
