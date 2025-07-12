# app.py - Enhanced Flask App
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory
import os
import threading
import json
import datetime
from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback
import logging
import openpyxl
from datetime import datetime
from collections import defaultdict
import contextlib
import threading
import uuid
# Import your evaluation functions
from evaluate_ml_supervised_mlflow import run_ml_evaluation, get_ml_progress, get_ml_results, clear_ml_progress, update_progress, export_results_to_json, list_available_results
from evaluate_llm import get_history, run_evaluation, _save_enhanced_results 
from custom_evaluate_ml import (
    run_custom_ml_evaluation_task,
    clear_custom_ml_results,
    get_custom_ml_status,
    export_custom_ml_excel,
    export_custom_ml_csv,
    custom_evaluation_results,           # <-- add this
    custom_evaluation_progress           # <-- add this
)
import numpy as np
from history_ml_supervised import load_ml_history

## ML Imports #
from routes.ml.supervised.tool.mlflow.ml_supervised_tool_mlflow import ml_s_t_mlflow_bp
from routes.ml.supervised.custom.ml_supervised_custom import ml_s_c_bp
from routes.ml.supervised.history.ml_supervised_history import ml_s_h_bp

## LLM Imports #
from routes.llm.tool.bigbench.llm_tool_bigbench import llm_t_bb_bp
from routes.llm.custom.llm_custom import llm_custom_bp

## DL Imports #
from routes.dl.nlp.tool.dl_nlp_tool import dl_nlp_t_bp

## Genai Imports #
from routes.genai.genai_tool import genai_t_bp
from routes.genai.genai_custom import genai_c_bp
from routes.genai.genai_history import genai_h_bp

# Add logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




app = Flask(__name__)
## ML Blueprints ##
app.register_blueprint(ml_s_t_mlflow_bp)
app.register_blueprint(ml_s_c_bp)
app.register_blueprint(ml_s_h_bp)

## LLM Blueprints ##
app.register_blueprint(llm_t_bb_bp)
app.register_blueprint(llm_custom_bp)

## DL Blueprints ##
app.register_blueprint(dl_nlp_t_bp)

## Genai Blueprints ##
app.register_blueprint(genai_t_bp)
app.register_blueprint(genai_c_bp)
app.register_blueprint(genai_h_bp)

app.secret_key = 'your-secret-key-here'
model_base_path = "models"
processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Add this global variable with other globals


categories = {
    "llm": [],
    "genai": [],
    "dl": [],
    "ml": []
}

# Create folders
def create_model_directories():
    """Create model directories with subcategory support."""
    categories = {
        "llm": [],
        "genai": [],
        "dl": ["computer_vision", "nlp", "others"],
        "ml": ["supervised", "unsupervised", "reinforcement", "others"]
    }

    # Create folders
    for category, subcategories in categories.items():
        category_path = os.path.join(model_base_path, category)
        os.makedirs(category_path, exist_ok=True)
        
        # If subcategories exist, create them
        if isinstance(subcategories, list) and subcategories:
            for subcategory in subcategories:
                subcategory_path = os.path.join(category_path, subcategory)
                os.makedirs(subcategory_path, exist_ok=True)
                print(f"Created: {subcategory_path}")
create_model_directories()

js_function = '''
<script>
function setBenchmark(type, index) {
    const dropdown = document.getElementById('benchmark-' + type + '-' + index);
    const hiddenInput = document.getElementById('benchmark-input-' + type + '-' + index);
    if (dropdown && hiddenInput) {
        hiddenInput.value = dropdown.value;
    }
}
</script>
'''
LLM_BENCHMARKS = [
    "BIG-Bench", "MMLU", "HellaSwag", "PIQA", "SocialIQA", "BooIQ", "WinoGrande",
    "CommonsenseQA", "OpenBookQA", "ARC-e", "ARC-c", "TriviaQA", "Natural Questions",
    "HumanEval", "MBPP", "GSM8K", "MATH", "AGIEval"
]

ML_BENCHMARKS = [
    "MLflow", "scikit-learn", "Yellowbrick", "Evidently AI", "Weights & Biases", "AutoML (TPOT, H2O)"
]

DL_BENCHMARKS = []  # Add DL benchmarks if needed

GENAI_BENCHMARKS = LLM_BENCHMARKS


@app.route('/')
def index():
    """Updated index route to handle subcategories for ML and DL models."""
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
            if display_name == "ML Models":
                # Handle ML Models with subcategories
                model_data[display_name] = {
                    "supervised": [],
                    "unsupervised": [],
                    "reinforcement": [],
                    "others": []
                }
                # Check for subcategory folders or categorize models
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        # Check if it's a subcategory folder
                        if item in ["supervised", "unsupervised", "reinforcement", "others"]:
                            models = [model for model in os.listdir(item_path) 
                                    if os.path.isdir(os.path.join(item_path, model))]
                            model_data[display_name][item] = models
                        else:
                            # Default to 'others' if not categorized
                            model_data[display_name]["others"].append(item)
                            
            elif display_name == "DL Models":
                # Handle DL Models with subcategories
                model_data[display_name] = {
                    "computer_vision": [],
                    "nlp": [],
                    "others": []
                }
                # Check for subcategory folders or categorize models
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        # Check if it's a subcategory folder
                        if item in ["computer_vision", "nlp", "others"]:
                            models = [model for model in os.listdir(item_path) 
                                    if os.path.isdir(os.path.join(item_path, model))]
                            model_data[display_name][item] = models
                        else:
                            # Default to 'others' if not categorized
                            model_data[display_name]["others"].append(item)
            else:
                # Handle LLMs and GenAI as before (flat structure)
                models = [model for model in os.listdir(path) 
                         if os.path.isdir(os.path.join(path, model))]
                model_data[display_name] = models
        else:
            if display_name in ["ML Models", "DL Models"]:
                model_data[display_name] = {
                    "supervised": [] if display_name == "ML Models" else [],
                    "unsupervised": [] if display_name == "ML Models" else [],
                    "reinforcement": [] if display_name == "ML Models" else [],
                    "computer_vision": [] if display_name == "DL Models" else [],
                    "nlp": [] if display_name == "DL Models" else [],
                    "others": []
                }
            else:
                model_data[display_name] = []

    return render_template("index.html", model_data=model_data,    llm_benchmarks=LLM_BENCHMARKS,
        ml_benchmarks=ML_BENCHMARKS,
        dl_benchmarks=DL_BENCHMARKS,
        genai_benchmarks=GENAI_BENCHMARKS)




plot_lock = threading.Lock()

@contextlib.contextmanager
def thread_safe_plotting():
    """Context manager for thread-safe matplotlib operations."""
    with plot_lock:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        try:
            yield plt
        finally:
            plt.close('all')


# def convert_numpy_types(obj):
#     """Recursively convert numpy types to native Python types for JSON serialization."""
#     import numpy as np
#     if isinstance(obj, dict):
#         return {k: convert_numpy_types(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy_types(v) for v in obj]
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.bool_):
#         return bool(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj
 

# @app.route('/api/ml_results/<model_name>')
# def get_results(model_name):
#     """API endpoint to get evaluation results"""
#     results = get_ml_results(model_name)
#     if not results:
#         return jsonify({"error": "No results found"}), 404
#     return jsonify(results)




# @app.route('/plots/<model_name>/<filename>')
# def serve_plot(model_name, filename):
#     """Serve plot images from static/plots/{model_name}/."""
#     return send_from_directory(f'static/plots/{model_name}', filename)








# @app.route('/custom_ml/<model_name>/<subcategory>')
# def custom_ml(model_name, subcategory):
#     """Custom evaluation page for ML models."""
#     # Get existing uploaded files for this model
#     model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
#     os.makedirs(model_upload_dir, exist_ok=True)
#     uploaded_files = []
#     if os.path.exists(model_upload_dir):
#         uploaded_files = [f for f in os.listdir(model_upload_dir) 
#                          if os.path.isfile(os.path.join(model_upload_dir, f))]
#     print(f"Uploaded files for {model_name}: {uploaded_files}")
#     # Get evaluation results if available
#     results = custom_evaluation_results.get(f"{model_name}_ml", {})
    
#     return render_template('custom_ml.html', 
#                          model_name=model_name, 
#                          subcategory=subcategory,
#                          uploaded_files=uploaded_files,
#                          evaluation_results=results)


# Add this route to your Flask app to properly pass results to template


if __name__ == '__main__':
    app.run(debug=True)