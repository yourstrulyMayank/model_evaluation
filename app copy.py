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
from werkzeug.utils import secure_filename
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

# Add logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# PDF generation
try:
    from weasyprint import HTML, CSS
    from jinja2 import Template
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️ PDF generation not available. Install: pip install weasyprint")
    PDF_AVAILABLE = False

app = Flask(__name__)
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

@app.route('/evaluate_ml/<model_name>/<subcategory>', methods=['POST'])
def evaluate_ml(model_name, subcategory):
    """Evaluate ML models with subcategory support."""
    benchmark = request.form.get('benchmark', 'MLflow')
    
    print(f"Evaluating ML model {model_name} (subcategory: {subcategory}) on benchmark: {benchmark}")   
    
    try:        
        # Correct path structure for your model
        model_path = os.path.join('models', 'ml', subcategory, model_name, 'model')
        dataset_path = os.path.join('models', 'ml', subcategory, model_name, 'dataset')        
        test_csv_path = os.path.join(dataset_path, 'test.csv')  # Look for test.csv in model directory
        model_file_path = os.path.join(model_path, 'model.pkl')  # Look for model.pkl
                
        
        # Validate paths exist
        if not os.path.exists(model_path):
            flash(f"Model directory not found: {model_path}")
            return redirect(url_for('index'))
        
        
        if not os.path.exists(model_file_path):
            flash(f"Model file not found: {model_file_path}")
            return redirect(url_for('index'))
        else:
            print(f'Found model: {model_file_path}')

        
        if not os.path.exists(test_csv_path):
            flash(f"Test CSV file not found: {test_csv_path}")
            return redirect(url_for('index'))
        else:
            print(f'Found test csv: {test_csv_path}')

        from concurrent.futures import ThreadPoolExecutor
        import threading
        evaluation_lock = threading.Lock()
        # Start evaluation in background thread
        # Start evaluation with proper resource management
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_ml_evaluation_wrapper,
                model_name, model_file_path, test_csv_path, benchmark
            )
        
        flash(f"ML model evaluation started for {model_name}. Check progress on the evaluation page.")
        
        return render_template('evaluate_ml_supervised_mlflow.html', 
                             model_name=model_name, 
                             subcategory=subcategory,
                             benchmark=benchmark)
        
    except Exception as e:
        print(f"Error starting ML evaluation: {e}")
        flash(f"Error starting evaluation: {str(e)}")
        return redirect(url_for('index'))
    #     evaluation_thread = threading.Thread(
    #         target=run_ml_evaluation_wrapper,
    #         args=(model_name, model_file_path, test_csv_path, benchmark),
    #         daemon=True
    #     )
    #     evaluation_thread.start()
        
    #     flash(f"ML model evaluation started for {model_name}. Check progress on the evaluation page.")
        
    #     return render_template('evaluate_ml_supervised_mlflow.html', 
    #                          model_name=model_name, 
    #                          subcategory=subcategory,
    #                          benchmark=benchmark)
        
    # except Exception as e:
    #     print(f"Error starting ML evaluation: {e}")
    #     flash(f"Error starting evaluation: {str(e)}")
    #     return redirect(url_for('index'))

def run_ml_evaluation_wrapper(model_name, model_file_path, test_csv_path, benchmark):
    """Wrapper function to run ML evaluation in background thread."""
    try:
        # Ensure matplotlib uses non-GUI backend in thread
        import matplotlib
        matplotlib.use('Agg')
        
        print(f"Starting ML evaluation for {model_name}")
        print(f"Model file: {model_file_path}")
        print(f"Test data: {test_csv_path}")
        
        # Run the evaluation
        results = run_ml_evaluation(model_name, model_file_path, test_csv_path)
        
        print(f"ML evaluation completed for {model_name}")
        
        # Clean up matplotlib resources
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except Exception as e:
        print(f"Error in ML evaluation thread: {e}")
        update_progress(model_name, f"Error: {str(e)}", 0)
        # Clean up on error
        import matplotlib.pyplot as plt
        plt.close('all')

import contextlib
import threading

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

def extract_test_csv_if_needed(dataset_path):
    """Extract test.csv from dataset.zip if it doesn't exist."""
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    dataset_zip_path = os.path.join(dataset_path, 'dataset.zip')
    
    # If test.csv already exists, return its path
    if os.path.exists(test_csv_path):
        return test_csv_path
    
    # If dataset.zip exists, try to extract test.csv
    if os.path.exists(dataset_zip_path):
        try:
            import zipfile
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                # Look for test.csv in the zip
                if 'test.csv' in zip_ref.namelist():
                    zip_ref.extract('test.csv', dataset_path)
                    return test_csv_path
                else:
                    # Look for any CSV file that might be the test data
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        # Extract the first CSV file and rename it to test.csv
                        zip_ref.extract(csv_files[0], dataset_path)
                        extracted_path = os.path.join(dataset_path, csv_files[0])
                        os.rename(extracted_path, test_csv_path)
                        return test_csv_path
        except Exception as e:
            print(f"Error extracting from dataset.zip: {e}")
    
    return None

# API endpoints for progress tracking and results
@app.route('/api/ml_progress/<model_name>')
def get_ml_evaluation_progress(model_name):
    """Get current progress of ML model evaluation."""
    progress = get_ml_progress(model_name)
    return jsonify(progress)




def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
@app.route('/api/ml_results/<model_name>')
def get_ml_evaluation_results(model_name):
    """Get evaluation results for a specific model."""
    results = get_ml_results(model_name)
    results = convert_numpy_types(results)
    return jsonify(results)

@app.route('/history_ml/<category>/<model_name>')
def history_ml(category, model_name):
    """
    Display ML evaluation history for a specific model and category.
    """
    try:
        history_data, benchmark_list, benchmark_averages, benchmark_stats = load_ml_history(model_name, category)
        return render_template(
            'history_ml.html',
            model_name=model_name,
            category=category,
            history_data=history_data,
            benchmark_list=benchmark_list,
            benchmark_averages=benchmark_averages,
            benchmark_stats=benchmark_stats
        )
    except Exception as e:
        print(f"Error loading ML history: {e}")
        return render_template(
            'history_ml.html',
            model_name=model_name,
            category=category,
            history_data=[],
            benchmark_list=[],
            benchmark_averages={},
            benchmark_stats={'benchmarks_tested': 0, 'overall_average': 0, 'best_score': 0}
        )
    
@app.route('/api/download_report/<model_name>')
def download_ml_report(model_name):
    """Download comprehensive evaluation report."""
    try:
        results = get_ml_results(model_name)
        if not results or 'error' in results:
            flash("No results available for download")
            return redirect(url_for('index'))
        
        # Generate report content
        report_content = generate_ml_report(results)
        
        # Create response
        response = make_response(report_content)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=ml_evaluation_report_{model_name}.json'
        
        return response
        
    except Exception as e:
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('index'))

def generate_ml_report(results):
    """Generate comprehensive ML evaluation report."""
    report = {
        'model_name': results.get('model_name'),
        'evaluation_timestamp': results.get('timestamp'),
        'problem_type': results.get('problem_type'),
        'dataset_info': results.get('dataset_info'),
        'performance_metrics': results.get('metrics'),
        'mlflow_run_id': results.get('mlflow_run_id'),
        'summary': {
            'evaluation_completed': True,
            'total_samples': results.get('dataset_info', {}).get('n_samples', 0),
            'total_features': results.get('dataset_info', {}).get('n_features', 0)
        }
    }
    
    # Add key performance indicators
    metrics = results.get('metrics', {})
    if results.get('problem_type') == 'classification':
        report['summary']['key_metrics'] = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0)
        }
        if 'roc_auc' in metrics:
            report['summary']['key_metrics']['roc_auc'] = metrics['roc_auc']
    else:
        report['summary']['key_metrics'] = {
            'r2_score': metrics.get('r2_score', 0),
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0),
            'mape': metrics.get('mean_absolute_percentage_error', 0)
        }
    
    # Add cross-validation summary
    if 'cv_scores' in metrics and metrics['cv_scores']:
        report['cross_validation'] = {
            'mean_score': metrics['cv_scores']['mean'],
            'std_score': metrics['cv_scores']['std'],
            'individual_scores': metrics['cv_scores']['scores']
        }
    
    return json.dumps(report, indent=2, default=str)

# Additional helper route for clearing evaluation data
@app.route('/api/clear_ml_evaluation/<model_name>', methods=['POST'])
def clear_ml_evaluation_data(model_name):
    """Clear evaluation progress and results for a specific model."""
    try:
        clear_ml_progress(model_name)
        return jsonify({'status': 'success', 'message': f'Cleared evaluation data for {model_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    


@app.route('/api/ml_progress/<model_name>')
def get_progress(model_name):
    """API endpoint to get evaluation progress"""
    progress = get_ml_progress(model_name)
    return jsonify(progress)

@app.route('/api/ml_results/<model_name>')
def get_results(model_name):
    """API endpoint to get evaluation results"""
    results = get_ml_results(model_name)
    if not results:
        return jsonify({"error": "No results found"}), 404
    return jsonify(results)

@app.route('/api/start_evaluation/<model_name>', methods=['POST'])
def start_evaluation_api(model_name):
    """API endpoint to start model evaluation with proper resource management."""
    print('------------------------------------')
    print(f'Starting the evaluation api for model: {model_name}')
    try:
        # Get model and dataset paths from request or session
        model_path = request.json.get('model_path')
        dataset_path = request.json.get('dataset_path')
        print(f"Model path: {model_path}, Dataset path: {dataset_path}")
        
        if not model_path or not dataset_path:
            return jsonify({"error": "Model path and dataset path required"}), 400
        
        # Use ThreadPoolExecutor for better resource management
        from concurrent.futures import ThreadPoolExecutor
        
        def run_evaluation_task():
            try:
                run_ml_evaluation(model_name, model_path, dataset_path)
            except Exception as e:
                print(f"Evaluation task failed: {e}")
                update_progress(model_name, f"Error: {str(e)}", 0)
        
        # Submit task to thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run_evaluation_task)
        
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_results/<model_name>/<format>')
def export_results_api(model_name, format):
    """API endpoint to export results"""
    try:
        results = get_ml_results(model_name)
        if not results:
            return jsonify({"error": "No results found"}), 404
        
        if format == 'json':
            output_path = export_results_to_json(model_name)
            return send_file(output_path, as_attachment=True)
        elif format == 'csv':
            # Create CSV export
            output_dir = f"results/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"ml_evaluation_{model_name}.csv")
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(csv_path, index=False)
            return send_file(csv_path, as_attachment=True)
        else:
            return jsonify({"error": "Invalid format"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plots/<model_name>/<filename>')
def serve_plot(model_name, filename):
    """Serve plot images from static/plots/{model_name}/."""
    return send_from_directory(f'static/plots/{model_name}', filename)


@app.route('/api/available_models')
def get_available_models():
    """API endpoint to get list of available models"""
    models = list_available_results()
    return jsonify(models)


@app.route('/evaluate_dl/<model_name>/<subcategory>', methods=['POST'])
def evaluate_dl(model_name, subcategory):
    """Evaluate DL models with subcategory support."""
    benchmark = request.form.get('benchmark', '')
    
    print(f"Evaluating DL model {model_name} (subcategory: {subcategory}) on benchmark: {benchmark}")
    
    # For now, show not supported message
    flash(f"DL model evaluation is not yet implemented.")
    return redirect(url_for('index'))

@app.route('/evaluate_genai/<model_name>', methods=['POST'])
def evaluate_genai(model_name):
    """Evaluate GenAI models."""
    benchmark = request.form.get('benchmark', 'BIG-Bench')
    
    print(f"Evaluating GenAI model {model_name} on benchmark: {benchmark}")
    
    if benchmark != "BIG-Bench":
        flash(f"Evaluation for {benchmark} is not yet supported.")
        return redirect(url_for('index'))

    model_path = os.path.join(model_base_path, "genai", model_name)
    if not os.path.exists(model_path):
        flash(f"Model '{model_name}' not found.")
        return redirect(url_for('index'))

    eval_params = {
        'num_examples': 25,
        'max_tokens': 128,
        'full_benchmark': False
    }
    
    run_evaluation_in_background(model_name, model_path, eval_params)
    return render_template('loading.html', model_name=model_name)

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

@app.route('/upload_ml_files/<model_name>', methods=['POST'])
def upload_ml_files(model_name):
    """Upload model and test files for custom ML evaluation."""
    try:
        model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
        os.makedirs(model_upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        # Handle model file upload
        if 'model_file' in request.files:
            model_file = request.files['model_file']
            if model_file.filename != '':
                filename = secure_filename(model_file.filename)
                model_path = os.path.join(model_upload_dir, filename)
                model_file.save(model_path)
                uploaded_files.append(f"Model: {filename}")
        
        # Handle test data upload
        if 'test_file' in request.files:
            test_file = request.files['test_file']
            if test_file.filename != '':
                filename = secure_filename(test_file.filename)
                test_path = os.path.join(model_upload_dir, filename)
                test_file.save(test_path)
                uploaded_files.append(f"Test Data: {filename}")
        
        if uploaded_files:
            return jsonify({
                'status': 'success', 
                'message': f'Uploaded: {", ".join(uploaded_files)}'
            })
        else:
            return jsonify({'status': 'error', 'message': 'No files were uploaded'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add this route to your Flask app to properly pass results to template
@app.route('/custom_ml/<model_name>/<subcategory>')
def custom_ml(model_name, subcategory):
    """Custom ML evaluation page with results display"""
    try:
        # Get uploaded files
        upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
        uploaded_files = []
        if os.path.exists(upload_dir):
            uploaded_files = os.listdir(upload_dir)
        
        # Get evaluation results - THIS IS THE KEY FIX
        evaluation_results = None
        results_key = f"{model_name}_ml"
        if results_key in custom_evaluation_results:
            evaluation_results = custom_evaluation_results[results_key]
            logger.info(f"Found evaluation results for {model_name}: {evaluation_results.keys()}")
        
        return render_template('custom_ml.html', 
                             model_name=model_name,
                             subcategory=subcategory,
                             uploaded_files=uploaded_files,
                             evaluation_results=evaluation_results)  # Pass results here
    
    except Exception as e:
        logger.error(f"Error in custom_ml route: {str(e)}")
        flash(f"Error loading page: {str(e)}")
        return redirect(url_for('index'))
    
@app.route('/run_custom_ml_evaluation/<model_name>', methods=['POST', 'GET'])
def run_custom_ml_evaluation(model_name):
    """Run custom ML evaluation with uploaded files and optional steps file."""
    try:
        upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'No files uploaded for evaluation'}), 400
        
        # Find model, test, and steps files
        files = os.listdir(upload_dir)
        model_file = None
        test_file = None
        steps_file = None
        print(files)
        for file in files:
            if file.endswith(('.pkl', '.joblib', '.model')):
                model_file = os.path.join(upload_dir, file)
            elif file.endswith(('.xlsx', '.xls', '.csv')):
                test_file = os.path.join(upload_dir, file)
            elif file.startswith('steps.') and file.split('.')[-1] in ('py', 'json', 'txt'):
                steps_file = os.path.join(upload_dir, file)
        
        if not model_file or not test_file:
            return jsonify({'error': 'Both model file (.pkl) and test file (.xlsx/.csv) are required'}), 400
        
        # Set initial status
        processing_status[f"{model_name}_ml_custom"] = "processing"
        
        def background_evaluation():
            """Background evaluation with comprehensive error handling"""
            try:
                logger.info(f"Starting ML evaluation for model: {model_name}")
                logger.info(f"Model file: {model_file}")
                logger.info(f"Test file: {test_file}")
                logger.info(f"Steps file: {steps_file}")
                
                # Run the evaluation
                result = run_custom_ml_evaluation_task(model_name, model_file, test_file, steps_file)
                
                logger.info(f"Evaluation completed for {model_name}")
                if result.get('error'):
                    logger.error(f"Evaluation error: {result['error']}")
                    processing_status[f"{model_name}_ml_custom"] = "error"
                else:
                    logger.info(f"Evaluation successful for {model_name}")
                    processing_status[f"{model_name}_ml_custom"] = "complete"
                    
            except Exception as e:
                error_msg = f"Background evaluation failed: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                # Store error in results
                custom_evaluation_results[f"{model_name}_ml"] = {
                    'error': error_msg,
                    'traceback': traceback.format_exc(),
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
                processing_status[f"{model_name}_ml_custom"] = "error"
        
        # Start background thread - removed daemon=True to prevent immediate shutdown
        thread = threading.Thread(target=background_evaluation)
        thread.start()
        
        logger.info(f"Background thread started for {model_name}")
        return jsonify({'status': 'started', 'message': 'ML evaluation started successfully'})
        
    except Exception as e:
        error_msg = f'Error starting evaluation: {str(e)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500


@app.route('/check_custom_ml_status/<model_name>')
def check_custom_ml_status(model_name):
    """Check status of custom ML evaluation - FIXED VERSION"""
    try:
        status = processing_status.get(f"{model_name}_ml_custom", "not_started")
        results = custom_evaluation_results.get(f"{model_name}_ml", {})
        progress = custom_evaluation_progress.get(model_name, {})

        
        
        if status == "complete" and results and not results.get('error'):
            return jsonify(convert_numpy_types({
                'status': 'complete',
                'results': results,
                'progress': progress
            }))
        elif status == "error" or results.get('error'):
            return jsonify(convert_numpy_types({
                'status': 'error',
                'results': results,
                'progress': progress
            }))
        else:
            return jsonify(convert_numpy_types({
                'status': 'processing',
                'progress': progress
            }))
            
    except Exception as e:
        logger.error(f"Error checking status for {model_name}: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500



@app.route('/download_custom_ml_report/<model_name>')
def download_custom_ml_report(model_name):
    """Download custom ML evaluation report as Excel."""
    try:
        excel_path = os.path.join('uploads', model_name, f"{model_name}_custom_ml_report.xlsx")
        print(excel_path)
        if not os.path.exists(excel_path):
            print("file not found")
            flash("Excel report not found. Please re-run evaluation.")
            return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))
        return send_file(excel_path, as_attachment=True, download_name=f"{model_name}_custom_ml_report.xlsx")
    except Exception as e:
        logger.error(f"Error downloading report for {model_name}: {str(e)}")
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))



@app.route('/download_custom_ml_testcases/<model_name>')
def download_custom_ml_testcases(model_name):
    """Download test cases with predictions as CSV."""
    try:
        csv_path = os.path.join('uploads', model_name, f"{model_name}_test_results.csv")
        print(csv_path)
        if not os.path.exists(csv_path):
            print("file not found")
            flash("CSV results not found. Please re-run evaluation.")
            return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))
        return send_file(csv_path, as_attachment=True, download_name=f"{model_name}_test_results.csv")
    except Exception as e:
        logger.error(f"Error downloading test cases for {model_name}: {str(e)}")
        flash(f"Error generating CSV: {str(e)}")
        return redirect(url_for('custom_ml', model_name=model_name, subcategory='supervised'))


@app.route('/clear_custom_ml_results/<model_name>', methods=['POST'])
def clear_custom_ml_results_route(model_name):
    """Clear custom evaluation results for a model."""
    try:
        clear_custom_ml_results(model_name)
        logger.info(f"Results cleared for {model_name}")
        return jsonify({'status': 'success', 'message': 'Results cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing results for {model_name}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/start_evaluation/<model_name>', methods=['POST'])
def api_start_evaluation(model_name):
    """
    API endpoint to start or restart ML evaluation for a model.
    This will clear previous results and start a new evaluation.
    """
    try:
        # You may want to get subcategory and benchmark from request if needed
        subcategory = request.json.get('subcategory', 'supervised')
        benchmark = request.json.get('benchmark', 'MLflow')

        # Paths for model and test data
        
        model_path = os.path.join('models', 'ml', subcategory, model_name, 'model')
        dataset_path = os.path.join('models', 'ml', subcategory, model_name, 'dataset')        
        test_csv_path = os.path.join(dataset_path, 'test.csv')  # Look for test.csv in model directory
        model_file_path = os.path.join(model_path, 'model.pkl')  # Look for model.pkl
        # Validate paths
        if not os.path.exists(model_path) or not os.path.exists(test_csv_path):
            return jsonify({'error': 'Model or test data not found'}), 400

        # Clear previous progress/results
        clear_ml_progress(model_name)

        # Start evaluation in background
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run_ml_evaluation, model_name, model_file_path, test_csv_path)

        return jsonify({'status': 'started', 'message': 'Evaluation started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/custom_dl/<model_name>/<subcategory>')
def custom_dl(model_name, subcategory):
    """Custom evaluation page for DL models."""
    # Get existing uploaded files for this model
    model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
    uploaded_files = []
    if os.path.exists(model_upload_dir):
        uploaded_files = [f for f in os.listdir(model_upload_dir) 
                         if os.path.isfile(os.path.join(model_upload_dir, f))]
    
    # For now, redirect to a generic custom evaluation page
    flash(f"Custom evaluation for DL models ({subcategory}) is not yet implemented.")
    return redirect(url_for('index'))

@app.route('/custom_genai/<model_name>')
def custom_genai(model_name):
    """Custom evaluation page for GenAI models."""
    # Get existing uploaded files for this model
    model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
    uploaded_files = []
    if os.path.exists(model_upload_dir):
        uploaded_files = [f for f in os.listdir(model_upload_dir) 
                         if os.path.isfile(os.path.join(model_upload_dir, f))]
    
    # Get evaluation results if available
    results = custom_evaluation_results.get(model_name, {})
    
    return render_template('custom_llm.html', 
                         model_name=model_name, 
                         uploaded_files=uploaded_files,
                         evaluation_results=results,
                         model_type='genai')

def run_evaluation_in_background(model_name, model_path, eval_params):
    """Enhanced background evaluation with progress tracking."""
    processing_status[model_name] = "processing"
    evaluation_progress[model_name] = {
        "current_task": "",
        "completed_tasks": 0,
        "total_tasks": 0,
        "progress_percent": 0
    }

    def background_task():
        try:
            # Update progress
            evaluation_progress[model_name]["current_task"] = "Initializing..."
            
            # Run enhanced evaluation
            result = run_evaluation(
                model_name=model_path,
                num_examples=eval_params.get('num_examples', 25),
                max_new_tokens=eval_params.get('max_tokens', 128),
                use_full_bigbench=eval_params.get('full_benchmark', False)
            )
            
            processing_status[model_name] = "complete"
            evaluation_progress[model_name]["progress_percent"] = 100
            evaluation_progress[model_name]["current_task"] = "Completed"
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            processing_status[model_name] = "error"
            evaluation_progress[model_name]["current_task"] = f"Error: {str(e)}"

    threading.Thread(target=background_task).start()



@app.route('/download_custom_excel/<model_name>')
def download_custom_excel(model_name):
    """Download custom evaluation results as Excel file."""
    try:
        # Get evaluation results
        results = custom_evaluation_results.get(model_name, {})
        
        if not results or results.get('error'):
            flash("No evaluation results found for this model.")
            return redirect(url_for('custom_llm', model_name=model_name))
        
        # Import pandas for Excel creation
        import pandas as pd
        from io import BytesIO
        
        # Create Excel file in memory
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main results sheet
            if 'ground_truth_comparison' in results:
                comparison_data = results['ground_truth_comparison']
                
                # Create DataFrame with original format plus model outputs
                df_results = pd.DataFrame([
                    {
                        'Prompt': item['prompt'],
                        'Ground_Truth_Actual': item['actual'],
                        'Model_Extracted': item['extracted'],
                        'Confidence_Score': item['score'],
                        'Test_Grade': item['grade'],
                        'Status': 'Pass' if item['grade'] == '✅ Pass' else 'Fail' if item['grade'] == '❌ Fail' else 'Intermittent'
                    }
                    for item in comparison_data
                ])
                
                df_results.to_excel(writer, sheet_name='Evaluation_Results', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Model Name',
                    'Evaluation Date',
                    'Total Tests',
                    'Tests Passed',
                    'Tests Failed', 
                    'Intermittent Tests',
                    'Overall Score (%)',
                    'Success Rate (%)',
                    'Average Score',
                    'Highest Score',
                    'Lowest Score',
                    'Files Processed'
                ],
                'Value': [
                    results.get('model_name', model_name),
                    results.get('timestamp', 'N/A'),
                    results.get('total_tests', 0),
                    results.get('pass_count', 0),
                    results.get('fail_count', 0),
                    results.get('intermittent_count', 0),
                    round(results.get('overall_score', 0), 2),
                    round(results.get('success_rate', 0), 2),
                    round(results.get('average_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('highest_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('lowest_score', 0), 2),
                    results.get('files_processed', 0)
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # File info sheet
            if 'file_info' in results:
                file_info = results['file_info']
                df_files = pd.DataFrame([
                    {'File Type': 'Image File', 'File Name': file_info.get('image_file', 'N/A')},
                    {'File Type': 'Transaction File', 'File Name': file_info.get('transaction_file', 'N/A')},
                    {'File Type': 'Ground Truth File', 'File Name': file_info.get('ground_truth_file', 'N/A')}
                ])
                df_files.to_excel(writer, sheet_name='File_Info', index=False)
        
        output.seek(0)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_custom_evaluation_results.xlsx"'
        
        return response
        
    except Exception as e:
        print(f"Error generating Excel file: {e}")
        flash(f"Error generating Excel file: {str(e)}")
        return redirect(url_for('custom_llm', model_name=model_name))
        
@app.route('/clear_custom_results/<model_name>', methods=['POST'])
def clear_custom_results(model_name):
    """Clear custom evaluation results for a model."""
    try:
        # Clear from global storage
        if model_name in custom_evaluation_results:
            del custom_evaluation_results[model_name]
        
        # Clear processing status
        status_key = f"{model_name}_custom"
        if status_key in processing_status:
            del processing_status[status_key]
        
        # Clear progress tracking from custom_evaluate_llm
        from custom_evaluate_llm import clear_progress
        clear_progress(model_name)
        
        return jsonify({'status': 'success', 'message': 'Results cleared successfully'})
        
    except Exception as e:
        print(f"Error clearing results for {model_name}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

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

    # Default evaluation parameters
    eval_params = {
        'num_examples': 25,
        'max_tokens': 128,
        'full_benchmark': False
    }
    
    run_evaluation_in_background(model_name, model_path, eval_params)
    return render_template('loading.html', model_name=model_name)

@app.route('/custom_llm/<model_name>')
def custom_llm(model_name):
    # Get existing uploaded files for this model
    model_upload_dir = os.path.join(UPLOAD_FOLDER, model_name)
    uploaded_files = []
    if os.path.exists(model_upload_dir):
        uploaded_files = [f for f in os.listdir(model_upload_dir) if os.path.isfile(os.path.join(model_upload_dir, f))]
    
    # Get evaluation results if available
    results = custom_evaluation_results.get(model_name, {})
    
    return render_template('custom_llm.html', 
                         model_name=model_name, 
                         uploaded_files=uploaded_files,
                         evaluation_results=results)


@app.route('/run_custom_evaluation/<model_name>', methods=['POST'])
def run_custom_evaluation_route(model_name):
    try:
        # Import custom evaluator
        from custom_evaluate_llm import run_custom_evaluation
        
        # Get model path
        model_path = os.path.join(model_base_path, "llm", model_name)
        upload_dir = os.path.join(UPLOAD_FOLDER)
        
        if not os.path.exists(upload_dir):
            return jsonify({'error': 'No files uploaded for evaluation'}), 400
        
        # Set processing status BEFORE starting background task
        processing_status[f"{model_name}_custom"] = "processing"
        
        def background_evaluation():
            try:
                print(f"Starting custom evaluation for {model_name}")
                # Run custom evaluation
                results = run_custom_evaluation(model_name, model_path, upload_dir)
                print(f"Custom evaluation completed for {model_name}")
                
                custom_evaluation_results[model_name] = results
                processing_status[f"{model_name}_custom"] = "complete"
                
            except Exception as e:
                print(f"Custom evaluation error for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
                processing_status[f"{model_name}_custom"] = "error"
                custom_evaluation_results[model_name] = {"error": str(e)}
        
        # Run in background
        threading.Thread(target=background_evaluation, daemon=True).start()
        
        # Return success response
        return jsonify({'status': 'started', 'message': 'Evaluation started successfully'})
        
    except Exception as e:
        print(f"Error starting evaluation: {e}")
        processing_status[f"{model_name}_custom"] = "error"
        return jsonify({'error': f'Error starting evaluation: {str(e)}'}), 500







@app.route('/check_custom_status/<model_name>')
def check_custom_status(model_name):
    status_key = f"{model_name}_custom"
    status = processing_status.get(status_key, "not_started")
    results = custom_evaluation_results.get(model_name, {})
    from custom_evaluate_llm import get_progress
    # Get progress information from custom_evaluate_llm
    progress_info = get_progress(model_name)
    
    print(f"Status check for {model_name}: {status}, Progress: {progress_info}")  # Debug log
    
    response_data = {
        "status": status,
        "results": results,
        "progress": progress_info,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response_data)

@app.route('/evaluate_llm/<model_name>', methods=['POST','GET'])
def evaluate_llm(model_name):
    benchmark = request.form.get('benchmark', 'BIG-Bench')
    num_examples = int(request.form.get('num_examples', 25))
    max_tokens = int(request.form.get('max_tokens', 128))
    full_benchmark = request.form.get('full_benchmark') == 'on'
    
    print(f"Evaluating {model_name} on benchmark: {benchmark}")
    if benchmark != "BIG-Bench":
        flash(f"Evaluation for {benchmark} is not yet supported.")
        return redirect(url_for('index'))

    model_path = os.path.join(model_base_path, "llm", model_name)
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    eval_params = {
        'num_examples': num_examples,
        'max_tokens': max_tokens,
        'full_benchmark': full_benchmark
    }
    
    run_evaluation_in_background(model_name, model_path, eval_params)
    return render_template('loading.html', model_name=model_name)

@app.route('/check_status/<model_name>')
def check_status(model_name):
    status = processing_status.get(model_name, "not_started")
    progress = evaluation_progress.get(model_name, {})
    
    return jsonify({
        "status": status,
        "progress": progress
    })

@app.route('/history/<category>/<model_name>')
def history(category, model_name):
    """Display benchmark history for a specific model."""
    try:
        # Load history data
        history_file = "evaluation_results/allbenchmarkhistory.json"
        history_data = []
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)
            
            # Filter data for this specific model
            model_history = [entry for entry in all_history if model_name in entry.get("model_path", "")]
            
            # Sort by run number
            model_history.sort(key=lambda x: x.get("run", 0))
            
            # Process the history data for the template
            history_data = []
            for entry in model_history:
                processed_entry = {
                    'run': f"Run {entry.get('run', 1)}",
                    'scores': {entry.get('benchmark', 'BIG-Bench'): entry.get('average', 0)},
                    'average': entry.get('average', 0)
                }
                history_data.append(processed_entry)
            
        # Define all benchmarks we want to track
        benchmark_list = [
            "MMLU", "HellaSwag", "PIQA", "SocialIQA", "BooIQ", 
            "WinoGrande", "CommonsenseQA", "OpenBookQA", "ARC-e", 
            "ARC-c", "TriviaQA", "Natural Questions", "HumanEval", 
            "MBPP", "GSM8K", "MATH", "AGIEval", "BIG-Bench"
        ]
        
        # Calculate benchmark averages
        benchmark_averages = {}
        benchmark_counts = defaultdict(int)
        benchmark_sums = defaultdict(float)
        
        for entry in history_data:
            for benchmark, score in entry['scores'].items():
                if score != 'N/A' and isinstance(score, (int, float)):
                    benchmark_sums[benchmark] += score
                    benchmark_counts[benchmark] += 1
        
        for benchmark in benchmark_list:
            if benchmark_counts[benchmark] > 0:
                benchmark_averages[benchmark] = benchmark_sums[benchmark] / benchmark_counts[benchmark]
            else:
                benchmark_averages[benchmark] = 'N/A'
        
        # Calculate summary statistics
        all_scores = []
        benchmarks_tested = set()
        
        for entry in history_data:
            for benchmark, score in entry['scores'].items():
                if score != 'N/A' and isinstance(score, (int, float)):
                    all_scores.append(score)
                    benchmarks_tested.add(benchmark)
        
        benchmark_stats = {
            'benchmarks_tested': len(benchmarks_tested),
            'overall_average': sum(all_scores) / len(all_scores) if all_scores else 0,
            'best_score': max(all_scores) if all_scores else 0
        }
        
        return render_template('history.html',
                             model_name=model_name,
                             category=category,
                             history_data=history_data,
                             benchmark_list=benchmark_list,
                             benchmark_averages=benchmark_averages,
                             benchmark_stats=benchmark_stats)
    
    except Exception as e:
        print(f"Error loading history: {e}")
        return render_template('history.html',
                             model_name=model_name,
                             category=category,
                             history_data=[],
                             benchmark_list=[],
                             benchmark_averages={},
                             benchmark_stats={'benchmarks_tested': 0, 'overall_average': 0, 'best_score': 0})

def extract_score_from_results(results):
    """Extract a score from various result formats."""
    try:
        # Handle different possible result structures
        if isinstance(results, dict):
            # Look for common score fields
            score_fields = ['accuracy', 'score', 'exact_match', 'f1', 'bleu', 'rouge_l']
            
            for field in score_fields:
                if field in results:
                    value = results[field]
                    if isinstance(value, (int, float)):
                        return value * 100 if value <= 1.0 else value
                    elif isinstance(value, dict) and 'mean' in value:
                        mean_val = value['mean']
                        return mean_val * 100 if mean_val <= 1.0 else mean_val
            
            # If no direct score field, look for nested structures
            if 'metrics' in results:
                return extract_score_from_results(results['metrics'])
            
            if 'summary' in results:
                return extract_score_from_results(results['summary'])
        
        elif isinstance(results, (int, float)):
            return results * 100 if results <= 1.0 else results
    
    except:
        pass
    
    return None


@app.route('/results/<model_name>')
def analyze(model_name):
    """Enhanced results page with comprehensive metrics display."""
    try:
        # Determine category for this model
        category = None
        categories_mapping = {
            "LLMs": "llm",
            "Other GenAI Models": "genai", 
            "DL Models": "dl",
            "ML Models": "ml"
        }
        
        # Find which category this model belongs to
        for display_name, folder in categories_mapping.items():
            path = os.path.join(model_base_path, folder)
            if os.path.exists(path):
                models = [model for model in os.listdir(path) if os.path.isdir(os.path.join(path, model))]
                if model_name in models:
                    category = display_name
                    break
        
        # Default to LLMs if not found
        if not category:
            category = "LLMs"
        
        # Load enhanced results
        history_file = "evaluation_results/history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)
            history_data = [entry for entry in all_history if model_name in entry["model_path"]]
        else:
            # Fallback to old format
            history_data = get_history(model_name)
            
    except Exception as e:
        print(f"Error loading results: {e}")
        history_data = []
        category = "LLMs"
    
    return render_template('results.html', model_name=model_name, history=history_data, category=category)




@app.route('/download_report/<model_name>')
def download_report(model_name):
    """Generate and download PDF report."""
    if not PDF_AVAILABLE:
        flash("PDF generation not available. Please install weasyprint.")
        return redirect(url_for('analyze', model_name=model_name))
    
    try:
        # Load results data
        history_file = "evaluation_results/history.json"        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)                
            history_data = [entry for entry in all_history if model_name in entry["model_path"]]
        else:
            history_data = []
            
        if not history_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Render HTML template for PDF
        html_content = render_template('pdf_report.html', 
                                     model_name=model_name, 
                                     history=history_data)
        
        # Generate PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Create response
        response = make_response(pdf_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_evaluation_report.pdf"'
        
        return response
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        flash(f"Error generating PDF report: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))

@app.route('/export_json/<model_name>')
def export_json(model_name):
    """Export results as JSON file."""
    try:
        history_file = "evaluation_results/history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)
            history_data = [entry for entry in all_history if model_name in entry["model_path"]]
        else:
            history_data = []
            
        if not history_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Create JSON response
        response = make_response(json.dumps(history_data, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_results.json"'
        
        return response
        
    except Exception as e:
        flash(f"Error exporting JSON: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))

if __name__ == '__main__':
    app.run(debug=True)