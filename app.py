# app.py - Enhanced Flask App
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response
import os
import threading
import json
import datetime
from io import BytesIO
import base64
import os
from datetime import datetime
from collections import defaultdict
from werkzeug.utils import secure_filename
import uuid
# Import your evaluation functions
from custom_evaluate_llm import run_custom_evaluation, get_progress
from evaluate_llm import get_history, run_evaluation, _save_enhanced_results 




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
custom_evaluation_results = {}  # Store custom evaluation results

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