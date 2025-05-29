# app.py - Enhanced Flask App
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response
import os
import threading
import json
import datetime
from io import BytesIO
import base64

# Import your evaluation functions

from evaluate_llm import get_history, run_evaluation, _save_enhanced_results 

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
    return render_template('custom_llm.html', model_name=model_name)

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

@app.route('/history/<model_name>')
def history(model_name):
    # Try to get enhanced results first, fallback to old format
    try:
        history_file = "evaluation_results/history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                all_history = json.load(f)
            history_data = [entry for entry in all_history if model_name in entry["model_path"]]
        else:
            history_data = get_history(model_name)  # Fallback to old system
    except:
        history_data = []
    
    return render_template('history.html', model_name=model_name, history=history_data)

@app.route('/results/<model_name>')
def analyze(model_name):
    """Enhanced results page with comprehensive metrics display."""
    try:
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
    
    return render_template('results.html', model_name=model_name, history=history_data)

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