from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory
import os
import json
from io import BytesIO
# from weasyprint import HTML
from .llm_tool_bigbench_utils import ( get_history,
                                        run_evaluation_in_background,
                                        extract_score_from_results)
from collections import defaultdict

llm_t_bb_bp = Blueprint('llm_t_bb', __name__)

model_base_path = "models"
processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress


@llm_t_bb_bp.route('/download_report/<model_name>')
def download_report(model_name):
    # PDF generation
    try:
        # from weasyprint import HTML, CSS
        from jinja2 import Template
        PDF_AVAILABLE = True
    except ImportError:
        print("⚠️ PDF generation not available. Install: pip install weasyprint")
        PDF_AVAILABLE = False
    """Generate and download PDF report."""
    if not PDF_AVAILABLE:
        flash("PDF generation not available. Please install weasyprint.")
        return redirect(url_for('analyze', model_name=model_name))
    
    try:
        # Load results data
        history_file = "evaluation_results/llm/history.json"        
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
        html_content = render_template('llm/pdf_report.html', 
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

@llm_t_bb_bp.route('/export_json/<model_name>')
def export_json(model_name):
    """Export results as JSON file."""
    try:
        history_file = "evaluation_results/llm/history.json"
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
    
@llm_t_bb_bp.route('/evaluate_model/<category>/<model_name>')
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
    return render_template('llm/loading.html', model_name=model_name)



@llm_t_bb_bp.route('/evaluate_llm/<model_name>', methods=['POST','GET'])
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
    return render_template('llm/loading.html', model_name=model_name)

@llm_t_bb_bp.route('/check_status/<model_name>')
def check_status(model_name):
    status = processing_status.get(model_name, "not_started")
    progress = evaluation_progress.get(model_name, {})
    
    return jsonify({
        "status": status,
        "progress": progress
    })

@llm_t_bb_bp.route('/history/<category>/<model_name>')
def history(category, model_name):
    """Display benchmark history for a specific model."""
    try:
        # Load history data
        history_file = "evaluation_results/llm/allbenchmarkhistory.json"
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
        
        return render_template('llm/history.html',
                             model_name=model_name,
                             category=category,
                             history_data=history_data,
                             benchmark_list=benchmark_list,
                             benchmark_averages=benchmark_averages,
                             benchmark_stats=benchmark_stats)
    
    except Exception as e:
        print(f"Error loading history: {e}")
        return render_template('llm/history.html',
                             model_name=model_name,
                             category=category,
                             history_data=[],
                             benchmark_list=[],
                             benchmark_averages={},
                             benchmark_stats={'benchmarks_tested': 0, 'overall_average': 0, 'best_score': 0})



@llm_t_bb_bp.route('/results/<model_name>')
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
        history_file = "evaluation_results/llm/history.json"
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
    
    return render_template('llm/tool_evaluate.html', model_name=model_name, history=history_data, category=category)
