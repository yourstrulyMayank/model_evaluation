from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory
import os
import threading
from datetime import datetime


llm_custom_bp = Blueprint('llm_custom', __name__)

UPLOAD_FOLDER = 'uploads'
model_base_path = "models"
processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress
custom_evaluation_results = {}
processing_status = {}
custom_evaluation_progress = {} 
model_base_path = "models"

@llm_custom_bp.route('/custom_llm/<model_name>')
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


@llm_custom_bp.route('/run_custom_evaluation/<model_name>', methods=['POST'])
def run_custom_evaluation_route(model_name):
    try:
        # Import custom evaluator
        from llm_custom_utils import run_custom_evaluation
        
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


@llm_custom_bp.route('/clear_custom_results/<model_name>', methods=['POST'])
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





@llm_custom_bp.route('/check_custom_status/<model_name>')
def check_custom_status(model_name):
    status_key = f"{model_name}_custom"
    status = processing_status.get(status_key, "not_started")
    results = custom_evaluation_results.get(model_name, {})
    from llm_custom_utils import get_progress
    # Get progress information from llm_custom_utils
    progress_info = get_progress(model_name)
    
    print(f"Status check for {model_name}: {status}, Progress: {progress_info}")  # Debug log
    
    response_data = {
        "status": status,
        "results": results,
        "progress": progress_info,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response_data)

@llm_custom_bp.route('/download_custom_excel/<model_name>')
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
        
