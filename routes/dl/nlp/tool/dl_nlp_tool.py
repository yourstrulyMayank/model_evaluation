from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory


dl_nlp_t_bp = Blueprint('dl_nlp_t', __name__)

@dl_nlp_t_bp.route('/evaluate_dl/<model_name>/<subcategory>', methods=['POST'])
def evaluate_dl(model_name, subcategory):
    """Evaluate DL models with subcategory support."""
    benchmark = request.form.get('benchmark', '')
    
    print(f"Evaluating DL model {model_name} (subcategory: {subcategory}) on benchmark: {benchmark}")
    
    # For now, show not supported message
    flash(f"DL model evaluation is not yet implemented.")
    return redirect(url_for('index'))


    
