from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory


dl_nlp_c_bp = Blueprint('dl_nlp_c', __name__)

@dl_nlp_c_bp.route('/custom_dl/<model_name>/<subcategory>')
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