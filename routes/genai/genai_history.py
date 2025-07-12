from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory


genai_h_bp = Blueprint('genai_h', __name__)

@genai_h_bp.route('/evaluate_genai/<model_name>', methods=['POST'])
def history(model_name):
    pass