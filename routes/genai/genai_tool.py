from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory


genai_t_bp = Blueprint('genai_t', __name__)

@genai_t_bp.route('/evaluate_genai/<model_name>', methods=['POST'])
def evaluate_genai(model_name):
    pass