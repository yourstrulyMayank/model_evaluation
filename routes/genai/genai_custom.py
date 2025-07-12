from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory


genai_c_bp = Blueprint('genai_c', __name__)

@genai_c_bp.route('/custom_genai/<model_name>', methods=['POST'])
def custom_genai(model_name):
    pass