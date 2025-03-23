from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Página principal con interfaz de reconocimiento facial"""
    return render_template('index.html') 