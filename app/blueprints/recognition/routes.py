from flask import Blueprint, jsonify
from app.blueprints.camera.routes import camera_instance

# Crear el blueprint
recognition_bp = Blueprint('recognition', __name__, url_prefix='/recognition')

@recognition_bp.route('/update_model', methods=['POST'])
def update_model():
    """Actualiza el modelo de reconocimiento facial"""
    result = camera_instance.update_model()
    
    if not result.get('success', True):
        return jsonify(result), 500
        
    return jsonify(result) 
