from flask import Blueprint, jsonify, request, Response
from app.camera import Camera

# Instancia singleton de la cámara
camera_instance = Camera()

# Crear el blueprint
camera_bp = Blueprint('camera', __name__, url_prefix='/camera')

@camera_bp.route('/start', methods=['POST'])
def start_camera():
    """Iniciar la cámara y el proceso de reconocimiento"""
    # Obtener ID de cámara del request
    data = request.get_json() or {}
    camera_id = data.get('camera_id', '0')
    
    result = camera_instance.start_camera(camera_id)
    
    if not result.get('success', True):
        return jsonify(result), 500
        
    return jsonify(result)

@camera_bp.route('/stop', methods=['POST'])
def stop_camera():
    """Detener la cámara y el proceso de reconocimiento"""
    result = camera_instance.stop_camera()
    return jsonify(result)

@camera_bp.route('/reset', methods=['POST'])
def reset():
    """Resetear la aplicación en caso de error"""
    result = camera_instance.reset()
    return jsonify(result)

@camera_bp.route('/video_feed')
def video_feed():
    """Stream de video con reconocimiento facial"""
    if not camera_instance.is_camera_running:
        # Si la cámara no está activa, devolver una imagen estática
        return Response('No hay transmisión de video activa', mimetype='text/plain')
    
    return Response(camera_instance.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
