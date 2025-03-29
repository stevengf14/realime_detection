from flask import Flask
import os

def create_app(test_config=None):
    """Aplicación factory que crea y configura la instancia de la aplicación Flask"""
    
    # Crear y configurar la aplicación
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        # Otras configuraciones
    )

    # Asegurar que existe el directorio para el modelo
    os.makedirs(os.path.join('model'), exist_ok=True)
    
    # Registrar blueprints
    from app.blueprints.camera.routes import camera_bp
    from app.blueprints.recognition.routes import recognition_bp
    
    app.register_blueprint(camera_bp)
    app.register_blueprint(recognition_bp)
    
    # Ruta principal
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app 