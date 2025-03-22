from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import time
import threading
import numpy as np
from PIL import Image
import pickle
import traceback
from face_recognition_model import FaceNetRecognizer

# Configuración para optimización de rendimiento
FRAME_RESIZE_FACTOR = 0.5  # Reducir el tamaño del frame para procesar menos píxeles
PROCESS_EVERY_N_FRAMES = 3  # Procesar cada N frames para mejorar rendimiento

app = Flask(__name__)

# Variables globales
camera = None
output_frame = None
lock = threading.Lock()
is_camera_running = False
face_recognizer = None
face_tracking = {}
FACE_MEMORY_FRAMES = 15
tracking_id = None

# Ruta donde guardar/cargar el modelo
MODEL_PATH = "model/facenet_encodings.pkl"

def load_faces():
    global face_recognizer
    
    try:
        print("\n=== INICIANDO CARGA DE ROSTROS ===")
        
        # Eliminar el modelo anterior si existe
        if os.path.exists(MODEL_PATH):
            print(f"Eliminando modelo anterior: {MODEL_PATH}")
            try:
                os.remove(MODEL_PATH)
                print("Modelo anterior eliminado correctamente")
            except Exception as e:
                print(f"No se pudo eliminar el modelo anterior: {e}")
        
        # Crear directorio para el modelo si no existe
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Inicializar el reconocedor facial con ajustes más permisivos
        face_recognizer = FaceNetRecognizer(
            model_path=None,  # Forzar recarga desde imágenes
            people_folder='people',
            face_confidence=0.8  # Más permisivo para detección
        )
        
        # Verificar si se cargaron rostros
        if len(face_recognizer.known_face_names) == 0:
            print("\n⚠️ ADVERTENCIA: No se cargaron rostros conocidos")
            print("Verificando si existe Ivan.jpg en la carpeta 'people'...")
            
            ivan_path = os.path.join('people', 'Ivan.jpg')
            if os.path.exists(ivan_path):
                print(f"¡Se encontró {ivan_path}! Intentando cargar manualmente...")
                
                # Cargar manualmente la imagen de Ivan
                success = face_recognizer._process_ivan_image(ivan_path, "Ivan")
                
                if success:
                    print("✅ Ivan cargado manualmente con éxito")
                else:
                    print("❌ No se pudo cargar Ivan manualmente")
            else:
                print(f"❌ No se encontró {ivan_path}")
                print("Por favor, añade imágenes a la carpeta 'people'")
        
        # Guardar el modelo para uso futuro
        if len(face_recognizer.known_face_names) > 0:
            try:
                face_recognizer.save_model(MODEL_PATH)
                print(f"✅ Modelo guardado en {MODEL_PATH} con {len(face_recognizer.known_face_names)} rostros")
            except Exception as e:
                print(f"❌ Error al guardar modelo: {e}")
                traceback.print_exc()
        
        print(f"\n=== ROSTROS CARGADOS: {len(face_recognizer.known_face_names)} ===")
        if len(face_recognizer.known_face_names) > 0:
            print(f"Nombres: {', '.join(face_recognizer.known_face_names)}")
        print("=====================================\n")
            
    except Exception as e:
        print(f"Error grave al cargar rostros: {e}")
        traceback.print_exc()
        # Crear un reconocedor vacío para evitar errores
        face_recognizer = FaceNetRecognizer(
            model_path=None,
            people_folder='people'
        )

def detect_and_recognize_faces(frame):
    global face_recognizer, face_tracking, tracking_id
    
    # Redimensionar el frame para procesar más rápido
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
    
    # Variables para este frame
    current_face_locations = []
    current_face_names = []
    current_face_confidences = []
    
    # Verificar que el reconocedor facial está inicializado
    if face_recognizer is None:
        print("Error: FaceNetRecognizer no está inicializado")
        processed_frame = frame.copy()
        cv2.putText(processed_frame, "Error: Reconocedor no inicializado", 
                   (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return processed_frame
    
    # Verificar si Ivan está cargado
    ivan_loaded = "Ivan" in face_recognizer.known_face_names
    
    # Contamos frames para procesar solo algunos (mejora rendimiento)
    current_timestamp = time.time()
    frame_counter = int(current_timestamp * 10) % PROCESS_EVERY_N_FRAMES
    should_process = frame_counter == 0
    
    # Información para depuración
    if should_process:
        print(f"\n--- Procesando frame {int(current_timestamp)} ---")
    
    # Detectar rostros reales solo en algunos frames para mejorar rendimiento
    if should_process:
        try:
            # Método 1: Usar detector Haar Cascade (más rápido y estable)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Crear clasificador si no existe
            if not hasattr(face_recognizer, 'cascade'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_recognizer.cascade = cv2.CascadeClassifier(cascade_path)
                print("Inicializado detector Haar Cascade")
            
            # Detectar rostros
            faces = face_recognizer.cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                print(f"Detector Haar encontró {len(faces)} rostros")
                
                # Convertir detecciones Haar al formato esperado
                for (x, y, w, h) in faces:
                    # Guardar ubicación (x, y, w, h)
                    face_loc = (x, y, w, h)
                    
                    # Extraer la región del rostro
                    try:
                        face_region = small_frame[y:y+h, x:x+w]
                        if face_region.size == 0:
                            print("Región de rostro vacía, omitiendo")
                            continue
                            
                        # Usar FaceNet para comparar con rostros conocidos
                        name, confidence = face_recognizer.identify_face(face_region)
                        
                        # Solo agregar rostros conocidos con confianza suficiente
                        if name != "Desconocido" and confidence >= 40:
                            current_face_locations.append(face_loc)
                            current_face_names.append(name)
                            current_face_confidences.append(confidence)
                            
                            # Crear o actualizar tracking
                            track_id = f"{name}_track"
                            face_tracking[track_id] = {
                                'location': face_loc,
                                'name': name,
                                'confidence': confidence,
                                'frames_left': FACE_MEMORY_FRAMES
                            }
                            print(f"Reconocido: {name} con confianza {confidence:.1f}%")
                        else:
                            print(f"Rostro detectado pero no reconocido como {name}: confianza {confidence:.1f}% < umbral 40%")
                    except Exception as e:
                        print(f"Error al procesar rostro con Haar: {e}")
                
            # Método 2: Intentar también con MTCNN si Haar no encontró rostros conocidos
            if len(current_face_names) == 0 and face_recognizer is not None:
                try:
                    boxes, probs = face_recognizer.detect_faces(small_frame)
                    
                    if boxes is not None and len(boxes) > 0:
                        print("MTCNN detectó rostros")
                        
                        # Convertir detecciones MTCNN
                        for i, box in enumerate(boxes):
                            if isinstance(box, (list, np.ndarray)) and len(box) == 4:
                                try:
                                    box_array = np.array(box)
                                    x1, y1, x2, y2 = map(int, box_array.astype(int))
                                    w, h = x2 - x1, y2 - y1
                                    
                                    # Extraer región del rostro
                                    if y1 < 0: y1 = 0
                                    if x1 < 0: x1 = 0
                                    
                                    face_region = small_frame[y1:y2, x1:x2]
                                    if face_region.size == 0:
                                        print("Región de rostro MTCNN vacía, omitiendo")
                                        continue
                                        
                                    # Usar FaceNet para comparar con rostros conocidos
                                    name, confidence = face_recognizer.identify_face(face_region)
                                    
                                    # Solo agregar rostros conocidos con confianza suficiente
                                    if name != "Desconocido" and confidence >= 40:
                                        face_loc = (x1, y1, w, h)
                                        current_face_locations.append(face_loc)
                                        current_face_names.append(name)
                                        current_face_confidences.append(confidence)
                                        
                                        # Crear o actualizar tracking
                                        track_id = f"{name}_track"
                                        face_tracking[track_id] = {
                                            'location': face_loc,
                                            'name': name,
                                            'confidence': confidence,
                                            'frames_left': FACE_MEMORY_FRAMES
                                        }
                                        print(f"Reconocido con MTCNN: {name} con confianza {confidence:.1f}%")
                                    else:
                                        print(f"Rostro MTCNN detectado pero no reconocido como {name}: confianza {confidence:.1f}% < umbral 40%")
                                except Exception as e:
                                    print(f"Error al procesar box MTCNN: {e}")
                except Exception as e:
                    print(f"Error con MTCNN: {e}")
                    
        except Exception as e:
            print(f"Error en detección: {e}")
            traceback.print_exc()
    
    # Decrementar contador para todos los trackings
    keys_to_remove = []
    for track_id in face_tracking:
        face_tracking[track_id]['frames_left'] -= 1
        
        # Marcar para eliminar si expiró
        if face_tracking[track_id]['frames_left'] <= 0:
            keys_to_remove.append(track_id)
    
    # Eliminar tracks expirados
    for key in keys_to_remove:
        del face_tracking[key]
        if key == tracking_id:
            tracking_id = None
    
    # Dibujar todos los rostros en seguimiento
    processed_frame = frame.copy()
    
    # Depuración: verificar cuántos rostros estamos siguiendo
    print(f"Dibujando {len(face_tracking)} rostros en seguimiento")
    
    for track_id, face_data in face_tracking.items():
        try:
            # Extraer datos
            x, y, w, h = face_data['location']
            name = face_data['name']
            confidence = face_data['confidence']
            
            # Ajustar coordenadas al tamaño original
            x_orig = int(x / FRAME_RESIZE_FACTOR)
            y_orig = int(y / FRAME_RESIZE_FACTOR)
            w_orig = int(w / FRAME_RESIZE_FACTOR)
            h_orig = int(h / FRAME_RESIZE_FACTOR)
            
            # Color verde para rostros conocidos
            color = (0, 255, 0)  # Verde en BGR
            # Texto con nombre y confianza
            label = f"{name} {int(confidence)}%"
            
            print(f"Dibujando recuadro para {name} en ({x_orig},{y_orig},{w_orig},{h_orig})")
            
            # Dibujar el recuadro del rostro (más grueso para mayor visibilidad)
            cv2.rectangle(processed_frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 3)
            
            # Dibujar un fondo más grande para el texto
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)[0]
            cv2.rectangle(processed_frame, 
                        (x_orig, y_orig - 35), 
                        (x_orig + text_size[0] + 10, y_orig),
                        (0, 0, 0), -1)
            
            # Dibujar el nombre en AZUL (más grande y grueso)
            cv2.putText(processed_frame, 
                    label, 
                    (x_orig + 6, y_orig - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.8,
                    (255, 0, 0),  # Azul en BGR
                    2)  # Línea más gruesa
            
            # Añadir un recuadro adicional para mayor visibilidad
            cv2.rectangle(processed_frame, 
                        (x_orig-2, y_orig-2), 
                        (x_orig + w_orig+2, y_orig + h_orig+2), 
                        (255, 255, 255), 1)  # Borde blanco exterior
        except Exception as e:
            print(f"Error al dibujar rostro: {e}")
    
    return processed_frame

def generate_frames():
    global output_frame, lock, is_camera_running
    
    while is_camera_running:
        try:
            with lock:
                if output_frame is None:
                    continue
                
                # Codificar el frame a JPEG
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
            
            # Enviar el frame como respuesta de multipart
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_image) + b'\r\n')
                  
            # Pequeña pausa para evitar sobrecarga
            time.sleep(0.03)  # ~30 FPS máximo
            
        except Exception as e:
            print(f"Error en generate_frames: {e}")
            # Pequeña pausa para evitar bucles de error rápidos
            time.sleep(0.1)

def capture_frames():
    global output_frame, lock, camera, is_camera_running
    
    frame_count = 0
    fps = 0
    prev_time = 0
    error_count = 0
    max_errors = 5
    
    # Inicializar contador de frames para depuración
    total_frames = 0
    frames_with_faces = 0
    
    while is_camera_running:
        try:
            if camera is None or not camera.isOpened():
                print("Cámara cerrada o no disponible")
                is_camera_running = False
                break
                
            ret, frame = camera.read()
            if not ret:
                error_count += 1
                print(f"Error al leer frame de la cámara (intento {error_count}/{max_errors})")
                if error_count >= max_errors:
                    print("Demasiados errores consecutivos, deteniendo cámara")
                    is_camera_running = False
                    break
                time.sleep(0.1)
                continue
            
            # Resetear contador de errores si llegamos aquí
            error_count = 0
            
            # Calcular FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            # Procesar frame
            try:
                total_frames += 1
                
                # Imprimir estadísticas cada 100 frames
                if total_frames % 100 == 0:
                    if frames_with_faces > 0:
                        print(f"ESTADÍSTICAS: {frames_with_faces}/{total_frames} frames han tenido rostros detectados ({frames_with_faces/total_frames*100:.1f}%)")
                    else:
                        print(f"ESTADÍSTICAS: Ningún rostro detectado en {total_frames} frames")
                    
                # Guardar número de rostros antes del procesamiento
                faces_before = len(face_tracking) if face_tracking else 0
                
                processed_frame = detect_and_recognize_faces(frame)
                
                # Guardar número de rostros después del procesamiento
                faces_after = len(face_tracking) if face_tracking else 0
                
                # Actualizar contador si se detectaron rostros
                if faces_after > 0:
                    frames_with_faces += 1
                    
                # Imprimir info sobre cambios en los rostros detectados
                if faces_before != faces_after:
                    print(f"Cambio en rostros detectados: {faces_before} -> {faces_after}")
                
            except Exception as e:
                print(f"Error al procesar frame: {e}")
                processed_frame = frame.copy()  # Usar frame original si hay error
            
            # Mostrar FPS en el frame
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Actualizar el frame de salida
            with lock:
                output_frame = processed_frame.copy()
            
            frame_count += 1
            
            # Pequeña pausa para evitar sobrecarga de CPU
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error en capture_frames: {e}")
            # Pequeña pausa para evitar bucles de error rápidos
            time.sleep(0.1)

@app.route('/')
def index():
    """Página principal con interfaz de reconocimiento facial"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream de video con reconocimiento facial"""
    if not is_camera_running:
        # Si la cámara no está activa, devolver una imagen estática
        return Response('No hay transmisión de video activa', mimetype='text/plain')
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Iniciar la cámara y el proceso de reconocimiento"""
    global camera, is_camera_running
    
    if is_camera_running:
        return jsonify({"status": "La cámara ya está funcionando"})
    
    # Obtener ID de cámara del request
    data = request.get_json() or {}
    camera_id = data.get('camera_id', '0')
    
    print(f"Intentando abrir cámara con ID: {camera_id}")
    
    # Intentar varias opciones para la cámara
    camera = None
    error_messages = []
    
    if camera_id == 'dshow' and os.name == 'nt':  # DirectShow (solo Windows)
        try:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if camera.isOpened():
                print("Cámara abierta usando DirectShow")
            else:
                error_messages.append("No se pudo abrir la cámara usando DirectShow")
                camera.release()
                camera = None
        except Exception as e:
            error_messages.append(f"Error al intentar abrir cámara con DirectShow: {str(e)}")
    else:
        try:
            # Convertir a entero si es un número
            cam_index = int(camera_id) if camera_id.isdigit() else camera_id
            camera = cv2.VideoCapture(cam_index)
            if camera.isOpened():
                print(f"Cámara abierta en índice {cam_index}")
            else:
                error_messages.append(f"No se pudo abrir la cámara en índice {cam_index}")
                camera.release()
                camera = None
        except Exception as e:
            error_messages.append(f"Error al intentar abrir cámara en índice {camera_id}: {str(e)}")
    
    # Verificar si se pudo abrir la cámara
    if camera is None or not camera.isOpened():
        error_msg = "No se pudo acceder a la cámara. Errores: " + "; ".join(error_messages)
        print(f"ERROR: {error_msg}")
        return jsonify({"status": error_msg}), 500
    
    # Probar a leer un frame para verificar que funciona
    ret, test_frame = camera.read()
    if not ret or test_frame is None or test_frame.size == 0:
        camera.release()
        error_msg = "La cámara se abrió pero no devuelve imágenes válidas"
        print(f"ERROR: {error_msg}")
        return jsonify({"status": error_msg}), 500
    else:
        print(f"Lectura de cámara OK, tamaño de frame: {test_frame.shape}")
    
    # Cargar rostros conocidos
    load_faces()
    
    # Iniciar proceso
    is_camera_running = True
    threading.Thread(target=capture_frames, daemon=True).start()
    
    return jsonify({"status": "Cámara iniciada correctamente"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Detener la cámara y el proceso de reconocimiento"""
    global camera, is_camera_running
    
    if not is_camera_running:
        return jsonify({"status": "La cámara ya está detenida"})
    
    # Detener proceso
    is_camera_running = False
    time.sleep(1)  # Dar tiempo al thread para finalizar
    
    # Liberar cámara
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"status": "Cámara detenida correctamente"})

@app.route('/status')
def status():
    # Retorna el estado actual de la cámara y la información del modelo
    global camera, face_recognizer
    known_faces_count = len(face_recognizer.known_face_names) if face_recognizer else 0
    
    return jsonify({
        'camera_running': camera is not None and camera.isOpened(),
        'error_count': camera.error_count if camera and hasattr(camera, 'error_count') else 0,
        'known_faces': known_faces_count
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Resetear la aplicación en caso de error"""
    global camera, is_camera_running, output_frame
    
    # Detener la cámara si está activa
    if is_camera_running:
        is_camera_running = False
        time.sleep(1)
    
    # Liberar recursos
    if camera is not None:
        camera.release()
        camera = None
    
    # Limpiar frame de salida
    with lock:
        output_frame = None
    
    return jsonify({"status": "Sistema reseteado correctamente"})

@app.route('/update_model', methods=['POST'])
def update_model():
    global camera, face_recognizer
    
    # Verificar si la cámara está activa
    if camera is not None and camera.isOpened():
        return jsonify({'status': 'Error: Detenga la cámara antes de actualizar el modelo'}), 400
    
    try:
        # Reiniciar el reconocedor facial
        face_recognizer = None
        
        # Cargar nuevamente las caras conocidas
        load_faces()
        
        known_faces_count = len(face_recognizer.known_face_names) if face_recognizer else 0
        return jsonify({
            'status': 'Modelo actualizado exitosamente',
            'known_faces': known_faces_count
        })
    except Exception as e:
        app.logger.error(f"Error al actualizar el modelo: {str(e)}")
        return jsonify({'status': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    # Para producción, establecer threaded=True para mejor rendimiento con múltiples clientes
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True) 