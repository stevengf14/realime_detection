import cv2
import os
import time
import numpy as np
import pickle
import traceback
from face_recognition_model import FaceNetRecognizer

# Configuración para optimización de rendimiento
FRAME_RESIZE_FACTOR = 0.5  # Reducir el tamaño del frame para procesar menos píxeles
PROCESS_EVERY_N_FRAMES = 3  # Procesar cada N frames para mejorar rendimiento
FACE_MEMORY_FRAMES = 15  # Cuántos frames recordar un rostro
MODEL_PATH = "model/facenet_encodings.pkl"  # Ruta donde guardar/cargar el modelo

class FaceDetector:
    def __init__(self):
        self.face_recognizer = None
        self.face_tracking = {}
        self.tracking_id = None
        self.load_faces()
    
    def load_faces(self):
        """Carga los rostros conocidos desde la carpeta people"""
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
            self.face_recognizer = FaceNetRecognizer(
                model_path=None,  # Forzar recarga desde imágenes
                people_folder='people',
                face_confidence=0.8  # Más permisivo para detección
            )
            
            # Verificar si se cargaron rostros
            if len(self.face_recognizer.known_face_names) == 0:
                print("\n⚠️ ADVERTENCIA: No se cargaron rostros conocidos")
                print("Verificando si existe Ivan.jpg en la carpeta 'people'...")
                
                ivan_path = os.path.join('people', 'Ivan.jpg')
                if os.path.exists(ivan_path):
                    print(f"¡Se encontró {ivan_path}! Intentando cargar manualmente...")
                    
                    # Cargar manualmente la imagen de Ivan
                    success = self.face_recognizer._process_ivan_image(ivan_path, "Ivan")
                    
                    if success:
                        print("✅ Ivan cargado manualmente con éxito")
                    else:
                        print("❌ No se pudo cargar Ivan manualmente")
                else:
                    print(f"❌ No se encontró {ivan_path}")
                    print("Por favor, añade imágenes a la carpeta 'people'")
            
            # Guardar el modelo para uso futuro
            if len(self.face_recognizer.known_face_names) > 0:
                try:
                    self.face_recognizer.save_model(MODEL_PATH)
                    print(f"✅ Modelo guardado en {MODEL_PATH} con {len(self.face_recognizer.known_face_names)} rostros")
                except Exception as e:
                    print(f"❌ Error al guardar modelo: {e}")
                    traceback.print_exc()
            
            print(f"\n=== ROSTROS CARGADOS: {len(self.face_recognizer.known_face_names)} ===")
            if len(self.face_recognizer.known_face_names) > 0:
                print(f"Nombres: {', '.join(self.face_recognizer.known_face_names)}")
            print("=====================================\n")
                
        except Exception as e:
            print(f"Error grave al cargar rostros: {e}")
            traceback.print_exc()
            # Crear un reconocedor vacío para evitar errores
            self.face_recognizer = FaceNetRecognizer(
                model_path=None,
                people_folder='people'
            )
    
    def detect_and_recognize_faces(self, frame):
        """Detecta y reconoce rostros en el frame proporcionado"""
        # Redimensionar el frame para procesar más rápido
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
        
        # Variables para este frame
        current_face_locations = []
        current_face_names = []
        current_face_confidences = []
        
        # Verificar que el reconocedor facial está inicializado
        if self.face_recognizer is None:
            print("Error: FaceNetRecognizer no está inicializado")
            processed_frame = frame.copy()
            cv2.putText(processed_frame, "Error: Reconocedor no inicializado", 
                       (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame
        
        # Verificar si Ivan está cargado
        ivan_loaded = "Ivan" in self.face_recognizer.known_face_names
        
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
                if not hasattr(self.face_recognizer, 'cascade'):
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    self.face_recognizer.cascade = cv2.CascadeClassifier(cascade_path)
                    print("Inicializado detector Haar Cascade")
                
                # Detectar rostros
                faces = self.face_recognizer.cascade.detectMultiScale(
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
                            name, confidence = self.face_recognizer.identify_face(face_region)
                            
                            # Solo agregar rostros conocidos con confianza suficiente
                            if name != "Desconocido" and confidence >= 40:
                                current_face_locations.append(face_loc)
                                current_face_names.append(name)
                                current_face_confidences.append(confidence)
                                
                                # Crear o actualizar tracking
                                track_id = f"{name}_track"
                                self.face_tracking[track_id] = {
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
                if len(current_face_names) == 0 and self.face_recognizer is not None:
                    try:
                        print("Intentando detección con MTCNN...")
                        boxes, probs = self.face_recognizer.detect_faces(small_frame)
                        
                        if boxes is not None and len(boxes) > 0:
                            print(f"MTCNN detectó {len(boxes)} rostros")
                            
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
                                        name, confidence = self.face_recognizer.identify_face(face_region)
                                        
                                        # Solo agregar rostros conocidos con confianza suficiente
                                        if name != "Desconocido" and confidence >= 40:
                                            face_loc = (x1, y1, w, h)
                                            current_face_locations.append(face_loc)
                                            current_face_names.append(name)
                                            current_face_confidences.append(confidence)
                                            
                                            # Crear o actualizar tracking
                                            track_id = f"{name}_track"
                                            self.face_tracking[track_id] = {
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
                        print("MTCNN falló, usar solo Haar Cascade para detección")
            
            except Exception as e:
                print(f"Error en detección: {e}")
                traceback.print_exc()
        
        # Decrementar contador para todos los trackings
        keys_to_remove = []
        for track_id in self.face_tracking:
            self.face_tracking[track_id]['frames_left'] -= 1
            
            # Marcar para eliminar si expiró
            if self.face_tracking[track_id]['frames_left'] <= 0:
                keys_to_remove.append(track_id)
        
        # Eliminar tracks expirados
        for key in keys_to_remove:
            del self.face_tracking[key]
            if key == self.tracking_id:
                self.tracking_id = None
        
        # Dibujar todos los rostros en seguimiento
        processed_frame = frame.copy()
        
        # Depuración: verificar cuántos rostros estamos siguiendo
        print(f"Dibujando {len(self.face_tracking)} rostros en seguimiento")
        
        for track_id, face_data in self.face_tracking.items():
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