import cv2
import time
import threading
from app.face_detector import FaceDetector

class Camera:
    def __init__(self):
        self.camera = None
        self.output_frame = None
        self.lock = threading.Lock()
        self.is_camera_running = False
        self.face_detector = FaceDetector()
        
    def start_camera(self, camera_id='0'):
        """Iniciar la cámara con el ID especificado"""
        if self.is_camera_running:
            return {"status": "La cámara ya está funcionando"}
        
        print(f"Intentando abrir cámara con ID: {camera_id}")
        
        # Intentar varias opciones para la cámara
        self.camera = None
        error_messages = []
        
        if camera_id == 'dshow' and hasattr(cv2, 'CAP_DSHOW'):
            try:
                self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if self.camera.isOpened():
                    print("Cámara abierta usando DirectShow")
                else:
                    error_messages.append("No se pudo abrir la cámara usando DirectShow")
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                error_messages.append(f"Error al intentar abrir cámara con DirectShow: {str(e)}")
        else:
            try:
                # Convertir a entero si es un número
                cam_index = int(camera_id) if camera_id.isdigit() else camera_id
                self.camera = cv2.VideoCapture(cam_index)
                if self.camera.isOpened():
                    print(f"Cámara abierta en índice {cam_index}")
                else:
                    error_messages.append(f"No se pudo abrir la cámara en índice {cam_index}")
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                error_messages.append(f"Error al intentar abrir cámara en índice {camera_id}: {str(e)}")
        
        # Verificar si se pudo abrir la cámara
        if self.camera is None or not self.camera.isOpened():
            error_msg = "No se pudo acceder a la cámara. Errores: " + "; ".join(error_messages)
            print(f"ERROR: {error_msg}")
            return {"status": error_msg, "success": False}
        
        # Probar a leer un frame para verificar que funciona
        ret, test_frame = self.camera.read()
        if not ret or test_frame is None or test_frame.size == 0:
            self.camera.release()
            error_msg = "La cámara se abrió pero no devuelve imágenes válidas"
            print(f"ERROR: {error_msg}")
            return {"status": error_msg, "success": False}
        else:
            print(f"Lectura de cámara OK, tamaño de frame: {test_frame.shape}")
        
        # Iniciar proceso
        self.is_camera_running = True
        threading.Thread(target=self.capture_frames, daemon=True).start()
        
        return {"status": "Cámara iniciada correctamente", "success": True}
    
    def stop_camera(self):
        """Detener la cámara y el proceso de reconocimiento"""
        if not self.is_camera_running:
            return {"status": "La cámara ya está detenida"}
        
        # Detener proceso
        self.is_camera_running = False
        time.sleep(1)  # Dar tiempo al thread para finalizar
        
        # Liberar cámara
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        return {"status": "Cámara detenida correctamente"}
    
    def reset(self):
        """Resetear la cámara en caso de error"""
        # Detener la cámara si está activa
        if self.is_camera_running:
            self.is_camera_running = False
            time.sleep(1)
        
        # Liberar recursos
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # Limpiar frame de salida
        with self.lock:
            self.output_frame = None
        
        return {"status": "Sistema reseteado correctamente"}
    
    def capture_frames(self):
        """Captura frames de la cámara y los procesa para reconocimiento facial"""
        frame_count = 0
        fps = 0
        prev_time = 0
        error_count = 0
        max_errors = 5
        
        # Inicializar contador de frames para depuración
        total_frames = 0
        frames_with_faces = 0
        
        while self.is_camera_running:
            try:
                if self.camera is None or not self.camera.isOpened():
                    print("Cámara cerrada o no disponible")
                    self.is_camera_running = False
                    break
                    
                ret, frame = self.camera.read()
                if not ret:
                    error_count += 1
                    print(f"Error al leer frame de la cámara (intento {error_count}/{max_errors})")
                    if error_count >= max_errors:
                        print("Demasiados errores consecutivos, deteniendo cámara")
                        self.is_camera_running = False
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
                    faces_before = len(self.face_detector.face_tracking) if self.face_detector.face_tracking else 0
                    
                    processed_frame = self.face_detector.detect_and_recognize_faces(frame)
                    
                    # Guardar número de rostros después del procesamiento
                    faces_after = len(self.face_detector.face_tracking) if self.face_detector.face_tracking else 0
                    
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
                with self.lock:
                    self.output_frame = processed_frame.copy()
                
                frame_count += 1
                
                # Pequeña pausa para evitar sobrecarga de CPU
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error en capture_frames: {e}")
                # Pequeña pausa para evitar bucles de error rápidos
                time.sleep(0.1)
                
    def generate_frames(self):
        """Genera frames para el streaming de video"""
        while self.is_camera_running:
            try:
                with self.lock:
                    if self.output_frame is None:
                        continue
                    
                    # Codificar el frame a JPEG
                    (flag, encoded_image) = cv2.imencode(".jpg", self.output_frame)
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
    
    def update_model(self):
        """Actualiza el modelo de reconocimiento facial"""
        if self.camera is not None and self.is_camera_running:
            return {"status": "Error: Detenga la cámara antes de actualizar el modelo", "success": False}
        
        try:
            # Reiniciar el reconocedor facial
            self.face_detector = FaceDetector()
            
            known_faces_count = len(self.face_detector.face_recognizer.known_face_names) if self.face_detector.face_recognizer else 0
            return {
                "status": "Modelo actualizado exitosamente",
                "known_faces": known_faces_count,
                "success": True
            }
        except Exception as e:
            print(f"Error al actualizar el modelo: {str(e)}")
            return {"status": f"Error: {str(e)}", "success": False}
            
    def get_status(self):
        """Obtiene el estado actual de la cámara y el modelo"""
        known_faces_count = 0
        if self.face_detector and self.face_detector.face_recognizer:
            known_faces_count = len(self.face_detector.face_recognizer.known_face_names)
            
        return {
            'camera_running': self.camera is not None and self.camera.isOpened(),
            'error_count': 0,  # Ya no se usa
            'known_faces': known_faces_count
        } 
