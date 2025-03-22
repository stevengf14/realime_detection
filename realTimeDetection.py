import cv2
from ultralytics import YOLO
import os
import face_recognition
import time
import numpy as np

# Configuración para optimización de rendimiento
FRAME_RESIZE_FACTOR = 0.5  # Reducir el tamaño del frame para procesar menos píxeles
PROCESS_EVERY_N_FRAMES = 3  # Procesar sólo cada N frames
CONFIDENCE_THRESHOLD = 0.5  # Umbral de confianza para detecciones de YOLO
RECOGNITION_TOLERANCE = 0.6  # Umbral de tolerancia para reconocimiento facial

try:
    # Cargar el modelo YOLOv8 preentrenado (versión más ligera "n" en lugar de "x")
    print("Cargando modelo YOLO...")
    model = YOLO("yolov8n.pt")  # Modelo mucho más ligero, mejor para CPU
    # Configurar el modelo para usar CPU explícitamente
    model.to('cpu')
    print("Modelo YOLO cargado correctamente")
    
    # Inicializar la captura de video en tiempo real
    print("Inicializando cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo acceder a la cámara. Verifica que esté conectada y no esté siendo usada por otra aplicación.")
    print("Cámara inicializada correctamente")
    
    # Variables para el seguimiento de FPS
    prev_time = 0
    fps = 0
    frame_count = 0
    
    # Listar imágenes en la carpeta 'people'
    people_names = []
    known_face_encodings = []
    
    # Verificar que la ruta 'people' exista
    people_folder = "people"
    if not os.path.exists(people_folder):
        os.makedirs(people_folder)
        print(f"Se ha creado la carpeta '{people_folder}'. Por favor, agrega imágenes de personas en subcarpetas.")
    else:
        # Cargar las imágenes de referencia una sola vez al inicio
        print("Cargando imágenes de referencia...")
        person_count = 0
        for foldername in os.listdir(people_folder):
            folder_path = os.path.join(people_folder, foldername)
    
            if os.path.isdir(folder_path):
                print(f"Procesando carpeta: {foldername}")
                
                # Procesar solo una imagen por carpeta (suficiente para identificación)
                for filename in os.listdir(folder_path):
                    if filename.endswith((".jpg", ".jpeg", ".png")):
                        image_path = os.path.join(folder_path, filename)
                        
                        try:
                            # Cargar y redimensionar imagen para mejorar rendimiento
                            img = cv2.imread(image_path)
                            if img is None:
                                print(f"No se pudo leer: {image_path}")
                                continue
                                
                            # Reducir tamaño de la imagen para procesamiento más rápido
                            small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                            
                            # Obtener codificaciones faciales
                            encoding = face_recognition.face_encodings(rgb_small_img)
                            if encoding:  
                                known_face_encodings.append(encoding[0])
                                people_names.append(foldername) 
                                person_count += 1
                                break  # Solo usar la primera imagen válida por persona
                            else:
                                print(f"No se detectó rostro en: {image_path}")
                        except Exception as e:
                            print(f"Error al cargar {image_path}: {e}")
        print(f"Se cargaron {person_count} personas para reconocimiento facial")
    
    # Carga más rápida del clasificador de rostros
    print("Cargando clasificador de rostros...")
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        raise Exception(f"No se encontró el clasificador de rostros en {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print("Clasificador de rostros cargado correctamente")
    
    # Función optimizada para reconocer rostros
    def verify_person(face_encoding):
        # Si no hay encodings conocidos, retornar None
        if not known_face_encodings:
            return None
            
        # Calcular distancias con todos los rostros conocidos de una vez (vectorizado)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Si la mejor coincidencia está por debajo del umbral, retornar el nombre
        if face_distances[best_match_index] <= RECOGNITION_TOLERANCE:
            return people_names[best_match_index]
        return None
    
    # Variables para almacenar información entre frames
    detections = []
    face_locations = []
    
    print("Iniciando detección en tiempo real. Presiona 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de la cámara")
            break
        
        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Redimensionar el frame para procesar más rápido
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
        
        # Solo procesar cada N frames para mejorar rendimiento
        frame_count += 1
        process_this_frame = frame_count % PROCESS_EVERY_N_FRAMES == 0
        
        if process_this_frame:
            # Limpiar listas de detecciones previas
            detections = []
            face_locations = []
            
            try:
                # Procesar el fotograma y detectar objetos con YOLO
                results = model(small_frame, conf=CONFIDENCE_THRESHOLD)
                
                # Detectar rostros usando Haar Cascade (más rápido que face_recognition para detección)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # Convertir pequeño frame a RGB para face_recognition
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Almacenar detecciones para usar en frames no procesados
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        if model.names[class_id] == "person" and box.conf[0].item() >= CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Ajustar coordenadas al tamaño del frame original
                            x1, y1, x2, y2 = int(x1/FRAME_RESIZE_FACTOR), int(y1/FRAME_RESIZE_FACTOR), int(x2/FRAME_RESIZE_FACTOR), int(y2/FRAME_RESIZE_FACTOR)
                            detections.append((x1, y1, x2, y2))
                            
                # Procesar rostros solo si hay personas detectadas y hay rostros conocidos
                if len(detections) > 0 and len(faces) > 0 and len(known_face_encodings) > 0:
                    for (x, y, w, h) in faces:
                        # Ajustar coordenadas al tamaño del frame original
                        x, y, w, h = int(x/FRAME_RESIZE_FACTOR), int(y/FRAME_RESIZE_FACTOR), int(w/FRAME_RESIZE_FACTOR), int(h/FRAME_RESIZE_FACTOR)
                        face_locations.append((x, y, w, h))
                        
                        # Verificar que el rostro esté dentro de los límites del frame
                        if y < 0 or y + h >= frame.shape[0] or x < 0 or x + w >= frame.shape[1]:
                            continue
                            
                        # Extraer la región del rostro directamente del frame original para mejor calidad
                        face = frame[y:y+h, x:x+w]
                        if face.size == 0:  # Verificar que el rostro sea válido
                            continue
                            
                        # Redimensionar para facial recognition más rápido
                        face_resized = cv2.resize(face, (0, 0), fx=0.5, fy=0.5)
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        
                        # Obtener encodings del rostro
                        face_encodings = face_recognition.face_encodings(face_rgb)
                        if face_encodings:
                            person_name = verify_person(face_encodings[0])
                            if person_name:
                                for (x1, y1, x2, y2) in detections:
                                    # Verificar si el rostro está dentro de una detección de persona
                                    if (x1 <= x <= x2 and y1 <= y <= y2) or (x1 <= x+w <= x2 and y1 <= y+h <= y2):
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(frame, f"{person_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                    (255, 0, 0), 2)
                                        break
            except Exception as e:
                print(f"Error al procesar frame: {e}")
        else:
            # Para frames no procesados, mantener las detecciones anteriores
            for (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Para rostros ya identificados, mostrar etiquetas
            for (x, y, w, h) in face_locations:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                
        # Mostrar FPS en el frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar el video con las detecciones
        cv2.imshow("Detección en Tiempo Real (Optimizado para CPU)", frame)
    
        # Salir al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Liberar recursos
    print("Cerrando aplicación y liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada correctamente")

except Exception as e:
    print(f"Error en la aplicación: {e}")
    # Asegurar que los recursos se liberen incluso si hay un error
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
