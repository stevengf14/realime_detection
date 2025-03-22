import cv2
from ultralytics import YOLO
import os
import time
import numpy as np

# Configuración para optimización de rendimiento
FRAME_RESIZE_FACTOR = 0.5  # Reducir el tamaño del frame para procesar menos píxeles
PROCESS_EVERY_N_FRAMES = 2  # Procesar sólo cada N frames (reducido para mejor respuesta)
CONFIDENCE_THRESHOLD = 0.45  # Umbral de confianza para detecciones de YOLO (reducido para detectar más objetos)

# Lista de clases de interés (objetos comunes en una habitación)
OBJECTS_OF_INTEREST = [
    "person", "bed", "couch", "chair", "dining table", "laptop", "tv", "cell phone", 
    "book", "clock", "vase", "cup", "bottle", "keyboard", "mouse", "remote", 
    "microwave", "refrigerator", "oven", "toaster", "sink", "toilet", "backpack"
]

try:
    # Cargar el modelo YOLOv8 preentrenado (versión más ligera "n" en lugar de "x")
    print("Cargando modelo YOLO...")
    model = YOLO("yolov8n.pt")  # Modelo más ligero para CPU
    # Configurar el modelo para usar CPU explícitamente
    model.to('cpu')
    print("Modelo YOLO cargado correctamente")
    
    # Mostrar todas las clases que el modelo puede detectar
    print(f"El modelo puede detectar las siguientes clases: {model.names}")
    
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
    
    # Variables para almacenar información entre frames
    detections = []
    
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
            
            try:
                # Procesar el fotograma y detectar objetos con YOLO
                results = model(small_frame, conf=CONFIDENCE_THRESHOLD)
                
                # Almacenar detecciones para usar en frames no procesados
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = model.names[class_id]
                        
                        # Solo considerar objetos de interés o todos si la lista está vacía
                        if not OBJECTS_OF_INTEREST or class_name in OBJECTS_OF_INTEREST:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Ajustar coordenadas al tamaño del frame original
                            x1, y1, x2, y2 = int(x1/FRAME_RESIZE_FACTOR), int(y1/FRAME_RESIZE_FACTOR), int(x2/FRAME_RESIZE_FACTOR), int(y2/FRAME_RESIZE_FACTOR)
                            # Guardar la clase y confianza junto con las coordenadas
                            detections.append((x1, y1, x2, y2, class_name, confidence))
                
            except Exception as e:
                print(f"Error al procesar frame: {e}")
                
        # Mostrar todas las detecciones (ya sea en el frame procesado o el no procesado)
        for (x1, y1, x2, y2, class_name, confidence) in detections:
            # Seleccionar color según la clase (para distinguir diferentes objetos)
            color = (0, 255, 0)  # Verde por defecto
            if class_name == "person":
                color = (0, 255, 0)  # Verde para personas
            elif "phone" in class_name:
                color = (255, 0, 0)  # Azul para teléfonos
            elif class_name in ["bed", "couch", "chair"]:
                color = (0, 165, 255)  # Naranja para muebles
            elif class_name in ["book", "clock"]:
                color = (0, 0, 255)  # Rojo para objetos pequeños
                
            # Dibujar rectángulo alrededor del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Texto a mostrar (clase y confianza)
            text = f"{class_name} {int(confidence*100)}%"
            
            # Dibujar fondo negro para el texto (mayor contraste)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                        (x1, y1 - 30), 
                        (x1 + text_size[0] + 10, y1),
                        (0, 0, 0), -1)
            
            # Dibujar el nombre de la clase en AZUL con letras grandes
            cv2.putText(frame, 
                       text, 
                       (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7,  # Tamaño
                       (255, 0, 0),  # AZUL en BGR
                       2)  # Grosor
                
        # Mostrar FPS en el frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar el video con las detecciones
        cv2.imshow("Detección de Objetos (Optimizado para CPU)", frame)
    
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
