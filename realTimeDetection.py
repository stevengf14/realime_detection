import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
modelo = YOLO("yolov8m.pt")  # Puedes usar modelos más grandes como 'yolov8s.pt'

# Inicializar la captura de video en tiempo real
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara web, o reemplaza con la ruta de un video

objects_names = ["person", "cat", "dog", "cell phone"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones del frame
    altura, ancho, _ = frame.shape

    # Procesar el fotograma y detectar objetos
    resultados = modelo(frame)

    # Dibujar las detecciones en el fotograma
    for resultado in resultados:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])  # Coordenadas de la caja
            conf = caja.conf[0].item()  # Confianza del modelo
            clase = int(caja.cls[0].item())  # Clase detectada
            confianza2 = conf * 100
            etiqueta = f"{modelo.names[clase]} {confianza2:.2f}" + "%"  # Etiqueta con confianza

            if modelo.names[clase] in objects_names:
                # Color dinámico basado en la confianza
                color = (0, int(255 * conf), int(255 * (1 - conf)))

                # Dibujar la caja y la etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar el video con las detecciones
    cv2.imshow("YOLOv8 - Detección en Tiempo Real", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
