import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")  # Usa 'yolov8s.pt' o 'yolov8m.pt' si necesitas mayor precisión

# Inicializar la captura de video en tiempo real
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara web, o reemplaza con la ruta de un video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección de objetos en el fotograma
    results = model(frame)

    # Dibujar las cajas delimitadoras y etiquetas en la imagen
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
            conf = box.conf[0].item()  # Confianza de detección
            cls = int(box.cls[0].item())  # Clase detectada
            label = f"{model.names[cls]} {conf:.2f}"  # Etiqueta con confianza

            # Dibujar el rectángulo y la etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la imagen con detecciones
    cv2.imshow("Detección en Tiempo Real - YOLOv8", frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
