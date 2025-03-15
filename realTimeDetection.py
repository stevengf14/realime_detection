import cv2
import numpy as np
from ultralytics import YOLO
import os
import face_recognition

# Cargar el modelo YOLOv8 preentrenado
modelo = YOLO("yolov8m.pt")

# Inicializar la captura de video en tiempo real
cap = cv2.VideoCapture(0)

# Listar imágenes en la carpeta 'people'
people_images = []
people_names = []

# Verificar que la ruta 'people' exista
people_folder = "people"
if not os.path.exists(people_folder):
    print(f"¡La carpeta '{people_folder}' no existe!")
else:
    # Recorrer las carpetas dentro de 'people'
    for foldername in os.listdir(people_folder):
        folder_path = os.path.join(people_folder, foldername)

        # Asegurarse de que es una carpeta (no un archivo)
        if os.path.isdir(folder_path):
            print(f"Procesando la carpeta: {foldername}")

            # Recorrer las imágenes dentro de la subcarpeta
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)

                    try:
                        print(f"Cargando la imagen: {image_path}")
                        img = face_recognition.load_image_file(image_path)

                        # Obtener las codificaciones de rostro
                        encoding = face_recognition.face_encodings(img)
                        if encoding:  # Si se encuentra al menos un rostro en la imagen
                            people_images.append(encoding[0])
                            people_names.append(foldername)  # Nombre de la carpeta como nombre de la persona
                            print(f"Codificación encontrada para {foldername}: {encoding[0]}")
                        else:
                            print(f"No se encontró ningún rostro en {image_path}")
                    except Exception as e:
                        print(f"Error al cargar {image_path}: {e}")

# Imprimir los resultados de las personas cargadas
print(f"Personas encontradas: {people_names}")
print(f"Codificaciones de rostros: {people_images}")

# Función para comparar el rostro detectado con las imágenes conocidas
def verificar_persona(rostro_detectado):
    for i, person_encoding in enumerate(people_images):
        matches = face_recognition.compare_faces([person_encoding], rostro_detectado)
        # Puedes agregar un umbral para mejorar la comparación, por ejemplo:
        if True in matches:
            print(f"¡Coincidencia encontrada con {people_names[i]}!")
            return people_names[i]
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones del frame
    altura, ancho, _ = frame.shape

    # Procesar el fotograma y detectar objetos con YOLO
    resultados = modelo(frame)

    # Detectar rostros en el frame
    rostro_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = rostro_cascade.detectMultiScale(gray, 1.1, 4)

    for resultado in resultados:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])  # Coordenadas de la caja
            conf = caja.conf[0].item()  # Confianza del modelo
            clase = int(caja.cls[0].item())  # Clase detectada
            confianza2 = conf * 100
            etiqueta = f"{modelo.names[clase]} {confianza2:.2f}" + "%"  # Etiqueta con confianza

            if modelo.names[clase] == "person":
                # Dibujar la caja y la etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Extraer la región de la cara detectada
                for (x, y, w, h) in rostros:
                    rostro = frame[y:y + h, x:x + w]

                    # Convertir el rostro a RGB (face_recognition espera RGB, no BGR como OpenCV)
                    rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)

                    # Convertir el rostro a la codificación
                    rostro_codificado = face_recognition.face_encodings(rostro_rgb)
                    if rostro_codificado:
                        nombre_persona = verificar_persona(rostro_codificado[0])
                        if nombre_persona:
                            cv2.putText(frame, f"{nombre_persona}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 0, 0), 2)

    # Mostrar el video con las detecciones
    cv2.imshow("YOLOv8 - Detección en Tiempo Real", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
