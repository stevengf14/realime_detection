import cv2
from ultralytics import YOLO
import os
import face_recognition

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("yolov8x.pt")

# Inicializar la captura de video en tiempo real
cap = cv2.VideoCapture(0)

# Listar imágenes en la carpeta 'people'
people_images = []
people_names = []

# Verificar que la ruta 'people' exista
people_folder = "people"
if not os.path.exists(people_folder):
    print(f"¡Folder '{people_folder}' doesn't exist!")
else:
    # Recorrer las carpetas dentro de 'people'
    for foldername in os.listdir(people_folder):
        folder_path = os.path.join(people_folder, foldername)

        # Asegurarse de que es una carpeta (no un archivo)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {foldername}")

            # Recorrer las imágenes dentro de la subcarpeta
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)

                    try:
                        img = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(img)
                        if encoding:  # Si se encuentra al menos un rostro en la imagen
                            people_images.append(encoding[0])
                            people_names.append(foldername)  # Nombre de la carpeta como nombre de la persona
                        else:
                            print(f"Face not detected in: {image_path}")
                    except Exception as e:
                        print(f"Error uploading {image_path}: {e}")

# Función para comparar el rostro detectado con las imágenes conocidas
def verify_person(detected_face):
    for i, person_encoding in enumerate(people_images):
        matches = face_recognition.compare_faces([person_encoding], detected_face)
        # Puedes agregar un umbral para mejorar la comparación, por ejemplo:
        if True in matches:
            return people_names[i]
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el fotograma y detectar objetos con YOLO
    results = model(frame)

    # Detectar rostros en el frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
            if model.names[int(box.cls[0].item())] == "person":
                # Extraer la región del rostro detectado
                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]

                    # Convertir el rostro a RGB (face_recognition espera RGB, no BGR como OpenCV)
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    # Convertir el rostro a la codificación
                    encodig_face = face_recognition.face_encodings(face_rgb)
                    if encodig_face:
                        person_name = verify_person(encodig_face[0])
                        if person_name:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{person_name}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 0, 0), 2)

    # Mostrar el video con las detecciones
    cv2.imshow("YOLOv8 - Detección en Tiempo Real", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
