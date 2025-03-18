import cv2
import requests

url = "http://localhost:8000/detectar-objetos/"

cap = cv2.VideoCapture(0)  # Captura desde la webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"archivo": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        resultado = response.json()
        if resultado["objetos"]:
            print("Objetos detectados:", resultado["objetos"])
            for resultado_frame in resultado["objetos"]:
                x1, y1, x2, y2 = resultado_frame["coordenadas"]  # Coordenadas de la caja
                conf = resultado_frame["confianza"]  # Confianza del modelo
                clase = resultado_frame["objeto"]  # Clase detectada
                confianza2 = conf * 100
                etiqueta = f"{clase} {confianza2:.2f}" + "%"  # Etiqueta con

                # Color dinámico basado en la confianza
                color = (0, int(255 * conf), int(255 * (1 - conf)))

                # Dibujar la caja y la etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        print("Error:", response.text)

    cv2.imshow("Cliente - Video en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
