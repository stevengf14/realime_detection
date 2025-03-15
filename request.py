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
        print("Objetos detectados:", resultado["objetos"])
    else:
        print("Error:", response.text)

    cv2.imshow("Cliente - Video en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
