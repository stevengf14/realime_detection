from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Cargar el modelo YOLOv8
modelo_yolo = YOLO("yolov8n.pt")  # Asegúrate de descargar este modelo previamente

@app.post("/detectar-objetos/")
async def detectar_objetos(archivo: UploadFile = File(...)):
    # Leer la imagen del archivo recibido
    contents = await archivo.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Procesar con YOLO
    resultados = modelo_yolo(frame)

    # Extraer las detecciones
    detecciones = []
    for resultado in resultados:
        for box in resultado.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
            conf = float(box.conf[0])  # Confianza de la detección
            clase = int(box.cls[0])  # Clase detectada
            nombre_clase = modelo_yolo.names[clase]  # Nombre del objeto

            detecciones.append({
                "objeto": nombre_clase,
                "confianza": conf,
                "coordenadas": [x1, y1, x2, y2]
            })

    return {"objetos": detecciones}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
