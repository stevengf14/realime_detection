# Sistema de Reconocimiento Facial con FaceNet

Este proyecto es una aplicación web para reconocimiento facial en tiempo real usando Flask, PyTorch, FaceNet y MTCNN.

## Características

- Detección y reconocimiento de rostros en tiempo real con redes neuronales profundas
- Interfaz web responsive y amigable
- Seguimiento persistente de rostros detectados
- Marcado de rostros conocidos con nombres y porcentajes de similitud
- Motor de reconocimiento facial basado en embeddings faciales
- Detector MTCNN para localización precisa de rostros
- Modelo InceptionResnetV1 para extracción de características faciales

## Requisitos

- Python 3.8+
- Una cámara web
- GPU opcional (mejora el rendimiento, pero funciona en CPU)
- Las bibliotecas listadas en `requirements.txt`

## Instalación

1. Clonar este repositorio:
```
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```

2. Crear un entorno virtual (recomendado):
```
python -m venv venv
```

3. Activar el entorno virtual:
   - En Windows:
   ```
   venv\Scripts\activate
   ```
   - En macOS/Linux:
   ```
   source venv/bin/activate
   ```

4. Instalar las dependencias:
```
pip install -r requirements.txt
```

**Nota para usuarios de Windows**: Si experimenta problemas al instalar PyTorch, considere visitar la [página oficial de PyTorch](https://pytorch.org/get-started/locally/) para obtener el comando de instalación específico para su sistema.

## Configuración

1. Crear una carpeta llamada `people` en el directorio raíz del proyecto:
```
mkdir people
```

2. Añadir imágenes de rostros para reconocimiento:
   - Opción 1: Colocar imágenes directamente en la carpeta `people`. El nombre del archivo (sin extensión) se utilizará como identificador de la persona.
   - Opción 2: Crear subcarpetas con el nombre de cada persona dentro de `people` y colocar sus fotos dentro.

Las imágenes deben ser claras, bien iluminadas y mostrar el rostro frontal. Formatos soportados: JPG, JPEG, y PNG.

## Uso

1. Iniciar la aplicación:
```
python app.py
```

2. Abrir un navegador web y acceder a:
```
http://localhost:5000
```

3. En la interfaz web:
   - Haga clic en "Iniciar Cámara" para comenzar el reconocimiento
   - Su rostro será detectado y reconocido si está en la base de datos
   - Se mostrará un recuadro verde con su nombre y el porcentaje de similitud
   - Haga clic en "Detener Cámara" cuando termine
   - Use el botón "Actualizar Modelo" para recargar los rostros después de añadir nuevas imágenes
   - Use el botón "Reiniciar" si la aplicación deja de responder

## Estructura del Proyecto

- `app.py`: Aplicación principal de Flask
- `face_recognition_model.py`: Implementación del modelo de reconocimiento facial con FaceNet
- `templates/`: Plantillas HTML para la interfaz web
- `people/`: Carpeta donde se almacenan las imágenes de referencia
- `realTimeDetection.py`: Script original de reconocimiento facial (independiente de la versión web)

## Cómo Funciona

1. **Detección de Rostros**: MTCNN localiza rostros en cada fotograma con alta precisión
2. **Extracción de Características**: InceptionResnetV1 convierte cada rostro en un vector de 512 dimensiones
3. **Comparación**: Se calcula la similitud de coseno entre el rostro detectado y los rostros conocidos
4. **Reconocimiento**: Si la similitud supera un umbral, se identifica a la persona

## Solución de Problemas

- **Error al acceder a la cámara**: Asegúrese de que ninguna otra aplicación esté utilizando la cámara
- **Rostros no reconocidos**: Intente con imágenes de mejor calidad o en diferentes condiciones de iluminación
- **Rendimiento lento**: Reduzca la resolución de la cámara o active GPU si está disponible
- **La aplicación se bloquea**: Use el botón "Reiniciar" o recargue la página

## Personalización

- Modifique `FACE_MEMORY_FRAMES` en `app.py` para ajustar cuánto tiempo permanecen visibles los recuadros de rostros
- Ajuste `FRAME_RESIZE_FACTOR` para equilibrar entre rendimiento y precisión
- Cambie `SIMILARITY_THRESHOLD` para hacer el reconocimiento más o menos estricto

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.
