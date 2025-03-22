# Sistema de Reconocimiento Facial

Este proyecto es una aplicación web para reconocimiento facial en tiempo real usando Flask, OpenCV y face_recognition.

## Características

- Detección y reconocimiento de rostros en tiempo real
- Interfaz web responsive y amigable
- Seguimiento persistente de rostros detectados
- Marcado de rostros conocidos con nombres y porcentajes de confianza

## Requisitos

- Python 3.8+
- Una cámara web
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

**Nota para usuarios de Windows**: La instalación de `dlib` puede ser complicada. Si encuentra problemas, considere descargar un archivo wheel precompilado de [este repositorio](https://github.com/jloh02/dlib/releases).

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
   - Se mostrará un recuadro verde con su nombre y el porcentaje de confianza
   - Haga clic en "Detener Cámara" cuando termine

## Estructura del Proyecto

- `app.py`: Aplicación principal de Flask
- `templates/`: Plantillas HTML para la interfaz web
- `people/`: Carpeta donde se almacenan las imágenes de referencia
- `realTimeDetection.py`: Script original de reconocimiento facial (independiente de la versión web)

## Solución de Problemas

- **Error al acceder a la cámara**: Asegúrese de que ninguna otra aplicación esté utilizando la cámara
- **Rostros no reconocidos**: Intente con imágenes de mejor calidad o en diferentes condiciones de iluminación
- **Error con imágenes**: Si algunas imágenes no se cargan correctamente, intente convertirlas a formato JPG

## Personalización

- Modifique `FACE_MEMORY_FRAMES` en `app.py` para ajustar cuánto tiempo permanecen visibles los recuadros de rostros
- Ajuste `FRAME_RESIZE_FACTOR` para equilibrar entre rendimiento y precisión

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.
