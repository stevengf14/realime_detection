# Face Recognition using YOLOv8

## Overview
This project implements face recognition using YOLOv8 for real-time detection and identification. YOLOv8 is used for face detection, while a face recognition model is applied to match detected faces against a known database.

## Features
- Real-time face detection using YOLOv8.
- Face recognition using deep learning.
- Works with live video streams or image inputs.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-recognition-yolo8.git
   cd face-recognition-yolo8
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model weights**
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8/yolov8n.pt
   ```

## Usage

### Running Face Detection and Recognition
```bash
python detect.py --source 0 --model yolov8n.pt --face_db faces/
```
- `--source`: Set input source (0 for webcam, path for image/video file).
- `--model`: Path to the YOLOv8 model weights.
- `--face_db`: Directory containing known faces for recognition.

### Training Custom Face Recognition Model
If you want to fine-tune the recognition model, prepare labeled images and run:
```bash
python train.py --data face_dataset/ --epochs 50
```

## Project Structure
```
face-recognition-yolo8/
│── models/               # Pretrained and custom models
│── data/                 # Sample images/videos
│── faces/                # Known faces database
│── detect.py             # Face detection script
│── train.py              # Training script for face recognition
│── requirements.txt      # Required Python packages
│── README.md             # Project documentation
```

## Dependencies
- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- Torch & Torchvision
- Face recognition libraries (e.g., `face_recognition`)

## References
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Face Recognition Python](https://github.com/ageitgey/face_recognition)

## License
This project is licensed under the MIT License.
