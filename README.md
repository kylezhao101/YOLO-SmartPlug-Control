# YOLOv8 Smart Plug Control using Rock-Paper-Scissors Detection

This project uses a YOLOv8 model fine-tuned on two datasets, [Kyle & Terence Data](https://universe.roboflow.com/iat360kasa/iat360-test) and [Rock-Paper-Scissors Computer Vision Project](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw), to detect hand gestures through webcam input and control a TP-Link Kasa smart plug accordingly.

Note:The Roboflow API key is revoked. Replace with your own key.

## Main Files
- **IAT350_YOLO_KASA_plug.ipynb**: This notebook trains the YOLOv8 model using the specified datasets and saves the best model as `best.pt`.
- **main.py**: Uses `best.pt` to perform real-time object detection via a webcam feed, interact with the Kasa Smart Plug library, and trigger the plug based on detected gestures.

## main.py Overview
`main.py` reads webcam input, processes frames with YOLOv8, and identifies rock-paper-scissors gestures. When "Paper" is detected, the smart plug turns on; when "Rock" is detected, the plug turns off. This interaction relies on the **pyHS100** library to control the Kasa Smart Plug.

## Key Libraries
- **ultralytics**: YOLOv8 for object detection.
- **dotenv**: Loads environment variables for secure IP management.
- **pyHS100**: Controls the Kasa Smart Plug.
- **OpenCV (cv2)**: Captures and processes webcam frames for YOLO inference.
