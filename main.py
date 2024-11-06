from dotenv import load_dotenv
import os
from ultralytics import YOLO
from pyHS100 import SmartPlug  # https://github.com/GadgetReactor/pyHS100
import cv2 as cv
import numpy as np
import time

# Initialize YOLO model and SmartPlug
model = YOLO('best.pt')
load_dotenv()
kasa_plug_ip = os.getenv('KASA_PLUG_IP')
plug = SmartPlug(kasa_plug_ip)

cap = cv.VideoCapture(0)  # https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

# Plug Control Functions
def turn_on():
    if plug.state == "OFF":
        plug.turn_on()
        print("Plug turned ON")

def turn_off():
    if plug.state == "ON":
        plug.turn_off()
        print("Plug turned OFF")

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Parameters for control delay
CONTROL_DELAY = 2  # Time in seconds between plug control checks
last_control_time = time.time()
CONFIDENCE_THRESHOLD = 0.6

# Read the first frame to initialize
ret, prev_frame = cap.read()
if not ret:
    print("Can't receive initial frame. Exiting ...")
    cap.release()
    exit()

# Capture the original dimensions of the frame
original_height, original_width = prev_frame.shape[:2]
target_size = 640

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLO model inference on the resized frame
    input_frame = cv.resize(frame, (target_size, target_size))
    results = model(input_frame)

    # Calculate scaling factors
    width_scale = original_width / target_size
    height_scale = original_height / target_size

    # Track detection results
    paper_detected, rock_detected = False, False

    # Loop through each detection result
    for box in results[0].boxes:
        confidence = box.conf[0].item()  # Get the confidence score
        if confidence < CONFIDENCE_THRESHOLD:
            continue  # Skip detections below the confidence threshold

        class_id = int(box.cls[0].item())  # Get the class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

        # Scale bounding box coordinates back to original dimensions
        x1 = int(x1 * width_scale)
        y1 = int(y1 * height_scale)
        x2 = int(x2 * width_scale)
        y2 = int(y2 * height_scale)

        # Map class IDs to labels
        if class_id == 0:
            rock_detected = True
            label = "Paper"
        elif class_id == 1:
            paper_detected = True
            label = "Rock"
        elif class_id == 2:
            label = "Scissors"
        else:
            label = "Unknown"

        # Draw the bounding box and label on the frame
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
        cv.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label and confidence

    # Check plug control based on detection results with a delay
    current_time = time.time()
    if current_time - last_control_time > CONTROL_DELAY:
        if paper_detected:
            turn_on()
        elif rock_detected:
            turn_off()
        last_control_time = current_time

    # Add plug status to the frame
    plug_status = f"Plug Status: {plug.state}"
    cv.putText(frame, plug_status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)

    # Break on 'q' key press
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv.destroyAllWindows()
