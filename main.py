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

cap = cv.VideoCapture(0)

# Plug Control Functions
print("Current state: %s" % plug.state)
def turn_on():
    if plug.state == "OFF":
        plug.turn_on()
        print("Plug turned ON")

def turn_off():
    if plug.state == "ON":
        plug.turn_off()
        print("Plug turned OFF")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Parameters for control delay and frame processing
CONTROL_DELAY = 2  # Time in seconds between plug control checks
last_control_time = time.time()
MOVEMENT_THRESHOLD = 5000  # Threshold for detecting movement
PROCESS_EVERY_N_FRAMES = 5  # Run YOLO every nth frame if movement is detected
frame_count = 0
CONFIDENCE_THRESHOLD = 0.5

# Read the first frame to initialize movement detection
ret, prev_frame = cap.read()
if not ret:
    print("Can't receive initial frame. Exiting ...")
    cap.release()
    exit()

prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the current frame to grayscale for movement detection
    current_frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_count += 1

    # Calculate absolute difference between the previous and current frame
    frame_diff = cv.absdiff(prev_frame_gray, current_frame_gray)
    movement_score = np.sum(frame_diff)  # Summing pixel differences

    # Update the previous frame to the current frame
    prev_frame_gray = current_frame_gray

    # If movement is detected and it’s the nth frame
    if movement_score > MOVEMENT_THRESHOLD and frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Resize frame for faster YOLO inference
        input_frame = cv.resize(frame, (640, 640))

        # Run YOLO model inference
        results = model(input_frame)

        # Track detection results
        paper_detected, rock_detected = False, False

        # Loop through each detection result
        for box in results[0].boxes:

            confidence = box.conf[0].item()  # Get the confidence score
            if confidence < CONFIDENCE_THRESHOLD:
                continue  # Skip detections below the confidence threshold

            class_id = int(box.cls[0].item())  # Get the class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

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

    # Display the resulting frame
    cv.imshow('frame', frame)

    # Break on 'q' key press
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv.destroyAllWindows()
