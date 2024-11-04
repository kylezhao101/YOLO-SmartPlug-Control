from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n classification model
model.train(data="C:/Users/kylez/projects/IAT360-kasa/data/rcs/data.yaml", epochs=20)  # train the model

metrics = model.val(
    data="C:/Users/kylez/projects/IAT360-kasa/data/rcs/data.yaml",  # Path to your dataset YAML file
    save=True,                                         # Save evaluation predictions
    conf=0.25,                                         # Confidence threshold for predictions
    iou=0.6                                            # IoU threshold for NMS
)

# Print the evaluation metrics
print(metrics)
