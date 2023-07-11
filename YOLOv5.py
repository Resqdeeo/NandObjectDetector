import torch

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5",'custom', path='weights/yolov5.pt')
model.conf = 0.5

def yolo_model(frame, thresh):
    # Run YOLOv5
    results = model(frame)

    # Visualize the results on the frame
    results.render()

    # Get the detected objects
    objects = results.pandas().xyxy[0]

    # Check if any object is a hand
    yolo_model = False
    for _, obj in objects.iterrows():
        if obj["name"] == "hand" and obj["confidence"] > thresh:
            yolo_model = True
            break

    return yolo_model