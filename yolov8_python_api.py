from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run batched inference on a list of images
results = model(['images/img0.JPG', 'images/bus.jpg'])  # return a list of Results objects

print('\n')
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    print("result: ")
    print(result)
    print('\n')
