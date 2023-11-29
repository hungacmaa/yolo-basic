from PIL import Image
from ultralytics import YOLO
import requests
import numpy as np
import cv2

url = "https://firebasestorage.googleapis.com/v0/b/fir-15eec.appspot.com/o/images%2Fao_phao.jpge7d3f729-e9f8-43b2-85f0-2a0cfed814ca?alt=media&token=a9e2e4fb-ddea-4e70-94e4-7f916ac2a710"
response = requests.get(url)
img_array = np.frombuffer(response.content, dtype=np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

modelSrc = "clothing-detect.pt"
model = YOLO(modelSrc)

detection_outputs = model.predict(source=image, conf=0.5, save=True)