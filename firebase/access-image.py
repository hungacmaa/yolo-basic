import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2
from ultralytics import YOLO

cred = credentials.Certificate("./firebase/key.json")
app = firebase_admin.initialize_app(cred, { 'storageBucket' : 'wdyt-c4821.appspot.com' })

bucket = storage.bucket()
blob = bucket.get_blob("ao_phao.jpg")
arr = np.frombuffer(blob.download_as_string(), np.uint8) # array of bytes
img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555) # actual image

# print(type(img))
# cv2.imshow('image', img)
# cv2.waitKey(0)

modelSrc = "clothing-detect.pt"
model = YOLO(modelSrc)

detection_outputs = model.predict(source=img, conf=0.5, save=True)