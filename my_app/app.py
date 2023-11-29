from flask import Flask, jsonify, request
import requests
import cv2
import numpy as np
from ultralytics import YOLO
import json

modelSrc = "clothing-detect.pt"
model = YOLO(modelSrc)

app = Flask(__name__)

# http://localhost:5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    response = {}
    response['status'] = 200
    response['data'] = []
    try:
        # Lấy đường dẫn ảnh từ request
        img_urls = request.json.get('img_urls')

        response['message'] = "Detect sucessfully"

        for url in img_urls:

            object = {}
            object['url'] = url

            img = requests.get(url) # download ảnh
            img_array = np.frombuffer(img.content, dtype=np.uint8) # chuyển thành array bytes
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # dùng opencv đọc

            output = model.predict(source=image, conf=0.25, save=False)

            object['boxes'] = json.loads(output[0].tojson())

            response['data'].append(object)

        return jsonify(response)

    except Exception as e:
        response['message'] = str(e)
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
