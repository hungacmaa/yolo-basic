from ultralytics import YOLO
import json

# Load model
modelSrc = "clothing-detect.pt"
model = YOLO(modelSrc)

# List images
listImage = ['https://static-images.vnncdn.net/files/publish/2022/9/3/bien-vo-cuc-thai-binh-346.jpeg', 'https://image.bnews.vn/MediaUpload/Org/2021/04/16/94881840-3512380665444994-5140652141903347712-n.jpg']


detection_outputs = model.predict(source='https://static-images.vnncdn.net/files/publish/2022/9/3/bien-vo-cuc-thai-binh-346.jpeg', conf=0.25, save=False)
# Response object
result = {}
result['status'] = 200

try:
    detection_outputs = model.predict(source=listImage, conf=0.25, save=False)
except Exception  as e:
    result['message'] = "False to detect" 
    result['data'] = []
else:
    result['message'] = "Successfully" 
    result['data'] = []

    for i in range(len(listImage)):
        object = {}
        object['url'] = listImage[i]
        object['boxes'] = json.loads(detection_outputs[i].tojson())
        result['data'].append(object)
        
# To json
result = json.dumps(result, indent=2)

print(result)