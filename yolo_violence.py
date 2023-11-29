from ultralytics import YOLO
import json

# Load model
modelSrc = "violence-detect.pt"
model = YOLO(modelSrc)

# List images
listImage = ['images/dn1.jpg', 'images/danhnhau.webp']

# Response object
result = {}
result['status'] = 200

detection_outputs = model.predict(source="images/dn1.jpg", conf=0.25, save=True)

# try:
#     detection_outputs = model.predict(source="images/dn1.jpg", conf=0.25, save=False)
# except Exception  as e:
#     result['message'] = "False to detect" 
#     result['data'] = []
# else:
#     result['message'] = "Successfully" 
#     result['data'] = []

#     for i in range(len(listImage)):
#         object = {}
#         object['url'] = listImage[i]
#         object['boxes'] = json.loads(detection_outputs[i].tojson())
#         result['data'].append(object)
        
# # To json
# result = json.dumps(result, indent=2)

# print(result)