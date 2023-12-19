from ultralytics import YOLO

# Load model
modelSrc = "clothing-detect.pt"
model = YOLO(modelSrc)

# List images
listImage = ['./images/shirt-test.webp',
            './images/sweater-test.jpg',
            './images/pants-test.webp',
            './images/jacket-test.webp',
            './images/skirt-test.jpg',
            './images/pc-test.jpg']


detection_outputs = model.predict(source=listImage, conf=0.25, save=True)












# # Response object
# result = {}
# result['status'] = 200

# try:
#     detection_outputs = model.predict(source=listImage, conf=0.25, save=False)
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