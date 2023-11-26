from ultralytics import YOLO
import json
import numpy

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")

# predict on an image
bus_image = "images/bus.jpg"
img0 = "images/img0.JPG"
li = [bus_image, img0]
detection_outputs = model.predict(source=li, conf=0.25, save=False)

# Display tensor array
# resultObject = detection_outputs[1]
# print(resultObject.tojson())

# for output in detection_outputs:
#     # result = output[0]
#     print("result: ")
#     print(output.tojson())
#     print('\n')

# Display numpy arrat
# print(detection_output[0].numpy())
result = {}
result['status'] = 200
result['result'] = []
for i in range(len(li)):
    object = {}
    object['url'] = li[i]
    object['boxes'] = json.loads(detection_outputs[i].tojson())

    result['result'].append(object)

result = json.dump(result, indent=2)
print(result)
# print(json.dumps(result, indent=2))
# print(type(result['result'][0]))