import urllib.request 
from PIL import Image
from ultralytics import YOLO
  
# Retrieving the resource located at the URL 
# and storing it in the file name a.png 
url = "https://firebasestorage.googleapis.com/v0/b/fir-15eec.appspot.com/o/images%2Fao_phao.jpge7d3f729-e9f8-43b2-85f0-2a0cfed814ca?alt=media&token=a9e2e4fb-ddea-4e70-94e4-7f916ac2a710"
urllib.request.urlretrieve(url, "ao_phao.jpg") 
  
# Opening the image and displaying it (to confirm its presence) 
img = Image.open(r"ao_phao.jpg") 

modelSrc = "clothing-detect.pt"
model = YOLO(modelSrc)

detection_outputs = model.predict(source=img, conf=0.5, save=False)