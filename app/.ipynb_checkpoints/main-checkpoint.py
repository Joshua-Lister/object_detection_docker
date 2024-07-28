import io
import uvicorn
import numpy as np
import nest_asyncio
import cv2
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

dir_name = "uploaded_images"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

app = FastAPI(title='Object detection')

model_path = './app/best.pt'  # Replace with the path to your custom weights file
model = YOLO(model_path)

@app.get("/")
def home():
    return "Api is working"


@app.post("/predict") 
def prediction(file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    
    results = model(image)
    
    # Extract bounding boxes, labels, and confidences
    bbox = results[0].boxes.xyxy.tolist()  # Extract bounding boxes
    conf = results[0].boxes.conf.tolist()  # Extract confidence scores
    label = [model.names[int(cls)] for cls in results[0].boxes.cls]  # Extract class labels

    # Draw bounding boxes and labels on the image
    output_image = image.copy()
    for box, lbl, cf in zip(bbox, label, conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, f'{lbl} {cf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the result image
    output_path = f'{dir_name}/{filename}'
    cv2.imwrite(output_path, output_image)
    
    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    file_image = open(output_path, mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")