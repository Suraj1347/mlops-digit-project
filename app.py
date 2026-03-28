from fastapi import FastAPI, UploadFile, File
import pickle
import numpy as np
import cv2

app = FastAPI(title="Multi-Digit Recognition API")

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict/")
async def predict_multiple_digits(file: UploadFile = File(...)):
    # 1. Read the uploaded image using OpenCV
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # 2. Thresholding: Make the image pure black and white (invert to white text on black background)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 3. Find Contours: Draw invisible boxes around every shape it finds
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Filter out tiny specks of dust/noise (must be at least 5x10 pixels)
    bounding_boxes = [b for b in bounding_boxes if b[2] > 5 and b[3] > 10]
    
    # Sort the boxes from left to right so we read the numbers in the correct order!
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    predictions = []
    
    # 4. Loop through every number it found
    for (x, y, w, h) in bounding_boxes:
        # Cut out the number and add a small border
        pad = 5
        roi = thresh[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        
        if roi.size == 0: continue
            
        # Resize the cut-out to 8x8 to match our AI model
        resized = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Scale to 0-16 format and flatten
        scaled = (resized / 255.0) * 16
        flattened = scaled.reshape(1, -1)
        
        # Ask the AI to predict this specific digit
        pred = model.predict(flattened)
        predictions.append(int(pred[0]))
        
    return {"filename": file.filename, "detected_numbers": predictions}