
from fastapi import FastAPI, File, UploadFile
import numpy as np
import requests
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}
   

MODEL_URL = "http://localhost:8501/v1/models/my_cnn_model:predict"

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    processed_image = preprocess_image(image)
    
    response = requests.post(MODEL_URL, json={"instances": processed_image.tolist()})
    prediction = response.json()
    
    return {"prediction": prediction}


    