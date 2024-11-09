
import multipart
import PIL
import fastapi
import tensorflow as tf
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import requests
from PIL import Image
import io


app = FastAPI()

# URL for the TensorFlow Serving model
#MODEL_URL = "http://localhost:8501/v1/models/my_cnn_model:predict"
MODEL_URL = "http://my_cnn_model:8501/v1/models/my_cnn_model:predict"


def preprocess_image(image):
    """Resize and normalize image."""
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/")
async def root():  # read_root
    return {"message": "Welcome to CNN Model API"} # Welcome to PurseXTrack Model API

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image = preprocess_image(image)

        payload = {"instances": image.tolist()}
        response = requests.post(MODEL_URL, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        return {"error": str(e)}




#@app.post("/predict")
#async def predict(file: UploadFile = File(...)):
#    try:
#        # Load and preprocess the image
#        image = Image.open(io.BytesIO(await file.read()))
#        processed_image = preprocess_image(image)
#        
#        # Send a request to the TensorFlow Serving model
#        response = requests.post(MODEL_URL, json={"instances": processed_image.tolist()})
#        
#        # Parse the prediction
#        if response.status_code == 200:
#            prediction = response.json()
#            return {"prediction": prediction}
#        else:
#            raise HTTPException(status_code=response.status_code, detail=response.text)
#            # return {"error": f"Failed to get prediction, status code: {response.status_code}", "details": response.text}
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
#        # return {"error": str(e)}
 

#@app.get("/")
#async def read_root():
#    return {"Hello": "World"}
   
#MODEL_URL = "http://localhost:8501/v1/models/my_cnn_model:predict"

#@app.get("/")
#async def preprocess_image(image):
#    image = image.resize((128, 128))
#    image = np.array(image) / 255.0
#    image = np.expand_dims(image, axis=0)
#    return image

#@app.post("/predict")
#async def predict(data: dict):
#    response = requests.post("http://localhost:8501/v1/models/my_cnn_model:predict", json=data)
#    if response.status_code == 200:
#        return response.json()
#    else:
#        raise HTTPException(status_code=response.status_code, detail="Prediction failed.")

#@app.post("/predict")
#async def predict(data: dict):
#    url = "http://localhost:8501/v1/models/my_cnn_model:predict"
#    response = requests.post(url, json=data)
#    if response.status_code != 200:
#        raise HTTPException(status_code=response.status_code, detail=response.text)
#    return response.json()

#@app.post("/predict")
#async def predict(file: UploadFile = File(...)):
#    image = Image.open(io.BytesIO(await file.read()))
#    processed_image = preprocess_image(image)
    
#    response = requests.post(MODEL_URL, json={"instances": processed_image.tolist()})
#    prediction = response.json()
    
#    return {"prediction": prediction}


    