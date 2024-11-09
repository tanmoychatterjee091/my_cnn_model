
# CNN Model Deployment with TensorFlow Serving and FastAPI

## Overview
This repository provides a complete pipeline to train, serve, and deploy a CNN model for detecting diabetic retinopathy.

## Requirements
- Docker
- TensorFlow
- FastAPI
- Uvicorn

## Setup Instructions

1. **Train the Model**: Run `model_training.ipynb` to train and save the model.
2. **Build Docker Image**: 
    ```bash
    #docker build -t cnn_model_serving .
    docker-compose up --build
    ```
3. **Run TensorFlow Serving**:
    ```bash
    #docker run -p 8501:8501 cnn_model_serving
    docker run -p 8501:8501 --name=my_cnn_model -v "$(pwd)/saved_model: C:/Users/tanmo/Downloads/My_Projects_in_Github/cnn_deployment_docker_latest/my_cnn_model/models/my_cnn_model" -e MODEL_NAME=my_cnn_model tensorflow/serving
    ```
4. **Run FastAPI**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8501
    uvicorn app:app --reload
    ```
5. **Test Prediction**:
   Use the `/predict` endpoint by uploading an image to get a prediction.
    