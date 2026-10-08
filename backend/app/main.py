from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.app.predictor import predict_image


app = FastAPI(
    title="AI Image Detector API",
    description="API for detecting whether an image is real or AI-generated.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-image-detector-api",
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model": "Improved CNN",
        "dataset": "CIFAKE",
        "input_size": "32x32 RGB",
        "classes": ["REAL", "FAKE"],
        "test_accuracy": 0.9605,
        "test_f1": 0.9605,
        "test_roc_auc": 0.9933,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)

    result = predict_image(image)

    return {
        "filename": file.filename,
        **result,
    }