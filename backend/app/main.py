from fastapi import FastAPI, File, UploadFile
from PIL import Image

from backend.app.predictor import predict_image


app = FastAPI(
    title="AI Image Detector API",
    description="API for detecting whether an image is real or AI-generated.",
    version="1.0.0",
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-image-detector-api",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)

    result = predict_image(image)

    return {
        "filename": file.filename,
        **result,
    }