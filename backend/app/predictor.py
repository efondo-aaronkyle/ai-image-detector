from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ml.improved_model import ImprovedCNN


MODEL_PATH = Path("models/improved_cnn_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def load_model():
    model = ImprovedCNN()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        fake_probability = torch.sigmoid(logit).item()

    if fake_probability >= 0.5:
        label = "FAKE"
    else:
        label = "REAL"

    confidence = (
        fake_probability
        if label == "FAKE"
        else 1.0 - fake_probability
    )

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(1.0 - fake_probability, 4),
    }