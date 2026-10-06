import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from improved_model import ImprovedCNN


MODEL_PATH = "models/improved_cnn_best.pth"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image):
        self.model.zero_grad()

        output = self.model(image)

        # Positive class = FAKE
        output.backward()

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam


def load_model():
    model = ImprovedCNN()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    image_resized = image.resize((32, 32))

    image_tensor = torch.from_numpy(
        np.array(image_resized)
    ).float() / 255.0

    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor = (image_tensor - 0.5) / 0.5

    image_tensor = image_tensor.unsqueeze(0)

    return original, image_tensor


def create_visualization(image, cam):
    height, width = image.shape[:2]

    cam_resized = cv2.resize(
        cam,
        (width, height),
    )

    heatmap = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        0.5 * image +
        0.5 * heatmap
    ).astype(np.uint8)

    return heatmap, overlay


def main():
    model = load_model()

    print("Model loaded successfully.")

    # Use the final convolutional layer of the third block.
    target_layer = model.features[19]

    gradcam = GradCAM(
        model,
        target_layer,
    )

    print("Grad-CAM initialized.")

    # Find one test image.
    test_folders = {
        "REAL": Path("data/CIFAKE/test/REAL"),
        "FAKE": Path("data/CIFAKE/test/FAKE"),
    }

    output_dir = Path("results/figures/gradcam")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for true_label, folder in test_folders.items():

        test_images = list(folder.glob("*.jpg"))

        if not test_images:
            test_images = list(folder.glob("*.png"))

        image_path = test_images[0]

        print()
        print(f"True label: {true_label}")
        print(f"Image: {image_path}")

        original, image_tensor = preprocess_image(
            image_path
        )

        output = model(image_tensor)

        probability = torch.sigmoid(output).item()

        prediction = (
            "FAKE"
            if probability >= 0.5
            else "REAL"
        )

        print(f"Prediction: {prediction}")
        print(f"Fake probability: {probability:.4f}")

        cam = gradcam.generate(
            image_tensor
        )

        heatmap, overlay = create_visualization(
            original,
            cam,
        )

        output_path = (
            output_dir /
            f"gradcam_{true_label.lower()}.png"
        )

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title(f"Original ({true_label})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title("Grad-CAM")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f"Prediction: {prediction}")
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    main()