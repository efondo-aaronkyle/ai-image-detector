# AI Image Detector

A CNN-based image classification system for detecting whether an image is real or AI-generated.

## Dataset

This project uses the CIFAKE: Real and AI-Generated Synthetic Images dataset.

- Total images: 120,000
- Real images: 60,000
- AI-generated images: 60,000
- Image size: 32 × 32 RGB

## Planned Models

1. Custom CNN — baseline
2. Improved CNN
3. ResNet18 — transfer learning

## Evaluation

Models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

Grad-CAM will be used for model explainability.

## Technology Stack

- Python
- PyTorch
- torchvision
- scikit-learn
- FastAPI
- React
- TypeScript
- Docker

## Project Status

Currently setting up the dataset and PyTorch training pipeline.