import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for CIFAKE.

    Input:
        RGB image, 3 x 32 x 32

    Output:
        Single logit:
            positive -> FAKE
            negative -> REAL
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


if __name__ == "__main__":
    model = BaselineCNN()

    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)

    print(model)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nRaw logits:")
    print(output)

    print("\nNumber of parameters:")
    print(f"{sum(p.numel() for p in model.parameters()):,}")