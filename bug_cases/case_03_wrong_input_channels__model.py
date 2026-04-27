"""
BUG CASE 03: First convolution expects 3 input channels instead of 1.

Drop-in replacement for src/model.py.

ORIGINAL : self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, ...)
MODIFIED : self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, ...)

EXPECTED FAILURE:
    RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected
    input[256, 1, 28, 28] to have 3 channels, but got 1 channels instead.

REPRODUCE:
    cp bug_cases/case_03_wrong_input_channels__model.py src/model.py
    python src/train.py
"""

import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # BUG: MNIST is grayscale (1 channel). Configuring the first conv
        # for 3 channels makes the model incompatible with the data.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
