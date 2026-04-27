"""
CNN model for MNIST handwritten digit classification.
"""

import torch.nn as nn


class MNISTNet(nn.Module):
    """
    Simple Convolutional Neural Network for classifying 28x28 grayscale digits.

    Architecture:
        Conv2d(1,32,3,padding=1) -> ReLU -> MaxPool(2)
        Conv2d(32,64,3,padding=1) -> ReLU -> MaxPool(2)
        Flatten
        Linear(64*7*7, 128) -> ReLU
        Linear(128, 10)
    """

    def __init__(self):
        super().__init__()

        # First convolutional block: 1 input channel -> 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 28x28 -> 14x14

        # Second convolutional block: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 14x14 -> 7x7

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv block 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Conv block 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # Classifier
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)  # raw logits, no softmax
        return x
