"""
BUG CASE 02: Output layer has wrong number of classes.

Drop-in replacement for src/model.py.

ORIGINAL : self.fc2 = nn.Linear(128, 10)
MODIFIED : self.fc2 = nn.Linear(128, 9)

EXPECTED FAILURE:
    IndexError / "Target 9 is out of bounds." raised by CrossEntropyLoss
    on the first batch that contains a label of 9.

REPRODUCE:
    cp bug_cases/case_02_output_dim_mismatch__model.py src/model.py
    python src/train.py
"""

import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        # BUG: MNIST has 10 classes (digits 0-9). With out_features=9,
        # CrossEntropyLoss treats target=9 as out-of-bounds.
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
