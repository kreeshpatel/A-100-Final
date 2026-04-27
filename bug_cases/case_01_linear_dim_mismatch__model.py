"""
BUG CASE 01: Linear input dimension mismatch.

Drop-in replacement for src/model.py.

ORIGINAL : self.fc1 = nn.Linear(64 * 7 * 7, 128)
MODIFIED : self.fc1 = nn.Linear(64 * 14 * 14, 128)

EXPECTED FAILURE:
    RuntimeError: mat1 and mat2 shapes cannot be multiplied
        (256x3136 and 12544x128)

REPRODUCE:
    cp bug_cases/case_01_linear_dim_mismatch__model.py src/model.py
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
        # BUG: should be 64 * 7 * 7 (= 3136). 64 * 14 * 14 (= 12544) assumes
        # only one pooling stage instead of two.
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
