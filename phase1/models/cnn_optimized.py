import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNOptimized(nn.Module):
    def __init__(self):
        super(CNNOptimized, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier
        self.fc = nn.Linear(32, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.gap(x)          # (32,1,1)
        x = torch.flatten(x, 1)  # (32)

        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = CNNOptimized()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)

    print("Output shape:", y.shape)