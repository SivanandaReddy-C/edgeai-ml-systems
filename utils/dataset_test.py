import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download MNIST dataset
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Create dataloader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

# Check one batch
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)