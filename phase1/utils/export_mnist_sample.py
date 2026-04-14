from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from phase1.models.cnn import CNN

# Load MNIST 
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Pick one sample
image, label = dataset[0]

print("Ground truth label:", label)

# Convert to numpy
img = image.numpy()  # (1, 28, 28)

# Flatten
img_flat = img.flatten()

# Convert to uint8 (0–255)
img_uint8 = (img_flat * 255).astype(np.uint8)

# Print as C array
print("\nPaste this into STM32 main.c:\n")

print("static const uint8_t digit_image[784] = {")
for i in range(784):
    print(f"{img_uint8[i]},", end='')
    if (i + 1) % 28 == 0:
        print()
print("};")

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.colorbar()
plt.show()



digit_image = np.array([
    # paste the same 784 values from main.c here
], dtype=np.uint8)

x = digit_image.astype(np.float32) / 255.0
x = x.reshape(1, 1, 28, 28)

model = CNN()
model.load_state_dict(torch.load("phase1/best_cnn.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    out = model(torch.from_numpy(x))
    pred = out.argmax(dim=1).item()

print("Python prediction:", pred)
print("Logits:", out.numpy())