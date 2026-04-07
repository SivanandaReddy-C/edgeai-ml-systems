import torch
from phase1.models.cnn import CNN
import os

# Load trained CNN model
model = CNN()
model.load_state_dict(torch.load("phase1/best_cnn.pth"))
model.eval()

print("PyTorch model loaded")

# Create dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "phase2/onnx/cnn.onnx",
    input_names = ["input"],
    output_names = ["output"],
    opset_version = 17
)

print("ONNX export successful")

# Verify file creation
if os.path.exists("phase2/onnx/cnn.onnx"):
    print("ONN file exists")
else:
    print("ONN export failed")