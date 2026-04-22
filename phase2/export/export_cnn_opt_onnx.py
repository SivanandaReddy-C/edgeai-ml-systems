import os
import torch
from phase1.models.cnn_optimized import CNNOptimized

# Load model
model = CNNOptimized()
model.load_state_dict(torch.load("phase1/best_cnn_optimized.pth"))
model.eval()

print("PyTorch model loaded")

# Dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# Output path
output_path = "phase2/models/cnn_optimized.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# EXPORT
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamo=False
)

print(f"ONNX export successful: {output_path}")