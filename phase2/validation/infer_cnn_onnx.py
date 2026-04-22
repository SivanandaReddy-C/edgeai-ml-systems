import torch
import numpy as np
import onnxruntime as ort
from phase1.models.cnn import CNN

# Load PyTorch model
torch_model = CNN()
torch_model.load_state_dict(torch.load("phase1/best_cnn.pth"))
torch_model.eval()

# Load ONNX model
onnx_session = ort.InferenceSession("phase2/onnx/cnn.onnx")

# Get input dynamically
input_name = onnx_session.get_inputs()[0].name

# Create same input for both
input_torch = torch.randn(1, 1, 28, 28)
input_numpy = input_torch.numpy().astype(np.float32)

# PyTorch inference
with torch.no_grad():
    torch_output = torch_model(input_torch).numpy()

# ONNX inference
onnx_output = onnx_session.run(None, {input_name: input_numpy})[0]

# Compare outputs
difference = np.abs(torch_output - onnx_output).mean()

print("PyTorch Output:", torch_output)
print("ONNX Output:", onnx_output)
print(f"Mean Difference: {difference:.6f}")

# Validation check
torch_pred = np.argmax(torch_output)
onnx_pred = np.argmax(onnx_output)

if difference < 1e-4:
    print("Numerical match")
else:
    print("Numerical mismatch detected")

if torch_pred == onnx_pred:
    print("Prediction match")
else:
    print("Prediction mismatch detected")
