import torch
import numpy as np
import onnxruntime as ort
from phase1.models.transformer import TransformerClassifier

# Load PyTorch model
torch_model = TransformerClassifier()
torch_model.load_state_dict(torch.load("phase1/best_transformer.pth"))
torch_model.eval()

print("PyTorch Transformer loaded")

# Load ONNX model
onnx_model_path = "phase2/models/transformer.onnx"
onnx_session = ort.InferenceSession(onnx_model_path)

input_name = onnx_session.get_inputs()[0].name

print("ONNX model loaded")

# Create SAME input
input_torch = torch.randn(1, 28, 28)
input_numpy = input_torch.numpy().astype(np.float32)

# PyTorch inference
with torch.no_grad():
    torch_output = torch_model(input_torch).numpy()

# ONNX inference
onnx_output = onnx_session.run(None, {input_name: input_numpy})[0]

# Compare outputs
difference = np.abs(torch_output - onnx_output).mean()

# Prediction comparison
torch_pred = np.argmax(torch_output)
onnx_pred = np.argmax(onnx_output)

# Print results
print(f"Mean Difference: {difference:.6f}")
print(f"PyTorch Prediction: {torch_pred}")
print(f"ONNX Prediction: {onnx_pred}")

# Validation
if difference < 1e-4:
    print("Numerical match")

if torch_pred == onnx_pred:
    print("Prediction match")
else:
    print("Prediction mismatch")