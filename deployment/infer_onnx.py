import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("deployment/cnn.onnx")

# Prepare input
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

# Run inference
outputs = session.run(None,{"input":input_data})

print("Output shape:",outputs[0].shape)
print("Output:",outputs[0])