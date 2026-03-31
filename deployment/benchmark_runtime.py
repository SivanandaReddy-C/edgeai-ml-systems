import torch
import time
import numpy as np
import onnxruntime as ort

from models.cnn import CNN

# Load PyTorch model
torch_model = CNN()
torch_model.load_state_dict(torch.load("best_cnn.pth"))
torch_model.eval()

# Load ONNX model
ort_session = ort.InferenceSession("deployment/cnn.onnx")

# Input
input_torch = torch.randn(1, 1, 28, 28)
input_numpy = input_torch.numpy().astype(np.float32)

# Warmup
for _ in range(10):
    torch_model(input_torch)
    ort_session.run(None, {"input":input_numpy})

# Benchmark PyTorch
start = time.time()
for _ in range(100):
    torch_model(input_torch)
end = time.time()
torch_time = (end - start) / 100

# Benchmark ONNX
start = time.time()
for _ in range(100):
    ort_session.run(None,{"input":input_numpy})
end = time.time()
onnx_time = (end - start) / 100

print(f"PyTorch Latency: {torch_time * 1000 : .4f} ms")
print(f"ONNX Latency: {onnx_time * 1000 : .4f} ms")