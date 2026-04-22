import torch
import time
import numpy as np
import onnxruntime as ort
from phase1.models.cnn import CNN


# Load PyTorch model
torch_model = CNN()
torch_model.load_state_dict(torch.load("phase1/best_cnn.pth"))
torch_model.eval()

# Load ONNX model
onnx_session  = ort.InferenceSession("phase2/models/cnn.onnx")
input_name = onnx_session.get_inputs()[0].name

# Input
input_torch = torch.randn(1, 1, 28, 28)
input_numpy = input_torch.numpy().astype(np.float32)

# Warmup
for _ in range(20):
    torch_model(input_torch)
    onnx_session.run(None,{input_name: input_numpy})

# PyTorch Benchmark
runs = 1000

results = []

with torch.no_grad():
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(runs):
            torch_model(input_torch)
        end = time.perf_counter()

        results.append((end - start) / runs)

torch_latency = np.mean(results)

results = []

# ONNX Benchmark
for _ in range(5):
    start = time.perf_counter()
    for _ in range(runs):
        onnx_session.run(None, {input_name: input_numpy})
    end = time.perf_counter()

    results.append((end - start) / runs)

onnx_latency = np.mean(results)

# Results
print(f"PyTorch_Latency: {torch_latency * 1000:.4f} ms")
print(f"ONNX_Latency: {onnx_latency * 1000:.4f} ms")