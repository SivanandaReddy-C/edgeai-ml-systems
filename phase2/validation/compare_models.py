import time
import numpy as np
import torch
import onnxruntime as ort

from phase1.models.cnn import CNN
from phase1.models.transformer import TransformerClassifier

# Load PyTorch models
cnn = CNN()
cnn.load_state_dict(torch.load("phase1/best_cnn.pth"))
cnn.eval()

transformer = TransformerClassifier()
transformer.load_state_dict(torch.load("phase1/best_transformer.pth"))
transformer.eval()

# Load ONNX models
cnn_model_path = "phase2/models/cnn.onnx"
tr_model_path = "phase2/models/transformer.onnx"

cnn_sess = ort.InferenceSession(cnn_model_path)
tr_sess = ort.InferenceSession(tr_model_path)

cnn_input_name = cnn_sess.get_inputs()[0].name
tr_input_name = tr_sess.get_inputs()[0].name

# Inputs
cnn_input = torch.randn(1, 1, 28, 28)
tr_input = torch.randn(1, 28, 28)

cnn_np = cnn_input.numpy().astype(np.float32)
tr_np = tr_input.numpy().astype(np.float32)

# Warmup
for _ in range(10):
    cnn_sess.run(None, {cnn_input_name: cnn_np})
    tr_sess.run(None, {tr_input_name: tr_np})

# Benchmark function
def benchmark(session, input_name, data, runs=100):
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: data})
    end = time.perf_counter()
    return (end - start) / runs * 1000  # ms

# Run benchmarks
cnn_latency = benchmark(cnn_sess, cnn_input_name, cnn_np)
tr_latency = benchmark(tr_sess, tr_input_name, tr_np)

# Results
print(f"CNN ONNX Latency: {cnn_latency:.4f} ms")
print(f"Transformer ONNX Latency: {tr_latency:.4f} ms")