import time
import numpy as np
import onnxruntime as ort

# Load sessions
fp32_session = ort.InferenceSession(
    "phase2/models/transformer.onnx",
    providers=["CPUExecutionProvider"]
)

int8_session = ort.InferenceSession(
    "phase2/models/transformer_int8.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = fp32_session.get_inputs()[0].name

# Input
input_data = np.random.randn(1, 28, 28).astype(np.float32)

# Warmup
for _ in range(20):
    fp32_session.run(None, {input_name: input_data})
    int8_session.run(None, {input_name: input_data})

# FP32 benchmark
start = time.perf_counter()
for _ in range(1000):
    fp32_session.run(None, {input_name: input_data})
end = time.perf_counter()
fp32_latency = (end - start) / 1000

# INT8 benchmark
start = time.perf_counter()
for _ in range(1000):
    int8_session.run(None, {input_name: input_data})
end = time.perf_counter()
int8_latency = (end - start) / 1000

# Results
print(f"FP32 Latency: {fp32_latency * 1000: .4f} ms")
print(f"INT8 Latency: {int8_latency * 1000: .4f} ms")

speedup = fp32_latency / int8_latency
print(f"Speedup: {speedup:.2f}x")