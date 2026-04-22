import time
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("phase2/models/cnn.onnx")

# Get input name dynamically
input_name = session.get_inputs()[0].name

# Create input
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

# Warmup
for _ in range(20):
    session.run(None, {input_name: input_data})

# Benchmark
num_runs = 200

start = time.perf_counter()

for _ in range(num_runs):
    session.run(None, {input_name: input_data})

end = time.perf_counter()

latency = (end - start) / num_runs

# Report
print(f"Average Latency: {latency * 1000: .4f} ms")