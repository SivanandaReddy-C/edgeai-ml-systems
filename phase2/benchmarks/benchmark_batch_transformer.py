import time
import numpy as np
import onnxruntime as ort

# Load model (FP32 only)
session = ort.InferenceSession(
    "phase2/models/transformer.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# Batch sizes to test
batch_sizes = [1, 2, 4, 8, 16]

for batch in batch_sizes:
    input_data = np.random.randn(batch, 28, 28).astype(np.float32)

    # Warmup
    for _ in range(20):
        session.run(None, {input_name: input_data})

    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        session.run(None, {input_name: input_data})
    end = time.perf_counter()

    latency = (end - start) / 1000
    througput = batch / latency

    print(f"\nBatch size: {batch}")
    print(f"Latency: {latency * 1000:.4f} ms")
    print(f"Throughput: {througput:.2f} samples/sec")
