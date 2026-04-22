import time
import numpy as np
import onnxruntime as ort

providers = ["CPUExecutionProvider"]

input_data = np.random.randn(1, 28, 28).astype(np.float32)

for provider in providers:
    session = ort.InferenceSession(
        "phase2/models/transformer.onnx",
        providers = [provider]
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(20):
        session.run(None, {input_name: input_data})

    # Benchmark
    start = time.perf_counter()
    for _ in range(500):
        session.run(None, {input_name: input_data})
    end = time.perf_counter()

    latency = (end - start) / 500

    print(f"\nProvider: {provider}")
    print(f"Latency: {latency * 1000:.4f} ms")

    