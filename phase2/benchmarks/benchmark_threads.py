import time
import numpy as np
import onnxruntime as ort

# Thread settings to test
thread_settings = [1, 2, 4, 8]

# Input
input_data = np.random.randn(1, 28, 28).astype(np.float32)

for num_threads in thread_settings:
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads

    session = ort.InferenceSession(
        "phase2/models/transformer.onnx",
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(20):
        session.run(None, {input_name: input_data})

    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        session.run(None, {input_name: input_data})
    end = time.perf_counter()

    latency = (end - start) / 1000

    print(f"\nThreads: {num_threads}")
    print(f"Latency: {latency * 1000:.4f} ms")