import time
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession(
    "phase2/models/transformer.onnx",
    providers = ["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# Warmup
warmup_input = np.random.randn(1, 28, 28).astype(np.float32)
for _ in range(20):
    session.run(None, {input_name: warmup_input})

num_runs = 200

pre_times = []
infer_times = []
post_times = []
total_times = []

for _ in range(num_runs):
    t0 = time.perf_counter()

    # Preprocessing
    p0 = time.perf_counter()
    raw_input = np.random.randn(28, 28)
    input_data = raw_input.astype(np.float32).reshape(1, 28, 28)
    p1 = time.perf_counter()

    # Inference
    i0 = time.perf_counter()
    output = session.run(None, {input_name: input_data})[0]
    i1 = time.perf_counter()

    # Postprocessing
    o0 = time.perf_counter()
    pred = int(np.argmax(output))
    confidence = float(np.max(output))
    o1 = time.perf_counter()

    t1 = time.perf_counter()

    pre_times.append(p1 - p0)
    infer_times.append(i1 - i0)
    post_times.append(o1 - o0)
    total_times.append(t1 - t0)

print(f"Average Preprocessing Time: {np.mean(pre_times) * 1000:.4f} ms")
print(f"Average Inference Time :    {np.mean(infer_times) * 1000: .4f} ms")
print(f"Average Postprocessing Time: {np.mean(post_times) * 1000:.4f} ms")
print(f"Average Total Latency      : {np.mean(total_times) * 1000:.4f} ms")