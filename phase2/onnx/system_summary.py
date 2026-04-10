import os
import time
import numpy as np
import onnxruntime as ort


MODEL_PATH = "phase2/onnx/transformer.onnx"


def get_model_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024


def measure_load_time_ms(path: str, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"]
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


def measure_inference_latency_ms(path: str, runs: int = 200) -> float:
    session = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(1, 28, 28).astype(np.float32)

    for _ in range(20):
        session.run(None, {input_name: input_data})

    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: input_data})
    end = time.perf_counter()

    return ((end - start) / runs) * 1000


def measure_end_to_end_latency_ms(path: str, runs: int = 200) -> tuple[float, float, float, float]:
    session = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    warmup_input = np.random.randn(1, 28, 28).astype(np.float32)
    for _ in range(20):
        session.run(None, {input_name: warmup_input})

    pre_times = []
    infer_times = []
    post_times = []
    total_times = []

    for _ in range(runs):
        t0 = time.perf_counter()

        p0 = time.perf_counter()
        raw_input = np.random.randn(28, 28)
        input_data = raw_input.astype(np.float32).reshape(1, 28, 28)
        p1 = time.perf_counter()

        i0 = time.perf_counter()
        output = session.run(None, {input_name: input_data})[0]
        i1 = time.perf_counter()

        o0 = time.perf_counter()
        _ = int(np.argmax(output))
        _ = float(np.max(output))
        o1 = time.perf_counter()

        t1 = time.perf_counter()

        pre_times.append(p1 - p0)
        infer_times.append(i1 - i0)
        post_times.append(o1 - o0)
        total_times.append(t1 - t0)

    return (
        np.mean(pre_times) * 1000,
        np.mean(infer_times) * 1000,
        np.mean(post_times) * 1000,
        np.mean(total_times) * 1000,
    )


def main():
    size_kb = get_model_size_kb(MODEL_PATH)
    load_time_ms = measure_load_time_ms(MODEL_PATH)
    infer_ms = measure_inference_latency_ms(MODEL_PATH)
    pre_ms, infer_e2e_ms, post_ms, total_ms = measure_end_to_end_latency_ms(MODEL_PATH)

    print("\n=== System Summary: Transformer FP32 ===\n")
    print(f"Model size           : {size_kb:.2f} KB")
    print(f"Load time            : {load_time_ms:.2f} ms")
    print(f"Inference latency    : {infer_ms:.4f} ms")
    print(f"Preprocessing time   : {pre_ms:.4f} ms")
    print(f"Inference time       : {infer_e2e_ms:.4f} ms")
    print(f"Postprocessing time  : {post_ms:.4f} ms")
    print(f"End-to-end latency   : {total_ms:.4f} ms")


if __name__ == "__main__":
    main()

