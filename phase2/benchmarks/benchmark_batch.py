import sys
import time
import numpy as np
import onnxruntime as ort

WARMUP_RUNS = 20
BENCHMARK_RUNS = 1000


def get_model_config(model_name: str):
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "model_path": "phase2/models/cnn.onnx",
            "input_shape_fn": lambda batch: (batch, 1, 28, 28),
        }
    elif model_name == "transformer":
        return {
            "model_path": "phase2/models/transformer.onnx",
            "input_shape_fn": lambda batch: (batch, 28, 28),
        }
    else:
        raise ValueError("Model must be 'cnn' or 'transformer'")


def benchmark_batch_sizes(model_name: str):
    config = get_model_config(model_name)

    session = ort.InferenceSession(
        config["model_path"],
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    batch_sizes = [1, 2, 4, 8, 16]

    print(f"Model: {model_name}")

    for batch in batch_sizes:
        input_shape = config["input_shape_fn"](batch)
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(WARMUP_RUNS):
            session.run(None, {input_name: input_data})

        # Benchmark
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            session.run(None, {input_name: input_data})
        end = time.perf_counter()

        latency = (end - start) / BENCHMARK_RUNS
        throughput = batch / latency

        print(f"\nBatch size: {batch}")
        print(f"Latency: {latency * 1000:.4f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.benchmarks.benchmark_batch_transformer <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        benchmark_batch_sizes(model_name)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()