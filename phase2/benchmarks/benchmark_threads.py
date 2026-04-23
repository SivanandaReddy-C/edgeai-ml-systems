import sys
import time
import numpy as np
import onnxruntime as ort

# Thread settings to test
THREAD_SETTINGS = [1, 2, 4, 8]
WARMUP_RUNS = 20
BENCHMARK_RUNS = 1000


def get_model_config(model_name: str):
    """
    Return ONNX model path and input shape based on model name.

    Args:
        model_name (str): 'cnn' or 'transformer'

    Returns:
        dict: model configuration
    """
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "model_path": "phase2/models/cnn.onnx",
            "input_shape": (1, 1, 28, 28),
        }
    elif model_name == "transformer":
        return {
            "model_path": "phase2/models/transformer.onnx",
            "input_shape": (1, 28, 28),
        }
    else:
        raise ValueError("Model must be 'cnn' or 'transformer'")


def benchmark_model(model_name: str):
    """
    Benchmark ONNX Runtime latency across different thread settings.

    Args:
        model_name (str): 'cnn' or 'transformer'
    """
    config = get_model_config(model_name)
    input_data = np.random.randn(*config["input_shape"]).astype(np.float32)

    print(f"Model: {model_name}")

    for num_threads in THREAD_SETTINGS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads

        session = ort.InferenceSession(
            config["model_path"],
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(WARMUP_RUNS):
            session.run(None, {input_name: input_data})

        # Benchmark
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            session.run(None, {input_name: input_data})
        end = time.perf_counter()

        latency = (end - start) / BENCHMARK_RUNS

        print(f"\nThreads: {num_threads}")
        print(f"Latency: {latency * 1000:.4f} ms")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.benchmarks.benchmark_threads <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        benchmark_model(model_name)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()