import sys
import time
import numpy as np
import onnxruntime as ort

PROVIDERS = ["CPUExecutionProvider"]
WARMUP_RUNS = 20
BENCHMARK_RUNS = 500


def get_model_config(model_name: str):
    """
    Return model path and input shape for the selected model.
    """
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "display_name": "CNN",
            "model_path": "phase2/models/cnn.onnx",
            "input_shape": (1, 1, 28, 28),
        }

    if model_name == "transformer":
        return {
            "display_name": "Transformer",
            "model_path": "phase2/models/transformer.onnx",
            "input_shape": (1, 28, 28),
        }

    raise ValueError("Model must be 'cnn' or 'transformer'")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.benchmarks.benchmark_provider <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        config = get_model_config(model_name)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    input_data = np.random.randn(*config["input_shape"]).astype(np.float32)

    print(f"Model: {config['display_name']}")

    for provider in PROVIDERS:
        session = ort.InferenceSession(
            config["model_path"],
            providers=[provider]
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

        print(f"\nProvider: {provider}")
        print(f"Latency: {latency * 1000:.4f} ms")


if __name__ == "__main__":
    main()