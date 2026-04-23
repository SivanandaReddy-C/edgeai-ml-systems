import sys
import time
import numpy as np
import onnxruntime as ort

WARMUP = 20
RUNS = 1000


def get_model_config(model_name: str):
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "fp32_path": "phase2/models/cnn.onnx",
            "int8_path": "phase2/models/cnn_int8.onnx",
            "input_shape": (1, 1, 28, 28),
            "display_name": "CNN"
        }

    elif model_name == "transformer":
        return {
            "fp32_path": "phase2/models/transformer.onnx",
            "int8_path": "phase2/models/transformer_int8.onnx",
            "input_shape": (1, 28, 28),
            "display_name": "Transformer"
        }

    else:
        raise ValueError("Model must be 'cnn' or 'transformer'")


def benchmark(session, input_name, input_data):
    start = time.perf_counter()
    for _ in range(RUNS):
        session.run(None, {input_name: input_data})
    end = time.perf_counter()

    return (end - start) / RUNS


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.benchmarks.benchmark_onnxFP32_vs_onnxINT8 <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]

    config = get_model_config(model_name)

    print(f"\nModel: {config['display_name']}")

    # Load FP32 session
    fp32_session = ort.InferenceSession(
        config["fp32_path"],
        providers=["CPUExecutionProvider"]
    )

    input_name = fp32_session.get_inputs()[0].name
    input_data = np.random.randn(*config["input_shape"]).astype(np.float32)

    # Warmup FP32
    for _ in range(WARMUP):
        fp32_session.run(None, {input_name: input_data})

    fp32_latency = benchmark(fp32_session, input_name, input_data)

    print(f"FP32 Latency: {fp32_latency * 1000:.4f} ms")

    # Try INT8
    try:
        int8_session = ort.InferenceSession(
            config["int8_path"],
            providers=["CPUExecutionProvider"]
        )

        # Warmup INT8
        for _ in range(WARMUP):
            int8_session.run(None, {input_name: input_data})

        int8_latency = benchmark(int8_session, input_name, input_data)

        print(f"INT8 Latency: {int8_latency * 1000:.4f} ms")

        if int8_latency > 0:
            speedup = fp32_latency / int8_latency
            print(f"Speedup: {speedup:.2f}x")

    except Exception as e:
        print("\nINT8 model could not be executed")
        print(f"Reason: {e}")

    print("-" * 40)


if __name__ == "__main__":
    main()