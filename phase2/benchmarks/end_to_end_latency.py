import sys
import time
import numpy as np
import onnxruntime as ort

WARMUP_RUNS = 20
NUM_RUNS = 200


def get_model_config(model_name: str):
    """
    Return model-specific ONNX path and preprocessing behavior.
    """
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "model_path": "phase2/models/cnn.onnx",
            "warmup_shape": (1, 1, 28, 28),
            "prepare_input": prepare_cnn_input,
        }
    elif model_name == "transformer":
        return {
            "model_path": "phase2/models/transformer.onnx",
            "warmup_shape": (1, 28, 28),
            "prepare_input": prepare_transformer_input,
        }
    else:
        raise ValueError("Model must be 'cnn' or 'transformer'")


def prepare_cnn_input():
    """
    Simulate preprocessing for CNN input.
    Output shape: (1, 1, 28, 28)
    """
    raw_input = np.random.randn(28, 28)
    input_data = raw_input.astype(np.float32).reshape(1, 1, 28, 28)
    return input_data


def prepare_transformer_input():
    """
    Simulate preprocessing for Transformer input.
    Output shape: (1, 28, 28)
    """
    raw_input = np.random.randn(28, 28)
    input_data = raw_input.astype(np.float32).reshape(1, 28, 28)
    return input_data


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.benchmarks.end_to_end_latency <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]
    config = get_model_config(model_name)

    # Load model
    session = ort.InferenceSession(
        config["model_path"],
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    warmup_input = np.random.randn(*config["warmup_shape"]).astype(np.float32)
    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: warmup_input})

    pre_times = []
    infer_times = []
    post_times = []
    total_times = []

    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()

        # Preprocessing
        p0 = time.perf_counter()
        input_data = config["prepare_input"]()
        p1 = time.perf_counter()

        # Inference
        i0 = time.perf_counter()
        output = session.run(None, {input_name: input_data})[0]
        i1 = time.perf_counter()

        # Postprocessing
        o0 = time.perf_counter()
        pred = int(np.argmax(output))
        confidence = float(np.max(output))
        _ = (pred, confidence)  # keep variables intentionally used
        o1 = time.perf_counter()

        t1 = time.perf_counter()

        pre_times.append(p1 - p0)
        infer_times.append(i1 - i0)
        post_times.append(o1 - o0)
        total_times.append(t1 - t0)

    print(f"Model: {model_name}")
    print(f"Average Preprocessing Time : {np.mean(pre_times) * 1000:.4f} ms")
    print(f"Average Inference Time     : {np.mean(infer_times) * 1000:.4f} ms")
    print(f"Average Postprocessing Time: {np.mean(post_times) * 1000:.4f} ms")
    print(f"Average Total Latency      : {np.mean(total_times) * 1000:.4f} ms")


if __name__ == "__main__":
    main()