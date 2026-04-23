import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from phase1.models.cnn import CNN
from phase1.models.transformer import TransformerClassifier


RUNS = 1000
REPEATS = 5
WARMUP_RUNS = 20


def load_model_and_input(model_name: str):
    """
    Load the requested PyTorch model, ONNX model, and matching dummy input.

    Args:
        model_name (str): Either "cnn" or "transformer".

    Returns:
        tuple:
            torch_model: Loaded PyTorch model in eval mode.
            onnx_session: ONNX Runtime session.
            input_torch: Dummy torch input matching the model.
            input_numpy: Same input converted to NumPy float32.
    """
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        torch_model = CNN()
        torch_model.load_state_dict(
            torch.load("phase1/best_cnn.pth", map_location="cpu")
        )
        onnx_path = Path("phase2/models/cnn.onnx")
        input_torch = torch.randn(1, 1, 28, 28)

    elif model_name == "transformer":
        torch_model = TransformerClassifier()
        torch_model.load_state_dict(
            torch.load("phase1/best_transformer.pth", map_location="cpu")
        )
        onnx_path = Path("phase2/models/transformer.onnx")
        input_torch = torch.randn(1, 28, 28)

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Use 'cnn' or 'transformer'."
        )

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Export it first."
        )

    torch_model.eval()
    onnx_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    input_numpy = input_torch.numpy().astype(np.float32)

    return torch_model, onnx_session, input_torch, input_numpy


def warmup(torch_model, onnx_session, input_torch, input_numpy) -> None:
    """
    Run warmup iterations for both PyTorch and ONNX models.

    Args:
        torch_model: PyTorch model.
        onnx_session: ONNX Runtime session.
        input_torch: Torch input tensor.
        input_numpy: NumPy input array.
    """
    input_name = onnx_session.get_inputs()[0].name

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            torch_model(input_torch)
            onnx_session.run(None, {input_name: input_numpy})


def benchmark_pytorch(torch_model, input_torch) -> float:
    """
    Benchmark average PyTorch inference latency.

    Args:
        torch_model: PyTorch model.
        input_torch: Torch input tensor.

    Returns:
        float: Average latency in milliseconds.
    """
    results = []

    with torch.no_grad():
        for _ in range(REPEATS):
            start = time.perf_counter()
            for _ in range(RUNS):
                torch_model(input_torch)
            end = time.perf_counter()
            results.append((end - start) / RUNS)

    return float(np.mean(results) * 1000.0)


def benchmark_onnx(onnx_session, input_numpy) -> float:
    """
    Benchmark average ONNX Runtime inference latency.

    Args:
        onnx_session: ONNX Runtime session.
        input_numpy: NumPy input array.

    Returns:
        float: Average latency in milliseconds.
    """
    input_name = onnx_session.get_inputs()[0].name
    results = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(RUNS):
            onnx_session.run(None, {input_name: input_numpy})
        end = time.perf_counter()
        results.append((end - start) / RUNS)

    return float(np.mean(results) * 1000.0)


def main():
    """
    Entry point for PyTorch vs ONNX benchmarking.

    Usage:
        python -m phase2.benchmarks.benchmark_pytorch_vs_onnx cnn
        python -m phase2.benchmarks.benchmark_pytorch_vs_onnx transformer
    """
    if len(sys.argv) != 2:
        print(
            "Usage: python -m phase2.benchmarks.benchmark_pytorch_vs_onnx "
            "<cnn|transformer>"
        )
        sys.exit(1)

    model_name = sys.argv[1].lower().strip()

    try:
        torch_model, onnx_session, input_torch, input_numpy = load_model_and_input(
            model_name
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    warmup(torch_model, onnx_session, input_torch, input_numpy)

    torch_latency = benchmark_pytorch(torch_model, input_torch)
    onnx_latency = benchmark_onnx(onnx_session, input_numpy)

    print(f"Model: {model_name}")
    print(f"PyTorch Latency: {torch_latency:.4f} ms")
    print(f"ONNX Latency: {onnx_latency:.4f} ms")


if __name__ == "__main__":
    main()