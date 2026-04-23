import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from phase1.models.cnn import CNN
from phase1.models.transformer import TransformerClassifier


def load_config(model_name: str) -> dict:
    """
    Return model-specific configuration for PyTorch and ONNX validation.
    """
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "display_name": "CNN",
            "torch_model": CNN(),
            "weights_path": "phase1/best_cnn.pth",
            "onnx_path": "phase2/models/cnn.onnx",
            "input_shape": (1, 1, 28, 28),
        }

    if model_name == "transformer":
        return {
            "display_name": "Transformer",
            "torch_model": TransformerClassifier(),
            "weights_path": "phase1/best_transformer.pth",
            "onnx_path": "phase2/models/transformer.onnx",
            "input_shape": (1, 28, 28),
        }

    raise ValueError("Model must be 'cnn' or 'transformer'")


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.validation.validate_onnx_vs_pytorch <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        config = load_config(model_name)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    weights_path = Path(config["weights_path"])
    onnx_path = Path(config["onnx_path"])

    if not weights_path.exists():
        print(f"Error: PyTorch weights not found: {weights_path}")
        sys.exit(1)

    if not onnx_path.exists():
        print(f"Error: ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Load PyTorch model
    torch_model = config["torch_model"]
    torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    torch_model.eval()

    print(f"PyTorch {config['display_name']} loaded")

    # Load ONNX model
    onnx_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )
    input_name = onnx_session.get_inputs()[0].name

    print("ONNX model loaded")

    # Create same input for both
    input_torch = torch.randn(*config["input_shape"])
    input_numpy = input_torch.numpy().astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        torch_output = torch_model(input_torch).numpy()

    # ONNX inference
    onnx_output = onnx_session.run(None, {input_name: input_numpy})[0]

    # Compare outputs
    mean_difference = np.abs(torch_output - onnx_output).mean()
    max_difference = np.abs(torch_output - onnx_output).max()

    # Prediction comparison
    torch_pred = int(np.argmax(torch_output))
    onnx_pred = int(np.argmax(onnx_output))

    # Print results
    print(f"\nModel: {config['display_name']}")
    print(f"Mean Difference: {mean_difference:.6f}")
    print(f"Max Difference : {max_difference:.6f}")
    print(f"PyTorch Prediction: {torch_pred}")
    print(f"ONNX Prediction   : {onnx_pred}")

    # Validation summary
    if mean_difference < 1e-4:
        print("Numerical match")
    else:
        print("Numerical mismatch detected")

    if torch_pred == onnx_pred:
        print("Prediction match")
    else:
        print("Prediction mismatch detected")


if __name__ == "__main__":
    main()