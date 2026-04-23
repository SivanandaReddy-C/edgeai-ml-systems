import sys
import numpy as np
import onnxruntime as ort


def get_model_config(model_name: str) -> dict:
    """
    Return ONNX paths and input shape for the selected model.
    """
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return {
            "fp32_path": "phase2/models/cnn.onnx",
            "int8_path": "phase2/models/cnn_int8.onnx",
            "input_shape": (1, 1, 28, 28),
            "display_name": "CNN",
        }

    if model_name == "transformer":
        return {
            "fp32_path": "phase2/models/transformer.onnx",
            "int8_path": "phase2/models/transformer_int8.onnx",
            "input_shape": (1, 28, 28),
            "display_name": "Transformer",
        }

    raise ValueError("Model must be 'cnn' or 'transformer'")


def compare_models(fp32_path: str, int8_path: str, input_shape: tuple, model_name: str) -> None:
    """
    Compare FP32 and INT8 ONNX model outputs for the same input.
    """
    sess_fp32 = ort.InferenceSession(
        fp32_path,
        providers=["CPUExecutionProvider"]
    )

    try:
        sess_int8 = ort.InferenceSession(
        int8_path,
        providers=["CPUExecutionProvider"]
    )
    except Exception as e:
        print(f"\n{model_name}")
        print("INT8 model could not be loaded (expected for CNN)")
        print(f"Reason: {e}")
        print("-" * 40)
    return

    input_name = sess_fp32.get_inputs()[0].name
    input_data = np.random.randn(*input_shape).astype(np.float32)

    out_fp32 = sess_fp32.run(None, {input_name: input_data})[0]
    out_int8 = sess_int8.run(None, {input_name: input_data})[0]

    # Mean difference
    diff = np.abs(out_fp32 - out_int8).mean()

    # Prediction match
    pred_fp32 = int(np.argmax(out_fp32))
    pred_int8 = int(np.argmax(out_int8))

    print(f"\n{model_name}")
    print(f"Mean difference: {diff:.6f}")
    print(f"FP32 prediction: {pred_fp32}")
    print(f"INT8 prediction: {pred_int8}")
    print(f"Prediction match: {pred_fp32 == pred_int8}")
    print("-" * 40)


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m phase2.validation.compare_models <cnn|transformer>")
        sys.exit(1)

    model_name = sys.argv[1]

    try:
        config = get_model_config(model_name)
        compare_models(
            config["fp32_path"],
            config["int8_path"],
            config["input_shape"],
            config["display_name"]
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()