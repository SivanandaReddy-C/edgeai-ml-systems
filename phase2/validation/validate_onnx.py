import sys
import numpy as np
import onnxruntime as ort


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m phase2.validation.validate_onnx <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print("Input name :", input_info.name)
    print("Input shape:", input_info.shape)
    print("Input type :", input_info.type)

    print("Output name :", output_info.name)
    print("Output shape:", output_info.shape)
    print("Output type :", output_info.type)

    input_shape = [dim if isinstance(dim, int) else 1 for dim in input_info.shape]

    if "float16" in input_info.type:
        x = np.random.rand(*input_shape).astype(np.float16)
    elif "float" in input_info.type:
        x = np.random.rand(*input_shape).astype(np.float32)
    elif "int8" in input_info.type:
        x = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    elif "uint8" in input_info.type:
        x = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported input type: {input_info.type}")

    y = session.run(None, {input_info.name: x})

    print("\nInference successful")
    print("Output shape:", y[0].shape)
    print("Output sample:", y[0])


if __name__ == "__main__":
    main()