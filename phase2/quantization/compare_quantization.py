import numpy as np
import onnxruntime as ort

def compare_models(fp32_path, int8_path, input_shape, model_name):
    sess_fp32 = ort.InferenceSession(
    fp32_path,
    providers=["CPUExecutionProvider"]
    )

    sess_int8 = ort.InferenceSession(
    int8_path,
    providers=["CPUExecutionProvider"]
    )

    input_name = sess_fp32.get_inputs()[0].name

    input_data = np.random.randn(*input_shape).astype(np.float32)

    out_fp32 = sess_fp32.run(None, {input_name: input_data})[0]
    out_int8 = sess_int8.run(None, {input_name: input_data})[0]

    # Mean difference
    diff = np.abs(out_fp32 - out_int8).mean()

    # Prediction match
    pred_fp32 = np.argmax(out_fp32)
    pred_int8 = np.argmax(out_int8)

    print(f"\n{model_name}")
    print(f"Mean difference: {diff:.6f}")
    print(f"Prediction match: {pred_fp32 == pred_int8}")
    print("-" * 40)

# CNN
#compare_models(
#   "phase2/models/cnn.onnx",
#    "phase2/models/cnn_int8.onnx",
#    (1, 1, 28, 28),
#    "CNN"
#)

# Transformer
compare_models(
    "phase2/models/transformer.onnx",
    "phase2/models/transformer_int8.onnx",
    (1, 28, 28),
    "Transformer"
)