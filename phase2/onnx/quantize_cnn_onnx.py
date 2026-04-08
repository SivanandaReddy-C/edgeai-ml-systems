import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnx import shape_inference

# Load ONNX model
model_path = "phase2/onnx/cnn.onnx"
model = onnx.load(model_path)

# Run shape inference manually (FIX)
model = shape_inference.infer_shapes(model)

# Save updated model
onnx.save(model, model_path)

# Now quantize
quantize_dynamic(
    model_input=model_path,
    model_output="phase2/onnx/cnn_int8.onnx",
    weight_type=QuantType.QInt8
)

print("Quantization complete")