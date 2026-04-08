from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="phase2/onnx/transformer.onnx",
    model_output="phase2/onnx/transformer_int8.onnx",
    weight_type=QuantType.QInt8,
    per_channel=False 
)

print("Transformer quantization complete")