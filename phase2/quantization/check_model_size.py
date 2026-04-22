import os

cnn_fp32 = "phase2/models/cnn.onnx"
cnn_int8 = "phase2/models/cnn_int8.onnx"
transformer_fp32 = "phase2/models/transformer.onnx"
transformer_int8 = "phase2/models/transformer_int8.onnx"

cnn_fp32_size = os.path.getsize(cnn_fp32) / 1024
cnn_int8_size = os.path.getsize(cnn_int8) / 1024
transformer_fp32_size = os.path.getsize(transformer_fp32) / 1024
transformer_int8_size = os.path.getsize(transformer_int8) / 1024

print(f"CNN FP32 size: {cnn_fp32_size:.2f} KB")
print(f"CNN INT8 size: {cnn_int8_size:.2f} KB")
print(f"Transformer FP32 size: {transformer_fp32_size:.2f} KB")
print(f"Transformer INT8 size: {transformer_int8_size:.2f} KB")