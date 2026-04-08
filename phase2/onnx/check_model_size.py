import os

fp32 = "phase2/onnx/cnn.onnx"
int8 = "phase2/onnx/cnn_int8.onnx"

fp32_size = os.path.getsize(fp32) / 1024
int8_size = os.path.getsize(int8) / 1024

print(f"FP32 size: {fp32_size:.2f} KB")
print(f"INT8 size: {int8_size:.2f} KB")