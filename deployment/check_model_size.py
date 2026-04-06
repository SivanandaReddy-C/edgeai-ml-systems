import os

files = [
    "deployment/cnn.onnx",
    "deployment/cnn_int8.onnx"
]

for f in files:
    size = os.path.getsize(f) / (1024 * 1024)
    print(f"{f}: {size:.4f} MB")
