import os

def get_size(path):
    size_bytes = os.path.getsize(path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return size_bytes, size_kb, size_mb

models = {
    "CNN FP32": "phase2/models/cnn.onnx",
    "CNN INT8": "phase2/models/cnn_int8.onnx",
    "Transformer FP32": "phase2/models/transformer.onnx",
    "Transformer INT8": "phase2/models/transformer_int8.onnx",
}

print("\nModel Size Comparison:\n")

for name, path in models.items():
    size_b, size_kb, size_mb = get_size(path)
    print(f"{name:20s}: {size_kb:8.2f} KB | {size_mb:.4f} MB")