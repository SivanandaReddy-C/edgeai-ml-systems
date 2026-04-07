import torch
from phase1.models.transformer import TransformerClassifier
import os

# Load trained transformer
model = TransformerClassifier()
model.load_state_dict(torch.load("phase1/best_transformer.pth"))
model.eval()

print("Transformer model loaded")

# Define input shape
batch_size = 1
seq_len = 28
input_dim = 28

dummy_input = torch.randn(batch_size, seq_len, input_dim)
print("Dummy input shape:", dummy_input.shape)
out = model(dummy_input)
print("Output shape:", out.shape)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "phase2/onnx/transformer.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)
print("Transformer ONNX export complete")

# Verify file
if os.path.exists("phase2/onnx/transformer.onnx"):
    print("ONNX file created")
else:
    print("Export failed")