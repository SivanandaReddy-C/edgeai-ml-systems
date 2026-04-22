import os
import torch
from phase1.models.transformer import TransformerClassifier

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

# Output path
output_path = "phase2/models/transformer.onnx"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=18,
    do_constant_folding=False,
    dynamo=False
)

print("Transformer ONNX export complete")

# Verify file
if os.path.exists(output_path):
    print(f"ONNX file created: {output_path}")
else:
    print("Export failed")