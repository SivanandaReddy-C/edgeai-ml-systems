import torch
from models.transformer import TransformerClassifier

# Load model
model = TransformerClassifier()
model.load_state_dict(torch.load("best_transformer.pth"))
model.eval()

# Dummy input 
dummy_input = torch.randn(1, 28, 28)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "deployment/transformer.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("Transformer exported to ONNX")