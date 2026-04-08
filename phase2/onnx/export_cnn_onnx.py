import torch
from phase1.models.cnn import CNN

# Load model
model = CNN()
model.load_state_dict(torch.load("phase1/best_cnn.pth"))
model.eval()

print("PyTorch model loaded")

# Dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# 🔥 EXPORT (NO TRACE, NO SCRIPT)
torch.onnx.export(
    model,
    dummy_input,
    "phase2/onnx/cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,   # 🔥 match your PyTorch
    dynamo=False        # 🔥 VERY IMPORTANT
)

print("ONNX export successful")