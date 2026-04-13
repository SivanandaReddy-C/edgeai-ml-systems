import torch
from phase1.models.cnn_optimized import CNNOptimized

# Load model
model = CNNOptimized()
#model.load_state_dict(torch.load("phase1/best_cnn.pth"))
model.eval()

print("PyTorch model loaded")

# Dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# EXPORT
torch.onnx.export(
    model,
    dummy_input,
    "phase2/onnx/cnn_optimized.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,  
    dynamo=False      
)

print("ONNX export successful")