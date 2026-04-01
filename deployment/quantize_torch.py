import torch
import torch.quantization
from models.cnn import CNN

# Load model
model = CNN()
model.load_state_dict(torch.load("best_cnn.pth"))
model.eval()

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},    # quantize only Linear layers
    dtype=torch.qint8
)

print("Quantization complete!")

torch.save(quantized_model.state_dict(),"deployment/cnn_int8.pth")