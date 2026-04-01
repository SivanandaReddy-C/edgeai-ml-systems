import torch
import time

from models.cnn import CNN

# Load FP32 model
model_fp32 = CNN()
model_fp32.load_state_dict(torch.load("best_cnn.pth"))
model_fp32.eval()

# Load INT8 model
model_int8 = torch.load("deployment/cnn_int8.pth", weights_only=False)
model_int8.eval()

# Input
input_data = torch.randn(1, 1, 28, 28)

# Warmup
for _ in range(10):
    model_fp32(input_data)
    model_int8(input_data)

# Benchmark FP32
start = time.time()
for _ in range (100):
    model_fp32(input_data)
end = time.time()
fp32_time = (end - start) / 100

# Benchmark INT8
start = time.time()
for _ in range (100):
    model_int8(input_data)
end = time.time()
int8_time = (end - start) / 100

print(f"FP32 Latency: {fp32_time * 1000: .4f} ms")
print(f"INT8 Latency: {int8_time * 1000: .4f} ms")
