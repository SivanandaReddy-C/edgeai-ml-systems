# Edge AI ML Systems
## Phase 1 — Model Development & Benchmarking

## Project Overview

This project implements an end-to-end ML systems pipeline using CNN and Transformer models from scratch. 

The focus is on system-level understanding including training workflows, performance benchmarking, profiling, and analyzing computational tradeoffs between different model architectures. 


## Phase 1 Objectives

- Implement CNN training pipeline
- Implement Transformer Encoder from scratch
- Profile training performance
- Benchmark CNN vs Transformer

## System Pipeline
![Pipeline](docs/system_pipeline1.png)

## Phase 2 - Deployment Pipeline
PyTorch → ONNX → Runtime → Quantization → Edge Deployment  
Goal: Optimize trained models for real-world inference performance

## ONNX Deployment

### Model Export
- Exported CNN and Transformer models from PyTorch to ONNX format
- Used dummy inputs to trace computation graphs

### Validation
- Verified ONNX model structure using ONNX checker
- Ensured compatibility before inference

### Inference (ONNX Runtime)
- Ran inference using ONNX Runtime (C++ backend)
- Used NumPy arrays instead of PyTorch tensors

### Performance Benchmark
| Model | PyTorch Latency | ONNX Latency |
|------|----------------|-------------|
| CNN  | 5.6070 ms      | 0.1009 ms   |

### Key Insights
- ONNX Runtime significantly reduces inference latency by eliminating Python overhead
- Static graph execution enables runtime optimizations
- Proper validation prevents runtime errors
- Input/output names must match exactly during inference
- Warmup iterations are necessary for accurate benchmarking

## Quantization

### Objective
Optimize model inference performance by reducing numerical precision from FP32 to INT8.

### Motivation
- Reduce model size (~4x smaller)
- Improve inference latency
- Enable efficient deployment on resource-constrained devices

### Approach
- Applied dynamic quantization on PyTorch models
- Focused on Linear layers (most impactful for Transformer)

### Tradeoff
- Achieved faster inference with minimal accuracy degradation

### Key Insight
Quantization complements ONNX optimization by reducing computation cost, not just execution overhead.

### Observation

| Model | FP32 Latency | INT8 Latency |
|------|--------------|--------------|
| CNN  | 0.2623 ms    | 0.5262 ms    |

### Insight

Quantization did not improve performance for this CNN model. This is because:
- The model is relatively small
- Only Linear layers were quantized
- Quantization overhead exceeded computational savings

This highlights that optimization techniques must be evaluated contextually rather than assumed to always improve performance.

### Runtime Optimization

Applied ONNX Runtime graph optimizations and controlled threading.

| Model | Latency (Before) | Latency (After) |
|------|----------------|----------------|
| FP32 | 0.0700 ms     | 0.0400 ms      |
| INT8 | 0.2200 ms     | 0.0300 ms      |

**Insight:**
Runtime optimizations significantly improved performance and enabled INT8 to outperform FP32, highlighting the importance of execution-level tuning in deployment.

## Repository Structure

edgeai-ml-systems-phase1/

models/  
&nbsp;&nbsp;&nbsp;&nbsp;cnn.py  
&nbsp;&nbsp;&nbsp;&nbsp;transformer.py 

training/  
&nbsp;&nbsp;&nbsp;&nbsp;train.py 

utils/  
&nbsp;&nbsp;&nbsp;&nbsp;dataset.py  

configs/  
&nbsp;&nbsp;&nbsp;&nbsp;cnn.yaml

benchmarks/  
&nbsp;&nbsp;&nbsp;&nbsp;benchmark.py

docs/  
&nbsp;&nbsp;&nbsp;&nbsp;system_pipeline1.png  
&nbsp;&nbsp;&nbsp;&nbsp;transformer_encoder1.png

README.md  
requirements.txt


## CNN Architecture
The implemented CNN architecture follows this pipeline:

Input (1x28x28)

Conv2d (1 → 16, kernel=3, padding=1)  
ReLU  
MaxPool (2×2)

Conv2d (16 → 32, kernel=3, padding=1)  
ReLU  
MaxPool (2×2)

Flatten

Fully Connected (1568 → 128)  
ReLU

Fully Connected (128 → 10)

Total Parameters: 206,922

## Transformer Encoder Architecture
The Transformer encoder is implemented using multi-head self-attention and feed-forward layers stacked sequentially.
<p align="center">
  <img src="docs/transformer_encoder1.png" alt="Transformer">
</p>

## Training Pipeline
The training loop follows a model-agnostic PyTorch workflow applicable to both CNN and Transformer models:  
Forward Pass → Loss Calculation → Backpropagation → Optimizer Update

Components:
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Dataset: MNIST

## CNN vs Transformer Comparison

| Metric | CNN | Transformer |
|------|------|------------|
| Training Time (1 epoch) | 17.36 s | 30.45 s |
| Single Inference Latency | 0.301 ms | 1.302 ms |
| Batch Inference Latency (32) | 1.164 ms | 2.641 ms |
| Parameters | 206,922 | 102,474 |
| Peak Memory Usage | ~335 MB | ~335 MB |
| Best DataLoader Workers | \- | 2 |

## Key Insights

- Transformer models have fewer parameters but higher computational complexity due to attention mechanisms (O(n²)).
- CNN is significantly faster for image-based tasks due to localized convolution operations.
- Transformer shows slower training and inference despite lower parameter count.
- Batch inference reduces the performance gap but CNN remains more efficient.
- Peak memory usage is similar for both models in this setup, indicating that activations and runtime dominate memory usage.
- DataLoader performance depends on system configuration and is independent of model architecture.

## Engineering Learnings

- Parameter count alone does not determine model efficiency.
- Attention mechanisms introduce quadratic complexity with sequence length.
- Profiling tools like cProfile and memory_profiler are essential for identifying bottlenecks.
- Backpropagation is the most computationally expensive step in training.
- Data loading can become a bottleneck without proper tuning.
- Benchmarking should evaluate training, inference, and memory — not just accuracy.

## Highlights

- Built CNN and Transformer models from scratch
- Implemented modular training pipeline with YAML configuration
- Performed system-level benchmarking and profiling
- Compared architectures based on performance, not just accuracy

## Conclusion

This project demonstrates a complete ML systems pipeline including model implementation, training, profiling, and benchmarking.

CNN models are more efficient for image-based tasks due to optimized convolution operations, while Transformer models provide architectural flexibility but introduce higher computational overhead.

This comparison highlights the importance of evaluating machine learning models from a systems perspective rather than relying solely on accuracy or parameter count.

## How to Run
### Create environment
conda create -n edgeai python=3.10  
conda activate edgeai

### Install dependencies
pip install -r requirements.txt

### Train the model

Default (CNN):
python -m training.train

Transformer:
python -m training.train --model_name transformer

### Run benchmarks
python -m benchmarks.benchmark

## Author
C. Sivananda Reddy