# Edge AI ML Systems

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Focus](https://img.shields.io/badge/Focus-ML%20Systems-orange)

> 🚀 End-to-end ML Systems project covering training, deployment, and performance optimization for edge AI applications.

---

## 🚀 Project Overview

This project implements an **end-to-end ML systems pipeline**, covering both:

- **Phase 1:** Model Development & Benchmarking  
- **Phase 2:** Deployment & Inference Optimization  

The focus is on **system-level understanding**, including:
- Training workflows
- Performance benchmarking
- Profiling and optimization
- Deployment for efficient inference

---

## ❓ Why This Project Matters

Modern ML systems are not just about training models but deploying them efficiently.

This project focuses on:
- Bridging the gap between research and real-world deployment  
- Understanding hardware-aware optimization  
- Building production-ready ML pipelines  

---

## 💡 Highlights

- Built end-to-end ML deployment pipeline (**PyTorch → ONNX → Optimization**)  
- Achieved **~5–10× inference speedup** using ONNX Runtime  
- Implemented **INT8 quantization** and analyzed real-world limitations  
- Identified backend constraints (e.g., **ConvInteger not supported on CPU**)  
- Performed **system-level benchmarking**: latency, throughput, threading  
- Compared **CNN vs Transformer** from both model and deployment perspectives  

---

# 🔹 Phase 1 — Model Development & Benchmarking

## 🎯 Objectives

- Implement CNN training pipeline
- Build Transformer Encoder from scratch
- Profile training performance
- Benchmark CNN vs Transformer

---

## 🧠 System Pipeline

![Pipeline](docs/system_pipeline1.png)

---
## 🧱 Models Implemented

### CNN Architecture

Input (1×28×28)

- Conv2d (1 → 16, kernel=3, padding=1)  
- ReLU  
- MaxPool (2×2)

- Conv2d (16 → 32, kernel=3, padding=1)  
- ReLU  
- MaxPool (2×2)

- Flatten  
- FC (1568 → 128) → ReLU  
- FC (128 → 10)

**Total Parameters:** 206,922

---

### Transformer Encoder

<p align="center">
  <img src="docs/transformer_encoder1.png" alt="Transformer">
</p>

- Multi-head self-attention  
- Feed-forward layers  
- Layer stacking architecture  

---

## ⚙️ Training Pipeline

The training loop follows a model-agnostic PyTorch workflow applicable to both CNN and Transformer models:  
Forward Pass → Loss Calculation → Backpropagation → Optimizer Update

**Components:**
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Dataset: MNIST 

---

## 📊 CNN vs Transformer Benchmark

| Metric | CNN | Transformer |
|------|------|------------|
| Training Time (1 epoch) | 17.36 s | 30.45 s |
| Single Inference Latency | 0.301 ms | 1.302 ms |
| Batch Latency (32) | 1.164 ms | 2.641 ms |
| Parameters | 206,922 | 102,474 |
| Peak Memory | ~335 MB | ~335 MB |
| Best DataLoader Workers | \- | 2 |
---

## 🔍 Key Insights (Phase 1)

- Transformer models have fewer parameters but higher computational complexity due to attention mechanisms (O(n²)).
- CNN is significantly faster for image-based tasks due to localized convolution operations.
- Transformer shows slower training and inference despite lower parameter count.
- Batch inference reduces the performance gap but CNN remains more efficient.
- Peak memory usage is similar for both models in this setup, indicating that activations and runtime dominate memory usage.
- DataLoader performance depends on system configuration and is independent of model architecture.

---

## 🧠 Engineering Learnings

- Parameter count alone does not determine model efficiency.
- Attention mechanisms introduce quadratic complexity with sequence length.
- Profiling tools like cProfile and memory_profiler are essential for identifying bottlenecks.
- Backpropagation is the most computationally expensive step in training.
- Data loading can become a bottleneck without proper tuning.
- Benchmarking should evaluate training, inference, and memory — not just accuracy.

---

# 🔹 Phase 2 — Deployment & Optimization

## 🎯 Objective

Convert trained models into **efficient deployment-ready systems**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyTorch → ONNX → Validation → Benchmarking → Optimization  

---
## ⚙️ Deployment Pipeline

PyTorch Model  
↓  
ONNX Export  
↓  
ONNX Runtime Inference  
↓  
Quantization (INT8)  
↓  
Benchmarking (Latency / Throughput / Threads)

---
## ⚙️ ONNX Export
- Exported CNN and Transformer models from PyTorch to ONNX
- Carefully handled input shapes for both architectures
- Ensured comparibility with ONNX runtime
---

## ✅ Validation
Validated ONNX models against PyTorch outputs:

- Mean numerical difference: ~1e-6
- Prediction consistency: 100% match  
This ensures correctness of deployment pipeline.
---
## ⚡ Performance benchmarking
### 📊 PyTorch vs ONNX
| Model | PyTorch Latency | ONNX Latency |
|------|----------------|-------------|
| CNN  | ~0.7314 ms        | ~0.0926 ms    |

ONNX Runtime achieved **~5–10× latency reduction** through graph optimizations and efficient execution backends.


### 📊 CNN vs Transformer (ONNX)
| Model | Latency |
|------|--------|
| CNN  | 0.0901 ms |
| Transformer | 0.3912 ms |
---

## 📊 Deployment Results

### 🔹 Latency Comparison (CPU)

| Model        | FP32 Latency | INT8 Latency | Speedup |
|-------------|-------------|-------------|---------|
| CNN         | ~0.73 ms     | ❌ Not supported | — |
| Transformer | ~0.35 ms    | ~0.40 ms    | ~1.0x |

---

### 🔹 Model Size Comparison

| Model        | FP32 Size | INT8 Size |
|-------------|----------|----------|
| CNN         | 810 KB   | 208 KB   |
| Transformer | 1677 KB  | 1686 KB  |

---

### 🔹 Thread Optimization (Transformer)

| Threads | Latency |
|--------|--------|
| 1      | 0.1277 ms |
| 2      | 0.1773 ms |
| 4      | 0.1608 ms |
| 8      | 0.2017 ms |

---

### 🔹 Batch Throughput (Transformer)

| Batch Size | Latency | Throughput |
|-----------|--------|------------|
| 1         | 0.2781 ms | 3595 samples/sec |
| 2         | 0.4311 ms | 4639 samples/sec |
| 4         | 0.6364 ms | 6285 samples/sec |
| 8         | 1.0101 ms | 7920 samples/sec |
| 16        | 1.5844 ms | 10098 samples/sec |

---

## 🔍 Key Insight

CNN achieves lower latency due to localized convolution operations, while Transformer incurs higher compute cost due to attention mechanisms.

---

## 🔍 Key Observations (Deployment)

- ONNX significantly reduces inference latency compared to PyTorch  
- INT8 quantization reduces model size but does not always improve latency  
- CNN INT8 model failed due to unsupported ConvInteger operator in CPUExecutionProvider  
- Transformer INT8 model runs successfully on CPU  
- Increasing CPU threads degraded performance due to overhead  
- Larger batch sizes improve throughput but increase latency  

---

## 🧠 Key Learnings

- Model optimization alone is not sufficient; backend support is critical  
- Execution provider determines real-world performance  
- Quantization benefits depend on hardware and operator support  
- Small models do not benefit from multi-threading  
- Throughput vs latency trade-off is key in deployment systems  

---

# 🗂️ Repository Structure

edgeai-ml-systems/

phase1/  
  &nbsp;&nbsp;&nbsp;&nbsp;models/  
  &nbsp;&nbsp;&nbsp;&nbsp;training/  
  &nbsp;&nbsp;&nbsp;&nbsp;benchmarks/  
  &nbsp;&nbsp;&nbsp;&nbsp;utils/      
  &nbsp;&nbsp;&nbsp;&nbsp;configs/    

phase2/  
  &nbsp;&nbsp;&nbsp;&nbsp;onnx/  
  

docs/  
&nbsp;&nbsp;&nbsp;&nbsp;system_pipeline1.png  
&nbsp;&nbsp;&nbsp;&nbsp;transformer_encoder1.png

README.md  
requirements.txt

---

# 🏁 Conclusion

This project demonstrates a complete ML deployment and optimization pipeline:

- Model development (CNN, Transformer)  
- ONNX-based deployment  
- Quantization and optimization  
- System-level benchmarking  

### Key Takeaways

- Model performance depends on architecture and execution backend  
- CNNs are more latency-efficient for vision tasks  
- Transformers introduce higher compute cost  
- Quantization benefits depend on runtime support  
- Real-world deployment requires system-level understanding  

---

# ▶️ How to Run
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

---

# 👨‍💻 Author
### C. Sivananda Reddy
---