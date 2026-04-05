# Edge AI ML Systems

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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyTorch → ONNX → Runtime → Quantization → Optimization  

---

## ⚙️ ONNX Deployment

### Model Export
- Converted PyTorch models to ONNX  
- Used dummy inputs to trace graph 

### Validation
- Verified using ONNX checker  
- Ensured inference compatibility  


### Inference (ONNX Runtime)
- Executed using ONNX Runtime (C++ backend)  
- Used NumPy arrays instead of PyTorch tensors

### 📊 Performance Comparison

| Model | PyTorch Latency | ONNX Latency |
|------|----------------|-------------|
| CNN  | 5.6070 ms      | 0.1009 ms   |

### Key Insight
ONNX significantly reduces latency by removing Python overhead and enabling graph-level optimizations.

---

## ⚡ Quantization (FP32 → INT8)

### Objective
Reduce precision to improve efficiency.

### Results

| Model | FP32 Latency | INT8 Latency |
|------|--------------|--------------|
| CNN  | 0.2623 ms    | 0.5262 ms    |

### Key Insight 
Quantization **did not improve performance** for this model because:
- Model is small  
- Limited layers were quantized  
- Overhead > compute savings  

👉 This highlights that optimization is **context-dependent**, not guaranteed.


---

## 🚀 Runtime Optimization

Applied ONNX Runtime graph optimizations + threading control.

| Model | Before | After |
|------|--------|-------|
| FP32 | 0.0700 ms | 0.0400 ms |
| INT8 | 0.2200 ms | 0.0300 ms |

### Key Insight 
Runtime optimizations significantly improved performance and enabled INT8 to outperform FP32, highlighting the importance of execution-level tuning in deployment.

---

## 📦 Batch Size Analysis

| Batch | Batch Latency (ms) | Per-sample Latency (ms) |
|------|--------------------|-------------------------|
| 1    | 0.0365             | 0.0365                  |
| 2    | 0.0497             | 0.0249                  |
| 4    | 0.0901             | 0.0225                  |
| 8    | 0.1583             | 0.0198                  |
| 16   | 0.2868             | 0.0179                  |
| 32   | 0.5527             | 0.0173                  |

### Key Insights 
- Batch latency increases with batch size
- Per-sample latency decreases due to better compute utilization
- Diminishing returns observed beyond batch size 16–32
---

## ⚡ Threading & Parallelism

### Results

#### Batch = 1

| Threads | Latency |
|--------|--------|
| 1 | 0.0361 ms |
| 2 | 0.0364 ms |
| 4 | 0.0360 ms |
| 8 | 0.0400 ms |

👉 No improvement due to small workload  



#### Batch = 8

| Threads | Latency |
|--------|--------|
| 1 | 0.1574 ms |
| 2 | 0.0966 ms |
| 4 | 0.0784 ms |
| 8 | 0.0742 ms |

👉 ~2× speedup using parallelism  



### Key Insights

- Small workloads → threading overhead dominates  
- Larger workloads → parallelism improves performance  
- Gains saturate due to CPU limits  

### 💡Practical Insight 
- Real-world deployment requires tuning both batch size and threading together. Optimal performance is achieved by balancing latency, throughput, and hardware utilization.

---

# 🗂️ Repository Structure

edgeai-ml-systems-phase1/

models/  
&nbsp;&nbsp;&nbsp;&nbsp;cnn.py  
&nbsp;&nbsp;&nbsp;&nbsp;transformer.py 

training/  
&nbsp;&nbsp;&nbsp;&nbsp;train.py 

deployment/  
&nbsp;&nbsp;&nbsp;&nbsp;export_onnx.py  
&nbsp;&nbsp;&nbsp;&nbsp;quantize_onnx.py   
&nbsp;&nbsp;&nbsp;&nbsp;benchmark_runtime.py 

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


# 🏁 Conclusion

This project demonstrates a **complete ML systems lifecycle**:

- Model design → Training → Profiling  
- Benchmarking → Deployment → Optimization  

### Final Takeaways

- Performance depends on **system-level factors**, not just model design  
- Optimization techniques are **workload-dependent**  
- Real-world ML systems require balancing:
  - Latency  
  - Throughput  
  - Hardware utilization  

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