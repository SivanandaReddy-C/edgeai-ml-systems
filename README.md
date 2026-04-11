# Edge AI ML Systems

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Focus](https://img.shields.io/badge/Focus-ML%20Systems-orange)

> 🚀 End-to-end ML Systems project covering training, deployment, and performance optimization for edge AI applications.

---

## 🚀 Project Overview

This project implements an **end-to-end ML systems pipeline**, covering:

- **Phase 1:** Model Development & Benchmarking  
- **Phase 2:** Deployment & Inference Optimization  

Focus areas:

- Model training and architecture design  
- Performance profiling and benchmarking  
- Deployment optimization using ONNX Runtime  
- System-level analysis (latency, throughput, load time)


---

## ❓ Why This Project Matters

Modern ML systems are not just about training models — but deploying them efficiently.

This project focuses on:

- Bridging research → production gap  
- Understanding runtime constraints  
- Building deployment-ready ML pipelines  
- Evaluating real-world system performance  

---

## 💡 Highlights

- Built CNN and Transformer models from scratch  
- Exported models from **PyTorch → ONNX** for deployment  
- Achieved **~5–10× inference speedup** using ONNX Runtime  
- Implemented **INT8 quantization** and analyzed real-world limitations  
- Identified backend constraints (e.g., **ConvInteger unsupported on CPU**)  
- Performed **system-level benchmarking**: latency, throughput, threads, memory  
- Measured **end-to-end latency and cold start performance**  

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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyTorch → ONNX → Validation → Quantization → Benchmarking → System Analysis  

---
## 📌 Understanding Latency Metrics

- **Inference Latency:** Model execution only  
- **End-to-End Latency:** Preprocessing + inference + postprocessing  
- **Batch Latency:** Multi-input processing  
- **Threaded Latency:** CPU parallelism impact  
- **Load Time (Cold Start):** Model initialization time  

👉 These metrics are **not directly comparable**, but together give a complete system view.

---
## ⚙️ Step 1 — ONNX Export

Converted PyTorch models into ONNX format for deployment.

- Exported CNN and Transformer models  
- Handled input shapes carefully  
- Ensured ONNX Runtime compatibility  

---

## ✅ Step 2 — Validation

Validated ONNX outputs against PyTorch.

- Mean numerical difference: **~1e-6**  
- Prediction match: **100%**

📌 Confirms correctness of conversion  

---
## ⚙️ Step 3 — Quantization

Applied INT8 quantization to optimize model size.

### Model Size Comparison

| Model | FP32 | INT8 |
|------|------|------|
| CNN | 810.15 KB | 208.24 KB |
| Transformer | 1681.38 KB | 1692.51 KB |

### Observations

- CNN achieves **~4× reduction**  
- Transformer shows **minimal change**  
- CNN INT8 fails due to `ConvInteger` operator limitation  

📌 Quantization effectiveness depends on architecture and runtime support  

---

## ⚡ Step 4 — Performance Benchmarking

### PyTorch vs ONNX

| Model | PyTorch Latency | ONNX Latency |
|------|----------------|-------------|
| CNN | ~0.7314 ms | ~0.0926 ms |

📌 ONNX Runtime achieves **~5–10× speedup**

---

### CNN vs Transformer (ONNX — Quick Comparison)

| Model | Latency |
|------|--------|
| CNN | 0.0901 ms |
| Transformer | 0.3912 ms |

📌 Single-run comparison for quick insight  
📌 CNN is faster due to convolution efficiency  
📌 Transformer has higher compute cost due to attention  

---

### Inference Latency (Baseline — Averaged)

📌 Note:

- Quick comparison above = **single-run latency**
- Below = **averaged latency over multiple runs (more reliable)**

| Model | FP32 | INT8 |
|------|------|------|
| CNN | ~0.2–0.3 ms | ❌ Not Supported |
| Transformer | ~0.27 ms | ~0.4–0.5 ms |

📌 Represents stable model execution time  

---


## 🧠 Step 5 — System Analysis

### End-to-End Latency (Transformer)

| Stage | Time |
|------|------|
| Preprocessing | 0.0145 ms |
| Inference | 0.2540 ms |
| Postprocessing | 0.0073 ms |
| **Total** | **0.2762 ms** |

📌 ~92% latency from inference → compute-bound system  

---
### Thread Optimization (Transformer on CPU)

| Threads | Latency |
|--------|--------|
| 1 | 0.1277 ms |
| 2 | 0.1773 ms |
| 4 | 0.1608 ms |
| 8 | 0.2017 ms |

📌 Observations:
- Increasing threads **degraded performance**
- Overhead dominates for small models  

---

### Batch Processing (Transformer - Throughput vs Latency)

| Batch | Latency | Throughput |
|------|--------|------------|
| 1 | 0.2781 ms | 3595 samples/sec |
| 2 | 0.4311 ms | 4639 samples/sec |
| 4 | 0.6364 ms | 6285 samples/sec |
| 8 | 1.0101 ms | 7920 samples/sec |
| 16 | 1.5844 ms | 10098 samples/sec |

📌 Observations:
- Larger batches increase throughput  
- Latency increases → trade-off exists  

---
### Load Time (Cold Start)

| Model | Load Time |
|------|----------|
| CNN FP32 | 6.86 ms |
| CNN INT8 | ❌ Failed |
| Transformer FP32 | 17.83 ms |
| Transformer INT8 | 22.31 ms |

📌 Observations:
- Load time >> inference latency  
- Critical for real-time systems and APIs  

---
### System Summary (Transformer)

| Metric | Value |
|------|------|
| Model Size | 1681.38 KB |
| Load Time | 16.21 ms |
| Inference Latency | 0.2688 ms |
| End-to-End Latency | 0.3048 ms |

📌 Cold start ≈ **60× slower than inference**

---

## ⚠️ Deployment Constraints

- CNN INT8 fails due to unsupported `ConvInteger` operator  
- Runtime operator support determines deployability  
- Quantization success ≠ execution success  

📌 Model conversion does not guarantee deployment  

---

## 🔍 Key Insights (Phase 2)

- ONNX significantly reduces inference latency  
- Quantization reduces size but does not guarantee speedup  
- CNN INT8 failed due to backend limitations  
- Transformer INT8 works but offers limited benefit  
- Increasing threads can degrade performance  
- Batch size improves throughput but increases latency  
- Inference dominates total latency (~90%)  
- Cold start latency is a major deployment bottleneck  

---

## 🧠 Engineering Learnings

### 1. Quantization is Not Always Beneficial
- May not improve latency  
- Depends on hardware and operator support  



### 2. Runtime Compatibility Matters
- Conversion ≠ Deployment  
- Backend determines feasibility  



### 3. System is Compute-Bound
- Optimize inference, not pipeline  



### 4. Cold Start vs Steady State
- Load time is significantly higher than inference  



### 5. Throughput vs Latency Trade-off
- Larger batches improve throughput  
- Smaller batches reduce latency  

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

- ONNX enables significant inference acceleration  
- Model efficiency depends on architecture + runtime backend  
- CNN is latency-efficient for vision tasks  
- Transformer introduces higher compute cost  
- Quantization depends on backend support  
- Deployment requires system-level understanding  

📌 **Final Insight:**  
Optimizing ML systems is not just about models — it is about understanding the entire execution stack.

---

# ▶️ How to Run
## Setup Environment
conda create -n edgeai python=3.10  
conda activate edgeai  
pip install -r requirements.txt

## Phase 1
### Model Training & Benchmarking
Default (CNN):  
python -m phase1.training.train

Transformer:  
python -m phase1.training.train --model_name transformer

### Benchmarking:    
python -m phase1.benchmarks.benchmark

## Phase2 
### Export Models to ONNX  
python -m phase2.onnx.export_cnn_onnx  
python -m phase2.onnx.export_transformer_onnx  

### Validate ONNX Models
python -m phase2.onnx.compare_models

### Quantization (INT8)
python -m phase2.onnx.quantize_cnn_onnx  
python -m phase2.onnx.quantize_transformer_onnx

### Performance Benchmarking
#### PyTorch vs ONNX
python -m phase2.onnx.benchmark_pytorch_vs_onnx

#### CNN ONNX Benchmark
python -m phase2.onnx.benchmark_cnn_onnx

#### Transformer INT8 Benchmark
python -m phase2.onnx.benchmark_transformer_quant

### System-Level Analysis
#### End-to-End Pipeline Timing
python -m phase2.onnx.pipeline_timing

#### Thread Optimization
python -m phase2.onnx.benchmark_threads

#### Batch Throughput
python -m phase2.onnx.benchmark_batch_transformer

#### Load Time (Cold Start)
python -m phase2.onnx.load_time

#### Model Size Comparison
python -m phase2.onnx.check_model_size

---

# 👨‍💻 Author
### C. Sivananda Reddy
---