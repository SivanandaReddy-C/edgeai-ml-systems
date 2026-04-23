# 🚀 Edge AI ML Systems

![Python](https://img.shields.io/badge/Python-3.10-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)  
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)  
![Status](https://img.shields.io/badge/Status-Active-success)  
![Focus](https://img.shields.io/badge/Focus-ML%20Systems-orange)

> End-to-end ML systems project covering **model development → deployment → embedded execution → system-level analysis**

---

# 🔥 Project Overview

This project implements a **complete ML systems pipeline** across three phases:

| Phase | Focus |
|------|------|
| Phase 1 | Model development, profiling, benchmarking |
| Phase 2 | ONNX deployment, quantization, system optimization |
| Phase 3 | Embedded deployment on STM32 (Cube.AI + CMSIS-NN) |

---

## ❓ Why This Project Matters

Most ML work stops at training.

This project answers:

- What happens **after training**?
- What breaks during **deployment**?
- What actually limits performance:
  - compute?
  - memory?
  - toolchain?

---

## 💡 Highlights

- Built CNN and Transformer from scratch  
- Achieved **~5–10× speedup using ONNX Runtime**  
- Implemented INT8 quantization and analyzed limitations  
- Identified real deployment blockers (`ConvInteger`)  
- Deployed models on **STM32 Cortex-M4**  
- Reduced Flash usage by **~24× via architecture redesign**  
- Built full **CMSIS-NN manual inference pipeline**  
- Debugged real-world issues:
  - quantization scaling  
  - tensor layout mismatch  
  - architecture mismatch  

---

# 🔹 Phase 1 — Model Development & Benchmarking

## 🎯 Objective
Build models from scratch and understand their performance characteristics.

---

## 🧠 System Pipeline

![Pipeline](docs/system_pipeline1.png)

---
## 🧱 Models Implemented

### 1. CNN Architecture

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

### 2. Transformer Encoder

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

**Command used:**
```  
python -m phase1.benchmarks.benchmark
```
---

## 🔍 Phase 1 - Key Insights 

- CNN faster despite more parameters  
- Transformer slower due to **O(n²) attention cost**  
- Batch improves throughput but not latency parity  
- Memory dominated by runtime activations  

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

**Commands used:**
```    
python -m phase2.export.export_cnn_onnx  
python -m phase2.export.export_transformer_onnx
```
---

## ✅ Step 2 — ONNX Validation

Validated ONNX outputs against PyTorch.

- Mean numerical difference: **~1e-6**  
- Prediction match: **100%** 

**Commands used:**
```  
python -m phase2.validation.validate_onnx_vs_pytorch cnn
python -m phase2.validation.validate_onnx_vs_pytorch transformer
```
### 🧠 Observations
- ONNX outputs closely match PyTorch outputs for both models  
- No prediction mismatch observed

📌 Key Insight:  
- Conversion to ONNX preserves both numerical correctness and model behavior
---
## ⚙️ Step 3 — Quantization

Applied INT8 quantization to optimize model size.

### 📊 Model Size Comparison

| Model | FP32 | INT8 |
|------|------|------|
| CNN | 810.15 KB | 208.24 KB |
| Transformer | 1681.38 KB | 1692.51 KB |

### 📊 FP32 vs INT8 Output Comparison
| Metric | CNN FP32 vs INT8 | Transformer FP32 vs INT8 |
|------|-------------|-------------|
| Mean Output Difference | — | Small (~1e-2 to 1e-1) |
| Prediction Match | — | ✅ Yes |
| Runtime Execution | ❌ Not Supported  | ✅ Supported |

**Commands used:**
```   
python -m phase2.quantization.quantize_cnn_onnx
python -m phase2.quantization.quantize_transformer_onnx
python -m phase2.quantization.model_size
python -m phase2.quantization.compare_models cnn
python -m phase2.quantization.compare_models transformer
```
### 🧠 Observations
- CNN achieves ~4× size reduction after quantization
- Transformer shows minimal size change
- CNN INT8 model fails due to unsupported ConvInteger operator
- Transformer INT8 model executes successfully 

📌 Key Insight:
- Quantization effectiveness depends on:
  - **model architecture**
  - **operator support in runtime**

👉 Quantization success ≠ deployment success

---

## ⚡ Step 4 — Performance Benchmarking

### 📊 PyTorch vs ONNX

| Model | PyTorch Latency | ONNX Latency |
|------|----------------|-------------|
| CNN | ~0.2902 ms | ~0.0926 ms |
| Transformer | ~0.8869 ms | ~0.2826 ms |

**Commands used:**
```  
python -m phase2.benchmarks.benchmark_pytorch_vs_onnx cnn
python -m phase2.benchmarks.benchmark_pytorch_vs_onnx transformer
```
#### 🧠 Observations:
- ONNX provides consistent latency reduction (~3–5×)
- CNN benefits more due to efficient convolution operations
- Transformer remains slower due to attention complexity (O(n²))

📌 Key Insight:  
- ONNX improves execution efficiency, but model architecture dominates latency
---

### 📊 ONNX FP32 vs ONNX INT8 

| Model | FP32 | INT8 |
|------|------|------|
| CNN | ~0.0926 ms | ❌ Not Supported |
| Transformer | ~0.27 ms | ~0.4–0.5 ms |

**Commands used:**
```  
python -m phase2.benchmarks.benchmark_onnxFP32_vs_onnxINT8 cnn
python -m phase2.benchmarks.benchmark_onnxFP32_vs_onnxINT8 transformer
```
#### 🧠 Observations:
- CNN INT8 model cannot be executed
- Transformer INT8 runs but shows no speedup

📌 Key Insight:
- Quantization does not guarantee performance improvement
---

## 🧠 Step 5 — System Analysis

### 📊 End-to-End Latency

| Stage | CNN Time | Transformer Time |
|------|------|------|
| Preprocessing | 0.0174 ms | 0.0145 ms |
| Inference | 0.0687 ms | 0.2540 ms |
| Postprocessing | 0.0066 ms | 0.0073 ms |
| **Total** | **0.0930 ms** | **0.2762 ms** |

**Commands used:**
```  
python -m phase2.benchmarks.end_to_end_latency cnn
python -m phase2.benchmarks.end_to_end_latency transformer
```

#### 🧠 Observations:
- Inference dominates (~90%) of total latency
- CNN is faster due to lower compute complexity

📌 Key Insight:
- System is **compute-bound**  

---
### 📊 Thread Optimization (CPU)

| Threads | CNN Latency | Transformer Latency |
|--------|--------|--------|
| 1 | 0.0383 ms |0.1277 ms |
| 2 | 0.0368 ms | 0.1773 ms |
| 4 | 0.0323 ms | 0.1608 ms |
| 8 | 0.0398 ms | 0.2017 ms |

**Commands used:**
```  
python -m phase2.benchmarks.benchmark_threads cnn
python -m phase2.benchmarks.benchmark_threads transformer
```

#### 🧠 Observations:
- Increasing threads does not improve performance
- Overhead dominates for small models

📌 Key Insight:
- Multi-threading is ineffective for lightweight workloads

---

### 📊 Batch Processing 

| Batch | CNN Latency | CNN Throughput | Transformer Latency | Transformer Throughput |
|------|------------|----------------|---------------------|------------------------|
| 1    | 0.0799 ms  | 12510.18 samples/sec | 0.2781 ms | 3595 samples/sec |
| 2    | 0.0953 ms  | 20994.20 samples/sec | 0.4311 ms | 4639 samples/sec |
| 4    | 0.0927 ms  | 43140.31 samples/sec | 0.6364 ms | 6285 samples/sec |
| 8    | 0.1233 ms  | 64870.51 samples/sec | 1.0101 ms | 7920 samples/sec |
| 16   | 0.1457 ms  | 109826.82 samples/sec | 1.5844 ms | 10098 samples/sec |

**Commands used:**
```  
python -m phase2.benchmarks.benchmark_batch cnn
python -m phase2.benchmarks.benchmark_batch transformer
```
#### 🧠 Observations:
- Throughput increases with batch size
- Latency also increases → trade-off

📌 Key Insight:
- Batching improves throughput but not suitable for real-time systems
---
### 📊 Load Time (Cold Start)

| Model | Load Time |
|------|----------|
| CNN FP32 | 6.86 ms |
| CNN INT8 | ❌ Failed |
| Transformer FP32 | 17.83 ms |
| Transformer INT8 | 22.31 ms |

**Commands used:**
```  
python -m phase2.benchmarks.load_time
```

#### 🧠 Observations:

- Load time >> inference latency

📌 Key Insight:
- Cold start latency is a **critical bottleneck** in deployment scenarios  

---
### 📊 System Summary

| Metric | CNN | Transformer |
|------|------|------|
| Model Size | 809.77 KB | 1681.38 KB |
| Load Time | 6.32 ms | 16.21 ms |
| Inference Latency | 0.0754 ms | 0.2688 ms |
| End-to-End Latency | 0.1161 ms | 0.3048 ms |

**Commands used:**
```  
python -m phase2.benchmarks.system_summary cnn
python -m phase2.benchmarks.system_summary transformer
```

#### 🧠 Observations:

- CNN consistently outperforms Transformer

📌 Key Insight:
- Model architecture is the primary driver of system performance
---

## ⚙️ Execution Providers

### 🔍 Available Providers in my system

**Command used:**
```
python -m phase2.validation.check_providers
```
Output: 
```
Available providers:
['AzureExecutionProvider', 'CPUExecutionProvider']
```

### ⚡ Provider Benchmark Results
| Model | Provider | Latency |
|------|------|------|
| CNN | CPUExecutionProvider | ~0.0775 ms |
| Transformer | CPUExecutionProvider | ~0.2775 ms | 

**Commands used:**
```
python -m phase2.benchmarks.benchmark_provider cnn
python -m phase2.benchmarks.benchmark_provider transformer
```

#### 🧠 Observations

- All inference in this project is executed using CPUExecutionProvider  

📌 Key Insight:
Execution backend significantly impacts performance

### ⚠️ Why Other Providers Were Not Used
- AzureExecutionProvider:
  - Designed for cloud-based execution
  - Not relevant for local benchmarking
- GPU providers:
  - Not available in current environment
  - No CUDAExecutionProvider or hardware acceleration backend

---

## 🔍 Phase 2 - Key Insights

- ONNX Runtime provides **consistent inference acceleration across models**  
- Model architecture strongly influences performance:
  - CNN → efficient and low-latency  
  - Transformer → compute-intensive  

- Quantization:
  - reduces model size 
  - but does not guarantee speedup  
  - depends on runtime operator support  

- CNN INT8 fails due to backend limitations (`ConvInteger`)  
- Transformer INT8 executes but offers **limited performance benefit**

- Increasing threads does not improve performance for small models  
- Batch size improves throughput but increases latency  

- Inference dominates system latency (~90%)  
- Cold start (load time) is a major deployment bottleneck  

📌 **Final Insight:**
- Efficient ML deployment depends on **model architecture + runtime support + system constraints**, not just model accuracy.

---

# 🔹 Phase 3 — Embedded Deployment on STM32

## 🎯 Objective
Deploy models on **Cortex-M4** and analyze real constraints.

---
## Model Validation (Pre-Deployment)

Validated ONNX models before deploying to embedded target:

| Model | Status | Observation |
|------|--------|------------|
| CNN FP32 | ✅ Success | Correct input/output shapes |
| CNN INT8 | ❌ Failed | `ConvInteger` unsupported |
| Transformer FP32 | ✅ Success | Dynamic batch supported |
| Transformer INT8 | ✅ Success | Runs correctly |

📌 Insight:  
Deployment feasibility depends on **operator support**, not just model correctness.

---
## ⚙️ Cube.AI Deployment 
---
### CNN

| Metric | Value |
|------|------|
| Flash | ~822 KB |
| RAM | ~22 KB |
| Latency | ~125.67 ms |

📌 Observation:  
- Fully Connected (FC) layer dominates memory (~97% of Flash)

---

### ⚠️ Transformer
❌ Failed on STM32:   
Unsupported: LayerNormalization
👉 Transformer not feasible on Cortex-M4 using Cube.AI  
---

### CNN model Optimization  

Redesigned CNN architecture for embedded deployment:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Conv → Conv → Global Averaging Pooling → FC(32 → 10) 


📌 Key Changes:
- Removed Flatten + large FC layer  
- Introduced Global Average Pooling (GAP)  

📌 Result:
- Drastic reduction in parameters  

 Metric | CNN | Optimized CNN |
|------|-------------|--------------|
| Parameters | ~206K | ~5K |
| Flash | ~822 KB | ~34 KB |
| RAM | ~21.5 KB | ~21.5 KB |
| MACs | ~1.25M | ~1.05M |
| Latency | ~125.67 ms | ~107.08 ms |

📌 Insight:
- Flash dominated by FC layer (~97%)
- Optimization removes FC bottleneck → ~24× reduction
- RAM unchanged → activation dominated (~21 KB)
- Conv2 layer dominates (~85% MACs)
- System is **compute-bound**
- FP32 arithmetic is primary bottleneck

👉 Weight optimization ≠ runtime memory optimization  
👉 Parameter reduction does NOT linearly reduce latency  

---

### Cube.AI Optimization Modes

| Mode | Latency | RAM |
|------|--------|-----|
| Balanced | ~107 ms | ~21 KB |
| Time | ~108 ms | ~56 KB |

👉 Memory increase does not improve speed  
👉 Optimization affects layout, not computation  

---

### ⚠️ INT8 Deployment Limitation

INT8 model failed during Cube.AI Analyze:

Unsupported operators:
- ConvInteger  
- MatMulInteger  
- DynamicQuantizeLinear  

📌 Insight:
- ONNX INT8 ≠ STM32-compatible INT8  
- External quantization not supported  

👉 Deployment depends on **operator + toolchain compatibility**

---

### 🧠 System-Level Insights

- Embedded ML is dominated by:
  - Convolution compute
  - FP32 precision cost  

- Memory vs Compute:
  - Flash → easy to reduce  
  - Compute → hard to reduce  

- Toolchain matters:
  - ONNX success ≠ STM32 compatibility  

---


## 🔥 CMSIS-NN Manual Pipeline 

### 🎯 Objective

Build full CNN inference manually using INT8.

---

### ⚙️ Implementation Details

- CMSIS-NN manually integrated into STM32 project
- INT8 weights and activations used across all layers
- INT32 accumulation used to avoid overflow
- Explicit requantization using **multiplier + shift**
- Custom fully connected layer (`linear_s8`) implemented
- Real PyTorch-learned parameters exported into C arrays and executed on STM32

---
### 📊 Final Results

| Metric | Value |
|------|------|
| Cycles | ~103.8M |
| Latency | ~865.46 ms |
| Prediction | Matches PyTorch |



---

### ⚠️ Key Debugging Challenges

#### 1. Quantization Scaling Issues

- Incorrect multiplier/shift caused valid outputs to collapse to zero
- Required layer-wise tuning of output scales
- Verified correctness using min/max and saturation checks

📌 Insight:  
Quantization correctness is critical — wrong scaling destroys meaningful inference.

---

#### 2. Input Representation

- Direct int8 handling of image input caused incorrect mapping
- Fixed by:
  - using uint8 input
  - converting properly to int8 before inference

📌 Insight:  
Input representation directly affects downstream convolution correctness.

---

#### 3. Architecture Mismatch (Critical Issue)

During PyTorch vs STM32 comparison, the key mismatch was identified:

- STM32 FC1 input size: `800`
- Trained PyTorch model FC1 input size: `1568`

**Root cause**
- STM32 path used no-padding style output shape at that stage
- Trained PyTorch model used `padding = 1`, so final feature map before FC1 was `32 × 7 × 7 = 1568`

📌 Impact:  
Even correct integer math could not match the trained model because the deployed architecture itself was different.

---

#### 4. Flatten Layout Mismatch (Critical Issue)

Initial assumption:
- Flatten needed a manual HWC → CHW reorder before FC1

Actual behavior:
- The pooled CMSIS-NN output was already aligned with the FC layer expectation for the working pipeline
- Manual reorder scrambled the data and broke FC behavior

**Final fix**
for (int i = 0; i < 7 * 7 * 32; i++) {
    flatten_out[i] = conv2_pool_out[i];
}

---


## ⚖️ CMSIS-NN vs Cube.AI Comparison

### ⚡ Performance Perspective (Same CNN)

| Metric | Cube.AI (CNN) | CMSIS-NN (CNN) |
|------|--------------|----------------|
| Inference Latency | ~125.67 ms | ~865.46 ms |
| Inference Cycles | Not exposed | ~103.8 Million cycles |
| Flash Usage | ~822 KB | Similar (weights dominated) |
| RAM Usage | ~21.5 KB | ~21.5 KB |
| Optimization Level | Tool-driven | Manual (baseline implementation) |

📌 Result:
👉 CMSIS-NN implementation is **~6.8× slower than Cube.AI** for the same CNN model  

---
### 🧠 Trade-off

| Cube.AI | CMSIS-NN |
|--------|----------|
| Fast | Flexible |
| Black-box | Transparent |
| Easy | Complex |

---

### 🎯 Accuracy & Validation
#### 📊 Training Accuracy

- CNN validation accuracy: ~99% (MNIST)

#### 🔁 Cross-Platform Output Consistency

To ensure deployment correctness, outputs were compared across:

- PyTorch (baseline)
- ONNX Runtime (CPU)
- STM32 CMSIS-NN implementation

#### ✅ Observations

- ONNX outputs closely match PyTorch outputs  
- STM32 outputs match integer reference implementation  
- Final predicted class matches PyTorch for tested samples  

📌 Example:

| Platform | Predicted Class |
|---------|----------------|
| PyTorch | 7 |
| ONNX | 7 |
| STM32 (CMSIS-NN) | 7 |


#### ⚠️ Quantization Impact

- INT8 quantization introduces minor numerical differences  
- No classification mismatch observed in tested samples  
- Full dataset accuracy evaluation on STM32 not performed  

#### 🧠 Key Insight

> Deployment correctness was validated through **cross-platform consistency**,  
> not full dataset accuracy evaluation on device

---

### 🚀 Final Insight

> CMSIS-NN is not a faster alternative by default  

👉 It is a **low-level toolkit**, not a full deployment engine  

👉 In this project:

- Cube.AI provides **better performance out-of-the-box**  
- CMSIS-NN provides **deeper control and visibility**  

👉 Performance advantage depends on **manual optimization effort**, not just the library itself  

---

# 📊 Final System Comparison

| Stage | Latency |
|------|--------|
| PyTorch CNN | ~0.73 ms |
| ONNX CNN | ~0.09 ms |
| STM32 Cube.AI | ~125 ms |
| STM32 CMSIS | ~865 ms |
| Transformer STM32 | ❌ Not deployable |
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

phase3/  
  &nbsp;&nbsp;&nbsp;&nbsp;stm32_cube_ai/  
  &nbsp;&nbsp;&nbsp;&nbsp;docs/ 
  

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

CNN Optimized:
python -m phase1.training.train --model_name cnn_optimized

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