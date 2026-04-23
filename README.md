# 🚀 Edge AI ML Systems  
> Built for understanding real-world ML deployment constraints across software and hardware stacks

![Python](https://img.shields.io/badge/Python-3.10-blue)  
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)  
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)  
![Status](https://img.shields.io/badge/Status-Active-success)  
![Focus](https://img.shields.io/badge/Focus-ML%20Systems-orange)

> End-to-end ML systems project covering   
**model development → deployment → embedded execution → system-level analysis**

---

# 🔥 Project Overview

This project builds and analyzes a **complete ML systems pipeline**, tracking the journey from model creation to embedded deployment:

| Phase | Focus |
|------|------|
| Phase 1 | Model design, training, profiling, benchmarking |
| Phase 2 | ONNX deployment, quantization, system optimization |
| Phase 3 | Embedded deployment on STM32 (Cube.AI + CMSIS-NN) |

The goal is not just to build models, but to understand:

- What happens **after training**
- What breaks during **deployment**
- What truly limits performance:
  - compute  
  - memory  
  - toolchain  

---

# ❓ Why This Project Matters

Most ML workflows stop at training accuracy.

Real-world systems require answering:

- Can the model **run efficiently on target hardware?**
- Does quantization actually **improve performance?**
- What are the **hidden constraints** in deployment toolchains?

👉 This project bridges the gap between:
- ML development  
- systems engineering  
- embedded deployment  

---

# 💡 Highlights

- Built CNN and Transformer models from scratch  
- Achieved **~3–5× inference speedup using ONNX Runtime**  
- Implemented INT8 quantization and analyzed real limitations  
- Identified deployment blockers (`ConvInteger`, `LayerNorm`)  
- Deployed models on **STM32 Cortex-M4**  
- Reduced Flash usage by **~24× via architecture redesign**  
- Built full **CMSIS-NN manual inference pipeline**  
- Validated **cross-platform correctness**:
  - PyTorch → ONNX → Cube.AI → CMSIS-NN  

---

# ⚡ Key Results

- ONNX Runtime → **~3–5× faster than PyTorch**
- CNN latency → **~0.29 ms → ~0.09 ms → ~125 ms (MCU)**
- Transformer → **not deployable on STM32 (LayerNorm limitation)**
- Quantization:
  - CNN → **~4× size reduction**
  - Transformer → **no benefit**
- CMSIS-NN → **~6.8× slower than Cube.AI (baseline)**  
- Full pipeline validated:
  - PyTorch → ONNX → STM32 → **same predictions**
--- 

# 🔹 Phase 1 — Model Development & Benchmarking
## 🎯 Objective
Build ML models from scratch and analyze their performance characteristics:

Model Design → Training → Profiling → Benchmarking → System Understanding


## 🧠 System Pipeline

<p align="center">
  <img src="docs/system_pipeline1.png" alt="Pipeline">
</p>

## 🧱 Step 1 — Model Architectures

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


### Transformer Encoder

<p align="center">
  <img src="docs/transformer_encoder1.png" alt="Transformer">
</p>

- Multi-head self-attention  
- Feed-forward layers  
- Layer stacking architecture  


## ⚙️ Step 2 — Training Pipeline

Model-agnostic PyTorch training workflow:  
Forward Pass → Loss → Backpropagation → Optimizer Update 

**Components:**
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Dataset: MNIST 


## 📊 Step 3 — Performance Benchmarking

### CNN vs Transformer

| Metric | CNN | Transformer |
|------|------|------------|
| Training Time (1 epoch) | 17.36 s | 30.45 s |
| Single Inference Latency | 0.301 ms | 1.302 ms |
| Batch Latency (32) | 1.164 ms | 2.641 ms |
| Parameters | 206,922 | 102,474 |
| Peak Memory | ~335 MB | ~335 MB |

<details>
  <summary>Click to see Benchmark Command</summary>

  ```bash
  python -m phase1.benchmarks.benchmark
  ```
</details>



### 🧠 Observations

- CNN achieves lower latency despite having more parameters  
- Transformer is slower due to **attention complexity (O(n²))**  
- Batch processing improves throughput for both models  
- Memory usage is dominated by runtime activations rather than parameters  

📌 **Key Insight:**
- Model efficiency is not determined by parameter count alone  
- Architectural design strongly influences performance  

👉 CNN is better suited for **low-latency inference**  
👉 Transformer introduces higher computational overhead due to attention  


## 🔍 Phase 1 — Final Insights

- CNN provides efficient performance for image-based tasks  
- Transformer offers flexibility but at higher computational cost  
- Batch processing improves throughput but not latency parity  
- Memory consumption is driven by runtime operations  

📌 **Final Insight:**

Understanding model behavior at the system level is essential before moving to deployment and optimization

---

# 🔹 Phase 2 — Deployment & Optimization

## 🎯 Objective

Convert trained models into **efficient deployment-ready systems**:    
PyTorch → ONNX → Validation → Quantization → Benchmarking → System Analysis  

## 📌 Understanding Latency Metrics

- **Inference Latency:** Model execution only  
- **End-to-End Latency:** Preprocessing + inference + postprocessing  
- **Batch Latency:** Multi-input processing  
- **Threaded Latency:** CPU parallelism impact  
- **Load Time (Cold Start):** Model initialization time  

👉 These metrics are **not directly comparable**, but together give a complete system view.

## ⚙️ Step 1 — ONNX Export

Converted PyTorch models into ONNX format for deployment.

- Exported CNN and Transformer models  
- Handled input shapes carefully  
- Ensured ONNX Runtime compatibility  

<details>
  <summary>Click to see ONNX Export Commands</summary>

  ```bash
  python -m phase2.export.export_cnn_onnx  
  python -m phase2.export.export_transformer_onnx
  ```
</details>



## ✅ Step 2 — ONNX Validation

Validated ONNX outputs against PyTorch.

- Mean numerical difference: **~1e-6**  
- Prediction match: **100%** 

<details>
  <summary>Click to see ONNX Validation Commands</summary>

  ```bash
  python -m phase2.validation.validate_onnx_vs_pytorch cnn
  python -m phase2.validation.validate_onnx_vs_pytorch transformer
  ```
</details>

### 🧠 Observations
- ONNX outputs closely match PyTorch outputs for both models  
- No prediction mismatch observed

📌 **Key Insight:**  
- Conversion to ONNX preserves both numerical correctness and model behavior

## ⚙️ Step 3 — Quantization

Applied INT8 quantization to optimize model size.

### 📊 Model Size Comparison

| Model | FP32 | INT8 |
|------|------|------|
| CNN | 810.15 KB | 208.24 KB |
| Transformer | 1681.38 KB | 1692.51 KB |

### 📊 FP32 vs INT8 Output Comparison
| Metric | CNN | Transformer |
|------|-------------|-------------|
| Mean Output Difference | — | Small (~1e-2 to 1e-1) |
| Prediction Match | — | ✅ Yes |
| Runtime Execution | ❌ Not Supported  | ✅ Supported |

<details>
  <summary>Click to see Quantization Commands</summary>

  ```bash
  python -m phase2.quantization.quantize_cnn_onnx
  python -m phase2.quantization.quantize_transformer_onnx
  python -m phase2.quantization.model_size
  python -m phase2.quantization.compare_models cnn
  python -m phase2.quantization.compare_models transformer
  ```
</details>

### 🧠 Observations
- CNN achieves ~4× size reduction after quantization
- Transformer shows minimal size change
- CNN INT8 model fails due to unsupported ConvInteger operator
- Transformer INT8 model executes successfully 

📌 **Key Insight:**
- Quantization effectiveness depends on:
  - **model architecture**
  - **operator support in runtime**

👉 Quantization success ≠ deployment success

## ⚡ Step 4 — Performance Benchmarking

### 📊 PyTorch vs ONNX

| Model | PyTorch Latency | ONNX Latency |
|------|----------------|-------------|
| CNN | ~0.2902 ms | ~0.0926 ms |
| Transformer | ~0.8869 ms | ~0.2826 ms |

<details>
  <summary>Click to see PyTorch vs ONNX Commands</summary>

  ```bash
  python -m phase2.benchmarks.benchmark_pytorch_vs_onnx cnn
  python -m phase2.benchmarks.benchmark_pytorch_vs_onnx transformer
  ```
</details>

#### 🧠 Observations:
- ONNX provides consistent latency reduction (~3–5×)
- CNN benefits more due to efficient convolution operations
- Transformer remains slower due to attention complexity (O(n²))

📌 **Key Insight:**  
- ONNX improves execution efficiency, but model architecture dominates latency

### 📊 ONNX FP32 vs ONNX INT8 

| Model | FP32 | INT8 |
|------|------|------|
| CNN | ~0.0926 ms | ❌ Not Supported |
| Transformer | ~0.27 ms | ~0.4–0.5 ms |

<details>
  <summary>Click to see FP32 vs INT8 Commands</summary>

  ```bash
  python -m phase2.benchmarks.benchmark_onnxFP32_vs_onnxINT8 cnn
  python -m phase2.benchmarks.benchmark_onnxFP32_vs_onnxINT8 transformer
  ```
</details>

#### 🧠 Observations:
- CNN INT8 model cannot be executed
- Transformer INT8 runs but shows no speedup

📌 **Key Insight:**
- Quantization does not guarantee performance improvement

## 🧠 Step 5 — System Analysis

### 📊 End-to-End Latency

| Stage | CNN Time | Transformer Time |
|------|------|------|
| Preprocessing | 0.0174 ms | 0.0145 ms |
| Inference | 0.0687 ms | 0.2540 ms |
| Postprocessing | 0.0066 ms | 0.0073 ms |
| **Total** | **0.0930 ms** | **0.2762 ms** |


<details>
  <summary>Click to see End-to-End Latency Commands</summary>

  ```bash
  python -m phase2.benchmarks.end_to_end_latency cnn
  python -m phase2.benchmarks.end_to_end_latency transformer
  ```
</details>

#### 🧠 Observations:
- Inference dominates (~90%) of total latency
- CNN is faster due to lower compute complexity

📌 **Key Insight:**
- System is **compute-bound**  

### 📊 Thread Optimization (CPU)

| Threads | CNN Latency | Transformer Latency |
|--------|--------|--------|
| 1 | 0.0383 ms |0.1277 ms |
| 2 | 0.0368 ms | 0.1773 ms |
| 4 | 0.0323 ms | 0.1608 ms |
| 8 | 0.0398 ms | 0.2017 ms |

<details>
  <summary>Click to see Thread Optimization Commands</summary>

  ```bash
  python -m phase2.benchmarks.benchmark_threads cnn
  python -m phase2.benchmarks.benchmark_threads transformer
  ```
</details>

#### 🧠 Observations:
- Increasing threads does not improve performance
- Overhead dominates for small models

📌 **Key Insight:**
- Multi-threading is ineffective for lightweight workloads

### 📊 Batch Processing 

| Batch | CNN Latency | CNN Throughput | Transformer Latency | Transformer Throughput |
|------|------------|----------------|---------------------|------------------------|
| 1    | 0.0799 ms  | 12510.18 samples/sec | 0.2781 ms | 3595 samples/sec |
| 2    | 0.0953 ms  | 20994.20 samples/sec | 0.4311 ms | 4639 samples/sec |
| 4    | 0.0927 ms  | 43140.31 samples/sec | 0.6364 ms | 6285 samples/sec |
| 8    | 0.1233 ms  | 64870.51 samples/sec | 1.0101 ms | 7920 samples/sec |
| 16   | 0.1457 ms  | 109826.82 samples/sec | 1.5844 ms | 10098 samples/sec |

<details>
  <summary>Click to see Batch Processing Commands</summary>

  ```bash
  python -m phase2.benchmarks.benchmark_batch cnn
  python -m phase2.benchmarks.benchmark_batch transformer
  ```
</details>

#### 🧠 Observations:
- Throughput increases with batch size
- Latency also increases → trade-off

📌 **Key Insight:**
- Batching improves throughput but not suitable for real-time systems

### 📊 Load Time (Cold Start)

| Model | Load Time |
|------|----------|
| CNN FP32 | 6.86 ms |
| CNN INT8 | ❌ Failed |
| Transformer FP32 | 17.83 ms |
| Transformer INT8 | 22.31 ms |

<details>
  <summary>Click to see Load Time Commands</summary>

  ```bash
  python -m phase2.benchmarks.load_time
  ```
</details>

#### 🧠 Observations:

- Load time >> inference latency

📌 **Key Insight:**
- Cold start latency is a **critical bottleneck** in deployment scenarios  

### 📊 System Summary

| Metric | CNN | Transformer |
|------|------|------|
| Model Size | 809.77 KB | 1681.38 KB |
| Load Time | 6.32 ms | 16.21 ms |
| Inference Latency | 0.0754 ms | 0.2688 ms |
| End-to-End Latency | 0.1161 ms | 0.3048 ms |

<details>
  <summary>Click to see System Summary Commands</summary>

  ```bash
  python -m phase2.benchmarks.system_summary cnn
  python -m phase2.benchmarks.system_summary transformer
  ```
</details>


#### 🧠 Observations:

- CNN consistently outperforms Transformer

📌 **Key Insight:**
- Model architecture is the primary driver of system performance

## ⚙️ Execution Providers

### 🔍 Available Providers in my system
 
```
Available providers:
['AzureExecutionProvider', 'CPUExecutionProvider']
```

<details>
  <summary>Click to see Check Providers Command</summary>

  ```bash
  python -m phase2.validation.check_providers
  ```
</details>

### ⚡ Provider Benchmark Results
| Model | Provider | Latency |
|------|------|------|
| CNN | CPUExecutionProvider | ~0.0775 ms |
| Transformer | CPUExecutionProvider | ~0.2775 ms | 

<details>
  <summary>Click to see Provider Benchmark Commands</summary>

  ```bash
  python -m phase2.benchmarks.benchmark_provider cnn
  python -m phase2.benchmarks.benchmark_provider transformer
  ```
</details>

#### 🧠 Observations

- All inference in this project is executed using CPUExecutionProvider  

📌 **Key Insight:**
Execution backend significantly impacts performance

### ⚠️ Why Other Providers Were Not Used
- AzureExecutionProvider:
  - Designed for cloud-based execution
  - Not relevant for local benchmarking
- GPU providers:
  - Not available in current environment
  - No CUDAExecutionProvider or hardware acceleration backend

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

Deploy trained models on **STM32 Cortex-M4** and analyze real-world constraints:  
Model → ONNX → Cube.AI → CMSIS-NN → System Analysis

## 🔍 Step 1 — Model Validation (Pre-Deployment)

Validated ONNX models before deploying to embedded target.

| Model | Status | Observation |
|------|--------|------------|
| CNN FP32 | ✅ Success | Correct input/output shapes |
| CNN INT8 | ❌ Failed | `ConvInteger` unsupported |
| Transformer FP32 | ✅ Success | Dynamic batch supported |
| Transformer INT8 | ✅ Success | Runs correctly |

### 🧠 Observations

- CNN INT8 fails due to unsupported operators  
- Transformer models pass validation but include unsupported layers for deployment  

📌 **Key Insight:**
Deployment feasibility depends on **operator support**, not just model correctness  

## ⚙️ Step 2 — Cube.AI Deployment

### 📊 CNN (Baseline)

| Metric | Value |
|------|------|
| Flash | ~822 KB |
| RAM | ~22 KB |
| Latency | ~125.67 ms |

### 🧠 Observations

- Fully Connected layer dominates **~97% of Flash usage**  

📌 **Key Insight:**  
Memory bottleneck is driven by **architecture design (FC layers)**  

### ⚠️ Transformer Deployment

❌ Failed on STM32  
Unsupported: `LayerNormalization`

### 🧠 Observations

- Transformer architecture relies on unsupported operators  

📌 **Key Insight:**  
Transformer is **not deployable on Cortex-M4 using Cube.AI**

## ⚙️ Step 3 — CNN Architecture Optimization

Redesigned CNN for embedded efficiency:  
Conv → Conv → Global Average Pooling → FC(32 → 10)

### 📊 Comparison

| Metric | CNN | Optimized CNN |
|------|------|--------------|
| Parameters | ~206K | ~5K |
| Flash | ~822 KB | ~34 KB |
| RAM | ~21.5 KB | ~21.5 KB |
| MACs | ~1.25M | ~1.05M |
| Latency | ~125.67 ms | ~107.08 ms |

### 🧠 Observations

- Flash reduced by **~24×**  
- RAM unchanged → dominated by activations  
- Latency improvement is **limited**

📌 **Key Insight:**
- Parameter reduction ≠ proportional latency reduction  
- System remains **compute-bound (convolutions dominate)**  

## ⚙️ Step 4 — Cube.AI Optimization Modes (on optimized CNN)

| Mode | Latency | RAM |
|------|--------|-----|
| Balanced | ~107 ms | ~21 KB |
| Time | ~108 ms | ~56 KB |

### 🧠 Observations

- Increasing RAM does not improve latency  

📌 **Key Insight:**  
Optimization modes affect **memory layout**, not computation cost  

## ⚠️ Step 5 — INT8 Deployment Limitation

INT8 model failed during Cube.AI Analyze.

Unsupported operators:
- ConvInteger  
- MatMulInteger  
- DynamicQuantizeLinear  

### 🧠 Observations

- External quantization is incompatible with Cube.AI  

📌 **Key Insight:**  
ONNX INT8 ≠ STM32-compatible INT8  

👉 Deployment depends on **toolchain compatibility**

## 🧠 Step 6 — System-Level Insights (Cube.AI)

### 🧠 Observations

- Convolution layers dominate compute (~85% MACs)  
- Flash memory is easy to reduce  
- Compute cost is difficult to reduce  

📌 **Key Insight:**

- Embedded ML is primarily:
  - **compute-bound**
  - limited by **FP32 arithmetic**

## 🔥 Step 7 — CMSIS-NN Manual Pipeline

### 🎯 Objective

Build full CNN inference manually using INT8.

### ⚙️ Implementation Details

- CMSIS-NN manually integrated  
- INT8 weights and activations  
- INT32 accumulation  
- Manual requantization (multiplier + shift)  
- Custom FC (`linear_s8`) implementation  
- PyTorch weights exported and used directly  

### 📊 Final Results

| Metric | Value |
|------|------|
| Cycles | ~103.8M |
| Latency | ~865.46 ms |
| Prediction | Matches PyTorch |

### ⚠️ Key Debugging Challenges

#### 1. Quantization Scaling

#### 🧠 Observations

- Incorrect scaling collapsed outputs to zero  
- Required layer-wise tuning  

📌 **Key Insight:**  
Quantization correctness is critical for valid inference  

#### 2. Input Representation

#### 🧠 Observations

- Direct int8 input caused incorrect mapping  

📌 **Key Insight:**  
Input representation affects entire inference pipeline  

#### 3. Architecture Mismatch 

#### 🧠 Observations

- FC mismatch: 800 vs 1568  

📌 **Key Insight:**  
Model and deployment architecture must be identical  

#### 4. Flatten Layout Issue

#### 🧠 Observations

- Manual reorder broke FC behavior  

📌 **Key Insight:**  
Tensor layout assumptions must match framework expectations  

## ⚖️ Step 8 — CMSIS-NN vs Cube.AI

### ⚡ Performance Comparison (baseline CNN)

| Metric | Cube.AI | CMSIS-NN |
|------|---------|----------|
| Inference Latency | ~125.67 ms | ~865.46 ms |
| Inference Cycles | Not exposed | ~103.8 Million cycles |
| Flash Usage | ~822 KB (network report) | ~265.6 KB (full firmware image) |
| RAM Usage | ~21.5 KB (network report) | ~2.46 KB (full firmware image) |
| Optimization Level | Tool-driven | Manual (baseline implementation) |

📌 Note:
Cube.AI memory figures are taken from the generated network report and represent network-level resource usage. CMSIS-NN memory figures are taken from STM32 build output and represent total firmware size, including application code and global buffers.Therefore, Flash and RAM values are useful for deployment context, but they are not strict like-for-like network-only comparisons.

### 🧠 Observations

- CMSIS-NN is **~6.8× slower** in current implementation  

📌 **Key Insight:**  
CMSIS-NN requires **manual optimization to outperform Cube.AI**

## 🎯 Step 10 — Accuracy & Validation

### 📊 Results (baseline CNN)

| Platform | Predicted Class |
|---------|----------------|
| PyTorch | 7 |
| ONNX | 7 |
| STM32 (Cube.AI) | 7 |
| STM32 (CMSIS-NN) | 7 |

### 🧠 Observations

- Outputs match across all platforms  
- Minor numerical differences due to quantization  

📌 **Key Insight:**  
Correctness validated through **cross-platform consistency**

## 🔍 Phase 3 — Key Insights

- CNN deployable; Transformer not feasible on Cortex-M4  
- Model architecture defines deployment feasibility  
- Cube.AI provides efficient out-of-box performance  
- CMSIS-NN provides control but requires heavy optimization  
- Quantization depends on toolchain compatibility  
- Embedded ML is compute-bound  

📌 **Final Insight:**  
Efficient embedded ML requires **co-design of model + toolchain + hardware**, not just model optimization

---

# 📊 Final System Comparison & Summary

This section consolidates results across all phases to provide a **complete system-level view** of the ML deployment pipeline.

## ⚡ End-to-End Latency Comparison

| Stage | CNN Latency | Transformer Latency |
|------|------------|---------------------|
| PyTorch (Phase 1) | ~0.290 ms | ~0.886 ms |
| ONNX Runtime (Phase 2) | ~0.092 ms | ~0.282 ms |
| STM32 Cube.AI (Phase 3) | ~125 ms | ❌ Not Deployable |
| STM32 CMSIS-NN (Phase 3) | ~865 ms | ❌ Not Implemented |

## 📦 Model Size Comparison

| Model | FP32 | INT8 |
|------|------|------|
| CNN | ~810 KB | ~208 KB |
| Transformer | ~1681 KB | ~1692 KB |

## ⚙️ Deployment Feasibility

| Model | ONNX Runtime | STM32 Cube.AI | STM32 CMSIS-NN |
|------|-------------|--------------|----------------|
| CNN FP32 | ✅ | ✅ | ✅ |
| CNN INT8 | ❌ | ❌ | ⚠️ Manual |
| Transformer FP32 | ✅ | ❌ | ❌ |
| Transformer INT8 | ✅ | ❌ | ❌ |

## 🧠 Observations

- ONNX Runtime provides **~3–5× speedup** over PyTorch across models  
- CNN consistently outperforms Transformer due to **lower computational complexity**  

- Quantization:
  - Effective for CNN (size reduction)
  - Ineffective for Transformer (no size or speed benefit)

- CNN INT8 fails due to **ConvInteger operator limitation**  
- Transformer fails on STM32 due to **unsupported layers (LayerNorm)**  

- Embedded deployment introduces **massive latency increase**:
  - ONNX → STM32 Cube.AI: ~1000× slower  
  - CMSIS-NN (baseline) even slower without optimization  

- Despite differences in execution:
  - Predictions remain **consistent across all platforms**

## 📌 Key System-Level Insights

- Model architecture is the **primary driver of performance and deployability**  
- Runtime/backend support determines whether a model can actually execute  

- Optimization trade-offs:
  - ONNX → speed  
  - Quantization → size (conditional)  
  - Cube.AI → ease of deployment  
  - CMSIS-NN → control + complexity  

- Embedded ML systems are:
  - **compute-bound**
  - constrained by **hardware + toolchain limitations**

## 🚀 Final Takeaway

> Building an ML model is only the beginning.  
> Real-world deployment requires alignment between:

- model architecture  
- runtime support  
- hardware constraints  
👉 Efficient ML systems are achieved through **co-design of model + deployment stack + hardware**, not model optimization alone.
---

# 🗂️ Repository Structure
```text
edgeai-ml-systems/
phase1/  
  models/    
  training/    
  benchmarks/    
  utils/        
  configs/      

phase2/  
  benchmarks/    
  cmsis/  
  export/  
  models/  
  quantization/  
  validation/  

phase3/  
  stm32_cube_ai/    
  notes/  
  debug/   
 
docs/  
  system_pipeline1.png    
  transformer_encoder1.png

README.md  
requirements.txt
```
---

# 🏁 Conclusion

This project demonstrates an end-to-end ML systems pipeline spanning:

- Model development (CNN, Transformer)  
- ONNX-based deployment and optimization  
- System-level benchmarking and analysis  
- Embedded deployment on STM32 (Cube.AI + CMSIS-NN)  

## 🧠 Key Takeaways

- Model performance is governed by **architecture, not just parameter count**  
- ONNX Runtime enables **consistent inference acceleration (~3–5×)**  
- Quantization effectiveness depends on **backend operator support**  
- CNN is well-suited for edge deployment, while Transformer faces **deployment limitations**  
- Embedded inference is **compute-bound**, with significant latency overhead  
- Toolchains (Cube.AI vs CMSIS-NN) introduce **different trade-offs between performance and control**

## 📌 Final Insight

> Machine learning systems are not defined by models alone.  
> They are defined by the interaction between:

- model architecture  
- runtime/backend  
- hardware constraints  

👉 Efficient ML systems are achieved through **co-design of model + deployment stack + hardware**, not model optimization alone.

---

# ▶️ How to Run
## Setup Environment

```
conda create -n edgeai python=3.10  
conda activate edgeai  
pip install -r requirements.txt
```

## Train Models (Phase 1)

```
python -m phase1.training.train
python -m phase1.training.train --model_name transformer
python -m phase1.training.train --model_name cnn_optimized
```

## Export & Validate (Phase 2) 

```
python -m phase2.export.export_cnn_onnx
python -m phase2.export.export_transformer_onnx

python -m phase2.validation.validate_onnx_vs_pytorch cnn
python -m phase2.validation.validate_onnx_vs_pytorch transformer
```

### Benchmark Performance (Phase 2)

```
python -m phase2.benchmarks.benchmark_pytorch_vs_onnx cnn
python -m phase2.benchmarks.benchmark_pytorch_vs_onnx transformer
```
### Embedded Deployment (Phase 3)
- 1. Import ONNX model into STM32CubeMX
- 2. Generate code using X-CUBE-AI
- 3. Build and flash using STM32CubeIDE
- 4. Observe inference output via UART

---

# 👨‍💻 Author
### C. Sivananda Reddy
---