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
- **Phase 2:** Deployment & Optimization (ONNX Runtime)  
- **Phase 3:** Embedded Deployment on STM32  
  - Cube.AI (high-level)
  - CMSIS-NN (low-level manual pipeline)  

Focus areas:

- Model training and architecture design  
- Performance profiling and benchmarking  
- Deployment optimization using ONNX Runtime  
- Embedded AI execution on STM32 (Cortex-M4)  
- System-level analysis (latency, throughput, memory)


---

## ❓ Why This Project Matters

Most ML work stops at training.

This project answers:

- How does a model behave **after deployment?**
- What breaks when moving to **edge devices?**
- What really limits performance — **compute, memory, or tooling?**

---

## 💡 Highlights

- Built CNN and Transformer from scratch  
- Achieved **~5–10× speedup** using ONNX Runtime  
- Implemented INT8 quantization and analyzed limitations  
- Identified runtime constraints (`ConvInteger` unsupported)  
- Performed full system benchmarking (latency, throughput, memory)  
- Deployed models on **STM32 Cortex-M4**  
- Reduced Flash usage by **~24× via architecture redesign**  
- Built **manual CMSIS-NN inference pipeline from scratch**  
- Debugged real deployment issues (quantization, layout, architecture mismatch)

---

# 🔹 Phase 1 — Model Development & Benchmarking

## 🎯 Objectives

- CNN training pipeline
- Transformer encoder from scratch
- Profiling & benchmarking

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

# 🔹 Phase 3 — Embedded Deployment on STM32

## 🎯 Objective
Deploy optimized ML models on **resource-constrained embedded hardware (STM32 Cortex-M4)** and analyze real execution behavior beyond desktop environments.

---

## ⚙️ Step 1 — Environment Setup

- Installed STM32CubeIDE  
- Verified CubeMX configuration  
- Installed X-CUBE-AI package  
- Created STM32 project for B-L4S5I-IOT01A  

📌 Build status: **0 errors**

---

## ⚙️ Step 2 — Model Validation (Pre-Deployment)

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

## ⚙️ Step 3 — Cube.AI Model Analysis

Imported CNN model into STM32Cube.AI:

### Baseline CNN

- Flash: ~822 KB  
- RAM: ~22 KB  

📌 Observation:  
- Fully Connected (FC) layer dominates memory (~97% of Flash)

---

## 🧠 Step 4 — Model Optimization  

Redesigned CNN architecture for embedded deployment:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Conv → Conv → Global Averaging Pooling → FC(32 → 10) 


### Key Changes

- Removed Flatten + large FC layer  
- Introduced Global Average Pooling (GAP)  

📌 Result:
- Drastic reduction in parameters  

---

## 🔥 Step 5 — Memory Optimization Results

| Model | Flash | RAM |
|------|------|-----|
| Original CNN | ~822 KB | ~22 KB |
| Optimized CNN | ~34 KB | ~21.5 KB |

👉 **~24× reduction in Flash memory**

📌 Insight:
- Flash → dominated by weights  
- RAM → dominated by activations  

---

## ⚙️ Step 6 — STM32 Firmware Integration

- Generated STM32 project using CubeMX  
- Imported optimized model into Cube.AI  
- Used **low-level `network.c` API** for integration  
- Configured UART for logging  
- Implemented `printf` retarget using `_write()`  

---

## 🚀 Step 7 — On-Device Inference Execution

Successfully executed inference on STM32:

- Input: dummy tensor `[1, 28, 28, 1]`  
- Output: 10-class logits  
- Output observed via UART  

### Example Output Behavior

- Stable across iterations  
- Deterministic for fixed input  

📌 Confirms:
- Correct model execution  
- Proper firmware integration  

---

## ⚠️ Challenges Faced

- CNN INT8 model failed due to unsupported operators (`ConvInteger`)  
- ONNX export issues due to opset mismatch  
- Reshape/Flatten incompatibility in Cube.AI  
  → resolved using tensor indexing  
- UART `printf` not working initially  
  → resolved using `_write()` retarget and correct initialization order  

---

## 🔥 Phase 3 — Extended System Analysis

This section extends Phase 3 beyond deployment into **detailed system-level evaluation** using real STM32 execution data.

---

## 📊 Model Comparison on STM32

| Metric | Baseline CNN | Optimized CNN |
|------|-------------|--------------|
| Parameters | ~206K | ~5K |
| Flash | ~822 KB | ~34 KB |
| RAM | ~21.5 KB | ~21.5 KB |
| MACs | ~1.25M | ~1.05M |
| Latency | ~125.67 ms | ~107.08 ms |

---

## 🧠 Memory Behavior

- Flash dominated by FC layer (~97%)
- Optimization removes FC bottleneck → ~24× reduction
- RAM unchanged → activation dominated (~21 KB)

👉 Weight optimization ≠ runtime memory optimization  

---

## ⚙️ Compute & Latency Analysis

- Conv2 layer dominates (~85% MACs)
- System is **compute-bound**
- FP32 arithmetic is primary bottleneck

👉 Parameter reduction does NOT linearly reduce latency  

---

## ⚙️ Cube.AI Optimization Modes

| Mode | Latency | RAM |
|------|--------|-----|
| Balanced | ~107 ms | ~21 KB |
| Time | ~108 ms | ~56 KB |

👉 Memory increase does not improve speed  
👉 Optimization affects layout, not computation  

---

## ⚠️ INT8 Deployment Limitation

INT8 model failed during Cube.AI Analyze:

Unsupported operators:
- ConvInteger  
- MatMulInteger  
- DynamicQuantizeLinear  

### Insight

- ONNX INT8 ≠ STM32-compatible INT8  
- External quantization not supported  

👉 Deployment depends on **operator + toolchain compatibility**

---

## 🧪 Stability & Robustness Testing

Tested with:
- Valid MNIST input  
- Zero input  
- Max input  
- Noise input  
- 200 repeated runs  

### Results

- Failed runs: 0  
- Invalid outputs: 0  
- Prediction changes: 0  
- Stable latency (~126 ms)

### Edge Behavior

| Input | Prediction |
|------|-----------|
| Zero | 1 |
| Max | 5 |
| Noise | 3 |

👉 Model always produces valid output  
👉 Stable but not semantically reliable for invalid inputs  

---

## 🧠 System-Level Insights

- Embedded ML is dominated by:
  - Convolution compute
  - FP32 precision cost  

- Memory vs Compute:
  - Flash → easy to reduce  
  - Compute → hard to reduce  

- Toolchain matters:
  - ONNX success ≠ STM32 compatibility  

---

## ⚠️ Limitations

- FP32 inference latency is high (~100+ ms)  
- INT8 deployment unsupported  
- No CMSIS-NN optimization yet  

---

# 🔥 Step 7 — CMSIS-NN Manual Inference Pipeline (Week 3)

## 🎯 Objective

Move beyond Cube.AI and build a **manual CNN inference pipeline using CMSIS-NN (INT8)** to understand low-level execution on Cortex-M4.

Target pipeline:

Input → Conv1 → ReLU → MaxPool  
→ Conv2 → ReLU → MaxPool  
→ Flatten → FC1 → ReLU → FC2

---

## ⚙️ Implementation Details

- CMSIS-NN manually integrated into STM32 project
- INT8 weights and activations used across all layers
- INT32 accumulation used to avoid overflow
- Explicit requantization using **multiplier + shift**
- Custom fully connected layer (`linear_s8`) implemented
- Stage-wise outputs printed via UART for validation
- Real PyTorch-learned parameters exported into C arrays and executed on STM32

---

## 🧪 Week 3 Progress Summary

| Day | Milestone | Result |
|------|----------|--------|
| Day 15 | CMSIS-NN integration into STM32 project | ✅ Successful |
| Day 16 | Fixed-point inference study | ✅ Understood and validated |
| Day 17 | Minimal CMSIS-NN convolution test | ✅ Correct output verified |
| Day 18 | Real Conv1 weights on STM32 | ✅ Working |
| Day 19 | Conv1 requantization fix | ✅ Stable, non-saturated output |
| Day 20 | Conv1 → ReLU → MaxPool pipeline | ✅ Working |
| Day 21 | Added Conv2 | ✅ Working |
| Day 22 | Added Conv2 → ReLU → MaxPool | ✅ Working |
| Day 23 | Added Flatten + FC1 + FC2 | ✅ Full pipeline executed |
| Day 24 | PyTorch vs STM32 validation | ✅ Root cause identified |
| Day 25 | Architecture + layout fix | ✅ STM32 prediction matched PyTorch |

---

## 📊 Stage-Wise Results

### Day 17 — First Real CMSIS-NN Convolution

Minimal convolution layer validated successfully on STM32.

**Validation setup**
- Input tensor: `4 × 4`
- Kernel: `3 × 3`
- Output tensor: `2 × 2`

**Expected raw output**
- `[-6, -6, -6, -6]`

**Observed STM32 output**
- `[-6, -6, -6, -6]`

**Status**
- `ARM_CMSIS_NN_SUCCESS (0)`

📌 Result:  
Verified that CMSIS-NN convolution was executing correctly independent of later multi-layer pipeline issues.

---

### Day 18 — Real Conv1 Layer on STM32

Integrated real Conv1 weights exported from PyTorch.

**Validation setup**
- Input: MNIST-style `28 × 28` image
- Kernel size: `3 × 3`
- Output tensor: `26 × 26 × 16`

**Results**
- Conv1 status: `0`
- Output sample: non-zero and varied
- Saturation count: `+127 = 0`, `-128 = 0`

📌 Result:  
First real learned CNN layer successfully executed on STM32.

---

### Day 19 — Conv1 Requantization Fix

Corrected multiplier/shift handling for Conv1.

**Results**
- Conv1 status: `0`
- Conv1 output min/max: `-40 / 47`
- Saturation: `+127 = 0`, `-128 = 0`

📌 Result:  
Conv1 output became numerically meaningful, stable, and suitable for chaining into later layers.

---

### Day 20 — Conv1 → ReLU → MaxPool

Extended the single-layer test into a mini inference pipeline.

**Results**
- Conv1 output sample: structured signed values
- ReLU output sample: negative values removed correctly
- MaxPool output sample: strongest activations preserved
- Conv1 min/max: `-40 / 47`
- ReLU min/max: `0 / 47`
- MaxPool min/max: `0 / 47`

📌 Result:  
Confirmed stable tensor propagation across Conv1, ReLU, and MaxPool.

---

### Day 21 — Added Conv2

Integrated Conv2 using pooled Conv1 output as input.

**Results**
- Conv2 status: `0`
- Conv2 output sample: structured signed values
- Conv2 min/max: `-106 / 80`
- Saturation: `+127 = 0`, `-128 = 0`

📌 Result:  
Two-layer quantized CNN pipeline became functional on STM32.

---

### Day 22 — Conv2 → ReLU → MaxPool

Extended the pipeline deeper and validated downstream feature flow.

**Results**
- Conv2 output sample: structured signed values
- Conv2 ReLU output: negatives removed correctly
- Conv2 MaxPool output: strong activations preserved
- Conv2 min/max: `-106 / 80`
- Conv2 ReLU min/max: `0 / 80`
- Conv2 MaxPool min/max: `0 / 80`

📌 Result:  
Confirmed stable multi-stage CNN feature extraction on STM32.

---

### Day 23 — Full CNN Inference Path

Added Flatten, FC1, ReLU, and FC2 to build complete end-to-end inference.

**Results**
- Flatten output sample: valid pooled feature vector observed
- FC1 output sample: structured signed values
- FC1 ReLU output sample: negatives removed correctly
- FC2 logits: `-12 -11 8 -25 9 -26 -25 -5 -9 -15`
- Predicted class: `4`

**Ranges**
- FC1 min/max: `-62 / 70`
- FC1 ReLU min/max: `0 / 70`
- FC2 min/max: `-26 / 9`

📌 Result:  
End-to-end CNN inference executed successfully on STM32, but prediction still did not match PyTorch.

---

## ⚠️ Key Debugging Challenges

### 1. Quantization Scaling Issues

- Incorrect multiplier/shift caused valid outputs to collapse to zero
- Required layer-wise tuning of output scales
- Verified correctness using min/max and saturation checks

📌 Insight:  
Quantization correctness is critical — wrong scaling destroys meaningful inference.

---

### 2. Input Representation

- Direct int8 handling of image input caused incorrect mapping
- Fixed by:
  - using uint8 input
  - converting properly to int8 before inference

📌 Insight:  
Input representation directly affects downstream convolution correctness.

---

### 3. Architecture Mismatch (Critical Issue)

During PyTorch vs STM32 comparison, the key mismatch was identified:

- STM32 FC1 input size: `800`
- Trained PyTorch model FC1 input size: `1568`

**Root cause**
- STM32 path used no-padding style output shape at that stage
- Trained PyTorch model used `padding = 1`, so final feature map before FC1 was `32 × 7 × 7 = 1568`

📌 Impact:  
Even correct integer math could not match the trained model because the deployed architecture itself was different.

---

### 4. Flatten Layout Mismatch (Critical Issue)

Initial assumption:
- Flatten needed a manual HWC → CHW reorder before FC1

Actual behavior:
- The pooled CMSIS-NN output was already aligned with the FC layer expectation for the working pipeline
- Manual reorder scrambled the data and broke FC behavior

**Final fix**
```c
for (int i = 0; i < 7 * 7 * 32; i++) {
    flatten_out[i] = conv2_pool_out[i];
}

---


# ⚖️ CMSIS-NN vs Cube.AI Comparison

## 🎯 Objective

Compare two deployment approaches on STM32 using the **same CNN architecture**:

- CNN:  
  Conv → Conv → Flatten → FC(1568 → 128 → 10)

This ensures a **fair comparison** across identical model structure.

---

## 📊 Comparison Summary

| Aspect | Cube.AI | CMSIS-NN |
|-------|--------|----------|
| Integration Effort | Very low | High (manual implementation) |
| Control over pipeline | Limited | Full control |
| Development Time | Fast | Slow |
| Debug visibility | Limited | Full (layer-wise inspection) |
| Flexibility | Low | High |
| Quantization control | Hidden (tool-driven) | Explicit (manual scaling) |

---

## ⚡ Performance Perspective (Same CNN)

| Metric | Cube.AI (CNN) | CMSIS-NN (CNN) |
|------|--------------|----------------|
| Inference Latency | ~125.67 ms | ~865.46 ms |
| Inference Cycles | Not exposed | ~103.8 Million cycles |
| Flash Usage | ~822 KB | Similar (weights dominated) |
| RAM Usage | ~21.5 KB | ~21.5 KB |
| Optimization Level | Tool-driven | Manual (baseline implementation) |

---

## 📊 CMSIS-NN Measured Performance

Benchmark setup:

- 2 warmup runs + 10 measured runs  
- Cycle-level measurement using DWT counter  
- Full inference pipeline measured:
  Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC1 → ReLU → FC2  

### Results (STM32 Cortex-M4)

- Average Cycles: ~103,855,240  
- Average Latency: ~865.46 ms  

---

## ⚠️ Key Observation

👉 CMSIS-NN implementation is **~6.8× slower than Cube.AI** for the same CNN model  

---

## 🧠 Why CMSIS-NN is Slower (In This Implementation)

Although CMSIS-NN provides optimized kernels:

- Only convolution layers use CMSIS optimized functions  
- Pooling and ReLU are implemented manually  
- Fully connected layers use custom `linear_s8` implementation  
- No operator fusion (Conv + ReLU + Pool are separate)  
- Memory access patterns are not optimized  

👉 Result:

The system behaves like a **correct reference pipeline**,  
not a fully optimized CMSIS implementation.

---

## 🔥 Real Debugging Insight (From This Project)

Using CMSIS-NN enabled identification of issues that are hard to detect in tool-driven pipelines:

- FC1 dimensional mismatch (800 vs 1568)  
- Flatten layout mismatch  
- Quantization scaling inconsistencies  

👉 These issues are difficult to diagnose using Cube.AI alone due to abstraction.

---

## 🧠 Engineering Trade-offs

### Cube.AI

**Pros**
- Fast deployment  
- Automatically optimized execution  
- Efficient memory handling  
- Lower latency out-of-the-box  

**Cons**
- Limited visibility into internal execution  
- Hard to debug layer-level issues  
- Less control over quantization behavior  

---

### CMSIS-NN

**Pros**
- Full control over:
  - data flow  
  - quantization  
  - memory layout  
- Enables detailed debugging  
- Provides deeper understanding of inference pipeline  

**Cons**
- High development effort  
- Easy to introduce structural and scaling errors  
- Performance depends heavily on manual optimization  

---

## 🧠 Final Takeaway

> Cube.AI is optimized for **deployment and performance out-of-the-box**  
> CMSIS-NN is designed for **control, debugging, and understanding**

---

## 🎯 When to Use What

| Scenario | Recommended Approach |
|---------|---------------------|
| Quick deployment | Cube.AI |
| Production systems | Cube.AI |
| Debugging model behavior | CMSIS-NN |
| Custom architecture control | CMSIS-NN |
| Learning / research | CMSIS-NN |

---

## 🚀 Final Insight

> CMSIS-NN is not a faster alternative by default  

👉 It is a **low-level toolkit**, not a full deployment engine  

👉 In this project:

- Cube.AI provides **better performance out-of-the-box**  
- CMSIS-NN provides **deeper control and visibility**  

👉 Performance advantage depends on **manual optimization effort**, not just the library itself  

---

## 🧠 Performance Reality (This Work)

| Observation | Result |
|------------|-------|
| CMSIS vs Cube.AI | ~6.8× slower |
| Implementation type | Baseline (non-optimized) |
| Correctness | ✅ Verified |
| Optimization stage | Not yet performed |

👉 This represents a **baseline CMSIS implementation**,  
not an optimized one.

---
## 🔥 Phase 3 Key Achievement
> End-to-end pipeline validated:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyTorch → ONNX → Cube.AI → STM32 → UART → System Analysis

--- 

## 🎯 Model Accuracy & Validation

### 📊 Training Accuracy (Phase 1)

- CNN validation accuracy: ~99% (MNIST)

---

### 🔁 Cross-Platform Output Consistency

To ensure deployment correctness, outputs were compared across:

- PyTorch (baseline)
- ONNX Runtime (CPU)
- STM32 CMSIS-NN implementation

---

### ✅ Observations

- ONNX outputs closely match PyTorch outputs  
- STM32 outputs match integer reference implementation  
- Final predicted class matches PyTorch for tested samples  

📌 Example:

| Platform | Predicted Class |
|---------|----------------|
| PyTorch | 7 |
| ONNX | 7 |
| STM32 (CMSIS-NN) | 7 |

---

### ⚠️ Quantization Impact

- INT8 quantization introduces minor numerical differences  
- No classification mismatch observed in tested samples  
- Full dataset accuracy evaluation on STM32 not performed  

---

### 🧠 Key Insight

> Deployment correctness was validated through **cross-platform consistency**,  
> not full dataset accuracy evaluation on device

---

### 🚀 Conclusion

- Model retains functional correctness after deployment  
- Pipeline verified from training → ONNX → embedded inference  
- Accuracy degradation (if any) is negligible for tested cases

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