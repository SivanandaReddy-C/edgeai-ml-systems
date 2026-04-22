# Final Benchmark Summary — End-to-End ML System

## Objective

Summarize performance across all stages of the pipeline:

Training → ONNX → Embedded Deployment

---

## 📊 Benchmark Table

| Model | Platform | Precision | Latency | Status | Notes |
|------|--------|-----------|--------|--------|------|
| CNN | PyTorch (CPU) | FP32 | ~0.73 ms | ✅ Working | Training baseline |
| CNN | ONNX Runtime | FP32 | ~0.09 ms | ✅ Working | ~8× faster than PyTorch |
| CNN | ONNX Runtime | INT8 | — | ❌ Failed | ConvInteger not supported |
| Transformer | ONNX Runtime | FP32 | ~0.27 ms | ✅ Working | Baseline inference |
| Transformer | ONNX Runtime | INT8 | ~0.30 ms | ✅ Working | Slight overhead |
| CNN | STM32 Cube.AI | INT8 | ~125.67 ms | ✅ Working | Tool-optimized deployment |
| CNN | STM32 CMSIS-NN | INT8 | ~865.46 ms | ✅ Working | Manual baseline implementation |
| Transformer | STM32 Cube.AI | FP32 | — | ❌ Not deployable | LayerNormalization unsupported |

---

## 📌 Key Observations

### 1. Performance Evolution
- ONNX provides significant speedup over PyTorch  
- Embedded deployment introduces large latency increase  

---

### 2. Embedded Constraints
- CNN successfully deploys on STM32  
- Transformer fails due to unsupported operators  

---

### 3. CMSIS vs Cube.AI
- Cube.AI is significantly faster (~6–7×)  
- CMSIS implementation is correctness-focused, not optimized  

---

### 4. Quantization Reality
- INT8 does not always improve performance (ONNX case)  
- Toolchain support determines usability  

---

## 🧠 Final Insight

> ML model performance is not only about architecture  
> It depends heavily on deployment platform and toolchain support

---

## 🚀 Conclusion

- CNN is practical for edge deployment  
- Transformer is not suitable for MCU deployment (current setup)  
- Tool-driven optimization (Cube.AI) outperforms manual pipelines unless heavily tuned  