# Transformer Deployment Failure Analysis (STM32)

## Model
- Architecture: Transformer Encoder
- Source: Phase 2 ONNX export

## Tool
- STM32Cube.AI (X-CUBE-AI)

---

## Failure Observation

Model analysis failed with error:

Unsupported layer:
- LayerNormalization

---

## Root Cause

Transformer architecture depends on operations not supported on MCU:

### 1. Layer Normalization
- Required after attention and feedforward blocks
- Not supported by STM32Cube.AI

### 2. Computational Complexity
- Attention requires O(n²) operations
- High compute demand not suitable for MCU

### 3. Memory Requirements
- Large intermediate tensors
- Exceeds typical MCU RAM limits

---

## Comparison with CNN

| Feature | CNN | Transformer |
|--------|-----|-------------|
| Operator support | Fully supported | Not supported |
| Memory usage | Low | High |
| Complexity | Linear | Quadratic |

---

## Conclusion

Transformer models are not suitable for deployment on STM32 MCU using current tools.

CNN-based models are far more feasible for embedded deployment.