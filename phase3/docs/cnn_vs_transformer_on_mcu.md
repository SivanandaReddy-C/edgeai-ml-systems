# CNN vs Transformer Feasibility on STM32 MCU

## Objective

Compare CNN and Transformer architectures for deployment on STM32 microcontrollers.

---

## 1. Toolchain Compatibility

### CNN
- Successfully analyzed and deployed using STM32Cube.AI
- All required layers supported
- End-to-end inference executed on STM32

### Transformer
- Failed during STM32Cube.AI analysis
- Unsupported operator:
  - LayerNormalization

### Observation
CNN is compatible with the STM32 deployment toolchain, while Transformer is not deployable in its current form.

---

## 2. Computational Complexity

### CNN
- Convolution-based computation
- Local receptive fields
- Lower complexity for image tasks

### Transformer
- Self-attention based computation
- Complexity grows as O(n²)
- Heavier compute demand for embedded MCU

### Observation
Transformer is significantly more compute-intensive than CNN, making it less suitable for MCU-class devices.

---

## 3. Memory Feasibility

### CNN
- Fits within STM32 memory constraints
- Flash and RAM usage measurable and manageable

### Transformer
- Larger intermediate tensors
- Less suitable for limited MCU memory
- Deployment blocked before full memory analysis due to unsupported ops

### Observation
CNN is memory-feasible on STM32, while Transformer is not practical under current constraints.

---

## 4. Deployment Outcome

| Aspect | CNN | Transformer |
|--------|-----|-------------|
| Cube.AI Analysis | Success | Failed |
| Operator Support | Supported | LayerNormalization unsupported |
| Inference on STM32 | Successful | Not deployable |
| Feasibility on MCU | High | Low |

---

## Final Conclusion

CNN is a practical and deployable architecture for STM32 MCU deployment.

Transformer is not suitable for STM32 deployment in its current form due to:
- unsupported operators
- higher computational complexity
- poor fit with MCU memory and toolchain constraints

---

## Key Insight

MCU deployment is not only about model accuracy.

It depends strongly on:
- supported operators
- memory footprint
- computational complexity
- toolchain compatibility