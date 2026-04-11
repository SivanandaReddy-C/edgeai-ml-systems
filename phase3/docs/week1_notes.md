# Day 1 - STM32 Environment Setup
- STM32CubeIDE installed: yes
- CubeMX working: yes
- X-CUBE-AI installed: yes
- Project build status: sucess
- Issues faced: 0
- Commit message: "Set up STM32CubeIDE, CubeMX, and X-CUBE-AI for B-L4S5I-IOT01A"

# Day 2 - Prepare model for STM32 deployment
- Model used: CNN FP32, CNN INT8, Transformer FP32, Transformer INT8
- Model paths: phase2/onnx/
- CNN FP32:
  - Input shape: [1, 1, 28, 28]
  - Output shape: [1, 10]
  - Inference: success
- CNN INT8:
  - Inference: failed
  - Issue: ConvInteger not implemented in ONNX Runtime CPUExecutionProvider
- Transformer FP32:
  - Input shape: dynamic batch_size
  - Output shape: dynamic batch_size
  - Inference: success
- Transformer INT8:
  - Inference: success
- Issues faced:
  - CNN INT8 incompatible with ONNX Runtime CPU due to ConvInteger
- Commit message: Prepared and validated ONNX models for STM32 deployment

# Day 3 - Import model into STM32Cube.AI
- CNN FP32 import into Cube.AI: success
- Input shape: [1, 28, 28, 1]
- Output shape: [1, 10]
- Flash usage: ~822 KB
- RAM usage: ~22 KB
- Warnings: none
- Unsupported layers: none

- CNN INT8 import into Cube.AI: failed
- Warnings:
- Unsupported layers:
  - ConvInteger
  - DynamicQuantizeLinear
  - MatMulInteger

- Issues faced:
  - CNN INT8 model uses QOperator-based quantization not supported by Cube.AI
- Commit message:
  Imported CNN models into Cube.AI; FP32 works, INT8 unsupported due to quantization operators

# Day 4 — Memory Optimization Insights
- Main bottleneck identified: CNN Flash usage dominated by FC layer
- Flash usage: ~822KB
- RAM usage: ~22KB
- Optimization ideas:
  1. Reduce FC size (1568 → 32)
  2. Replace Flatten with Global Average Pooling
- Chosen strategy: Keep the original CNN as baseline and create an optimized CNN variant using Global Average Pooling with a smaller classifier head
- Commit message: Analyzed CNN memory bottleneck and selected GAP-based optimization strategy for embedded deployment

# Day 5 - Build Optimized CNN (GAP-based)
- Replaced Flatten + FC with Global Average Pooling
- Reduced parameters drastically (~200K → ~320)

## Key Insight:
Global Average Pooling removes spatial redundancy and is highly efficient for embedded deployment.

## Completion report:
- Optimized model created: yes
- ONNX export: success
- ONNX validation: success
- Output shape: [1, 10]
- Observations:
  - Export issue fixed by using opset_version=18 and dynamo=False
  - Optimized model runs correctly in ONNX Runtime
- Commit message:
  Implemented optimized CNN with Global Average Pooling and successfully exported and validated ONNX model