# Day 1 - STM32 Environment Setup
- STM32CubeIDE installed: yes
- CubeMX working: yes
- X-CUBE-AI installed: yes
- Project build status: sucess
- Issues faced: 0
- Commit message: "Set up STM32CubeIDE, CubeMX, and X-CUBE-AI for B-L4S5I-IOT01A"

# Day 2 - Prepare model for STM32 deployment
- File(s) created/used: phase2/onnx/validate_onnx.py
- How to run validate_onnx.py
  - TO VALIDATE CNN FP32: python -m phase2.onnx.validate_onnx phase2/onnx/cnn.onnx
  - TO VALIDATE TRANSFORMER FP32: python -m phase2.onnx.validate_onnx phase2/onnx/transformer.onnx
  - TO VALIDATE CNN INT8: python -m phase2.onnx.validate_onnx phase2/onnx/cnn_int8.onnx
  - TO VALIDATE TRANSFORMER INT8: python -m phase2.onnx.validate_onnx phase2/onnx/transformer_int8.onnx
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
- File(s) created/updated: phase3/stm32_cube_ai/<your_project>.ioc
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
- File(s) created/updated: Nothing
- Main bottleneck identified: CNN Flash usage dominated by FC layer (Identified through Cube.AI report)
- Flash usage: ~822KB
- RAM usage: ~22KB
- Optimization ideas:
  1. Reduce FC size (1568 → 32)
  2. Replace Flatten with Global Average Pooling
- Chosen strategy: Keep the original CNN as baseline and create an optimized CNN variant using Global Average Pooling with a smaller classifier head
- Commit message: Analyzed CNN memory bottleneck and selected GAP-based optimization strategy for embedded deployment

# Day 5 - Build Optimized CNN (GAP-based)
- File(s) created/updated: phase1/models/cnn_optimized.py
- What to do:
  - Create: phase1/models/cnn_optimized.py; phase2/scripts/export_onnx.py
  - Export: python phase2/scripts/export_onnx.py cnn_opt phase2/onnx/cnn_optimized.onnx
  - Validate: python -m phase2.onnx.validate_onnx phase2/onnx/cnn_optimized.onnx
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

# Day 6 - Import optimized CNN into Cube.AI and compare
Import cnn_optimized.onnx into STM32Cube.AI and compare it against the original CNN on:
- Flash
- RAM
- layer support
- memory bottleneck shift

## Completion report:
- Optimized CNN import into Cube.AI: success
- Input shape: [1, 28, 28, 1]
- Output shape: [1, 10]

- Total Flash: ~33.7 KB
- Weights: ~20.5 KB
- Total RAM: ~21.5 KB
- Activations: ~19.3 KB

- Warnings: none
- Unsupported layers: none

- Comparison vs original CNN:
  - Flash reduced from ~822 KB → ~34 KB (~24× reduction)
  - Weights reduced from ~808 KB → ~20 KB (~40× reduction)
  - RAM remains ~same (~22 KB)

- Commit message:
  Imported optimized CNN into Cube.AI and demonstrated ~24x flash reduction using GAP-based architecture

# Day 7: Generate project and prepare first inference path
Goal:

- By the end of today, you should have:
  - Cube.AI code generated for the optimized CNN
  - project opened in STM32CubeIDE
  - build successful
  - UART enabled for printf
  - clear place identified for input buffer, inference call, and output print

## Completion report:
Day 7 Complete:
- Optimized model selected in Cube.AI: yes
- Code generation: success
- CubeIDE build: success
- UART configured: yes
- printf working: yes
- AI wrapper integration: manual (network.c API)
- Inference execution: success
- Output observed on UART: yes
- Issues faced:
  - printf not working initially due to missing stdio.h and UART init order
  - resolved using _write() retarget and correct initialization order
- Commit message:
  Integrated optimized CNN into STM32 firmware and successfully executed inference with UART output