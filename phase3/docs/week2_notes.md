# Day 8 - Measure Inference Latency on STM32
## 🎯 Goal

By end of today, you must have:

✅ Inference latency measured
✅ Printed on UART
✅ Stable across runs
✅ Ready to compare later

## Completion report:
Day 8 Complete:
- Model: Baseline CNN (original architecture)
- Deployment: STM32 X-CUBE-AI integration successful

- Memory Analysis:
  - Parameters: ~206,922
  - Flash: ~841,924 B (~822 KB)
  - RAM: ~22,044 B (~21.5 KB)

- Compute Complexity:
  - Total MACs: ~1.25M operations
  - Dominant compute: Conv2 layer (~73% of operations)
  - FC layer contributes significant compute (~16%)

- Observations:
  - Flash memory is heavily dominated by FC1 layer (~97% of total weights)
  - Convolution layers dominate runtime computation
  - RAM usage remains relatively small and stable (~21 KB)
  - Model size is too large for efficient embedded deployment due to FC layer

- Insights:
  - Fully Connected layer is the primary bottleneck for Flash usage
  - Convolution layers are the primary bottleneck for computation/latency
  - Highlights the need for architectural optimization (e.g., Global Average Pooling)

- Conclusion:
  - Baseline CNN is functional but memory-inefficient for embedded deployment
  - Provides reference point for optimization comparison (Day 9)

- Commit message:
  Analyzed baseline CNN memory and compute profile on STM32; identified FC layer as major Flash bottleneck

# Day 9 - Compare baseline vs optimized model on STM32
## 🎯 Goal
By the end of today, you should have:

- baseline CNN imported/generated on STM32
- latency measured for baseline CNN
- direct comparison against optimized CNN
- clear conclusion on memory vs latency tradeoff

## Completion report:
Day 9 Complete:
- Baseline CNN generated on STM32: yes
- Baseline CNN inference: success

- Baseline Model Details:
  - Parameters: ~206,922
  - Flash: ~841,924 B (~822 KB)
  - RAM: ~22,044 B (~21.5 KB)
  - MACs: ~1.25M operations

- Optimized Model Details:
  - Parameters: ~5,130
  - Flash: ~33,756 B (~34 KB)
  - RAM: ~21,500 B (~21.5 KB)
  - MACs: ~1.05M operations

- Latency Measurements:
  - Baseline latency: ~125.67 ms
  - Optimized latency: ~107.08 ms

- Comparison summary:
  - Parameter reduction: ~40×
  - Flash reduction: ~24×
  - RAM: nearly unchanged (activation-dominated)
  - MAC reduction: ~16% (1.25M → 1.05M)
  - Latency improvement: ~15%

- Key observations:
  - Baseline CNN Flash is dominated by FC layer (~97% of weights) :contentReference[oaicite:0]{index=0}
  - Optimized CNN removes FC bottleneck using Global Average Pooling
  - Convolution layers dominate compute in both models (~85–90%)
  - RAM usage remains similar because activations (~19 KB) dominate memory in both cases :contentReference[oaicite:1]{index=1}
  - All operations are FP32 (100%), limiting performance gains :contentReference[oaicite:2]{index=2}

- Insights:
  - Memory optimization is highly effective for reducing Flash usage
  - Removing FC layers drastically reduces parameters but only moderately reduces computation
  - Latency improvement is limited because Conv2 layer dominates MAC operations
  - Performance bottleneck is compute-bound, not memory-bound

- Conclusion:
  - Architectural optimization (GAP-based CNN) achieves significant memory savings (~24× Flash reduction)
  - Latency improvement (~15%) is modest due to FP32 computation and convolution dominance
  - Further speedup requires quantization (INT8) or optimized kernels (e.g., CMSIS-NN)

- Commit message:
  Compared baseline vs optimized CNN on STM32; achieved ~24× Flash reduction with modest (~15%) latency improvement and identified compute bottlenecks

  # Day 10 - Speed Optimization (INT8/CMSIS-NN Direction)
  ## 🎯 Goal
  By end of Day 10, you should:
  - Understand why FP32 is slow
  - Try INT8 path (where possible)
  - Explore Cube.AI optimizations
  - Measure latency improvement (if any)
  - Build strong system-level explanation

  ## Completion report:
Day 10 Complete:
- Cube.AI optimization modes tested: yes
- Modes tested: Balanced, Time

- Latency Results:
  - Balanced latency: ~107.08 ms
  - Time latency: ~108.12 ms
  - Observation: negligible difference (~1%)

- Model Characteristics (same across modes):
  - MACs: ~1.05M operations
  - Parameters: ~5,130
  - Compute graph unchanged

- Memory Differences:
  - Balanced mode:
    - Activations: ~19 KB
    - Total RAM: ~21.5 KB
  - Time mode:
    - Activations: ~53 KB (~2.7× increase)
    - Total RAM: ~56 KB

- Key observations:
  - Optimization mode does NOT change compute (MACs remain constant)
  - Conv2 layer dominates computation (~85% of MACs) :contentReference[oaicite:5]{index=5}
  - Time optimization increases RAM significantly by allocating larger intermediate buffers
  - Despite higher RAM usage, latency improvement is negligible
  - FP32 operations dominate (~98–100%), limiting execution speed :contentReference[oaicite:6]{index=6}

- Insights:
  - Performance is compute-bound, not memory-bound
  - Increasing memory (Time mode) does not improve speed on Cortex-M4
  - Optimization modes mainly affect memory layout, not arithmetic workload
  - Conv2 layer remains the critical bottleneck regardless of optimization mode

- INT8 Evaluation:
  - INT8 model tested: yes
  - Status: not supported (ConvInteger unsupported in CPUExecutionProvider)
  - Indicates limitation in deployment pipeline for quantized CNN

- Conclusion:
  - Cube.AI optimization modes have minimal impact on latency for compute-heavy FP32 models
  - Time mode trades higher RAM (~2.7×) for negligible performance gain
  - True speedup requires:
    - INT8 quantization support, or
    - optimized kernels (CMSIS-NN), or
    - architecture-level compute reduction

- Commit message:
  Evaluated Cube.AI optimization modes; identified compute-bound behavior and confirmed limited latency gains despite increased memory usage

# Day 11 — Model Comparison (FP32 vs INT8)
## 🎯 Goal
Compare FP32 vs INT8
→ Latency
→ Memory

## Completion report:
Day 11 Complete:
- FP32 model deployment on STM32: verified
- INT8 model import attempt: completed

- FP32 Results:
  - Baseline CNN:
    - Flash: ~822 KB
    - RAM: ~21.5 KB
    - Latency: ~125.67 ms
  - Optimized CNN:
    - Flash: ~34 KB
    - RAM: ~21.5 KB
    - Latency: ~107.08 ms

- INT8 Evaluation:
  - Model: CNN INT8 (ONNX)
  - Status: Failed during Cube.AI Analyze step
  - Error:
    - Unsupported layer types:
      - ConvInteger
      - MatMulInteger
      - DynamicQuantizeLinear

- Comparison summary:
  - FP32 models: fully functional and deployable on STM32
  - INT8 model: not deployable due to unsupported operators
  - Latency comparison not possible (INT8 execution failed)
  - Memory comparison not available for INT8 in Cube.AI

- Key observations:
  - INT8 ONNX model introduces quantized operators (ConvInteger, MatMulInteger)
  - These operators are not supported in STM32Cube.AI toolchain
  - Cube.AI supports FP32 models and its own internal optimization flow
  - External ONNX quantization is not directly compatible with STM32 deployment

- Insights:
  - Quantization alone does not guarantee deployability on embedded systems
  - Operator compatibility is a critical constraint in deployment pipelines
  - There is a mismatch between ONNX Runtime capabilities and STM32Cube.AI support
  - Deployment success depends on toolchain support, not just model format

- Conclusion:
  - FP32 models are currently the only reliable deployment path on STM32 in this setup
  - INT8 deployment is limited by lack of support for quantized ONNX operators
  - Achieving INT8 acceleration requires alternative approaches such as:
    - Cube.AI internal quantization (if supported)
    - CMSIS-NN optimized kernels
    - Architecture-level redesign

- Commit message:
  Evaluated FP32 vs INT8 deployment on STM32; identified ConvInteger limitation preventing INT8 execution in Cube.AI