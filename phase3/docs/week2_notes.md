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
- Latency
- Memory

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

# Day 12 — Stress Testing
## 🎯 Goal
Test robustness of inference on STM32
- Stability
- Consistency
- Edge cases

## Completion report:
Day 12 Complete:
- Continuous inference loop: implemented
- Multi-input stress testing: completed
- Edge case testing: completed

- Test scenarios:
  - Valid MNIST input
  - All-zero input
  - All-255 input
  - Deterministic noise input
  - 200 repeated inferences per case

- Results:
  - Failed runs: 0 (no execution failures)
  - Invalid outputs: 0 (no NaN or corrupted outputs)
  - Prediction changes: 0 (deterministic inference)
  - Latency: ~126.7 ms (consistent across all tests)

- Observations:
  - Inference stable across repeated executions
  - No memory corruption or crashes observed
  - Output values remained valid for all input types
  - Model produces consistent predictions even for invalid inputs

- Edge case behavior:
  - Zero input → predicted class: 1
  - Max input → predicted class: 5
  - Noise input → predicted class: 3
  - Indicates model always produces a valid classification

- Key insights:
  - Model is robust but not semantically reliable for invalid inputs
  - Neural networks inherently map any input to a class
  - Stability does not imply correctness for out-of-distribution data
  - FP32 inference is numerically stable on Cortex-M4

- Conclusion:
  - STM32 inference pipeline is stable and reliable under sustained execution
  - System is safe for real-time deployment scenarios
  - Additional input validation may be required in real-world applications

- Commit message:
  Performed stress testing of STM32 inference pipeline; validated stability, determinism, and robustness across valid and edge-case inputs

# Day 13 — Internal Documentation: Week 2 Insights

## 1. Baseline vs Optimized CNN
- Baseline CNN:
  - Flash: ~822 KB
  - RAM: ~21.5 KB
  - Latency: ~125.67 ms
- Optimized CNN:
  - Flash: ~34 KB
  - RAM: ~21.5 KB
  - Latency: ~107.08 ms

## 2. Memory Insights
- Baseline CNN Flash usage is dominated by the FC layer.
- Optimized CNN removes the large FC bottleneck using Global Average Pooling.
- RAM remains almost unchanged because activations dominate runtime memory.

## 3. Compute Insights
- Conv2 is the main compute bottleneck in both baseline and optimized models.
- Parameter reduction does not translate linearly into latency reduction.
- FP32 execution on Cortex-M4 is the primary reason latency remains high.

## 4. Toolchain Limitations
- INT8 ONNX CNN could not be imported into STM32Cube.AI.
- Unsupported operators observed:
  - ConvInteger
  - MatMulInteger
  - DynamicQuantizeLinear
- Conclusion: external ONNX quantization is not directly deployable in current STM32Cube.AI flow.

## 5. Stability Insights
- Stress testing across valid and invalid inputs showed:
  - 0 failed runs
  - 0 invalid outputs
  - 0 prediction changes for repeated identical inputs
- The pipeline is deterministic and numerically stable.

## 6. Overall Week 2 Conclusion
- Architectural optimization gave major Flash savings but only modest latency improvement.
- Current deployment path is reliable with FP32 models.
- Significant speedup will require:
  - quantization support inside the deployment toolchain, or
  - CMSIS-NN / lower-level optimized kernels.