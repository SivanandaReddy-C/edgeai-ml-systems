# Day 8 - Measure Inference Latency on STM32
## 🎯 Goal

By end of today, you must have:

✅ Inference latency measured
✅ Printed on UART
✅ Stable across runs
✅ Ready to compare later

## Completion report:
Day 8 Complete:
- Latency measurement: success
- Method: DWT cycle counter + 100-run validation
- Cycles: ~1,285,000 cycles
- Latency: ~107.08 ms per inference
- Stability: highly stable across runs
- Validation:
  - 100 runs took ~10708 ms
  - Confirms ~107 ms per inference
- Observations:
  - Latency dominated by FP32 computation
  - Cycle count consistent across runs
  - Measurement verified using both cycle counter and time-based averaging
- Conclusion:
  - Current model runs at ~107 ms on STM32 Cortex-M4
- Commit message:
  Measured and validated CNN inference latency on STM32 (~107 ms) using DWT and batch timing

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
- Baseline Flash: ~841,924 B (~822 KB)
- Baseline RAM: ~22,044 B (~21.5 KB)
- Baseline latency: ~125.67 ms
- Optimized Flash: ~33,756 B (~34 KB)
- Optimized RAM: ~21.5 KB
- Optimized latency: ~107.08 ms

- Comparison summary:
  - Parameter reduction: ~40×
  - Flash reduction: ~24×
  - RAM: nearly unchanged
  - Latency improvement: ~15%

- Key insight:
  - Memory optimization significantly reduced Flash usage
  - Latency improvement limited because convolution layers dominate compute
  - FP32 execution on Cortex-M4 remains the primary bottleneck

- Commit message:
  Compared baseline and optimized CNN performance on STM32 (memory vs latency tradeoff analysis)

  # Day 10 - Speed Optimization (INT8/CMSIS-NN Direction)
  ## 🎯 Goal
  By end of Day 10, you should:
  - Understand why FP32 is slow
  - Try INT8 path (where possible)
  - Explore Cube.AI optimizations
  - Measure latency improvement (if any)
  - Build strong system-level explanation

  ## Completion report
  Day 10 Complete:
- Cube.AI optimization modes tested: yes
- Modes tested: Balanced, Time
- Balanced latency: ~107.08 ms
- Time latency: ~108.12 ms

- INT8 model tested: yes
- INT8 support status: not supported (ConvInteger issue)

- Key observations:
  - Changing optimization mode had negligible impact on latency
  - Conv2 layer dominates computation (~85% MACs)
  - FP32 operations (~98%) are the main performance bottleneck
  - Cube.AI settings cannot significantly reduce compute-heavy workload

- Conclusion:
  - Latency is dominated by model architecture and numerical precision
  - Significant speedup requires INT8 quantization or optimized kernels (CMSIS-NN)
  - Cube.AI optimization modes provide only minor improvements

- Commit message:
  Evaluated Cube.AI optimization modes; confirmed FP32 compute bottleneck and limited impact on latency