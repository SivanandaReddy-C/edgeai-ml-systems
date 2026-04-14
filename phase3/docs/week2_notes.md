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