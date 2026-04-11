# Day 3 - Cube.AI Analysis

## CNN FP32
- Import: SUCCESS
- Flash: ~822 KB
- RAM: ~22 KB
- All layers supported

## CNN INT8
- Import: FAILED
- Unsupported ops:
  - ConvInteger
  - MatMulInteger
  - DynamicQuantizeLinear

## Key Insight
QOperator-based INT8 models are not supported by STM32Cube.AI.