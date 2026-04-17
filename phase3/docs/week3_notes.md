# Day 15 - Integrate CMSIS-NN into your STM32 project
## Completion Report:
Day 15 Complete:
- New STM32 project created: yes
- CMSIS-NN integrated manually: yes
- Core include path added: yes
- NN include path added: yes
- CMSIS-NN source files added to build: yes
- Build status: success
- Basic kernel integration test: success
- UART project remains functional: yes

- Observations:
  - CMSIS-NN is not exposed directly through the current CubeIDE GUI workflow
  - Manual integration is required for this setup
  - Successful build confirms headers, include paths, and source linkage are correct

- Key takeaway:
  - CMSIS-NN integration requires careful project-level setup before any model work can begin
  - Low-level embedded ML work is fundamentally different from Cube.AI middleware usage

- Commit message:
  Integrated CMSIS-NN library into project

# Day 16 - FIXED-POINT + CONVOLUTION (CORE UNDERSTANDING)
## 🎯 Goal
Understand how CNN computation actually runs on Cortex-M using CMSIS-NN

## Completion report
Day 16 Complete:
- Fixed-point inference pipeline studied: yes
- INT8 / INT32 data flow understood: yes
- Convolution computation breakdown: completed
- Manual dot-product + scaling test: implemented and verified

- Key observations:
  - INT8 operations significantly reduce memory and computation cost
  - Accumulation must be done in INT32 to avoid overflow
  - Scaling (requantization) is critical to map results back to INT8 range
  - Incorrect scaling leads to wrong predictions

- Key takeaway:
  - Embedded ML is fundamentally integer arithmetic, not floating-point
  - Model accuracy depends heavily on quantization and scaling correctness

- Commit message:
  Studied fixed-point inference and CMSIS-NN computation pipeline

# Day 17 - FIRST REAL CMSIS-NN CONVOLUTION
## 🎯 Goal
Run a minimal convolution using CMSIS-NN (not full CNN)

## Completion report
Day 17 Complete:
- CMSIS-NN convolution API integrated: yes
- Minimal convolution layer executed on STM32: yes
- Build and link issues resolved: yes
- End-to-end kernel execution verified: yes

- Validation:
  - Test input: 4×4 tensor
  - Kernel: 3×3 filter
  - Output: 2×2 tensor
  - Expected raw output: [-6, -6, -6, -6]
  - Observed output: [-6, -6, -6, -6]
  - Status: ARM_CMSIS_NN_SUCCESS (0)

- Key debugging steps:
  - Resolved API signature mismatch (added upscale_dims)
  - Fixed linker errors by including required NNSupportFunctions sources
  - Identified and corrected quantization issue causing zero outputs
  - Switched to near-identity quantization for validation

- Observations:
  - CMSIS-NN convolution includes requantization and activation internally
  - Incorrect multiplier/shift can collapse valid outputs to zero
  - Kernel execution depends on multiple internal helper functions
  - Successful execution requires correct dims, buffers, and quant params

- Key insight:
  - Verified correctness of convolution computation independent of quantization artifacts
  - Established a reliable low-level validation pipeline for CMSIS-NN kernels

- Conclusion:
  - CMSIS-NN convolution kernel successfully validated on STM32
  - System is ready to integrate real model weights and proper quantization

- Commit message:
  Validated CMSIS-NN convolution kernel on STM32 with correct output and quantization handling

# Day 18 
## 🎯 Goal
real CNN weights → C arrays → STM32

## Completion report
Day 18 Complete:
- Conv1 weights exported from PyTorch: yes
- Conv1 bias exported and quantized: yes
- Real Conv1 layer integrated into CMSIS-NN test project: yes
- STM32 execution status: success

- Validation:
  - Input: MNIST-style 28×28 image
  - Conv1 kernel size: 3×3
  - Output tensor: 26×26×16
  - CMSIS-NN status: 0 (success)
  - Output sample: non-zero and varied
  - Saturation count: +127=0, -128=0

- Key fixes made:
  - Replaced raw int8 image storage with uint8 input + explicit conversion
  - Corrected input mapping to avoid signed overflow issues
  - Updated Conv1 bias to larger properly scaled int32 values
  - Reduced output saturation using smaller temporary multiplier

- Observations:
  - Real exported Conv1 weights now execute successfully on STM32
  - Output is no longer dominated by clipping
  - Layer behavior is structurally correct and numerically stable
  - Output quantization is still approximate and not yet fully calibrated

- Key takeaway:
  - PyTorch Conv1 weights can be transferred into CMSIS-NN and executed correctly on STM32
  - Correct input representation and bias scaling are critical for meaningful output

- Conclusion:
  - First real CNN layer has been successfully validated using CMSIS-NN on STM32
  - System is ready to move toward multi-layer chaining and pipeline building

- Commit message:
  Integrated real Conv1 weights into CMSIS-NN pipeline and validated first CNN layer on STM32

# Day 19 
## 🎯 Goal
Make Conv1 output numerically meaningful by implementing correct requantization (multiplier + shift)
- (int32 accumulator → correct int8 output)