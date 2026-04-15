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