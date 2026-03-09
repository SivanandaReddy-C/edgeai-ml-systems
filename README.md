# Edge AI ML Systems - Phase 1

This repository contains implementations built from scratch for learning ML systems engineering.
The goal of phase 1 is to build a clean ML pipeline and measure performance characteristics.

## Phase 1 Objectives

- Implement CNN training pipeline
- Implement Transformer Encoder from scratch
- Profile training performance
- Benchmark CNN vs Transformer

## System Pipeline
![Pipeline](docs/system_pipeline.png)
## Repository Structure

edgeai-ml-systems-phase1/

models/  
&nbsp;&nbsp;&nbsp;&nbsp;cnn.py  

training/  
&nbsp;&nbsp;&nbsp;&nbsp;train.py 

utils/  
&nbsp;&nbsp;&nbsp;&nbsp;dataset.py  

configs/
&nbsp;&nbsp;&nbsp;&nbsp;cnn.yaml 

benchmarks/
&nbsp;&nbsp;&nbsp;&nbsp;benchmark.py 

## CNN Architecture
The implemented CNN architecture follows this pipeline:

Input (1x28x28)

Conv2d (1 → 16, kernel=3, padding=1)  
ReLU  
MaxPool (2×2)

Conv2d (16 → 32, kernel=3, padding=1)  
ReLU  
MaxPool (2×2)

Flatten

Fully Connected (1568 → 128)  
ReLU

Fully Connected (128 → 10)

Total Parameters:206,922

## Training Pipeline
The training loop follows the standard PyTorch workflow:
Forward Pass → Loss Calculation → Backpropagation → Optimizer Update

Components:
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Dataset: MNIST

## Benchmark Results
| Metric | Result |
|------|------|
Training Time (1 epoch) | 19.09 s |
Single Inference Latency | 0.050 ms |
Batch Inference Latency (32) | 1.215 ms |
Model Parameters | 206,922 |

## How to Run
### Create environment
conda create -n edgeai python=3.10
conda activate edgeai

### Install dependencies
pip install -r requirements.txt

### Train the model
python -m training.train

### Run benchmarks
python -m benchmarks.benchmark

## Lessons Learned
Key engineering insights from this phase:
-Modular ML project design
-Importance of dataset abstraction
-Training loop mechanics
-Benchmarking inference latency
-Configuration-driven experiments

## Author
C Sivananda Reddy