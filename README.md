# N-Bit Memory Task with JAX and Equinox

## Overview
This project implements and trains a continuous-time recurrent neural network (RNN) to solve varying tasks.

## Tasks
The models are trained with varying time resolutions (`dts`) and evaluated to compare their performance.

## Directory Structure
- `src/`: Source code for data generation, model utilities, training, and plotting.
- `experiments/`: Experiment scripts and notebooks.
- `tests/`: Unit tests for various components.
- `saved_models/`: Directory for storing trained models and metadata (ignored by `.gitignore`).

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nbit-memory-task.git
   cd nbit-memory-task