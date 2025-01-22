# N-Bit Memory Task with JAX and Equinox

## Overview
This repo compiles my work evaluating dynamical systems on RNNs trained on various tasks

## Experiments so far:
1. 3BitFF-dt-experiment.ipynb: The models are trained with varying time resolutions (`dts`) and evaluated on data with varying time resolutions (`dts`). We observe interesting dynamics when the model is trained on high resolution data and used to evaluate low resolution data.

## Directory Structure
- `src/`: Source code for data generation, model utilities, training, and plotting.
- `experiments/`: Experiment scripts and notebooks.
- `saved_models/`: Directory for storing trained models and metadata. The models are saved with unique `model_id` which is a hash of the metadata.
   - `{model_id}/`: 
      - `metadata.json`: Stores model hyperparameters, data hyperparameters and optimizer hyperparameters.
      - `model.eqx`: Model parameters
      - `loss.csv`: If the model has been trained, logs the training loss and the training step which the loss was recorded.

