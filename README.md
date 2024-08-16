# **Frequency-Masked Embedding Inference: A Non-Contrastive Approach for Time Series Representation Learning**
This repository contains the official code for Frequency-masked Embedding Inference (FEI). 
This version corresponds to the initial submission of the paper to AAAI-2025.

# Repository Structure
```
- config/                All model and training configurations
- datasets_clsa/         Classification dataset construction
- datasets_reg/          Regression dataset construction
- models/                Contains Baseline implementations and FEI architecture code
- train/                 Base class for all training code, as well as saving training logs and results
- util/                  All utility methods, including the frequency masking code for FEI
- experiments.py         Entry point for running training and testing
```

# Requirements
```
numpy~=1.24.3
torch~=2.0.1
scikit-learn~=1.3.0
matplotlib~=3.7.2
tsaug~=0.2.1
pandas~=2.0.3
```

# Preparing Datasets
## Classification Datasets
All classification datasets should follow the structure of train.pt/test.pt/val.pt, where each .pt file contains a dictionary with keys "samples" and "labels," corresponding to the sample and label data. See the TF-C dataset structure for details.

After downloading, place the datasets in the datasets_clsa folder. For example, the correct directory structure for the Gesture dataset should be as follows:
```
- datasets_clsa
  - Gesture
    - train.pt
    - test.pt
    - val.pt
```

## Regression Datasets
No additional processing is needed for regression datasets. Simply place them in the datasets_reg folder:
```
- datasets_clsa
  - CMAPSS
    - FD001
    - FD002
    - FD003
    - FD004
```

# Quick Start
To quickly start pre-training, use the following command:
> python ./experiment.py --task_type=p --method=FEI

After pre-training, you can find the corresponding logs and results in the `train/model_result/` directory. To validate the pre-trained model, use the following command:
> python ./experiment.py --model=./train/model_result/your_model_path --task_type=l --task=c --dataset=FDB --method=FEI

You can adjust the task type, dataset, and other parameters by modifying the arguments like --task_type and --dataset. For more help with run parameters, use:
> python ./experiment.py -h

Further code details will be described after the paper is accepted and the code is made publicly available.
