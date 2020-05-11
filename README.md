# RippleNet

## Description

This repository contains files for RippleNet, a recurrent neural network with LSTM
layers for detecting sharp-wave ripples in single-channel LFP signals measured
in hippocampus CA1.

Author: Espen Hagen (https://github.com/espenhgn)

LICENSE: https://github.com/espenhgn/RippleNet/blob/master/LICENSE

[![DOI](https://zenodo.org/badge/257224892.svg)](https://zenodo.org/badge/latestdoi/257224892)

## Clone

These codes can be downloaded using git (www.git-scm.com):

    cd <Repositories> # whatever download destination
    git clone https://github.com/espenhgn/RippleNet
    cd RippleNet

Some binary files like `.h5` and `.pkl` may be tracked using Git LFS (https://git-lfs.github.com)

## dependencies

- `python>=3`
- `numpy`
- `scipy`
- `ipympls`
- `matplotlib`
- `h5py`
- `pandas`
- `seaborn`
- `notebook`
- `jupyter`
- `tensorflow>=2.0`
- `tensorflow-gpu` (optional)

Dependencies can be installed in your existing Python environment using
the `requirements.txt` file and the `pip` utility:

    pip install -r requirements.txt

To install an Anaconda Python (www.anaconda.com) environment with these dependencies, issue

    conda env create -f environment.yml
    conda activate ripplenet

This will not install `tensorflow-gpu` for hardware acceleration by default.


## Binder

You may mess around with the RippleNet notebooks on MyBinder.org:
https://mybinder.org/v2/gh/espenhgn/RippleNet/master

Retraining networks is not recommended (no GPU access)!

## Files and folders:

- `README.md`: This file
- `LICENSE`: License file
- `environment.yml`: Conda environment file
- `RippleNet_training_bidirectional.ipynb`: Jupyter notebook for training bidirectional RippleNet
- `RippleNet_training_unidirectional.ipynb`: Notebook for training unidirectional RippleNet
- `RippleNet_manuscript_figures.ipynb`: Notebook for generating figures 2-7 in Hagen E. et al. (2020)
- `RippleNet_timeseries_prediction.ipynb`: Notebook	for generating figures 8-11 in Hagen E. et al. (2020)
- `RippleNet_interactive_prototype.ipynb`: Notebook with user-interactive detection and rejection of ripple events
- `trained_networks/`
    - `ripplenet_*directional_random_seed*.h5`: trained RippleNet instances of uni- or bidirectional types
    - `ripplenet_*directional_best_random_seed*.h5`: best-performing model on validation set during training
    - `ripplenet_*directional_history_random_seed*.csv`: training history (.csv format)
    - `ripplenet_*directional_history_random_seed*.pkl`: training history (.pickle format)
- `ripplenet/`
    - `common.py`: shared methods and functions
    - `models.py`: function declarations for `tensorflow.keras` models
- `data/`
    - `train_00.h5`: Training data set (mouse)
    - `train_tingley_00.h5`: Training data set (rat)
    - `validation_00.h5`: Validation data set (mouse)
    - `validation_00.h5`: Validation data set (rat)
    - `test_00.h5`: Test data set (mouse)
    - `m4029_session1.h5`: Test data set (mouse, continuous)
