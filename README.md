# RippleNet

## Description

This repository contains files for RippleNet, a recurrent neural network with LSTM
layers for detecting sharp-wave ripples in single-channel LFP signals measured
in hippocampus CA1.

Author: Espen Hagen (https://github.com/espenhgn)

LICENSE: https://github.com/espenhgn/RippleNet/blob/master/LICENSE

[![DOI](https://zenodo.org/badge/257224892.svg)](https://zenodo.org/badge/latestdoi/257224892)

## Citation

RippleNet and its application is described in a preprint, which can be cited as:

**RippleNet: A Recurrent Neural Network for Sharp Wave Ripple (SPW-R) Detection**  
Espen Hagen, Anna R. Chambers, Gaute T. Einevoll, Klas H. Pettersen, Rune Enger, Alexander J. Stasik  
*bioRxiv* 2020.05.11.087874; doi: https://doi.org/10.1101/2020.05.11.087874

BibTex format:
```
@article {Hagen2020.05.11.087874,
	author = {Hagen, Espen and Chambers, Anna R. and Einevoll, Gaute T. and Pettersen, Klas H. and Enger, Rune and Stasik, Alexander J.},
	title = {RippleNet: A Recurrent Neural Network for Sharp Wave Ripple (SPW-R) Detection},
	elocation-id = {2020.05.11.087874},
	year = {2020},
	doi = {10.1101/2020.05.11.087874},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Hippocampal sharp wave ripples (SPW-R) have been identified as key bio-markers of important brain functions such as memory consolidation and decision making. SPW-R detection typically relies on hand-crafted feature extraction, and laborious manual curation is often required. In this multidisciplinary study, we propose a novel, self-improving artificial intelligence (AI) method in the form of deep Recurrent Neural Networks (RNN) with Long Short-Term memory (LSTM) layers that can learn features of SPW-R events from raw, labeled input data. The algorithm is trained using supervised learning on hand-curated data sets with SPW-R events. The input to the algorithm is the local field potential (LFP), the low- frequency part of extracellularly recorded electric potentials from the CA1 region of the hippocampus. The output prediction can be interpreted as the time-varying probability of SPW-R events for the duration of the input. A simple thresholding applied to the output probabilities is found to identify times of events with high precision. The reference implementation of the algorithm, named {\textquoteright}RippleNet{\textquoteright}, is open source, freely available, and implemented using a common open-source framework for neural networks (tensorflow.keras) and can be easily incorporated into existing data analysis workflows for processing experimental data.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2020/05/12/2020.05.11.087874},
	eprint = {https://www.biorxiv.org/content/early/2020/05/12/2020.05.11.087874.full.pdf},
	journal = {bioRxiv}
}
```

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
