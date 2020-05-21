#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Some functions and definitions used by different jupyter notebooks

Author: Espen Hagen (<https://github.com/espenhgn>)

LICENSE: <https://github.com/espenhgn/RippleNet/blob/master/LICENSE>
'''
# import os
import numpy as np
import scipy.signal as ss
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import tensorflow
# from tensorflow import keras
# from tcn import TCN
# import h5py
# from glob import glob
import pandas as pd
# import pickle
# from matplotlib import colors


rcParams = {
          'axes.labelpad': 0.0,
          'axes.labelsize': 'medium',
          'axes.labelweight': 'normal',
          'axes.titlesize': 14.,
          'axes.titleweight': 'normal',
          'axes.xmargin': 0.0,
          'axes.ymargin': 0.0,
          'figure.frameon': True,
          'figure.titlesize': 'large',
          'figure.titleweight': 'normal',
          'font.size': 14.0,
          'font.style': 'normal',
          'font.variant': 'normal',
          'font.weight': 'normal',
          'legend.fontsize': 'medium',
          'legend.title_fontsize': None,
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'ytick.major.size': 3.5,
          'ytick.minor.size': 2.0,
}


def annotate_subplot(ax, ncols=1, nrows=1, letter='A',
                     linear_offset=0.025, fontsize=16,
                     fontweight='demibold'):
    '''add a subplot annotation'''
    ax.text(-ncols*linear_offset, 1+nrows*linear_offset, letter,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize,
            fontweight=fontweight,
            transform=ax.transAxes)


def get_metrics(X, Y, Y_pred, labels, all_labels,
                threshold, distance, width,
                y_label_width=62,
                Fs=1250,
                decimals=3):
    '''
    Compute TP, FP, FN counts and precision, recall and F1 scores.

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 / (precision**-1 + recall**-1)

    Parameters
    ----------
    X: ndarray
    Y: ndarray
    Y_pred: ndarray
    labels: structured ndarray
    all_labels: structured ndarray
    threshold: float
    distance: int
    width: int
    Fs: float
    decimals: int

    Returns
    -------
    TP: int
    FP: int
    FN: int
    precision: float
    recall: float
    F1: float
    '''
    # network is not trained on negatives, so we quantify true/false positives
    # and false negatives
    TP = 0
    # TN = 0
    FP = 0
    FN = 0
    # Determine time(s) and probability(ies) in sample as the time and
    # magnitude of local maxima following a threshold crossing from below
    for i in range(X.shape[0]):
        # find times of local peaks above threshold
        y = Y_pred[i, :, 0]
        peaks, _ = ss.find_peaks(y, height=threshold, distance=distance,
                                     width=width)

        def get_j():
            '''get time index of all labeled events'''
            idx = all_labels['session'] == labels['session'][i]
            lbls = all_labels['rippleLocs'][idx] - \
                labels[i]['rippleLocs'] - labels[i]['offset']
            return lbls[(lbls >= 0) & (lbls < Fs)]

        if peaks.size == 0:
            # no event was predicted, so all labeled event times are
            # counted as FNs
            for j in get_j():
                FN += 1
        else:
            # add labeled events to list of FNs if no peak found was when y==1
            for j in get_j():
                y_j = np.zeros(X.shape[1])
                y_j[j] = 1
                y_j = np.convolve(y_j, ss.boxcar(y_label_width), 'same')
                if np.all(y_j[peaks] == 0):
                    FN += 1
        # predicted events must be TPs or FPs
        for j in peaks:
            # check if label == 1 at time of peaks, if so, count TP
            if Y[i, j, 0] == 1:
                TP += 1
            else:
                # peak is occurring at time where label == 0, count as FP
                FP += 1

    # Evaluation metrics
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    recall = TP / (TP + FN)
    try:
        F1 = 2 / (precision**-1 + recall**-1)
    except ZeroDivisionError:
        F1 = 0

    return (TP, FP, FN,
            np.round(precision, decimals=decimals),
            np.round(recall, decimals=decimals),
            np.round(F1, decimals=decimals))


def get_TPs_FPs_FNs(X, Y, Y_pred, X1, S, labels, all_labels,
                    threshold, distance, width,
                    Fs=1250, y_label_width=62):
    '''
    Extract data segments

    Parameters
    ----------
    X: ndarray
    Y: ndarray
    Y_pred: ndarray
    labels: structured ndarray
    all_labels: structured ndarray
    threshold: float
    distance: int
    width: int
    Fs: float
    decimals: int

    Returns
    -------
    TPs: structured ndarray
    FPs: structured ndarray
    FNs: structured ndarray
    '''
    # containers
    TPs = []
    # TNs = []
    FPs = []
    FNs = []
    # Determine time(s) and probability(ies) in sample as the time and
    # magnitude of local maxima following a threshold crossing from below
    for i in range(X.shape[0]):
        # find times of local peaks above threshold
        y = Y_pred[i, :, 0]
        peaks, _ = ss.find_peaks(y, height=threshold, distance=distance,
                                     width=width)

        def get_j():
            '''get time index of all labeled events'''
            idx = all_labels['session'] == labels['session'][i]
            lbls = all_labels['rippleLocs'][idx] - labels[i]['rippleLocs'] - \
                labels[i]['offset']
            return lbls[(lbls >= 0) & (lbls < Fs)]

        if peaks.size == 0:
            # no event was predicted, all labeled event times are counted as
            # FNs
            for j in get_j():
                FNs.append((X[i], Y[i], y, j, X1[i], S[i], i))
        else:
            # add labeled events to list of FNs if no peak found was when y==1
            for j in get_j():
                y_j = np.zeros(X.shape[1])
                y_j[j] = 1
                y_j = np.convolve(y_j, ss.boxcar(y_label_width), 'same')
                if np.all(y_j[peaks] == 0):
                    FNs.append((X[i], Y[i], y, j, X1[i], S[i], i))

        for j in peaks:
            # check if label == 1 at time of peaks, if so, count TP
            if Y[i, j, 0] == 1:
                TPs.append((X[i], Y[i], y, j, X1[i], S[i], i))
            else:
                # peak is occurring at time where label == 0, count as FP
                FPs.append((X[i], Y[i], y, j, X1[i], S[i], i))

    # cast to typed arrays to allow fancy indexing.
    dtype = [('X', 'O'), ('Y', 'O'), ('y', 'O'), ('j', '<i8'),
             ('X1', 'O'), ('S', 'O'), ('i', '<i8')]
    TPs = np.array(TPs, dtype=dtype)
    FPs = np.array(FPs, dtype=dtype)
    FNs = np.array(FNs, dtype=dtype)

    return TPs, FPs, FNs


def get_TPs_FPs_FNs_stats(Y, Y_pred, rippleLocs,
                          threshold, distance, width,
                          run_speed,
                          y_label_width=62, decimals=3):
    '''
    Parameters
    ----------
    Y: ndarray
    Y_pred: ndarray
    rippleLocs: ndarray
    labels: structured ndarray
    all_labels: structured ndarray
    threshold: float
    distance: int
    width: int
    run_speed: ndarray or None
    Fs: float
    decimals: int

    Returns
    -------
    TPs: structured ndarray
    FPs: structured ndarray
    FNs: structured ndarray
    '''
    # network is not trained on negatives, so we quantify true/false positives
    # and false negatives
    TPs = []
    # TNs = []
    FPs = []
    FNs = []
    # Determine time(s) and probability(ies) in sample as the time and
    # magnitude of local maxima following a threshold crossing from below
    peaks, _ = ss.find_peaks(Y_pred, height=threshold, distance=distance,
                             width=width)

    if run_speed is not None:
        # keep ripples where run_speed == 0:
        peaks = peaks[run_speed[peaks] == 0]

    # one-hot encoding of found peaks
    hat_y = np.zeros(Y.shape[0])
    hat_y[peaks] = 1
    hat_y = np.convolve(hat_y, ss.boxcar(y_label_width), 'same')

    if peaks.size == 0:
        # no event was predicted, so all labeled event times are counted as FNs
        for j in rippleLocs:
            FNs.append(j)
    else:
        # add labeled events to list of FNs if no peak found was when y==1
        for j in rippleLocs:
            y_j = np.zeros(Y.shape[0])
            y_j[j] = 1
            y_j = np.convolve(y_j, ss.boxcar(y_label_width), 'same')
            if np.all(y_j[peaks] == 0):
                FNs.append(j)
    # predicted events must be TPs or FPs
    for j in peaks:
        # check if label == 1 at time of peaks, if so, count TP
        if Y[j] == 1:
            TPs.append(j)
        else:
            # peak is occurring at time where label == 0, count as FP
            FPs.append(j)

    # Evaluation metrics
    TP = len(TPs)
    # TN = len(TNs)
    FP = len(FPs)
    FN = len(FNs)
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    recall = TP / (TP + FN)
    try:
        F1 = 2 / (precision**-1 + recall**-1)
    except ZeroDivisionError:
        F1 = 0

    stats = pd.DataFrame([[TP, FP, FN, FP+FN,
                           np.round(precision, decimals=decimals),
                           np.round(recall, decimals=decimals),
                           np.round(F1, decimals=decimals)]],
                         columns=['TP', 'FP', 'FN', 'FP+FN',
                                  'precision', 'recall', 'F_1'])

    return TPs, FPs, FNs, stats
