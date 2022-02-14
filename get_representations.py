#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to get representation of triplet
"""
import numpy as np
import os
from scipy.stats import norm


def get_triphone_mfccs(folder_data, triphone_name):
    return np.load(os.path.join(folder_data, triphone_name + '.npy')) # time x dim

def get_triphone_wav2vec(folder_data, triphone_name, layer_name):
    """ We need to return the representation with timexdim"""
    if 'conv' in layer_name:
        data = np.load(os.path.join(folder_data, triphone_name + '.npy'))
        return data.swapaxes(0,1) # we put time as first dimension
    else:
        data = np.load(os.path.join(folder_data, triphone_name + '.npy'))
        sh = data.shape
        data = data.reshape((sh[0], sh[-1]))
        return data


def get_triphone_hubert(folder_data, triphone_name, layer_name):
    """ We need to return the representation with timexdim"""
    if 'conv' in layer_name:
        data = np.load(os.path.join(folder_data, triphone_name + '.npy'))
        return data.swapaxes(0, 1)  # we put time as first dimension
    else:
        data = np.load(os.path.join(folder_data, triphone_name + '.npy'))
        return data

def get_triphone_deepspeech(folder_data, triphone_name):
    return np.load(os.path.join(folder_data, triphone_name + '.npy')) # time x dim

def get_triphone_cpc(folder_data, triphone_name):
    return np.load(os.path.join(folder_data, triphone_name + '.npy'))