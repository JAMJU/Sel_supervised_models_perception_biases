#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute mfccs or other acoustic features and save them
    with the same structure than the original dataset
"""

from copy_structure import copy_structure
import librosa
import os
import numpy as np


def compute_mfccs(filename):
    y, sr = librosa.load(filename)
    spect = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
    )

    spect = spect.T
    return spect

def compute_melfilterbanks(filename):
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
    )
    spect = librosa.amplitude_to_db(spect)
    spect = spect.T
    return spect


def transform_and_save(filename_in, filename_out, features):
    if features == 'mfccs':
        spect = compute_mfccs(filename_in)
        np.save(filename_out, spect)
    elif features == 'melfilterbanks':
        spect = compute_melfilterbanks(filename_in)
        np.save(filename_out, spect)
    else:
        print('The feature you asked for is not available')
        raise ValueError


def transform_all(folder_in, folder_out, features):
    # we copy the structure
    copy_structure(input_path=folder_in, output_path=folder_out)

    for root, dirs, files in os.walk(folder_in):
        for name in files:
            print(name)
            a = os.path.join(root, name)

            if not a.endswith('.wav'):
                continue
            path = ''.join(a.split(folder_in))
            path_output = os.path.join(folder_out, path).replace('.wav', '.npy')
            if not os.path.isfile(path_output):
                transform_and_save(a, path_output, features)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to analyze output from discrimination experiment')
    parser.add_argument('folder_perceptimatic', metavar='in', type=str,
                        help='folder where input are')
    parser.add_argument('folder_out', metavar='out', type=str,
                        help='folder where to put outputs')

    args = parser.parse_args()

    transform_all(folder_in=args.folder_perceptimatic,
                  folder_out=args.folder_out,
                  features = 'mfccs')






