#!/usr/bin/env python3

"""
Created by Juliette MILLET september 2022
script to extract from all the layers of hubert model

"""

from hubert_pretrained_extraction import HubertFeatureReader
import os
import numpy as np


def main(input_directory, csv_file, ckpt_path, layer, feat_dir, max_chunk):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    if csv_file != '': # then it means it is a csv file
        f = open(csv_file, 'r')
        ind = f.readline().replace('\n', '').split(',')
        for line in f:
            new_line = line.replace('\n', '').split(',')
            fili = new_line[ind.index('#file_extract')]
            feat = reader.get_feats(path=os.path.join(input_directory, fili))
            np.save(os.path.join(feat_dir, fili.replace('.wav', '.npy')), feat)
    else:
        for file in os.listdir(input_directory):
            if not file.endswith('.wav'):
                continue
            feat = reader.get_feats(path = os.path.join(input_directory, file))
            np.save(os.path.join(feat_dir, file.replace('.wav', '.npy')), feat)
    print('Done')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument("csv_file")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=str)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    #logger.info(args)

    main(**vars(args))