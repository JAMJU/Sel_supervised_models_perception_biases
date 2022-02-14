#!/usr/bin/env python
# coding: utf-8

"""
Script to copy structure of wav file perceptimatic : Useful to create repr
 """

import os

def copy_structure(input_path, output_path):

    for dirpath, dirnames, filenames in os.walk(input_path):

        print(dirpath)
        structure = os.path.join(output_path, dirpath[len(input_path) + 1:])
        print(structure)
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")
