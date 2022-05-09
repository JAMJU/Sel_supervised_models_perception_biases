#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created april 2021
    by Juliette MILLET
    add delta_values to human_results
"""
import os

def add_values(file_human, files_with_delta, folder_results,  file_out):
    results = {}
    for file in files_with_delta:
        name = file.split('.')[0]
        name = name.split('_triplet')[0]
        f_res = open(os.path.join(folder_results, file), 'r')
        ind = f_res.readline().replace('\n', '').split(',')
        for line in f_res:
            new_line = line.replace('\n', '').split(',')
            trip = new_line[ind.index('triplet_id')]
            # print(trip)
            if trip not in results:
                results[trip] = {}
            results[trip][name] = new_line[ind.index('delta')]
        f_res.close()

    f_human = open(file_human, 'r')
    f_out = open(file_out, 'w')

    ind = f_human.readline().replace('\n', '')
    f_out.write(ind )
    for file in files_with_delta:
        name = file.split('.')[0]
        name = name.split('_triplet')[0]
        f_out.write(',' + name)
    f_out.write('\n')

    ind = ind.split(',')

    for line in f_human:
        print(line)
        new_line = line.replace('\n', '').split(',')
        trip = new_line[ind.index('triplet_id')]
        f_out.write(','.join(new_line))
        for file in files_with_delta:
            name = file.split('.')[0]
            name = name.split('_triplet')[0]
            f_out.write(',' + results[trip][name])
        f_out.write('\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to analyze output from humans vs model\'s outputs and sample the results')
    parser.add_argument('folder_delta_files', metavar='f_delta', type=str,
                        help='folder where the delta files are')
    parser.add_argument('human_file', metavar='f_human', type=str,
                        help='file where the human results are')
    parser.add_argument('out_file', metavar='f_do', type=str,
                        help='file produced with human + delta info')

    args = parser.parse_args()
    #result_path = 'results_acl_paper/'
    result_path = args.folder_delta_file
    files_values = [file for file in os.listdir(result_path)]
    file_human = args.human_file
    file_out = args.out_file

    add_values(file_human=file_human, files_with_delta=files_values, file_out=file_out, folder_results=result_path)
