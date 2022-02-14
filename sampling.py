#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    functions for bottstrap computations
"""
import random as rd
import numpy as np
#import pandas as pd

def random_selection_and_mean(list_results, nb):
    #print(list_results)
    selection = rd.choices(list_results, weights = [1. for k in list_results], k = nb)

    return np.mean(np.asarray(selection))

def get_confidence_interval_90(list_results, nb_iteration, nb_selection):
    all_mean = []
    for i in range(nb_iteration):
        all_mean.append(random_selection_and_mean(list_results, nb_selection))

    global_mean = np.median(np.asarray(all_mean))

    all_mean.sort()
    # we get the 95% interval
    int_begin = int((nb_iteration)*5/100)
    int_end = int((nb_iteration)*95/100)

    return global_mean, global_mean - all_mean[int_begin], all_mean[int_end] - global_mean

def get_score(filename, norm=False):
    f = open(filename, 'r')
    ind = f.readline().replace('\n', '').split(',')

    dico_scores = {'english':{}, 'french':{}}
    dico_count = {'english':{}, 'french':{}}
    for line in f:

        newline = line.replace('\n', '').split(',')
        language_indiv = newline[ind.index('language_indiv')]

        language = newline[ind.index('language_stimuli')]
        answer = float(newline[ind.index('correct_answer')])
        TGT = newline[ind.index('TGT')]
        OTH = newline[ind.index('OTH')]  # ,prev_phone,next_phone




        key1 = ','.join([TGT,OTH,language])
        key2 = ','.join([OTH,TGT,language])
        if key1 not in dico_scores[language_indiv] and key2 not in dico_scores[language_indiv]:
            other_language = 'english' if language_indiv == 'french' else 'french'
            if key1 in dico_scores[other_language]:
                key = key1
            elif key2 in dico_scores[other_language]:
                key = key2
            else:
                key = key1
            dico_scores[language_indiv][key] = []
            dico_count[language_indiv][key] = 0
        elif key1 in dico_scores[language_indiv]:
            key = key1
        else:
            key = key2

        if not norm:
            dico_scores[language_indiv][key].append(answer)
        else:
            dico_scores[language_indiv][key].append(1. if answer > 0 else -1.)
        dico_count[language_indiv][key] += 1

    return dico_scores, dico_count


def compute_score_bootstrap(file, norm=False):
    dico_score, dico_count = get_score(filename=file, norm=norm)
    print(dico_count)
    dico_results = {'english': {}, 'french': {}}
    dico_error_bars = {'english':{}, 'french': {}}
    for lang in ['english', 'french']:
        for key in dico_score[lang].keys():
            dico_error_bars[lang][key] = [0,0]
            dico_results[lang][key], dico_error_bars[lang][key][0], dico_error_bars[lang][key][1] = get_confidence_interval_90(list_results=dico_score[lang][key],
                                                                                                          nb_iteration= 10000,
                                                                                                          nb_selection=180)
    return dico_results, dico_error_bars



def get_ranking(filename, value_wanted):
    f = open(filename, 'r')
    ind = f.readline().replace('\n', '').split(',')

    dico_scores = {'english':{}, 'french':{}}
    dico_count = {'english':{}, 'french':{}}
    for line in f:

        newline = line.replace('\n', '').split(',')
        language_indiv = newline[ind.index('language_indiv')]

        language = newline[ind.index('language_stimuli')]
        answer = newline[ind.index(value_wanted)]

        TGT = newline[ind.index('TGT')]
        OTH = newline[ind.index('OTH')]  # ,prev_phone,next_phone

        key1 = ','.join([TGT,OTH,language])
        key2 = ','.join([OTH,TGT,language])
        if key1 not in dico_scores[language_indiv] and key2 not in dico_scores[language_indiv]:
            other_language = 'english' if language_indiv == 'french' else 'french'
            if key1 in dico_scores[other_language]:
                key = key1
            elif key2 in dico_scores[other_language]:
                key = key2
            else:
                key = key1
            dico_scores[language_indiv][key] = []
            dico_count[language_indiv][key] = 0
        elif key1 in dico_scores[language_indiv]:
            key = key1
        else:
            key = key2

        if answer == 'None':
            #answer = float(newline[ind.index('overlap_score_naive')])
            continue
        answer = float(answer) #if (value_wanted == 'overlap_score_naive' or value_wanted=="gamma_norm") else float(answer)

        dico_scores[language_indiv][key].append(answer)
        dico_count[language_indiv][key] += 1

    return dico_scores, dico_count

def get_dico_corres_file(data_file, french = True, english = True, dataset = ''):
    dico ={}
    f = open(data_file, 'r')
    ind = f.readline().replace('\n', '').split(',')
    count = 0
    for line in f:
        newline = line.replace('\n', '').split(',')
        language_indiv = newline[ind.index('subject_language')]
        dataset_= newline[ind.index('dataset')]
        #print(language_indiv)
        if (not french and language_indiv == 'FR'): # we perform the analysis only on the right language
            #print('out')
            count += 1
            continue
        if (not english and language_indiv == 'EN'): # we perform the analysis only on the right language
            count += 1
            continue

        if dataset != '' and dataset_ != dataset: # we perform the analysis only on the right dataset
            count += 1
            continue

        if newline[ind.index('triplet_id')] in dico:
            dico[newline[ind.index('triplet_id')]].append(count)
        else:
            dico[newline[ind.index('triplet_id')]] = [count]
        count += 1
    f.close()
    return dico


def sample_lines(dico_line_files):
    # we sample five results per filename
    list_lines = []
    for filename in dico_line_files:
        list_lines = list_lines + [dico_line_files[filename][rd.randrange(0,stop= len(dico_line_files[filename]))],
                                   dico_line_files[filename][rd.randrange(0, stop=len(dico_line_files[filename]))],
                                   dico_line_files[filename][rd.randrange(0, stop=len(dico_line_files[filename]))],
                                   dico_line_files[filename][rd.randrange(0, stop=len(dico_line_files[filename]))],
                                   dico_line_files[filename][rd.randrange(0, stop=len(dico_line_files[filename]))]]
    return list_lines


def compute_scores(file, value_wanted):

    dico_score, dico_count = get_ranking(filename=file, value_wanted=value_wanted)
    print(dico_score)
    dico_std = {'english':{}, 'french':{}}
    dico_results = {'english':{}, 'french':{}}
    for lang in ['english', 'french']:
        for key in dico_score[lang].keys():
            dico_results[lang][key] = np.asarray(dico_score[lang][key]).mean()
            dico_std[lang][key] = np.asarray(dico_score[lang][key]).std()
    print(dico_results)
    return dico_results, dico_std