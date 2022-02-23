#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    evaluate model on native effect
"""
import os
nb = "30"
os.environ["OMP_NUM_THREADS"] = nb
os.environ["OPENBLAS_NUM_THREADS"] = nb
os.environ["MKL_NUM_THREADS"] = nb
os.environ["VECLIB_MAXIMUM_THREADS"] = nb
os.environ["NUMEXPR_NUM_THREADS"] = nb
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from multiprocessing import Pool
from sampling import get_dico_corres_file, sample_lines



def commpute_differences_faster(filename_, levels,  values_comparison_french, values_comparison_english, sampled_lines, it):
    """ Compute diff french minus english"""
    data = pd.read_csv(filename_, sep=',', encoding='utf-8')

    all_val = []

    data = data.iloc[sampled_lines]

    print(data.loc[data['dataset'] == "WorldVowels", ['user_ans']])

    data.loc[data['dataset'] == "WorldVowels", ['user_ans']] = data.loc[data['dataset'] == "WorldVowels", ['user_ans']] / 3.
    data.loc[data['dataset'] == "zerospeech", ['user_ans']] = data.loc[data['dataset'] == "zerospeech", ['user_ans']] / 3.

    print(data.loc[data['dataset'] == "WorldVowels", ['user_ans']])

    data_fr = data[data['subject_language'] == 'FR'].copy()
    data_en = data[data['subject_language'] == 'EN'].copy()

    # We normalize english and french side for the model so they are comparable
    for i in range(len(values_comparison_english)):
        data_fr[values_comparison_french[i]] = data_fr[values_comparison_french[i]] / data_fr[
            values_comparison_french[i]].std()
        data_en[values_comparison_english[i]] = data_en[values_comparison_english[i]] / data_en[
            values_comparison_english[i]].std()

    data_en['triplet_compo'] = data_en['phone_TGT'] + ';' + data_en['phone_OTH'] + ';' + data_en['prev_phone'] + ';' + data_en[
        'next_phone'] + ';' + data_en['language_TGT'] + ';' + data_en['language_OTH'] + ';' + data_en['dataset']
    data_fr['triplet_compo'] = data_fr['phone_TGT'] + ';' + data_fr['phone_OTH'] + ';' + data_fr['prev_phone'] + ';' + data_fr[
        'next_phone'] + ';' + data_fr['language_TGT'] + ';' + data_fr['language_OTH'] + ';' + data_fr['dataset']
    data_en['contrast'] = data_en['phone_TGT'] + ';' + data_en['phone_OTH'] + ';' +  data_en['language_OTH']  + ';' +  data_en['language_TGT'] + ';' +  data_en[ 'dataset']
    data_fr['contrast'] = data_fr['phone_TGT'] + ';' + data_fr['phone_OTH'] + ';' + data_fr['language_OTH'] + ';' + data_fr['language_TGT'] + ';' + data_fr[ 'dataset']
    print(data_en['contrast'])

    # we create temporary files to facilitate the computation
    data_fr.to_csv('temp_fr' + str(it) + '.csv')
    data_en.to_csv('temp_en' + str(it) + '.csv')
    data_fr = ''
    data_en = ''
    data_fr_file = {}
    data_en_file = {}
    data_fr_triplet = {}
    data_en_triplet = {}
    data_fr_contrast = {}
    data_en_contrast = {}

    for k in values_comparison_french + ['user_ans']:
        data_fr_file[k] = {}
        data_fr_triplet[k] = {}
        data_fr_contrast[k] = {}
    for k in values_comparison_english + ['user_ans']:
        data_en_file[k] = {}
        data_en_triplet[k] = {}
        data_en_contrast[k] = {}


    # now we use the results of these temp files
    f = open('temp_fr' + str(it) + '.csv', 'r')
    ind = f.readline().replace('\n', '').split(',')

    for line in f:
        new_line = line.replace('\n', '').split(',')
        file = new_line[ind.index('triplet_id')].replace('"', "")
        triplet_compo = new_line[ind.index('triplet_compo')].replace('"', "")
        contrast = new_line[ind.index('contrast')].replace('"', "")
        #print(contrast)
        for k in values_comparison_french + ['user_ans']:
            delta = float(new_line[ind.index(k)])
            data_fr_file[k][file] = data_fr_file[k].get(file, []) + [delta]
            data_fr_triplet[k][triplet_compo] = data_fr_triplet[k].get(triplet_compo, []) + [delta]
            data_fr_contrast[k][contrast] = data_fr_contrast[k].get(contrast, []) + [delta]
    f.close()

    f = open('temp_en' + str(it) + '.csv', 'r')
    ind = f.readline().replace('\n', '').split(',')

    for line in f:
        new_line = line.replace('\n', '').split(',')
        file = new_line[ind.index('triplet_id')].replace('"', "")
        triplet_compo = new_line[ind.index('triplet_compo')].replace('"', "")
        contrast = new_line[ind.index('contrast')].replace('"', "")
        for k in values_comparison_english + ['user_ans']:
            delta = float(new_line[ind.index(k)])
            data_en_file[k][file] = data_en_file[k].get(file, []) + [delta]
            data_en_triplet[k][triplet_compo] = data_en_triplet[k].get(triplet_compo, []) + [delta]
            data_en_contrast[k][contrast] = data_en_contrast[k].get(contrast, []) + [delta]
    f.close()

    # we suppress the temp files
    os.remove('temp_fr' + str(it) + '.csv')
    os.remove('temp_en' + str(it) + '.csv')


    # we average the results
    for k in values_comparison_english + ['user_ans']:
        for file in data_en_file[k]:
            data_en_file[k][file] = np.asarray(data_en_file[k][file]).mean()
        for triplet in data_en_triplet[k]:
            data_en_triplet[k][triplet] = np.asarray(data_en_triplet[k][triplet]).mean()
        for cont in data_en_contrast[k]:
            data_en_contrast[k][cont] = np.asarray(data_en_contrast[k][cont]).mean()
    for k in values_comparison_french + ['user_ans']:
        for file in data_fr_file[k]:
            data_fr_file[k][file] = np.asarray(data_fr_file[k][file]).mean()
        for triplet in data_fr_triplet[k]:
            data_fr_triplet[k][triplet] = np.asarray(data_fr_triplet[k][triplet]).mean()
        for cont in data_fr_contrast[k]:
            data_fr_contrast[k][cont] = np.asarray(data_fr_contrast[k][cont]).mean()


    # the native effect is computed averaging over contrasts
    if 'contrast' in levels:
        triplet_list = list(data_en_contrast[values_comparison_english[0]].keys())
        diff_humans = []
        diff_models = {}
        for i in range(len(values_comparison_english)):
            diff_models[values_comparison_english[i]] = []
        triplet_done = []
        for trip in triplet_list:
            if trip in triplet_done:
                continue
            # we average on TGT-OTH OTH-TGT
            other = trip.split(';')
            other = ';'.join([other[1], other[0], other[3], other[2], other[4]])
            triplet_done.append(other)
            triplet_done.append(trip)

            if trip in data_fr_contrast['user_ans'] and not trip in data_en_contrast['user_ans']:
                print('ERROR triplet not test on eng', trip)
                continue
            elif trip not in data_fr_contrast['user_ans'] and trip in data_en_contrast[
                'user_ans']:
                print('ERROR triplet not test on fre', trip)
                continue
            elif trip not in data_fr_contrast['user_ans'] and trip not in data_en_contrast[
                'user_ans']:
                print('ERROR triplet not test on fre and on en', trip)
                continue

            val_fr_human = (data_fr_contrast['user_ans'][trip] + data_fr_contrast['user_ans'].get(other, data_fr_contrast['user_ans'][trip])) / 2.
            val_en_human = (data_en_contrast['user_ans'][trip] + data_en_contrast['user_ans'].get(other,data_en_contrast['user_ans'][trip])) / 2.
            diff_humans.append(val_fr_human - val_en_human)
            for i in range(len(values_comparison_english)):
                val_fr_model = (data_fr_contrast[values_comparison_french[i]][trip] +
                                data_fr_contrast[values_comparison_french[i]].get(other,data_fr_contrast[values_comparison_french[i]][trip])) / 2.
                val_en_model = (data_en_contrast[values_comparison_english[i]][trip] +
                                data_en_contrast[values_comparison_english[i]].get(other,data_en_contrast[values_comparison_english[i]][trip])) / 2.

                diff_models[values_comparison_english[i]].append(val_fr_model - val_en_model)

        all_val += [np.asarray(diff_humans)]
        for i in range(len(values_comparison_english)):
            all_val += [diff_models[values_comparison_english[i]]]
    return all_val

def compute_correlation(diff_models, diff_humans):
    return pearsonr(diff_models, diff_humans)

def function_to_parallel(args):

    it = args[0]
    file_data = args[1]
    models_couples = args[2]
    models = args[3]
    dico_french = args[4]
    dico_english = args[5]


    english_lines = sample_lines(dico_english)
    french_lines = sample_lines(dico_french)
    lines_sampled = english_lines + french_lines
    # we compute the diff for models and humans
    diffs = commpute_differences_faster(filename_=file_data, levels=['contrast'],
                                        values_comparison_english=[models_couples[modi]['english'] for modi in models],
                                        values_comparison_french=[models_couples[modi]['french'] for modi in models],
                                        sampled_lines=lines_sampled, it = it)
    line_file = []
    line_triplet = []
    line_contrast = []
    line_file.append(str(it))
    line_triplet.append(str(it))
    line_contrast.append(str(it))
    # first file
    count = 0
    diff_humans = diffs[0]
    count += 1
    # we compute correlations for each model
    for i in range(len((models))):
        r, p = compute_correlation(diff_models=diffs[count], diff_humans=diff_humans)
        line_contrast.append(str(r))
        line_contrast.append(str(p))
        count += 1
    return line_file, line_triplet, line_contrast

def iterations(models_couples, file_data, outfile_contrast, nb_it):
    dico_english = get_dico_corres_file(data_file=file_data, french = False, english = True)
    dico_french = get_dico_corres_file(data_file=file_data, french = True, english = False)

    models = list(models_couples.keys())
    for fili in [outfile_contrast]:#[outfile_file, outfile_triplet, outfile_contrast]:
        out = open(fili, 'a')
        out.write('nb')
        for mod in models:
            out.write(',' + mod + ',' + mod+ '_p_val')
        out.write('\n')
        out.close()

    div = int(nb_it/int(nb))

    for k in range(div):
        with Pool(int(nb)) as p:
            lines = p.map(function_to_parallel, [[k*int(nb) + i, file_data, models_couples, models,
                                          dico_french, dico_english] for i in range(int(nb))])
            for li in lines:
                file, trip, cont = li
                out = open(outfile_contrast, 'a')
                out.write(','.join(cont))
                out.close()
                for fili in [outfile_contrast]:
                    out = open(fili, 'a')
                    out.write('\n')
                    out.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to analyze output from discrimination experiment')
    parser.add_argument('file_in', metavar='f_do', type=str,
                        help='file where human and model data are are')
    parser.add_argument('file_out', metavar='res', type=str,
                        help='results file')
    parser.add_argument('nb_it', metavar='nb', type=str,
                        help='number of iterations')

    args = parser.parse_args()

    # this is the models we use in our paper, you can change this list
    dico_models = {'wav2vec_transf4':{'english':'wav2vec_english_transf4', 'french':'wav2vec_french_transf4'},
                   'hubert':{'english':'hubert_english_transf_5', 'french':'hubert_french_transf_5'},
                   'deepspeech_phon':{'english':'deepspeech_english_rnn4', 'french':'deepspeech_french_rnn4'},
                   'cpc':{'english':'cpc_english_AR', 'french':'cpc_french_AR'},
                   'deepspeech_txt': {'english': 'deepspeech_englishtxt_rnn4', 'french': 'deepspeech_frenchtxt_rnn4'},
                   }
    iterations(models_couples=dico_models, file_data=args.file_in,
               outfile_contrast=args.file_out,  nb_it=int(args.nb_it))




