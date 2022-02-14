#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute spearman correlation at the phone level and at the context level
"""
import os
nb = "30"
os.environ["OMP_NUM_THREADS"] = nb
os.environ["OPENBLAS_NUM_THREADS"] = nb
os.environ["MKL_NUM_THREADS"] = nb
os.environ["VECLIB_MAXIMUM_THREADS"] = nb
os.environ["NUMEXPR_NUM_THREADS"] = nb
#nb = "1"

from sampling import get_dico_corres_file, sample_lines
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import random as rd
from multiprocessing import Pool



def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def get_power(f):
    pw = int(np.log10(f))
    return pw

def get_spearman_phone(df_english, df_french, value_evaluated):

    #df = pd.read_csv(filename_data, delimiter = ',')

    # We separate participants
    #df_french = df[df['language_indiv'] == 'french']
    #df_english = df[df['language_indiv'] == 'english']
    all_values = []
    for dff in [df_french, df_english]:
        #print(dff['subject_language'].iloc[0])
        # We get only what we need
        dff = dff[['triplet_id', 'phone_TGT', 'phone_OTH', 'prev_phone', 'next_phone', 'language_OTH','language_TGT', 'dataset','user_ans', value_evaluated]]

        # We adapt to some dataset that have a -3 / 3 scale
        dff.loc[dff['dataset'] == "WorldVowels", ['user_ans']] = dff.loc[
                                                                     dff['dataset'] == "WorldVowels", ['user_ans']] / 3.
        dff.loc[dff['dataset'] == "zerospeech", ['user_ans']] = dff.loc[
                                                                    dff['dataset'] == "zerospeech", ['user_ans']] / 3.
        #print(dff[0:10])
        #print(dff['user_ans'])
        #dff.loc[value_evaluated] = dff[value_evaluated].astype(float)
        #dff.loc['user_ans'] = dff['user_ans'].astype(float)


        # We average over triplet first
        gf = dff.groupby(['triplet_id', 'phone_TGT', 'phone_OTH', 'prev_phone', 'next_phone', 'language_OTH','language_TGT', 'dataset'], as_index = False)
        ans_fr = gf.user_ans.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        # Then we average over context
        gf = ans_fr.groupby(['phone_TGT', 'phone_OTH', 'prev_phone', 'next_phone', 'language_OTH','language_TGT', 'dataset'], as_index = False)
        ans_fr = gf.user_ans.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        # then we average over phone contrast
        gf = ans_fr.groupby(['phone_TGT', 'phone_OTH', 'language_OTH','language_TGT', 'dataset'], as_index=False)
        ans_fr = gf.user_ans.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]

        # the we average over order TGT-OTH or the other way around
        res = ans_fr.copy()
        res['phone_TGT'] = ans_fr['phone_OTH']
        res['phone_OTH'] = ans_fr['phone_TGT']
        res['language_OTH'] = ans_fr['language_TGT']
        res['language_TGT'] = ans_fr['language_OTH']

        #print('ANS_FR###',ans_fr)
        #print('res', res)
        total = pd.concat([ans_fr, res], axis=0, )
        #print('TOTAL#####', total)
        gf = total.groupby(['phone_TGT', 'phone_OTH', 'language_OTH','language_TGT', 'dataset'], as_index=False)
        ans_fr = gf.user_ans.mean()
        val_fr = gf[value_evaluated].mean()
        ans_fr[value_evaluated] = val_fr[value_evaluated]


        rho_fr, p_fr = spearmanr(ans_fr['user_ans'], ans_fr[value_evaluated])
        print(value_evaluated, rho_fr, p_fr)
        all_values.append([abs(rho_fr), p_fr])
    return all_values

def func_to_parallelize(args):
    dico_lines_french = args[0]
    dico_lines_english = args[1]
    list_names = args[2]
    it = args[3]
    data = args[4]
    # we sample
    list_sampled_french = sample_lines(dico_lines_french)
    list_sampled_english = sample_lines(dico_lines_english)
    list_res_french = [str(it)]
    list_res_english = [str(it)]
    for mod in list_names:
        # print(mod)
        corrs = get_spearman_phone(df_english=data.iloc[list_sampled_english], df_french=data.iloc[list_sampled_french],
                                   value_evaluated=mod)
        list_res_french.append(str(corrs[0][0]))  # ([str(corrs[0][0]), str(corrs[0][1])])
        list_res_english.append(str(corrs[1][0]))  # ([str(corrs[1][0]), str(corrs[1][1])])
    return list_res_french, list_res_english

def iteration_function(filename, nb_it, outfile):
    data = pd.read_csv(filename)
    dico_lines_french = get_dico_corres_file(filename, french=True, english=False)
    dico_lines_english = get_dico_corres_file(filename, french=False, english=True)
    f_names = open(filename, 'r')
    line_names = f_names.readline().replace('\n', '').split(',')
    list_names = []
    start = False
    for nam in line_names:
        if start:
            list_names.append(nam)
        elif not start and nam == "dataset": # end of info start of models
            start = True
    f_names.close()
    out_french = open(outfile+ '_french.csv', 'a')
    out_english = open(outfile + '_english.csv', 'a')
    out_french.write('nb,' + ','.join(list_names) + '\n')
    out_english.write('nb,' + ','.join(list_names) + '\n')
    out_french.close()
    out_english.close()
    print('Beginning')
    div = int(nb_it / int(nb))

    for k in range(div):
        with Pool(int(nb)) as p:
            lines = p.map(func_to_parallelize, [[dico_lines_french, dico_lines_english, list_names,k * int(nb) + i, data] for i in range(int(nb))])
            for li in lines:
                french, english = li
                out_french = open(outfile + '_frenchpretrained.csv', 'a')
                out_english = open(outfile + '_englishpretrained.csv', 'a')
                out_french.write(','.join(french))
                out_english.write(','.join(english))
                out_french.write('\n')
                out_english.write('\n')
                out_french.close()
                out_english.close()
    #func_to_parallelize([dico_lines_french, dico_lines_english, list_names,2 * int(nb) + 1, data])



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to compute cor')
    parser.add_argument('file', metavar='f_do', type=str,
                        help='model')
    args = parser.parse_args()

    data_ = args.file

    iteration_function(filename = data_, nb_it = 10000, outfile = 'results_bootstrap/spearman_all')














