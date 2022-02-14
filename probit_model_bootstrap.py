#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to test if gamma value predicts discrimination well
"""
import os
nb = "30"
os.environ["OMP_NUM_THREADS"] = nb
os.environ["OPENBLAS_NUM_THREADS"] = nb
os.environ["MKL_NUM_THREADS"] = nb
os.environ["VECLIB_MAXIMUM_THREADS"] = nb
os.environ["NUMEXPR_NUM_THREADS"] = nb

from multiprocessing import Pool
import pandas as pd
from statsmodels.formula.api import probit
from sampling import get_dico_corres_file, sample_lines


def model_probit_binarized(data_file,  model, lines_sampled): # for the model, you have to add the +
    #print(lines_sampled)
    data_ = pd.read_csv(data_file, sep=',', encoding='utf-8')
    #data_['inverted_gamma'] = 1./data_['gamma_value']

    data_['bin_user_ans'] = (data_['bin_user_ans'] + 1.) / 2  # we transform -1 1 into 0 1
    data_['TGT_first'] = data_['TGT_first'].astype(bool)
    data_['TGT_first_code'] = data_['TGT_first'].astype(int)
    #print(data_['TGT_first_code'])
    #data_['english'] = 1 - data_['']
    #data_['french'] = data_['language_indiv_code']

    nb_lines = len(data_)
    all_lines = list(range(nb_lines))
    #line_not_sampled = list(set(all_lines) - set(lines_sampled))

    data = data_.iloc[lines_sampled]
    # we normalize data
    for val in ['nb_stimuli'] + [mod.replace(' ', '')  for mod in model.split('+')]:
        data[val] = (data[val] -data[val].mean())/data[val].std()
    model_probit = probit("bin_user_ans ~ TGT_first_code + C(subject_id) + C(dataset) + nb_stimuli + " + model, data) #
    #result_probit = model_probit.fit(max_iter=100, disp=True)
    try:
        result_probit = model_probit.fit_regularized(max_iter=200, disp=True)
        #predictions = result_probit.predict(exog=to_test, linear = False)
        #real_label = to_test['binarized_answer']
        #auc_score = roc_auc_score(y_true=real_label.to_numpy(), y_score=predictions.to_numpy())
        #print(model_probit.loglike(result_probit.params))
        return model_probit.loglike(result_probit.params)#, auc_score
    except:
        return 'None'

def func_to_parallel(args):
    dico_lines = args[0]
    list_names= args[1]
    it = args[2]
    file_humans = args[3]
    list_sampled = sample_lines(dico_lines)
    list_log = [str(it)]
    #list_auc = []
    for mod in list_names:
        print(mod)
        log= model_probit_binarized(data_file=file_humans, model=mod,
                                                 lines_sampled=list_sampled)
        list_log.append(str(log))
        #list_auc.append(str(auc_result))
    return list_log


def iteration_model(filename, nb_it, outfile, french = True, english = True):
    dico_lines = get_dico_corres_file(filename, french=french, english = english)

    f_names = open(filename, 'r')
    line_names = f_names.readline().replace('\n', '').split(',')
    list_names = []
    start = False
    for nam in line_names:
        if start:
            list_names.append(nam)
        elif not start and nam == "dataset":  # end of info start of models
            start = True

    f_names.close()
    print(list_names)
    out = open(outfile, 'a')
    out.write('nb,' + ','.join(list_names))
    out.write('\n')

    out.close()


    print('Beginning')
    nb_= int(int(nb)/2)
    div = int(nb_it / nb_)

    for k in range(div):
        with Pool(int(nb_)) as p:
            lines = p.map(func_to_parallel,
                          [[dico_lines, list_names,  k * int(nb_) + i, filename] for i in
                           range(int(nb_))])
            for li in lines:
                out = open(outfile , 'a')
                out.write(','.join(li))
                out.write('\n')
                out.close()
    """for it in range(nb_it):
        line = func_to_parallel([dico_lines, list_names, it, filename])
        out = open(outfile, 'a')
        out.write(','.join(line))
        out.write('\n')
        out.close()"""




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to analyze output from humans vs model\'s outputs and sample the results')
    parser.add_argument('file_humans', metavar='f_do', type=str,
                        help='file with outputs humans t give')
    parser.add_argument('outfile', metavar='f_do', type=str,
                        help='file with log likelihood answers')
    parser.add_argument('nb_it', metavar='f_do', type=int,
                        help='nb of sampling')
    parser.add_argument('french', metavar='f_do', type=str,
                        help='if french participants used')
    parser.add_argument('english', metavar='f_do', type=str,
                        help='if english participants used')

    args = parser.parse_args()

    fr = True if args.french == 'True' else False
    en = True if args.english == 'True' else False
    print('french', fr,'english', en)

    iteration_model(filename=args.file_humans, nb_it=args.nb_it, outfile=args.outfile, french=fr, english=en)
    #get_ROC_average(couples)