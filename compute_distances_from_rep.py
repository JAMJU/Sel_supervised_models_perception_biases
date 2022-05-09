#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute distance from representation
"""
import os
nb = "10"
os.environ["OMP_NUM_THREADS"] = nb
os.environ["OPENBLAS_NUM_THREADS"] = nb
os.environ["MKL_NUM_THREADS"] = nb
os.environ["VECLIB_MAXIMUM_THREADS"] = nb
os.environ["NUMEXPR_NUM_THREADS"] = nb
from dtw_experiment import compute_dtw
from get_representations import get_triphone_mfccs, get_triphone_wav2vec, get_triphone_deepspeech, get_triphone_cpc, get_triphone_hubert


def get_distances_and_delta(triphone_TGT, triphone_OTH, triphone_X, get_func, distance):
    TGT = get_func(triphone_TGT)
    OTH = get_func(triphone_OTH)
    X = get_func(triphone_X)

    TGTX = compute_dtw(TGT,X, distance, norm_div=True)
    OTHX = compute_dtw(OTH,X, distance, norm_div=True)

    delta = OTHX - TGTX

    return TGTX, OTHX, delta


def get_distance_for_triplets(filename_triplet_list, file_out, get_func, distance):
    f = open(filename_triplet_list, 'r')
    ind = f.readline().replace('\n', '').split(',')
    print(ind)
    f_out = open(file_out, 'w')
    f_out.write(','.join(ind + ['TGTX', 'OTHX', 'delta', 'decision\n']))
    kee_dis = {}
    count = 0
    for line in f:
        if count % 100 == 0:
            print(count)
        count += 1
        new_line = line.replace('\n', '').split(',')
        OTH_item = new_line[ind.index('OTH_item')].replace('.wav', '')
        TGT_item = new_line[ind.index('TGT_item')].replace('.wav', '')
        X_item = new_line[ind.index('X_item')].replace('.wav', '')
        if TGT_item + ',' + OTH_item + ',' + X_item in kee_dis.keys():
            key = TGT_item + ',' + OTH_item + ',' + X_item
            TGTX = kee_dis[key][0]
            OTHX = kee_dis[key][1]
            delta = kee_dis[key][2]
        else:
            TGTX, OTHX, delta = get_distances_and_delta(triphone_TGT=TGT_item, triphone_OTH=OTH_item, triphone_X=X_item, get_func=get_func, distance = distance)
            kee_dis[TGT_item + ',' + OTH_item + ',' + X_item] = [TGTX, OTHX, delta]
        f_out.write(','.join(new_line + [str(TGTX), str(OTHX), str(delta), '1\n' if delta> 0. else '0\n']))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to compute distances')
    parser.add_argument('model', metavar='f_do', type=str,
                        help='model')
    parser.add_argument('layer', metavar='f_do', type=str,
                        help='layer wanted, only for neural network model')
    parser.add_argument('distance', metavar='f_do', type=str,
                        help='distance wanted')
    parser.add_argument('path_to_general_folder', metavar='f_do', type=str,
                        help='distance wanted')
    args = parser.parse_args()

    triplet_file = 'triplet_data.csv'
    path_to_scratch = args.path_to_general_folder
    wav2vec_french_path = os.path.join(path_to_scratch, 'wav2vec_french', args.layer)
    wav2vec_random_path = os.path.join(path_to_scratch, 'wav2vec_random', args.layer)
    wav2vec_english_path = os.path.join(path_to_scratch, 'wav2vec_english', args.layer)
    wav2vec_audioset_path = os.path.join(path_to_scratch, 'wav2vec_audioset', args.layer)
    wav2vec_librispeech_path = os.path.join(path_to_scratch, 'wav2vec_en_librispeech', args.layer)
    wav2vec_voxpopuli_path = os.path.join(path_to_scratch, 'wav2vec_fr_voxpopuli', args.layer)

    mfccs_path = os.path.join(path_to_scratch,'mfccs/')

    cpc_english_path = os.path.join(path_to_scratch, 'cpc_english', args.layer)
    cpc_french_path = os.path.join(path_to_scratch, 'cpc_french', args.layer)
    cpc_audioset_path = os.path.join(path_to_scratch, 'cpc_audioset', args.layer)

    deepspeech_english_path = os.path.join(path_to_scratch, 'deepspeech_english', args.layer)
    deepspeech_french_path = os.path.join(path_to_scratch, 'deepspeech_french', args.layer)
    deepspeech_englishtxt_path = os.path.join(path_to_scratch, 'deepspeech_englishtxt', args.layer)
    deepspeech_frenchtxt_path = os.path.join(path_to_scratch, 'deepspeech_frenchtxt', args.layer)

    hubert_english_path = os.path.join(path_to_scratch, 'hubert_english', args.layer)
    hubert_french_path = os.path.join(path_to_scratch, 'hubert_french', args.layer)
    hubert_audioset_path = os.path.join(path_to_scratch, 'hubert_audioset', args.layer)
    hubert_librispeech_path = os.path.join(path_to_scratch, 'hubert_en_librispeech', args.layer)

    mfccs = lambda x: get_triphone_mfccs(folder_data=mfccs_path, triphone_name=x)

    wav2vec_french = lambda x: get_triphone_wav2vec(folder_data=wav2vec_french_path, triphone_name=x,
                                                    layer_name=args.layer)
    wav2vec_english = lambda x: get_triphone_wav2vec(folder_data=wav2vec_english_path, triphone_name=x,
                                                     layer_name=args.layer)
    wav2vec_random = lambda x: get_triphone_wav2vec(folder_data=wav2vec_random_path, triphone_name=x,
                                                     layer_name=args.layer)
    wav2vec_audioset = lambda x: get_triphone_wav2vec(folder_data=wav2vec_audioset_path, triphone_name=x,
                                                     layer_name=args.layer)
    wav2vec_librispeech = lambda x: get_triphone_wav2vec(folder_data=wav2vec_librispeech_path, triphone_name=x,
                                                      layer_name=args.layer)
    wav2vec_voxpopuli = lambda x: get_triphone_wav2vec(folder_data=wav2vec_voxpopuli_path, triphone_name=x,
                                                      layer_name=args.layer)

    deepspeech_english = lambda x: get_triphone_deepspeech(folder_data=deepspeech_english_path, triphone_name=x)
    deepspeech_french = lambda x: get_triphone_deepspeech(folder_data=deepspeech_french_path, triphone_name=x)
    deepspeech_englishtxt = lambda x: get_triphone_deepspeech(folder_data=deepspeech_englishtxt_path, triphone_name=x)
    deepspeech_frenchtxt = lambda x: get_triphone_deepspeech(folder_data=deepspeech_frenchtxt_path, triphone_name=x)

    cpc_english = lambda x: get_triphone_cpc(folder_data=cpc_english_path, triphone_name=x)
    cpc_french = lambda x: get_triphone_cpc(folder_data=cpc_french_path, triphone_name=x)
    cpc_audioset = lambda x: get_triphone_cpc(folder_data=cpc_audioset_path, triphone_name=x)

    hubert_english = lambda x:get_triphone_hubert(folder_data = hubert_english_path, triphone_name=x, layer_name=args.layer)
    hubert_french = lambda x: get_triphone_hubert(folder_data=hubert_french_path, triphone_name=x,
                                                   layer_name=args.layer)
    hubert_audioset = lambda x: get_triphone_hubert(folder_data=hubert_audioset_path, triphone_name=x,
                                                  layer_name=args.layer)

    hubert_librispeech =lambda x: get_triphone_hubert(folder_data=hubert_librispeech_path, triphone_name=x,
                                                  layer_name=args.layer)

    if args.model == 'mfccs':
        func = mfccs
    elif args.model == 'wav2vec_french':
        func = wav2vec_french
    elif args.model == 'wav2vec_english':
        func = wav2vec_english
    elif args.model == 'wav2vec_random':
        func = wav2vec_random
    elif args.model == 'wav2vec_audioset':
        func = wav2vec_audioset
    elif args.model == 'deepspeech_english':
        func = deepspeech_english
    elif args.model == 'deepspeech_french':
        func = deepspeech_french
    elif args.model == 'deepspeech_englishtxt':
        func = deepspeech_englishtxt
    elif args.model == 'deepspeech_frenchtxt':
        func = deepspeech_frenchtxt
    elif args.model == 'cpc_english':
        func = cpc_english
    elif args.model == 'cpc_french':
        func = cpc_french
    elif args.model == 'cpc_audioset':
        func = cpc_audioset
    elif args.model == 'hubert_english':
        func = hubert_english
    elif args.model == 'hubert_french':
        func= hubert_french
    elif args.model == 'hubert_audioset':
        func = hubert_audioset
    elif args.model == 'hubert_librispeech':
        func = hubert_librispeech
    elif args.model == 'wav2vec_librispeech':
        func = wav2vec_librispeech
    elif args.model == 'wav2vec_voxpopuli':
        func = wav2vec_voxpopuli
    else:
        print('Error the model does not exist')


    get_distance_for_triplets(filename_triplet_list=triplet_file, file_out=os.path.join('results_acl_paper', args.model + '_'+ args.layer + '_'  + 'triplet_distances.csv'),
                              get_func=func, distance=args.distance)#'kl' if 'dpgmm' in args.model else 'cosine')



