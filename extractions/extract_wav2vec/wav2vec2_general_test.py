
# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings from a wav2vec2 model
"""

import argparse
import glob
import os
from shutil import copy

# import h5py
import numpy as np
import soundfile as sf
import torch
import tqdm
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from torch import nn
import logging
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)

def forward_for_conv_encoder(conv_encoder, x, nb = 10000):

        # BxT -> BxCxT
        count = 0
        x = x.unsqueeze(1)

        for conv in conv_encoder.conv_layers:
            x = conv(x)
            if count == nb:
                return x
            count += 1

        return x

def forward_for_model_wav2vec2(model, source, padding_mask=None, mask=True, features_only='z'):
    """ This is a copy of the wav2vec2 forward function, modified to output feature ate different level
    model: the wav2vec2 model
    source: the source as input
    features_only : the feature you want as output, what is available : conv_0, conv_1, conv_2, conv_3, conv_4
    conv_5, conv_6, z, q, c, transf_0 to 11  waarnings : q does not work for fine tuned models"""

    #print('in forward of the model')
    if model.feature_grad_mult > 0:
        if not 'conv' in features_only:
            features = forward_for_conv_encoder(model.feature_extractor,source)
        else:
            features = forward_for_conv_encoder(model.feature_extractor,source, nb=int(features_only.split('_')[1]))
        if model.feature_grad_mult != 1.0:
            features = GradMultiply.apply(features, model.feature_grad_mult)
    else:
        with torch.no_grad():
            if not 'conv' in features_only:
                features = forward_for_conv_encoder(model.feature_extractor,source)
            else:
                features = forward_for_conv_encoder(model.feature_extractor,source, nb=int(features_only.split('_')[1]))
    if 'conv' in features_only:
        return {"x": features}

    features_pen = features.float().pow(2).mean()

    features = features.transpose(1, 2)
    features = model.layer_norm(features)
    unmasked_features = features.clone()

    if padding_mask is not None:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)

    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)

    features = model.dropout_input(features)
    unmasked_features = model.dropout_features(unmasked_features)

    num_vars = None
    code_ppl = None
    prob_ppl = None
    curr_temp = None

    if model.input_quantizer:
        q = model.input_quantizer(features, produce_targets=False)
        features = q["x"]
        num_vars = q["num_vars"]
        code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]
        features = model.project_inp(features)

    if mask:
        x, mask_indices = model.apply_mask(features, padding_mask)
        if mask_indices is not None:
            y = unmasked_features[mask_indices].view(
                unmasked_features.size(0), -1, unmasked_features.size(-1)
            )
        else:
            y = unmasked_features
    else:
        x = features
        y = unmasked_features
        mask_indices = None
        
    if features_only == 'z':
        #print('in mod', 'z')
        return {"x": x, "padding_mask": padding_mask}
    
    if 'transf' in features_only:
        block_nb = features_only.split('_')[-1]
        x, layer_results = model.encoder(x, padding_mask=padding_mask, layer=int(block_nb))
        return {"x":x, "padding_mask":padding_mask}
    
    x = model.encoder(x, padding_mask=padding_mask)

    if features_only == 'z_after_encoder':
        #print('in mod', 'z')
        return {"x": x, "padding_mask": padding_mask}

    if model.quantizer:
        q = model.quantizer(y, produce_targets=False)
        y = q["x"]
        num_vars = q["num_vars"]
        code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]

        y = model.project_q(y)

        if model.negatives_from_everywhere:
            neg_cands, *_ = model.quantizer(unmasked_features, produce_targets=False)
            negs, _ = model.sample_negatives(neg_cands, y.size(1))
            negs = model.project_q(negs)

        else:
            negs, _ = model.sample_negatives(y, y.size(1))

        if model.codebook_negatives > 0:
            cb_negs = model.quantizer.sample_from_codebook(
                y.size(0) * y.size(1), model.codebook_negatives
            )
            cb_negs = cb_negs.view(
                model.codebook_negatives, y.size(0), y.size(1), -1
            )  # order doesnt matter
            cb_negs = model.project_q(cb_negs)
            negs = torch.cat([negs, cb_negs], dim=0)
    elif model.project_q:
        y = model.project_q(y)

        if model.negatives_from_everywhere:
            negs, _ = model.sample_negatives(unmasked_features, y.size(1))
            negs = model.project_q(negs)
        else:
            negs, _ = model.sample_negatives(y, y.size(1))
    if features_only == 'q':
        #print('in mod', 'q')
        return {"x": y}

    if features_only == 'c':
        #print('in mod', 'c')
        #print(x)
        x = model.final_proj(x[0])
        return {"x": x}

    x = x[mask_indices].view(x.size(0), -1, x.size(-1))

    if model.target_glu:
        y = model.target_glu(y)
        negs = model.target_glu(negs)

    x = model.final_proj(x)
    x = model.compute_preds(x, y, negs)

    result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

    if prob_ppl is not None:
        result["prob_perplexity"] = prob_ppl
        result["code_perplexity"] = code_ppl
        result["num_vars"] = num_vars
        result["temp"] = curr_temp

    return result

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3


class PretrainedWav2Vec2Model(nn.Module):
    def __init__(self, folder, fname, asr = False):
        super().__init__()
        self.asr = asr # this option needs to be true for fine tuned models
        print('formatting model')
        model = Wav2Vec2Model.from_pretrained(folder, fname).models
        #model = Wav2Vec2Model.from_pretrained(model_name_or_path = folder,checkpoint_file=fname, data_name_or_path = folder).models
        print(model)
        model = model[0]
        model.eval()
        if self.asr:
            self.model_enc_all = model.w2v_encoder
            self.model = model.w2v_encoder.w2v_model
        else:
            self.model = model

    def forward(self, x, feat_only):
        with torch.no_grad():
            #print('in model', x.shape)
            if not self.asr:
                z = forward_for_model_wav2vec2(self.model, x, mask = False, features_only = feat_only)
            else:
                if feat_only != 'c':
                    z = forward_for_model_wav2vec2(self.model, x, mask = False, features_only = feat_only)
                else:
                    z = self.model_enc_all.forward(x, padding_mask=None) # we use the all original function
                    z['x'] = z['encoder_out']
        return z['x']

class Prediction:
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, folder, fname, gpu=0, asr = False):
        print('initialising')
        self.gpu = gpu
        self.model = PretrainedWav2Vec2Model(folder, fname, asr = asr).cuda(gpu)

    def __call__(self, x, feat_only):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            #print('x shape', x.shape)
            #print('x uns', x.unsqueeze(0).shape)
            z = self.model(x.unsqueeze(0), feat_only)
        return z.squeeze(0).cpu().numpy()

class EmbeddingDatasetWriter(object):
    """Given a model and a wav2letter++ dataset, pre-compute and store embeddings


    """

    def __init__(
            self,
            input_root,  # where your wavfile are
            output_root, # where you want your outputs to be
            model_folder, # the folder of your checkpoint
            model_fname, # the path to your checkpoint
            extension="wav",
            gpu=0,
            split='',
            verbose=False,
            use_feat='z', # the feature you want
            asr = False, # True if you used a fine tuned model
            csv_file = '' # file with input data
    ):
        print('In writer')
        assert os.path.exists(model_fname)

        self.model_fname = model_fname
        self.model_folder = model_folder
        self.model = Prediction(self.model_folder, self.model_fname, asr = asr, gpu = gpu)

        self.input_root = input_root
        self.output_root = output_root
        self.split = split
        self.verbose = verbose
        self.extension = extension
        self.use_feat = use_feat
        self.csv_file = csv_file

        assert os.path.exists(self.input_path), "Input path '{}' does not exist".format(
            self.input_path
        )

    def _progress(self, iterable, **kwargs):
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        return iterable

    def require_output_path(self, fname=None):
        path = self.get_output_path(fname)
        os.makedirs(path, exist_ok=True)

    @property
    def input_path(self):
        return self.get_input_path()

    @property
    def output_path(self):
        return self.get_output_path()

    def get_input_path(self, fname=None):
        if fname is None:
            return os.path.join(self.input_root  )  # , self.split)
        return os.path.join(self.get_input_path(), fname)

    def get_output_path(self, fname=None):
        if fname is None:
            return os.path.join(self.output_root  )  # , self.split)
        return os.path.join(self.get_output_path(), fname)

    def copy_labels(self):
        self.require_output_path()

        labels = list(
            filter(
                lambda x: self.extension not in x, glob.glob(self.get_input_path("*"))
            )
        )
        for fname in tqdm.tqdm(labels):
            copy(fname, self.output_path)

    @property
    def input_fnames(self):
        if self.csv_file == '':
            return sorted(glob.glob(self.get_input_path("*.{}".format(self.extension))))
        else:
            listi = []
            f = open(self.csv_file, 'r')
            ind = f.readline().replace('\n', '').split(',')
            for line in f:
                new_line = line.replace('\n', '').split(',')
                fili = os.path.join(self.input_root, new_line[ind.index('#file_extract')])
                listi.append(fili)
            return listi

    def __len__(self):
        return len(self.input_fnames)

    def write_features(self):

        paths = self.input_fnames

        fnames_context = map(
            lambda x: os.path.join(
                self.output_path, x.replace("." + self.extension, ".npy")
            ),
            map(lambda x: x.replace(self.input_root, '')[1:], paths),
        )
        # count = 0
        for name, target_fname in self._progress(
                zip(paths, fnames_context), total=len(self)
        ):
            wav, sr = read_audio(name)
            feat   = self.model(wav, self.use_feat)

            np.save(target_fname, feat)

    def write_features_fmri(self):

        paths = self.input_fnames

        fnames_context = map(
            lambda x: os.path.join(
                self.output_path, x.replace("." + self.extension, ".npy")
            ),
            map(lambda x: x.replace(self.input_root, '')[1:], paths),
        )
        #count = 0
        for name, target_fname in self._progress(
                zip(paths, fnames_context), total=len(self)
        ):
            wav, sr = read_audio(name)
            # we cut the sound wav into pieces of 10s with a stride of 5s
            size_win = 10*sr
            stride_win = 5*sr
            print('size wav',len(wav)/sr)
            size_wav_cut = len(wav) - size_win
            nb_cuts = int(size_wav_cut//stride_win)

            for i in range(nb_cuts):
                wav_to_transform = wav[int(stride_win*i):int(stride_win*i + size_win)]
                feat   = self.model(wav_to_transform, self.use_feat)
                if 'conv' in self.use_feat: # we get time in first dimension
                    feat = feat.swapaxes(0,1)
                if i == 0:
                    size_equival_win = int(feat.shape[0])
                    size_equival_stride = int(size_equival_win * (stride_win / size_win))
                    size_all = int(size_equival_win * len(wav) / size_win)
                    print('size_equivalent_win',size_win/sr, 's is', size_equival_win)
                    nb_times = np.asarray([0. for i in range(size_all)])
                    size_feat = feat.shape[1]
                    feat_all = np.zeros((size_all,size_feat))

                feat_all[int(i*size_equival_stride):int(i*size_equival_stride + size_equival_win), :] += feat
                nb_times[int(i*size_equival_stride):int(i*size_equival_stride + size_equival_win)] += 1.
                final = i

            # We get what is missing
            wav_to_transform = wav[int(stride_win * final + size_win):]
            feat = self.model(wav_to_transform, self.use_feat)
            if 'conv' in self.use_feat:  # we get time in first dimension
                feat = feat.swapaxes(0, 1)
            feat_all[int(final*size_equival_stride + size_equival_win):int(final*size_equival_stride + size_equival_win + feat.shape[0]), :] += feat
            nb_times[int(final*size_equival_stride + size_equival_win):int(final*size_equival_stride + size_equival_win + feat.shape[0])] += 1

            # get where the zeros are
            limit = 0
            for i in range(nb_times.shape[0]):
                if nb_times[i] == 0:
                    limit = i
                    break

            # now we average over strides
            divider = np.asarray([nb_times for i in range(size_feat)])
            divider = divider.swapaxes(0,1)
            feat_all = feat_all[:limit] / divider[:limit]

            np.save(target_fname, feat_all)
            #if count == 0:
            #    return

    def __repr__(self):

        return "EmbeddingDatasetWriter ({n_files} files)\n\tinput:\t{input_root}\n\toutput:\t{output_root}\n\tsplit:\t{split})".format(
            n_files=len(self), **self.__dict__
        )