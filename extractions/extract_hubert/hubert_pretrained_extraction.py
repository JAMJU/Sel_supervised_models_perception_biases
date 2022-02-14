# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.modules import GradMultiply
import numpy as np


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


def conv_forward_for_featurizer(conv_encoder, x, nb):

        # BxT -> BxCxT
        x = x.unsqueeze(1)
        count = 0
        for conv in conv_encoder.conv_layers:
            x = conv(x)
            if count == nb:
                return x
            count += 1

        return x


def forward_features(model, source, nb):
    if model.feature_grad_mult > 0:
        features = conv_forward_for_featurizer(model.feature_extractor, source, nb)
        if model.feature_grad_mult != 1.0:
            features = GradMultiply.apply(features, model.feature_grad_mult)
    else:
        with torch.no_grad():
            features = conv_forward_for_featurizer(model.feature_extractor, source, nb)
    return features

def forward_for_featurizer(
        model,
        source,
        target_list = None,
        padding_mask= None,
        mask=True,
        features_only=False,
        output_layer= None,
) :
    """output layer is 1-based"""
    if 'conv' in output_layer:
        features = forward_features(model, source, int(output_layer.split('_')[1]))
        return features.cpu().numpy()
    else:
        features = model.forward_features(source)
    if target_list is not None:
        features, target_list = model.forward_targets(features, target_list)

    #features_pen = features.float().pow(2).mean()

    features = features.transpose(1, 2)
    features = model.layer_norm(features)
    #unmasked_features = features.clone()

    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(features, padding_mask)

    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)

    features = model.dropout_input(features)
    #unmasked_features = model.dropout_features(unmasked_features)

    if mask:
        x, mask_indices = model.apply_mask(
            features, padding_mask, target_list
        )
    else:
        x = features
        mask_indices = None

    # feature: (B, T, D), float
    # target: (B, T), long
    # x: (B, T, D), float
    # padding_mask: (B, T), bool
    # mask_indices: (B, T), bool
    x, _ = model.encoder(
        x,
        padding_mask=padding_mask,
        layer=None if output_layer is None else int(output_layer.split('_')[1])
    )

    return x.cpu().numpy()




class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]

                feat_chunk = forward_for_featurizer(model = self.model,
                    source = x_chunk,
                    padding_mask=None,
                    mask=False,
                    features_only=True,
                    output_layer=self.layer,
                )
                print(feat_chunk.shape)

                feat.append(feat_chunk)
        print(np.concatenate(feat, 1).shape)
        print(np.concatenate(feat, 1).squeeze(0).shape)
        return np.concatenate(feat, 1).squeeze(0)

