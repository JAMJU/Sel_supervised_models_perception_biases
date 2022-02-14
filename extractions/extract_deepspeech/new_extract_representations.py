import json
from typing import List

import hydra
import torch
from torch.cuda.amp import autocast
import numpy as np
import os
from Extraction.inference_config import TranscribeConfig
from Extraction.data_loader import SpectrogramParser
from Extraction.model_for_extraction import DeepSpeech
#from deepspeech_pytorch.utils import load_model



def load_model(device,
               model_path):
    model = DeepSpeech.load_from_checkpoint(checkpoint_path = hydra.utils.to_absolute_path(model_path), strict = False)
    model.eval()
    model = model.to(device)
    return model

def save_outputs(outputs, layer, i, destination_path):

    full_destination_path = os.path.join(destination_path, layer)
    os.makedirs(full_destination_path, exist_ok=True)
    save_filename = '{}.npy'.format(i)
    np.save(os.path.join(full_destination_path, save_filename), outputs)
    #print("Outputs were saved at: ", full_destination_path)


def reshape_outputs(outputs, layer):
    # W : time dimension
    # outputs WxDim

    if layer.startswith('input'):
        # image output of size [1, nb_channels, H, W]
        return outputs[0, 0, :, :].swapaxes(0,1)

    elif layer.startswith('conv'):
        # image output of size [1, nb_channels, H, W]
        nb_channels = outputs.shape[1]
        W = outputs.shape[3]
        H = outputs.shape[2]
        reshaped_output = np.zeros((H * nb_channels, W ))
        for i in range(nb_channels):
            reshaped_output[i * H:(i + 1) * H, :] = outputs[0, i, :, :].cpu().detach().numpy()
        reshaped_output = reshaped_output.swapaxes(0,1)
    elif layer.startswith('rnn'):
        # image output of size [W, 1, H]
        reshaped_output = outputs[:, 0, :].cpu().detach().numpy()
        #reshaped_output = reshaped_output.transpose()
    elif layer.startswith('lookahead'):
        # image output of size [W, 1, H]
        W = outputs.shape[0]
        H = outputs.shape[2]
        reshaped_output = outputs[:, 0, :].cpu().detach().numpy()
        #reshaped_output = reshaped_output.transpose()
    elif layer.startswith('fully_connected'):
        # image output of size [ 1, W, H]
        out = outputs.cpu().detach().numpy()
        reshaped_output = outputs[0, :, :].cpu().detach().numpy()
        #reshaped_output = reshaped_output.transpose()
    return reshaped_output


def run_extract(audio_path: str,
                   spect_parser: SpectrogramParser,
                   model: DeepSpeech,
                   device: torch.device,
                   precision: int,
                    layer: str):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    with autocast(enabled=precision == 16):
        #out, output_sizes = model(spect, input_sizes)
        out, output_sizes = model.intermediate_forward(spect, input_sizes, layer)
    #decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return out, output_sizes

def run_extract_fmri(audio_path: str,
                   spect_parser: SpectrogramParser,
                   model: DeepSpeech,
                   device: torch.device,
                   precision: int,
                    layer: str):
    spect_ = spect_parser.parse_audio(audio_path).contiguous()# dim x nb frames time
    spect_ = spect_.view(1, 1, spect_.size(0), spect_.size(1))
    print(spect_.size(2),spect_.size(3))
    size_total = spect_.size(3)

    # one window = 10ms = 0.01 s, we want 10s of win, five of stride
    size_win = 10./0.01
    stride_win = 5./0.01
    size_wav_cut = size_total - size_win
    nb_cuts = int(size_wav_cut // stride_win)

    for i in range(nb_cuts):
        spect = spect_[:,:,:,int(stride_win*i):int(stride_win*i + size_win)]
        spect = spect.to(device)
        input_sizes = torch.IntTensor([spect.size(3)]).int()
        with autocast(enabled=precision == 16):
            #out, output_sizes = model(spect, input_sizes)
            out, output_sizes = model.intermediate_forward(spect, input_sizes, layer)
        out = reshape_outputs(out, layer) # time x dim
        print(out.shape)
        if i == 0:
            size_equival_win = int(out.shape[0])
            size_equival_stride = int(size_equival_win * (stride_win / size_win))
            size_all = int(size_equival_win * size_total / size_win)
            print('size_equivalent_win', '10s', 's is', size_equival_win)
            nb_times = np.asarray([0. for i in range(size_all)])
            size_feat = out.shape[1]
            feat_all = np.zeros((size_all, size_feat))

        feat_all[int(i * size_equival_stride):int(i * size_equival_stride + size_equival_win), :] += out
        nb_times[int(i * size_equival_stride):int(i * size_equival_stride + size_equival_win)] += 1.
        final = i

    # we get what is missing
    spect = spect_[:, :, :, int(stride_win * final + size_win):]
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    with autocast(enabled=precision == 16):
        # out, output_sizes = model(spect, input_sizes)
        out, output_sizes = model.intermediate_forward(spect, input_sizes, layer)
    out = reshape_outputs(out, layer)  # time x dim
    if int(final * size_equival_stride + size_equival_win + out.shape[0]) > feat_all.shape[0]:
        feat_all[int(final * size_equival_stride + size_equival_win) - 1:int(
            final * size_equival_stride + size_equival_win + out.shape[0]), :] += out
        nb_times[int(final * size_equival_stride + size_equival_win) - 1:int(
            final * size_equival_stride + size_equival_win + out.shape[0])] += 1.
    else:
        feat_all[int(final * size_equival_stride + size_equival_win):int(final * size_equival_stride + size_equival_win + out.shape[0]), :] += out
        nb_times[int(final * size_equival_stride + size_equival_win):int(final * size_equival_stride + size_equival_win + out.shape[0])] += 1.

    #decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    # get where the zeros are
    """limit = 0"""
    for i in range(nb_times.shape[0]):
        if nb_times[i] == 0:
            print('is 0', i)
            nb_times[i] = 1.

    # now we average over strides
    divider = np.asarray([nb_times for i in range(size_feat)])
    divider = divider.swapaxes(0, 1)
    feat_all = feat_all / divider
    return feat_all, output_sizes

def extract(cfg: TranscribeConfig,
            layer: str,
            destination_path: str,
            list_file: str = '',
            fmri: bool = False):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    spect_parser = SpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )
    count = 0
    if list_file == '':
        for file in os.listdir(hydra.utils.to_absolute_path(cfg.audio_path)):
            if not file.endswith('.wav'):
                continue

            file_all = os.path.join(hydra.utils.to_absolute_path(cfg.audio_path), file)

            if fmri:
                new_outputs, sizes = run_extract_fmri(audio_path=file_all,
                            spect_parser=spect_parser,
                            model=model,
                            device=device,
                            precision=cfg.model.precision,
                            layer = layer)
            else:
                output, sizes = run_extract(
                    audio_path=file_all,
                    spect_parser=spect_parser,
                    model=model,
                    device=device,
                    precision=cfg.model.precision,
                    layer = layer
                )
                # Reshape the outputs to have time x dim
                new_outputs = reshape_outputs(output, layer)

            # Save the representations
            #dataset_name = 'dataset'
            #destination_path = '/gpfsscratch/rech/tub/uzz69cv/transfo_multi/deepspeech_english'
            save_outputs(new_outputs, layer = layer, i = file.replace('.wav', ''), destination_path=destination_path)
            count +=1
            if count % 100 == 0:
                print(count)
    else:
        # first we get
        f = open(list_file, 'r')
        ind = f.readline().replace('\n', '').split(',')
        for line in f:
            new_line = line.replace('\n', '').split(',')
            fili = new_line[ind.index('#file_extract')]

            file_all = os.path.join(hydra.utils.to_absolute_path(cfg.audio_path), fili)
            if fmri:
                new_outputs, sizes = run_extract_fmri(audio_path=file_all,
                            spect_parser=spect_parser,
                            model=model,
                            device=device,
                            precision=cfg.model.precision,
                            layer = layer)
            else:
                output, sizes = run_extract(
                    audio_path=file_all,
                    spect_parser=spect_parser,
                    model=model,
                    device=device,
                    precision=cfg.model.precision,
                    layer = layer
                )
                # Reshape the outputs to have time x dim
                new_outputs = reshape_outputs(output, layer)

            # Save the representations
            # dataset_name = 'dataset'
            # destination_path = '/gpfsscratch/rech/tub/uzz69cv/transfo_multi/deepspeech_english'
            save_outputs(new_outputs, layer=layer, i=fili.replace('.wav', ''), destination_path=destination_path)
            count += 1
            if count % 100 == 0:
                print(count)
    print('Done!')


