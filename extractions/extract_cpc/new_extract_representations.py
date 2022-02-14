from model_for_extraction import CPCModel, ConcatenatedModel
import torch
import os
import numpy as np
from torch.cuda.amp import autocast
import feature_loader_for_extraction as fl
from representations_config import TranscribeConfig
from data_loader import findAllSeqs

# Function for loading the model from a specific checkpoint
def load_model(checkpoint_path, device):
    cpcmodel, hidden_gar, hidden_encoder = fl.loadModel(checkpoint_path, device)
    print('\n Loaded CPC Model of type: {}'.format(type(cpcmodel)))
    return cpcmodel, hidden_gar, hidden_encoder

# Function for saving the outputs
def save_outputs(outputs, layer, name, destination_path):
    """

    :param outputs: (array) reshaped model outputs
    :param layer:  (str) name of the layer from which we extract a representation
    :param name: (str) name of the sound file (without path and extension)
    :param destination_path: (str) path where we save the outputs
    :return: Saved outputs on a specific layer
    """
    full_destination_path = os.path.join(destination_path, layer)
    os.makedirs(full_destination_path, exist_ok=True)
    save_filename = '{}.npy'.format(name)
    np.save(os.path.join(full_destination_path, save_filename), outputs)


# Function for reshaping the outputs
def reshape_outputs(outputs, layer):
    """

    :param outputs: (tensor) model layer outputs
    :param layer: (str) name of the layer from which we reshape the output
    :return:
        W: Time dimension
        outputs: W x Dim
    """
    if layer.startswith('input'):
        # image output of size [1, nb_channels, H, W]
        return outputs[0, :, :].swapaxes(0, 1)
    else:
        if layer.startswith('conv'):
            # image output of size [1, H, W]
            reshaped_output = outputs[0, :, :].cpu().detach().numpy()
            
            #if layer != 'conv5':
            reshaped_output = reshaped_output.swapaxes(0, 1)
            return reshaped_output

        elif layer == 'AR':
            return outputs[0,:,:]


# Function for extraction
def run_extract(wav_path,
                label,
                model,
                precision,
                layer):

    with autocast(enabled=precision == 16):
        c_feature, encoded_data, label = model.intermediate_forward(batch_data, label, layer)

    return c_feature, encoded_data, label



# Main function
def extract(cfg: TranscribeConfig, layer, destination_path, csv_file = ''):

    device = torch.device("cuda" if cfg.model.cuda else "cpu")
    print('\n   DEVICE used for loading the model is {}  {} \n'.format(cfg.model.cuda, device))
    cpcmodel, hidden_gar, hidden_encoder = load_model([cfg.model.checkpoint_path], device)
    seq_names, speakers = findAllSeqs(cfg.data.audio_path,
                                      extension=cfg.data.file_extension)
    # cpcmodel = torch.nn.DataParallel(cpcmodel, device_ids=range(cfg.nGPU)).cuda()
    # No need to filter sequences
    """
    dataset = AudioBatchData(cfg.data.audio_path,
                             cfg.data.size_window, seq_names,
                             phoneLabelsDict=None, # cfg.data.phone_labels,
                             nSpeakers=len(speakers),
                             nProcessLoader=cfg.data.n_process_loader,
                             MAX_SIZE_LOADED=cfg.data.max_size_loaded)
    
    
    print('Length of the dataset: {}'.format(len(dataset)))
    batch_size = 1
    audio_loader = dataset.getDataLoader(batch_size, 'sequential', False, numWorkers=1)                                       
    """
    count = 0
    print('\n Outputs of OS WALK', os.walk(cfg.data.audio_path))

    audio_data_path = cfg.data.audio_path
    if csv_file == '':
        for wav_file in os.listdir(audio_data_path):

            if wav_file.endswith('.wav'):
                # Get the full path to the audio file
                wav_full_path = os.path.join(audio_data_path, wav_file)

                # Generate the representation
                feature = fl.buildIntermediateFeature(cpcmodel, layer, wav_full_path, strict=False,
                                                      maxSizeSeq=300000, seqNorm=False) # 64000
                # Reshape the outputs to have time x dim
                new_outputs = reshape_outputs(feature, layer)

                if 'pilot-july' in cfg.data.audio_path:
                    wav_file = os.path.join(audio_data_path.split('/')[-3], audio_data_path.split('/')[-2], audio_data_path.split('/')[-1], wav_file)

                # Save the feature
                save_outputs(new_outputs, layer=layer, name=wav_file.replace('.wav', ''), destination_path=destination_path)

            count += 1
            if count % 100 == 0:
                print(count)
    else:
        f = open(csv_file, 'r')
        ind = f.readline().replace('\n', '').split(',')
        for line in f:
            new_line = line.replace('\n', '').split(',')
            fili = new_line[ind.index('#file_extract')]
            file = os.path.join(audio_data_path, fili)
            # Generate the representation
            feature = fl.buildIntermediateFeature(cpcmodel, layer, file, strict=False,
                                                  maxSizeSeq=300000, seqNorm=False)  # 64000
            # Reshape the outputs to have time x dim
            new_outputs = reshape_outputs(feature, layer)

            # Save the feature
            save_outputs(new_outputs, layer=layer, name=fili.replace('.wav', ''), destination_path=destination_path)

    print('Done!')




