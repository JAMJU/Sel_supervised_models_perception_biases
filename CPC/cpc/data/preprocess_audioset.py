""" From the Json manifests created for deepspeech, creates txt files containing the path directly to the wav files """

import os
import sys
import json
import argparse

CORRUPTED_LIST = ['/gpfsdswork/dataset/AudioSet/unbalanced_train/-/-zVGs8QmK1I_210.000_220.000.wav',
                  '/gpfsdswork/dataset/AudioSet/unbalanced_train/1/18tpMkI4f2I_30.000_40.000.wav']

def transfer_necessary_audioset_sound_files_to_scratch(path_to_audioset_folder, data_type, num_classes=183):

    # Create the folder to store the av files if not done
    os.makedirs(os.path.join(path_to_audioset_folder, 'wavs'), exist_ok=True)

    # Load the json file created for deepspeech
    json_full_filename = os.path.join(path_to_audioset_folder, '{}_manifest_{}.json'.format(data_type, num_classes))

    with open(json_full_filename, 'r') as json_file:
        db = json.load(json_file)['samples']

    i = 0

    for sample in db:
        i += 1
        sys.stdout.write('\r Checking if sample needs to be copied .... {} /{}'.format(i, len(db)))
        wav_name= sample['wav_path'].split('/')[-1]
        if not os.path.isfile(os.path.join(path_to_audioset_folder, 'wavs', wav_name)):
            os.system('cp {} {}'.format(sample['wav_path'], os.path.join(path_to_audioset_folder, 'wavs')))
    return None

def create_txt_manifest(path_to_audioset_folder, data_type, num_classes=183):

    # Create a txt file
    lines = []

    # Load the json file created for deepspeech
    json_full_filename = os.path.join(path_to_audioset_folder, '{}_manifest_{}.json'.format(data_type, num_classes))

    with open(json_full_filename, 'r') as json_file:
        db = json.load(json_file)['samples']
    i = 0
    for sample in db:
        i += 1
        sys.stdout.write('\r Treating txt file .... {} /{}'.format(i, len(db)))
        wav_name = sample['wav_path'].split('/')[-1]
        wav_name_no_extension = wav_name[:-4]

        if wav_name not in CORRUPTED_LIST:
            lines.append(wav_name_no_extension)

    full_destination_filename = os.path.join(path_to_audioset_folder, 'audioset/{}_manifest_{}.txt'.format(data_type, num_classes))

    with open(full_destination_filename, 'w') as txt_file:
        txt_file.writelines('\n'.join(lines))
    return None

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_audioset_folder', '-db',
                        default='/gpfsscratch/rech/rnt/uuj49ar',
                        help='Path to the folder containing audioset Json manifest (and which will contain ewisting wavs')
    parser.add_argument('--num_classes', '-nb', default=182)
    parser.add_argument('--data_type', '-d', choices=['small_train', 'small_validation', 'small_test'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_argparse()
    
    transfer_necessary_audioset_sound_files_to_scratch(args.path_to_audioset_folder, 'small_train',
                                                       num_classes=args.num_classes)
    transfer_necessary_audioset_sound_files_to_scratch(args.path_to_audioset_folder, 'small_validation',
                                                       num_classes=args.num_classes)
    transfer_necessary_audioset_sound_files_to_scratch(args.path_to_audioset_folder, 'small_test',
                                                       num_classes=args.num_classes)
    
    create_txt_manifest(args.path_to_audioset_folder, 'small_train', num_classes=182)
    print()
    create_txt_manifest(args.path_to_audioset_folder, 'small_validation', num_classes=182)
    print()
    create_txt_manifest(args.path_to_audioset_folder, 'small_test', num_classes=182)
    print()
