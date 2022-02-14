import json
import argparse
import os
from pathlib import Path, PosixPath
import pandas as pd
from tqdm.notebook import tqdm


def create_commonvoice_manifest_and_labels_bis(path_to_json_file, language, data_type):
    manifest = {
        'root_path': '/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11',
        'samples': []
    }
    csv_filename = os.path.join(path_to_json_file, language, '{}.tsv'.format(data_type))
    csv_data_df = pd.read_csv(csv_filename, sep='\t')

    for (i, row) in tqdm(csv_data_df.iterrows(), total=csv_data_df.shape[0]):

        transcript = row['sentence']

        if type(transcript) == str:
            print('\r  Transcript {}: '.format(i), type(transcript), transcript)
            filename = row['path'].replace('.mp3', '.txt')
            wav_filename = row['path'].replace('.mp3', '.wav')
            transcript_path = os.path.join(path_to_json_file, language, filename)

            f = open(transcript_path, 'w')
            f.write(transcript)
            f.close()

            new_sample = dict()
            new_sample['wav_path'] = os.path.join('/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11',
                                                  language[:2], 'clips_wav', wav_filename)
            new_sample['transcript_path'] = transcript_path
            manifest['samples'].append(new_sample)

    # Create TXT File

    full_destination_path = os.path.join(path_to_json_file,
                                         'commonvoice_{data}_manifest_{lang}.json'.format(data=data_type,
                                                                                          lang=language))

    with open(full_destination_path, 'w') as json_file:
        json.dump(manifest, json_file, indent=4)


def create_commonvoice_manifest_and_labels(path_to_json_file, language, data_type, create_labels_files=False):
    manifest = {
        'root_path': '/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11',
        'samples': []
    }
    csv_filename = os.path.join(path_to_json_file, language, '{}.tsv'.format(data_type))
    csv_data_df = pd.read_csv(csv_filename, sep='\t')

    old_json_file = os.path.join(path_to_json_file, language,
                                 'commonvoice_{data}_manifest_{lang}_louis.json'.format(data=data_type, lang=language))
    with open(old_json_file, 'r') as json_f:
        old_data = json.load(json_f)

    index = 1
    full_destination_path = os.path.join(path_to_json_file,
                                         'commonvoice_{data}_manifest_{lang}.json'.format(data=data_type,
                                                                                          lang=language))

    for sample in old_data['samples']:
        sample['wav_path'] = os.path.join('/', sample['wav_path'])

        split_name = sample['wav_path'].split('/')
        sample['wav_path'] = os.path.join('/gpfsscratch/rech/rnt/uuj49ar/CommonVoice/cv-corpus-6.1-2020-12-11',
                                          language[:2], 'clips_wav', split_name[-2], split_name[-1])
        transcript_name = sample['wav_path'].split('/')
        wav_name = transcript_name[-1]
        transcript_name = '/'.join([transcript_name[-2], transcript_name[-1].replace('.wav', '.txt')])
        os.makedirs(os.path.join(path_to_json_file, language, data_type, 'txt'), exist_ok=True)
        sample['transcript_path'] = os.path.join(path_to_json_file, language, data_type, 'txt', transcript_name)

        # create transcript file
        mp3_name = transcript_name.replace('.txt', '.mp3')
        transcript = csv_data_df[csv_data_df['path'] == mp3_name]['sentence'].item()
        print('\r Transcript {} / {}:'.format(index, len(old_data['samples'])), transcript)

        if index % 20000 == 0:
            with open(full_destination_path, 'w') as json_file:
                json.dump(manifest, json_file, indent=4)

        if create_labels_files:
            f = open(sample['transcript_path'], 'w')
            f.write(transcript)
            f.close()

        manifest['samples'].append(sample)
        index += 1

    with open(full_destination_path, 'w') as json_file:
        json.dump(manifest, json_file, indent=4)

def create_commonvoice_txt_files(path_to_json_file, language, data_type, create_labels_files=False):


    old_json_file = os.path.join(path_to_json_file, language,
                                 'commonvoice_{data}_manifest_{lang}_louis.json'.format(data=data_type, lang=language))
    with open(old_json_file, 'r') as json_f:
        old_data = json.load(json_f)['samples']

    new_data_filename = os.path.join(path_to_json_file, language,'commonvoice_{data}_manifest_{lang}.txt'.format(data=data_type, lang=language))
    txt_file = open(new_data_filename, 'w')
    text_list = []

    for i in range(len(old_data)):
        audio_filename = old_data[i]['wav_path']
        audio_filename = audio_filename.split('/')
        # audio_filename = '/'.join([audio_filename[-2], audio_filename[-1].split('.')[0]])
        audio_filename = audio_filename[-1].split('.')[0]
        print('\r   Audio {} /{}: {}'.format(i, len(old_data), audio_filename))

        if i != len(old_data) - 1:
            text_list.append(audio_filename + "\n")
        else:
            text_list.append(audio_filename)

    txt_file.writelines(text_list)
    txt_file.close()

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-d', choices=['train', 'dev', 'test', 'validated'])
    parser.add_argument('--language', '-l', choices=['french', 'english'])
    parser.add_argument('--path_to_json_file', '-db', default='/gpfsscratch/rech/rnt/uuj49ar/CommonVoice')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = build_argparse()
    # create_commonvoice_manifest_and_labels(args.path_to_json_file, args.language, args.data_type)
    create_commonvoice_txt_files(args.path_to_json_file, args.language, args.data_type, create_labels_files=False)
    print()
