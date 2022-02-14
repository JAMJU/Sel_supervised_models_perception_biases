#!/usr/bin/env python3
import os
import csv
import tqdm
from multiprocessing.pool import ThreadPool


def convert_to_txt(csv_file, target_dir, num_workers):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to target_dir.

    Args:
        csv_file: str, path to *.csv file with data description, usually start from 'cv-'
        target_dir: str, path to dir to save results; wav/ and txt/ dirs will be created
    """
    #wav_dir = os.path.join(target_dir, 'wav/')
    txt_dir = os.path.join(target_dir, 'txt/')
    #os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    #audio_clips_path = os.path.dirname(csv_file) + '/clips/'

    def process(x):
        file_path, text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.strip().upper()
        with open(os.path.join(txt_dir, file_name + '.txt'), 'w') as f:
            f.write(text)
        #audio_path = os.path.join(audio_clips_path, file_path)
        #output_wav_path = os.path.join(wav_dir, file_name + '.wav')

        #tfm = Transformer()
        #tfm.rate(samplerate=args.sample_rate)
        #tfm.build(
        #    input_filepath=audio_path,
        #    output_filepath=output_wav_path
        #)

    print('Converting tsv to txt for {}.'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        data = [(row['path'], row['sentence']) for row in reader]
        with ThreadPool(num_workers) as pool:
            list(tqdm.tqdm(pool.imap(process, data), total=len(data)))

if __name__ == '__main__':
    file_train = "/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr/train.tsv"
    file_test = "/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr/test.tsv"
    file_valid = "/gpfsdswork/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr/validated.tsv"

    target_dir = "/gpfsscratch/rech/tub/uzz69cv/text_common_voice/french"
    num = 10
    convert_to_txt(csv_file=file_train, target_dir=target_dir, num_workers=num)
    convert_to_txt(csv_file=file_test, target_dir=target_dir, num_workers=num)
    convert_to_txt(csv_file=file_valid, target_dir=target_dir, num_workers=num)
