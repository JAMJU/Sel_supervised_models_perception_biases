import wav2vec2_general_test
import os
# to work the code needs to have the stimuli as wav files at 16000Hz with 1c

# WARNINGS : the wav files need to be mono and sampled at 16000Hz
def main(layer,PATHOUTPUT, FOLDERCHECKPOINT, FNAMECHECKPOINT , stimuli_path_wav, stimuli_csv):

    writer = wav2vec2_general_test.EmbeddingDatasetWriter(
                input_root=stimuli_path_wav,
                output_root=PATHOUTPUT, # where you want to keep your extractions
                model_folder = FOLDERCHECKPOINT, # the folder where the checkpoint is (need to contain vocabulary file for the fine tuned models)
                model_fname=FNAMECHECKPOINT, # path to the checkpoint
                gpu=0, # do not change this
                extension='wav', # the extenson of your stimuli files
                use_feat=layer, # what kind of feature you want: 'conv_i' (i from 0 to 6) to extract from the convolutions of the encoder, 'z', 'q' or 'c' (following the naming of wav2vec 2.0 paper), or transf_0 to 12
                asr = False, # True if you are using a fine tuned model, False otherwise
                csv_file=stimuli_csv
            )

    print(writer)
    writer.require_output_path()

    print("Writing Features...")
    writer.write_features()
    print("Done.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to convert wav files')
    parser.add_argument('PATHOUT', metavar='f_do', type=str,
                        help='path output')
    parser.add_argument('folder_check', metavar='f_do', type=str,
                        help='destination folder')
    parser.add_argument('fname_check', metavar='f_do', type=str,
                        help='file checkpoint')
    parser.add_argument('stimuli_fold', metavar='f_do', type=str,
                        help='stimuli folder')
    parser.add_argument('stimuli_transf', metavar='f_do', type=str,
                        help='stimuli to transform')

    args = parser.parse_args()
    #stimuli_path_wav = "/gpfswork/rech/tub/uzz69cv/wav_to_transform/wordvowels16000"  # replace this by the path to your stimuli to transform (extracted not source otherwise it will not work)

    for lay in [  'transf_4']: # change this list to the layers you want
        main(lay, PATHOUTPUT=args.PATHOUT + '/' + ''.join(lay.split('_')), FOLDERCHECKPOINT=args.folder_check, FNAMECHECKPOINT=args.fname_check, stimuli_path_wav=args.stimuli_fold, stimuli_csv=args.stimuli_transf)
