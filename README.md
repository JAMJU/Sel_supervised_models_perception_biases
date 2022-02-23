# Sel_supervised_models_perception_biases

## Human data and sound stimuli retrieval
Human data and sound stimuli used during the experiments are available in https://docs.cognitive-ml.fr/perceptimatic/

The concatenated results are available in the file humans_and_models/file_data.csv, along with the results of the different models tested in the paper.

## Training models

### Getting the training data
You can download the commonvoice dataset in English and French : https://commonvoice.mozilla.org/

An the audioset dataset : https://research.google.com/audioset/download.html

The audio files needs to be downsampled at 16000Hz, and need to be transformed into mono channel. This can be done using sox (http://sox.sourceforge.net/).

The files we used for the training of our models are available in training_sets/

The labels we selected for the audioset dataset are available in training_sets/labels_audioset.json

### DeepSpeech

Refer to the README.md file in DeepSpeech/. This code is a modified version of the original github : https://github.com/SeanNaren/deepspeech.pytorch

Example of the model's training command are in DeepSpeech/multi_gpu.slurm for English, and DeepSpeech/multi_gpu_french.slurm for French.

### CPC

Refer to the README.md in CPC/. This code is an adaptation of the original github: https://github.com/facebookresearch/CPC_audio#cross-lingual-transfer

Example of model's training command are in CPC/train_audioset_cpc.slurm for auioset, CPC/train_english_cpc.slurm for English and CPC/train_french_cpc.slurm for French.

### Wav2vec 2.0
Follow the instructions in https://github.com/pytorch/fairseq/tree/main/examples/wav2vec , using the config file : wav2vec2_base_librispeech.yaml

### HuBERT
Follow the instructions in https://github.com/pytorch/fairseq/tree/main/examples/hubert , using the config file : hubert_base_librispeech.yaml

## Extracting representations
### MFCCs
To extract mfccs, go into extractions and do the following command:

`python compute_mfccs.py $folder_where_you_downloaded_perceptimatic $folder_mfccs`

### DeepSpeech

First you need to copy the structure of the perceptimatic dataset in a folder where the extractions will be, using the copy_structure function in extractions/copy_structure.py

Go to extraction/extract_deepspeech and do:

`python new_get_representations.py model.model_path=deepspeech/english.ckpt audio_path=wav_to_transform/perceptimatic/ destination=deepspeech_english layer=rnn4 list_file=stimuli_data.csv`

### CPC

First you need to copy the structure of the perceptimatic dataset in a folder where the extractions will be, using the copy_structure function in extractions/copy_structure.py

Go to extraction/extract_cpc and do:


`python new_get_representations.py model.checkpoint_path=checkpoint.pt data.phone_labels=None data.audio_path=perceptimatic/ destination=output/ layer=AR data.n_process_loader=1`

### Wav2vec 2.0

First you need to copy the structure of the perceptimatic dataset in a folder where the extractions will be, using the copy_structure function in extractions/copy_structure.py

Clone the fairseq repo: https://github.com/pytorch/fairseq/

Copy the files in extractions/extract_wav2vec in the fairseq folder, then do (in the fairseq folder):

`python extract_wav2vec_layers.py output_folder checkpoint_folder checkpoint_file wav_to_transform_folder path/to/stimuli_data.csv`


### HuBERT

First you need to copy the structure of the perceptimatic dataset in a folder where the extractions will be, using the copy_structure function in extractions/copy_structure.py

Clone the fairseq repo: https://github.com/pytorch/fairseq/

Copy the files in extractions/extract_hubert in the fairseq folder

then do (in the fairseq folder):

`python extract_from_hubert.py perceptimatic_folder path_to/stimuli_data.csv checkpoint_file.pt transf_5 folder_destination
`

## Computing metrics

### Computing delta values
Before evaluating the models, you need to extract delta values, for each ABX triplets, from them. The values already computed are available in humans_and_models/file_data.csv (each column is the results for one model or information about the triplet used, each row is a participant's response).

To re-construct this file from scratch, you need to compute the delta values for each model, and then add them to the file human_data.csv (which is the concatenation of all the human data in Perceptimatic).

You need to have the extracted representations for all your models in individual folder (one folder per model) and all of them in a general folder. The name of the folders need to be chosen carefully (compute_distances_from_rep.py for a detail of the names).

To compute the delta values for a given model do:

`python compute_distances_from_rep.py $model $layer $distance $general_folder`

distance can be kl or cosine. The layers for all the models we tested are: deepspeech: rnn4, cpc: AR, wav2vec: transf4, and Hubert: transf_5 (put None for mfccs).

Once you have the files with delta values for each model you want to test, put them all in a folder and do:

`python add_value_on_humans.py $folder_delta human_data.csv $file_produced.csv` 


### Predicting human data
Once you have your file with human data and models' delta values, here are the instructins to compute our different metrics. For all of them, the bootstraping operation will output a file with each line corresponding to one sampling, and each column to the result of one given model. 

#### Log-likelihood
To compute the log-likelihood metric using bootstrap over participants do (with file_data.csv the file with human data and models' delta values):

For English:

`python probit_model_bootstrap.py humans_and_models/file_data.csv $results.csv $nb_iterations False True`

For French:

`python probit_model_bootstrap.py humans_and_models/file_data.csv $results.csv $nb_iterations True False`


#### Spearman correlation 
To compute the spearman correlation metric using bootstrap over participants do (with file_data.csv the file with human data and models' delta values):

`python spearman_correlation_bootstrap.py humans_and_models/file_data.csv $beginning_outfile $nb_iterations`

This script will output one file for French and another for English participants (ending with _english.csv and _french.csv). 

### Native language effect

To compute the native language effectusing bootstrap over participants do: (with file_data.csv the file with human data and models' delta values):

`python native_effect_bootstrap.py humans_and_models/file_data.csv $file_out.csv $nb_iterations`