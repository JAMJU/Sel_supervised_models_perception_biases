# Sel_supervised_models_perception_biases

## Human data and sound stimuli retrieval
Human data and sound stimuli used during the experiments are available in https://docs.cognitive-ml.fr/perceptimatic/

The concatenated results are available in the file humans_and_models/file_data.csv, along with the results of the different models tested in the paper.

## Training models

## Getting the training data
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

### HuBERT

## Extracting representations
### MFCCs

### DeepSpeech

### CPC

### Wav2vec 2.0

### HuBERT


## Computing metrics

### Predicting human data

#### Log-likelihood

#### Spearman correlation 

### Native language effect