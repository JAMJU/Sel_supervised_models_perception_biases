# CPC_audio

This code implements the Contrast Predictive Coding algorithm on audio data, as described in the paper [Unsupervised Pretraining Transfers well Across Languages](https://arxiv.org/abs/2002.02848). This is an unsupervised method to train audio features directly from the raw waveform.


## Setup instructions

The installation is a tiny bit involved due to the torch-audio dependency.

0/ Clone the repo:
`git clone git@github.com:facebookresearch/CPC_audio.git && cd CPC_audio`

1/ Install libraries which would be required for torch-audio https://github.com/pytorch/audio :
 * MacOS: `brew install sox`
 * Linux: `sudo apt-get install sox libsox-dev libsox-fmt-all`

2/ `conda env create -f environment.yml && conda activate cpc37`

3/ Run setup.py
`python setup.py develop`

You can test your installation with:
`nosetests -d`

### CUDA driver

This setup is given for CUDA 9.2 if you use a different version of CUDA then please change the version of cudatoolkit in environment.yml.
For more information on the cudatoolkit version to use, please check https://pytorch.org/

### Standard datasets

We suggest to train the model either on [Librispeech](http://www.openslr.org/12/) or [libri-light](https://github.com/facebookresearch/libri-light).


## How to run a session


To run a new training session, use:

- On CommonVoice:
```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --arMode $PREDICTION_NETWORK
```
or with the sbatch file:
```bash
cd sbatchs
sbatch train_french_commonvoice.slurm
```
```bash
cd sbatchs
sbatch train_english_commonvoice.slurm
```
- On AudioSet:
```bash
python cpc/audioset_train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
```
or with the sbatch file:
```bash
cd sbatchs
sbatch train_audioset.slurm
```
Where:
- $PATH_AUDIO_FILES is the directory containing the audio files. 
  
The files should be arranged as below:
```
PATH_AUDIO_FILES  
│
└───speaker1
│   └───...
│         │   seq_11.{$EXTENSION}
│         │   seq_12.{$EXTENSION}
│         │   ...
│   
└───speaker2
    └───...
          │   seq_21.{$EXTENSION}
          │   seq_22.{$EXTENSION}
```
For those trainings, use the manifests stored in `results/datasets_files_and_parameters_for_training/cpc`.

Launching a training from scratch will generate in  $PATH_CHECKPOINT_DIR:
- a ```checkpoint_args.json``` file, gathering all the parameters used for the training and editable with argparse.
- a ```checkpoint_logs.json``` file, where all the train/validation losses and accuracy are written at each epoch.
- ```checkpoint_*.pt``` files: checkpoints at a given epoch.

The training will restart automatically from a checkpoint if $PATH_CHECKPOINT_DIR is not empty and contains both ```checkpoint_args.json```
and a ```checkpoint_*.pt``` file.

Please note that each speaker directory can contain an arbitrary number of subdirectories: the speaker label will always be retrieved from the top one. The name of the files isn't relevant. For a concrete example, you can look at the organization of the [Librispeech](http://www.openslr.org/12/) dataset.

- $PATH_CHECKPOINT_DIR in the directory where the checkpoints will be saved
- $TRAINING_SET is a path to a .txt file containing the list of the training sequences (see [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb) for example)
- $VALIDATION_SET is a path to a .txt file containing the list of the validation sequences
- $EXTENSION is the extension of each audio file (here wav files)
- $PREDICTION_NETWORK is the architecture of the prediction net

## Custom architectures

The code allows you to train a wide range of architectures. For example, to train the CPC method as described in [Van Den Oord's paper](https://arxiv.org/abs/1807.03748) just run:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --normMode batchNorm --rnnMode linear
```

Or if you want to train a model with a FFD prediction network instead of a transformer:
```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --rnnMode ffd --schedulerRamp 10
```

The --schedulerRamp option add a learning rate ramp at the beginning of the training: it barely affects the performance of a model with a transformer predictor but is necessary with other models.

Launch cpc/train.py -h to see all the possible options.

## How to restart a session

To restart a session from the last saved checkpoint just run
```bash
python cpc/train.py --pathCheckpoint $PATH_CHECKPOINT_DIR
```

## Citations
Please consider citing this project in your publications if it helps your research.

```
@misc{rivire2020unsupervised,
    title={Unsupervised pretraining transfers well across languages},
    author={Morgane Rivière and Armand Joulin and Pierre-Emmanuel Mazaré and Emmanuel Dupoux},
    year={2020},
    eprint={2002.02848},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

## License

CPC_audio is MIT licensed, as found in the LICENSE file.
