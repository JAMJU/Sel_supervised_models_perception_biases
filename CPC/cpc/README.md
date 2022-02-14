# Repository's architecture

train.py : main script

dataset.py : defintion of the Librispeech dataset format

model.py : Basic encoders and AR models

feature_loader.py: different tools to load and save a CPC model.

transformers.py: an implementation of transformers

unit_tests.py : unit tests

criterion/: definition of the training criterions. Three criterion are currently available: CPC (unsupervised), speaker classification and phone classification.

eval/: evaluation scripts.

utils/: system utilities and misc.



# BEWARE: Debugging on Oberon

to launch a training with identified sound files :
```
python cpc/train.py 
    --pathDB /scratch2/vbrami/deepspeech_model2/data/Freesound_dataset/FSDKaggle2019.audio_train_curated/wav 
    --pathCheckpoint checkpoints
    --n_process_loader=2 # Must not exceed the number of cpus requested !!
```
# IMPORTANT: Multiprocessing
the parameter n_process_loader ust not exceed the number of cpus requested !! Otherwise the training won't work correctly.


