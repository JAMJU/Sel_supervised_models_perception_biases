## Extraction from Deepspeech intermediate layers

This code enables any user of deepspeech.pytorch to extract intermediary representations from a trained model given some sound stimuli
Clone this repository in the deespeech.pytorch folder, and move the file 'new_get_representations.py' in the main folder.

### Launching the extraction

`python new_get_representations.py model.model_path=path/t/checkpoint.ckpt audio_path=/path/to/your/audio/folder/ destination=path/to/where/you/to/store/your/representations/ layer=layer_waanted`

layer_wanted can be conv1, conv2, rnn1, rnn2, rnn3, rnn4, rnn5 and fully_connected

### Launching multiple extractions

Examples of sbatch files to launch multiple extractions are available: extract_deepspeech_english.slurm and exract_deepspeech_french.slurm