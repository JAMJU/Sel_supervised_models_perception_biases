


This code implements a means to extract sound files representations on a CPC model (either trained on Common Voice or on Audioset).


# Environment installation
Same instructions as in cpc_audio model + pip install omegaconf hydra hydra-core

# Run extractions on a single layer
``` 
python new_get_representations.py 
model.checkpoint_path=path/to/your/checkpoint.ckpt 
audio_path=/path/to/wav/folder/
layer=desired_layer #conv1, conv2, conv3, conv4, conv5, AR
```

# Run extractions on all layers

### On zerospeech
Run `sbatch extract_random_cpc.slurm`, `sbatch extract_commonvoice_cpc.slurm`, `sbatch extract_audioset_cpc.slurm`.