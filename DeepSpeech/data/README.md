## Description of the CommonVoice Dataset

### Manifest files 

The three manifest files are CSV files containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```

The first path is to the audio file, and the second path is to a text file containing the transcript on one line. 



### Dataset 

The data after downloading are divided into three folders: train dev test, each composed of wav txt phon

```

wav : contains the .wav sound files 
txt : contains the text statements  
phon : contains the same statements but phonemized

```


### Phonemizer

You can use the script `phonemiz_fr.py` to phonemize the txt repository. You have to specifiy the manifest file, the language and the backend ( espeak / festival / espeak-mbrole). There is an example below : 

```
python phonemiz_fr.py commonvoice_train_manifest.csv --language fr-fr --backend espeak
```
