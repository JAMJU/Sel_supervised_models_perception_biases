#!/bin/bash
#SBATCH -A tub@cpu
#SBATCH --job-name=convetr            # Job name
#SBATCH --output=convert.log          # nom du fichier de sortie
#SBATCH --error=convert.out           # nom du fichier d'erreur (ici en commun avec la sortie)
#SBATCH --cpus-per-task=30            # Ask for 10 CPU cores
#SBATCH --time=15:00:00               # Time limit hrs:min:sec


 # go into the right directory
cd /gpfswork/rech/tub/uzz69cv/UCR/data

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load anaconda-py3/2020.11
module load libsndfile/1.0.28
module load sox/14.4.2
conda activate deepspeech2

# execute
python phonemiz_fr.py /gpfswork/rech/tub/uzz69cv/training_deepspeech/manifests/french/commonvoice_test_manifest_txt.csv --language fr-fr --backend espeak --phone_target_dir /gpfsscratch/rech/tub/uzz69cv/phone_common_voice/french/test/phon --txt_dir /gpfsscratch/rech/tub/uzz69cv/text_common_voice/french/txt