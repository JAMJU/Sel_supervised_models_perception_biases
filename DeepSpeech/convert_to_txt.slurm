#!/bin/bash
#SBATCH -A tub@cpu
#SBATCH --job-name=txt            # Job name
#SBATCH --error=converttext.out          # nom du fichier de sortie
#SBATCH --cpus-per-task=10            # Ask for 10 CPU cores
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=converttext.log            # Standard output and error log

echo "Running job on $hostname"

# on se place dans le repertoire de soumission
cd /gpfswork/rech/tub/uzz69cv/UCR

# nettoyage des modules charges en interactif et herites par defaut
module purge

# load conda environment
# chargement des modules
module load anaconda-py3/2020.11
module load libsndfile/1.0.28
module load sox/14.4.2
conda activate deepspeech2

# launch your computation
python get_only_text.py