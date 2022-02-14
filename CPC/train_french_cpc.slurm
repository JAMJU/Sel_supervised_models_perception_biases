#!/bin/bash
#SBATCH --job-name=FRCPC
#SBATCH --ntasks=1         # nombre de taches MPI par noeud
#SBATCH -A name
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gres=gpu:8                  # nombre de GPUs par nœud
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --cpus-per-task=20            # Ask for 20 CPU cores
##SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=fr_cpc_train.log         # Standard output and error log
#SBATCH --error=fr_cpc_train.out            # Standard output and error log

echo "Running job on $hostname"
cd cpc_model/

# nettoyage des modules charges en interactif et herites par defaut
module purge

# load conda environment
module load anaconda-py3/2020.11
module load libsndfile/1.0.28
conda activate cpc37_2

python cpc/train.py --save_step 2 --max_size_loaded 60000000 --pathDB commonvoice_french/all/ --pathVal commonvoice_datasets/french/dev_manifest_french.txt --pathTrain commonvoice_datasets/french/train_manifest_french.txt --pathCheckpoint checkpoints/french_cv --n_process_loader 8

