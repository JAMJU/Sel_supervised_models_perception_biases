#!/bin/bash
#SBATCH -A tub@gpu
#SBATCH --job-name=FRDEEP          # nom du job
#SBATCH --partition=gpu_p2l         # de-commente pour la partition gpu_p2
##SBATCH --gres=gpu:8 a tester +  avec gpu=2   ntasks-per-node=2 
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=12           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --qos=qos_gpu-t3             # SOQ
          
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=fr_deepspeech.log      # nom du fichier de sortie
#SBATCH --error=fr_deepspeech.out       # nom du fichier d'erreur (ici commun avec la sortie)


 # go into the right directory
cd UCR/

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load anaconda-py3/2020.11
module load libsndfile/1.0.28
module load sox/14.4.2
conda activate deepspeech2

# echo des commandes lancees
set -x
 
# execution du code
PL_TORCH_DISTRIBUTED_BACKEND=nccl python train_multi.py +experiment=UCR/configs/commonvoice.yaml data.train_path=training_deepspeech/manifests/french/commonvoice_train_manifest.json data.val_path=training_deepspeech/manifests/french/commonvoice_dev_manifest.json data.batch_size=32 trainer.gpus=4 data.labels_path=UCR/labels_frenchJZ.json checkpoint.save_top_k=10 checkpoint.monitor="cer" checkpoint.verbose=True trainer.accelerator=ddp data.augmentation.spec_augment=True data.num_workers=4 data.spect.language=french trainer.resume_from_checkpoint=outputs/2021-09-28/10-48-04/lightning_logs/version_1279576/checkpoints/epoch113.ckpt
