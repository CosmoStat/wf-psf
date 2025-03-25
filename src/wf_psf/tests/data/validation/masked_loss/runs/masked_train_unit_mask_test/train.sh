#!/bin/bash
#SBATCH --mail-user=ezequiel.centofanti@cea.fr
#SBATCH --mail-type=NONE
#SBATCH --job-name=mask_umask   # nom du job
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread          # hyperthreading desactive
#SBATCH --time=20:00:00               # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=mask_psf_umask_test.out   # nom du fichier de sortie
#SBATCH --error=mask_psf_umask_test.err    # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@v100                   # specify the project

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.9.1

# echo launched commands
set -x

srun wavediff -c /gpfswork/rech/ynx/uds36vp/repos/mask_PSF/runs/masked_train_unit_mask_test/configs.yaml -r /gpfswork/rech/ynx/uds36vp/repos/wf-psf -o /gpfswork/rech/ynx/uds36vp/repos/mask_PSF/runs/masked_train_unit_mask_test
