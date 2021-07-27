#!/bin/bash
#SBATCH --job-name=2c_poly_tr_ev_click      # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=2c_poly_tr_ev_click%j.out  # nom du fichier de sortie
#SBATCH --error=2c_poly_tr_ev_click%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A xdy@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.4.1

# echo des commandes lancees
set -x

cd $WORK/repo/wf-psf/jz-submissions/slurm-logs/

srun python ./../scripts/train_eval_poly_200_click.py --n_epochs_param 2 2 --n_epochs_non_param 2 2 --id_name _test-coherent_euclid_200stars
