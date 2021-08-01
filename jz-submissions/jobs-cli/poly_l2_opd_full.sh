#!/bin/bash
#SBATCH --job-name=poly_l2_opd_full    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=poly_l2_opd_full%j.out  # nom du fichier de sortie
#SBATCH --error=poly_l2_opd_full%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A xdy@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-2

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.4.1

# echo des commandes lancees
set -x

opt[0]="--id_name poly_l2_opd_2c_ --train_dataset_file train_Euclid_res_200_TrainStars_id_001.npy  --n_epochs_param 30 30 --n_epochs_non_param 250 250 --l_rate_param 0.01 0.001 --l_rate_non_param 0.1 0.01"
opt[1]="--id_name poly_l2_opd_5c_ --train_dataset_file train_Euclid_res_500_TrainStars_id_001.npy  --n_epochs_param 20 20 --n_epochs_non_param 100 120 --l_rate_param 0.01 0.01 --l_rate_non_param 0.1 0.1"
opt[2]="--id_name poly_l2_opd_2k_ --train_dataset_file train_Euclid_res_2000_TrainStars_id_001.npy  --n_epochs_param 20 20 --n_epochs_non_param 150 100 --l_rate_param 0.01 0.001 --l_rate_non_param 0.1 0.01"


cd $WORK/repo/wf-psf/jz-submissions/slurm-logs/

srun python -u ./../../long-runs/l2_param_tr_ev.py \
    --model poly \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --saved_cycle cycle2 \
    --total_cycles 2 \
    --l2_param 1e-11 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \
