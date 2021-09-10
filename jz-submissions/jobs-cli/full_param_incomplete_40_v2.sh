#!/bin/bash
#SBATCH --job-name=param_incomplete_40_v2_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=param_incomplete_40_v2_%j.out  # nom du fichier de sortie
#SBATCH --error=param_incomplete_40_v2_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-3

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.4.1

# echo des commandes lancees
set -x

opt[0]="--id_name _incomplete_40_v2_2c --train_dataset_file train_Euclid_res_200_TrainStars_id_001.npy --n_epochs_param 40 40 --l_rate_param 0.005 0.001"
opt[1]="--id_name _incomplete_40_v2_5c --train_dataset_file train_Euclid_res_500_TrainStars_id_001.npy --n_epochs_param 30 30 --l_rate_param 0.005 0.001"
opt[2]="--id_name _incomplete_40_v2_1k --train_dataset_file train_Euclid_res_1000_TrainStars_id_001.npy --n_epochs_param 30 30 --l_rate_param 0.005 0.001"
opt[3]="--id_name _incomplete_40_v2_2k --train_dataset_file train_Euclid_res_2000_TrainStars_id_001.npy --n_epochs_param 20 20 --l_rate_param 0.005 0.001"


cd $WORK/repo/wf-psf/long-runs/

srun python -u ./train_eval_plot_script_click.py \
    --model param \
    --n_zernikes 40 \
    --saved_model_type checkpoint \
    --saved_cycle cycle2 \
    --total_cycles 2 \
    --use_sample_weights False \
    --l2_param 0. \
    --base_id_name _incomplete_40_v2_ \
    --suffix_id_name 2c --suffix_id_name 5c --suffix_id_name 1k --suffix_id_name 2k \
    --star_numbers 200 --star_numbers 500 --star_numbers 1000 --star_numbers 2000 \
    --plots_folder plots/param_incomplete_40_v2/ \
    ${opt[$SLURM_ARRAY_TASK_ID]} \
