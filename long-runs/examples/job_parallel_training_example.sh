#!/bin/bash
#SBATCH --job-name=5_cycles_256      # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=5_cycles_256%j.out  # nom du fichier de sortie
#SBATCH --error=5_cycles_256%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --array=0-3

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.7.0

# echo des commandes lancees
set -x

# n_bins ---> number of points per SED (n_bins + 1)
opt[0]="--id_name _5_cycles_256_proj_d2_45z --project_dd_features True --d_max 2 --n_zernikes 45"
opt[1]="--id_name _5_cycles_256_no_proj_d2_45z --project_dd_features False --d_max 2 --n_zernikes 45"
opt[2]="--id_name _5_cycles_256_proj_d5_45z --project_dd_features True --d_max 5 --n_zernikes 45"
opt[3]="--id_name _5_cycles_256_proj_d2_60z --project_dd_features True --d_max 2 --n_zernikes 60"


cd $WORK/repos/wf-SEDs/HD_projected_optimisation/scripts/

srun python -u ./train_project_click_multi_cycle.py \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --model poly \
    --base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/HD_projected_optimisation/wf-outputs/ \
    --log_folder log-files/ \
    --model_folder chkp/8_bins/ \
    --optim_hist_folder optim-hist/ \
    --chkp_save_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/HD_projected_optimisation/wf-outputs/chkp/8_bins/ \
    --plots_folder plots/ \
    --dataset_folder /gpfswork/rech/ynx/uds36vp/datasets/interp_SEDs/ \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_009_8_bins_sigma_0.npy \
    --test_dataset_file test_Euclid_res_id_009_8_bins.npy \
    --pupil_diameter 256 \
    --n_bins_lda 8 \
    --output_ 3. \
    --oversampling_rate 3. \
    --output_dim 32 \
    --d_max_nonparam 5 \
    --use_sample_weights True \
    --interpolation_type none \
    --l_rate_param_multi_cycle "0.01 0.0085 0.007 0.0055 0.004" \
    --l_rate_non_param_multi_cycle "0.1 0.09 0.08 0.07 0.06" \
    --n_epochs_param_multi_cycle "10 5 5 5 15" \
    --n_epochs_non_param_multi_cycle "25 25 25 25 50" \
    --save_all_cycles True \
    --total_cycles 5 \
    --cycle_def complete \
    --model_eval poly \
    --metric_base_path /gpfswork/rech/ynx/uds36vp/repos/wf-SEDs/HD_projected_optimisation/wf-outputs/metrics/ \
    --saved_model_type checkpoint \
    --saved_cycle cycle5 \
    --eval_batch_size 16 \
    --n_bins_gt 8 \
    --opt_stars_rel_pix_rmse True \
    --l2_param 0. \
    --base_id_name _5_cycles_256_ \
    --suffix_id_name proj_d2_45z --suffix_id_name no_proj_d2_45z --suffix_id_name proj_d5_45z --suffix_id_name proj_d2_60z \
    --star_numbers 1 --star_numbers 2 --star_numbers 3 --star_numbers 4 \
    ${opt[$SLURM_ARRAY_TASK_ID]} \

## --star_numbers is for the final plot's x-axis. It does not always represents the number of stars. 
## --id_name = --base_id_name + --suffix_id_name