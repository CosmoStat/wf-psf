#!/bin/bash
#SBATCH --job-name=PR_plot_all_wfv2_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=PR_plot_all_wfv2_%j.out  # nom du fichier de sortie
#SBATCH --error=PR_plot_all_wfv2_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-use=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-3

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.9.1

# echo des commandes lancees
set -x


wavediff -c /gpfswork/rech/ynx/ulx23va/wfv2/repos/phase-retrieval-PR/wf-psf/dev/config/PR_config/PR_plotting_config_all.yaml -r /gpfswork/rech/ynx/ulx23va/wfv2/repos/phase-retrieval-PR/wf-psf -o /gpfsscratch/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v2

