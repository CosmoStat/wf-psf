#!/bin/bash
#SBATCH --mail-user=jennifer.pollack@cea.fr
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=psf_phase_ret         # nom du job
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
##SBATCH -C v100-32g
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:35:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=psf_test-phase-train-metrics.out       # nom du fichier de sortie
#SBATCH --error=psf_test-phase-train-metrics.err        # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@v100                   # specify the project

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.11.0
export CUDA_DIR=$CUDA_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# echo launched commands
set -x

cd $WORK/wf-psf

srun wavediff -c ${WORK}/my_configs/config/configs_types/phase_retrieval/training_metrics/configs.yaml -r ${WORK}/wf-psf -o ${WORK}/my_configs/config/configs_types/phase_retrieval/training_metrics
