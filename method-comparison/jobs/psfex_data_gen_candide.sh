#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N psfex_data_gen
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n16:ppn=02

# Activate conda environment
# module load intelpython/3-2020.1
module load tensorflow/2.4
module load intel/19.0/2
source activate new_shapepipe

cd /home/tliaudat/github/wf-psf/

python ./method-comparison/scripts/dataset-conversion-PSFEx.py \
    --base_repo_path /home/tliaudat/github/wf-psf/ \
    --sex_exec_path sex \
    --psfex_saving_path /n05data/tliaudat/wf_exps/datasets/psfex/ \
    --verbose 1 \

# Return exit code
exit 0
