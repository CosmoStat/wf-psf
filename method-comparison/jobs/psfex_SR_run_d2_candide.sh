#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N psfex_SR_run_d2_shifts
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n03:ppn=02:hasgpu

# Activate conda environment
# module load intelpython/3-2020.1
module load tensorflow/2.4
module load intel/19.0/2
source activate new_shapepipe

cd /home/tliaudat/github/wf-psf/

python ./method-comparison/scripts/psfex_script_SR.py \
    --repo_base_path /home/tliaudat/github/wf-psf/ \
    --saving_dir /n05data/tliaudat/wf_exps/outputs/psfex_SR_d2_shifts/ \
    --dataset_path /n05data/tliaudat/wf_exps/datasets/psfex_shifts/ \
    --psfvar_degrees 2 \
    --psf_sampling 0.33 \
    --psf_size 64 \
    --run_id psfex_SR_run_d2 \
    --exec_path psfex \
    --verbose 1 \

# Return exit code
exit 0
