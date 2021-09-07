#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N rca_wf_exp_n4_shifts
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n02:ppn=02

# Activate conda environment
module load intelpython/3-2020.1
module load intel/19.0/2
source activate new_shapepipe

cd /home/tliaudat/github/wf-psf/method-comparison/scripts/

# Just 1 for OMP_NUM_THREADS for this Python script
export OMP_NUM_THREADS=1
# And let the low-level threading use all of the requested cores
export OPENBLAS_NUM_THREADS=$NSLOTS

python ./rca_script.py \
    --run_id rca_shifts_n4_up1_k1 \
    --n_comp 4 \
    --upfact 1 \
    --ksig 1. \
    --saving_dir /n05data/tliaudat/wf_exps/outputs/rca_shifts/ \
    --input_data_dir /n05data/tliaudat/wf_exps/datasets/rca_shifts/ \

# Return exit code
exit 0
