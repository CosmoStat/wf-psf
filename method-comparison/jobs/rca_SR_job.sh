#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N rca_SR
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n03:ppn=04:hasgpu

# Activate conda environment
# module load intelpython/3-2020.1
module load tensorflow/2.4
module load intel/19.0/2
source activate new_shapepipe

cd /home/tliaudat/github/wf-psf/method-comparison/scripts/

# Just 1 for OMP_NUM_THREADS for this Python script
export OMP_NUM_THREADS=1
# And let the low-level threading use all of the requested cores
export OPENBLAS_NUM_THREADS=$NSLOTS

# n8
python ./rca_script_SR.py \
    --saving_dir /n05data/tliaudat/wf_exps/outputs/rca_shifts/ \
    --input_data_dir /n05data/tliaudat/wf_exps/datasets/rca_shifts/ \
    --repo_base_path /home/tliaudat/github/wf-psf/ \
    --run_id rca_SR_shifts_n8_up3_k3 \
    --n_comp 8 \
    --upfact 3 \
    --ksig 3. \
    --psf_out_dim 64 \

# n12
python ./rca_script_SR.py \
    --saving_dir /n05data/tliaudat/wf_exps/outputs/rca_shifts/ \
    --input_data_dir /n05data/tliaudat/wf_exps/datasets/rca_shifts/ \
    --repo_base_path /home/tliaudat/github/wf-psf/ \
    --run_id rca_SR_shifts_n12_up3_k3 \
    --n_comp 12 \
    --upfact 3 \
    --ksig 3. \
    --psf_out_dim 64 \

# n16
python ./rca_script_SR.py \
    --saving_dir /n05data/tliaudat/wf_exps/outputs/rca_shifts/ \
    --input_data_dir /n05data/tliaudat/wf_exps/datasets/rca_shifts/ \
    --repo_base_path /home/tliaudat/github/wf-psf/ \
    --run_id rca_SR_shifts_n16_up3_k3 \
    --n_comp 16 \
    --upfact 3 \
    --ksig 3. \
    --psf_out_dim 64 \

# Return exit code
exit 0
