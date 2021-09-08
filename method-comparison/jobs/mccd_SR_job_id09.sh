#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N mccd_SR_id09
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

python ./mccd_script_SR.py \
    --config_file /home/tliaudat/github/wf-psf/method-comparison/config_files/mccd_configs/config_MCCD_SR_wf_exp_id09.ini \
    --repo_base_path /home/tliaudat/github/wf-psf/ \
    --run_id mccd_SR_id09 \
    --psf_out_dim 64 \


# Return exit code
exit 0
