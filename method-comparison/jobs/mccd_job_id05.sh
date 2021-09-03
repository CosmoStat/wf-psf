#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N mccd_wf_exp_id05
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n01:ppn=02

# Activate conda environment
module load intelpython/3-2020.1
module load intel/19.0/2
source activate new_shapepipe

cd /home/tliaudat/github/wf-psf/method-comparison/scripts/

python ./mccd_script.py \
    --config_file /home/tliaudat/github/wf-psf/method-comparison/config_files/mccd_configs/config_MCCD_wf_exp_id05.ini


# Return exit code
exit 0
