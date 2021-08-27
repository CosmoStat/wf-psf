#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################

# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N rca_wf_exp_n16
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=99:00:00
# Request number of cores
#PBS -l nodes=n03:ppn=02

# Activate conda environment
module load intelpython/3-2020.1
module load intel/19.0/2
source activate new_shapepipe

cd /home/tliaudat/github/mccd_develop/mccd/wf_exps/log_files/

python ./../scripts/rca_script.py \
    --run_id true_rca_n16_up1_k3 \
    --n_comp 16 \
    --upfact 1 \
    --ksig 3. \

# Return exit code
exit 0
