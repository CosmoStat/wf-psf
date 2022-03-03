#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobias.liaudat@cea.fr
#PBS -m ea
# Set a name for the job
#PBS -N re_eval_wf_psf
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=95:00:00
# Request number of cores
#PBS -l nodes=n03:ppn=4:hasgpu

# Full path to environment
export SPENV="$HOME/.conda/envs/shapepipe"
export CONFDIR="$HOME/github/aziz_repos/deep_mccd"

# Activate conda environment
module purge
module load tensorflow/2.4
# module load intel/19.0/2
# source activate shapepipe

cd /home/tliaudat/github/wf-psf/debug

# Run scripts
python  candide_helper_eval_plot_script.py

# Return exit code
exit 0
