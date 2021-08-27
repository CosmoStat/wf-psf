#!/bin/python

import mccd
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--config_file",
    default="/home/tliaudat/github/mccd_develop/mccd/wf_exps/config_files/config_MCCD_wf_exp.ini",
    type=str,
    help="MCCD configuration file path.")


def mccd_procedure(**args):
    # Run MCCD
    run_mccd = mccd.auxiliary_fun.RunMCCD(
        args['config_file'],
        fits_table_pos=1,
        verbose=True
    )
    run_mccd.run_MCCD()
    # run_mccd.validate_MCCD_models()

    # Calculate metrics
    val_output_dir = run_mccd.param_parser.get_extra_kw(
        'val_model_output_dir'
    )

    rmse_list = []
    rel_rmse_list = []
    catalogs = [200, 500, 1000, 2000]

    for it in range(len(catalogs)):
        
        # Load validation data
        file_name = 'validation_psf-%07d.fits'%(catalogs[it])
        mccd_val_dic = fits.open(val_output_dir + file_name)

        exp_psfs = mccd_val_dic[1].data['PSF_VIGNET_LIST']
        exp_stars = mccd_val_dic[1].data['VIGNET_LIST']
        
        # Calculate residuals
        residuals = np.sqrt(np.mean((exp_psfs - exp_stars)**2, axis=(1,2)))
        GT_star_mean = np.sqrt(np.mean((exp_stars)**2, axis=(1,2)))

        # RMSE calculations
        rmse = np.mean(residuals)
        rel_rmse = 100. * np.mean(residuals/GT_star_mean)
        
        rmse_list.append(rmse)
        rel_rmse_list.append(rel_rmse)

    rmse_list = np.array(rmse_list)
    rel_rmse_list = np.array(rel_rmse_list)

    metrics_dict = {
        'rmse': rmse_list,
        'rel_rmse': rel_rmse_list
    }
    metrics_save_name = 'mccd_metrics.npy'
    np.save(val_output_dir + metrics_save_name, metrics_dict, allow_pickle=True)


def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1)(
            delayed(mccd_procedure)(**args) for i in range(1)
        )

if __name__ == "__main__":
  main()
