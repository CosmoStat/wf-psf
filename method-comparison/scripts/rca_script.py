#!/bin/python

import numpy as np
from astropy.io import fits
import rca
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--n_comp",
    default=4,
    type=int,
    help="RCA number of eigenPSFs.")
@click.option(
    "--upfact",
    default=1,
    type=int,
    help="Upsampling factor.")
@click.option(
    "--ksig",
    default=3.,
    type=float,
    help="Denoising parameter K.")
@click.option(
    "--run_id",
    default="rca",
    type=str,
    help="Id used for the saved models and validations.")
@click.option(
    "--psf_size",
    default=1.25,
    type=float,
    help="PSF size in the type of the psf_size_type parameter.")
@click.option(
    "--psf_size_type",
    default="sigma",
    type=str,
    help="PSF size type.")
@click.option(
    "--saving_dir",
    default="/n05data/tliaudat/wf_exps/outputs/rca/",
    type=str,
    help="Path to the saving directory. Should include the directories /models, /validation and /metrics.")
@click.option(
    "--input_data_dir",
    default="/n05data/tliaudat/wf_exps/datasets/rca_shifts/",
    type=str,
    help="Input dataset directory. Should have /train and /test directories.")


def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1)(
            delayed(rca_procedure)(**args) for i in range(1)
        )

def rca_procedure(**args):
    # Model parameters
    n_comp = args['n_comp']
    upfact = args['upfact']
    ksig = args['ksig']

    run_id = args['run_id']


    # Load data
    saving_base_path = args['saving_dir']
    input_base_path = args['input_data_dir']
    model_save_dir_path = saving_base_path + 'models/'
    val_save_dir_path = saving_base_path + 'validation/'
    metrics_save_dir_path = saving_base_path + 'metrics/'

    catalog_ids = [200, 500, 1000, 2000]  # [200]
    ccd_tot = 36  # 3

    rmse_list = []
    rel_rmse_list = []


    # Iterate over the catalogs
    for catalog_id in catalog_ids:
        # Create lists for exposure metrics
        psf_interp_list = []
        star_list = []

        # Iterate over the ccds
        for ccd_it in range(ccd_tot):

            ## Training
            # Train data
            train_file_name = 'train/train_stars-%07d-%02d.fits'%(catalog_id, ccd_it)
            train_catalog = fits.open(input_base_path + train_file_name)

            # Prepare input data
            # In RCA format (batch dim at the end)
            obs_stars = rca.utils.rca_format(
                train_catalog[1].data['VIGNET']
            )
            obs_pos = np.array([
                train_catalog[1].data['XWIN_IMAGE'],
                train_catalog[1].data['YWIN_IMAGE']
            ]).T

            # Create RCA instance
            rca_inst = rca.RCA(
                n_comp=n_comp,
                upfact=upfact,
                ksig=ksig,
                verbose=1
            )

            # If we have 2 stars or less we skip the ccd
            if obs_pos.shape[0] <= 2:
                continue

            # fit it to stars
            _, _ = rca_inst.fit(
                obs_stars,
                obs_pos,
                psf_size=args['psf_size'],
                psf_size_type=args['psf_size_type']
            )
            # Save model
            save_path = model_save_dir_path + run_id + '_fitted_model-%07d-%02d.npy'%(catalog_id, ccd_it)
            rca_inst.quicksave(save_path)


            ## Validation
            # Test data
            test_file_name = 'test/test_stars-%07d-%02d.fits'%(catalog_id, ccd_it)
            test_catalog = fits.open(input_base_path + test_file_name)

            test_stars = rca.utils.rca_format(
                test_catalog[1].data['VIGNET']
            )

            # Test positions
            test_pos = np.array([
                test_catalog[1].data['XWIN_IMAGE'],
                test_catalog[1].data['YWIN_IMAGE']
            ]).T

            # Recover test PSFs
            interp_psfs_upfact1 = rca_inst.validation_stars(test_stars, test_pos, upfact=1)
            interp_psfs = rca_inst.validation_stars(test_stars, test_pos, upfact=upfact)

            # Save validation PSFs and Ground truth stars
            val_dict = {
                'PSF_VIGNETS': interp_psfs,
                'PSF_UPFACT1': interp_psfs_upfact1,
                'GT_VIGNET': test_catalog[1].data['GT_VIGNET'],
                'POS': test_pos,
            }
            val_save_name = run_id + '_validation-%07d-%02d.npy'%(catalog_id, ccd_it)
            np.save(val_save_dir_path + val_save_name, val_dict, allow_pickle=True)

            # Add images to lists
            psf_interp_list.append(np.copy(interp_psfs))
            star_list.append(np.copy(test_catalog[1].data['GT_VIGNET']))

        # Calculate RMSE metric on all the CCDs
        exp_psfs = np.concatenate(psf_interp_list)
        exp_stars = np.concatenate(star_list)

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

    # Save the metrics
    metrics_dict = {
        'rmse': rmse_list,
        'rel_rmse': rel_rmse_list
    }
    metrics_save_name = run_id + '_metrics.npy'
    np.save(metrics_save_dir_path + metrics_save_name, metrics_dict, allow_pickle=True)


if __name__ == "__main__":
  main()
