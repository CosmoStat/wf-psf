#!/bin/python

import numpy as np
from astropy.io import fits
import rca
import mccd
from wf_psf import method_comp_utils as comp_utils
from wf_psf.metrics import metrics as metrics
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
    default=3,
    type=int,
    help="Upsampling factor.")
@click.option(
    "--ksig",
    default=3.,
    type=float,
    help="Denoising parameter K. Default 3.")
@click.option(
    "--run_id",
    default="rca_SR",
    type=str,
    help="Id used for the saved models and validations.")
@click.option(
    "--psf_out_dim",
    default=64,
    type=int,
    help="Image dimension of the PSF vignet. Like the PSFEx variable (psf_size).")
@click.option(
    "--psf_size",
    default=1.25,
    type=float,
    help="PSF size in the type of the psf_size_type parameter. Default 1.25 (sigma).")
@click.option(
    "--psf_size_type",
    default="sigma",
    type=str,
    help="PSF size type. Default sigma.")
@click.option(
    "--saving_dir",
    default="/n05data/tliaudat/wf_exps/outputs/rca_SR_shifts/",
    type=str,
    help="Path to the saving directory. Should include the directories /models, /validation and /metrics.")
@click.option(
    "--input_data_dir",
    default="/n05data/tliaudat/wf_exps/datasets/rca_shifts/",
    type=str,
    help="Input dataset directory. Should have /train and /test directories.")
@click.option(
    "--repo_base_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/",
    type=str,
    help="Path of the wf-psf repository.")


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

    output_dim = [args['psf_out_dim'], args['psf_out_dim']]

    # Load data
    saving_base_path = args['saving_dir']
    input_base_path = args['input_data_dir']
    model_save_dir_path = saving_base_path + 'models/'
    val_save_dir_path = saving_base_path + 'validation/'
    metrics_save_dir_path = saving_base_path + 'metrics/'

    # Generate GT SR psfs
    test_wf_file_path = args['repo_base_path'] + 'data/coherent_euclid_dataset/test_Euclid_res_id_001.npy'
    GT_predictions, wf_test_pos, _ = metrics.gen_GT_wf_model(
        test_wf_file_path,
        pred_output_Q=1,
        pred_output_dim=args['psf_out_dim']
    )

    # Need to change positions to local coordinates
    loc2glob = mccd.mccd_utils.Loc2Glob_EUCLID_sim()
    ccd_tot = loc2glob.ccd_tot  # 36

    # Need to change positions to local coordinates
    # Shift origin to the global positions
    global_pos = wf_test_pos * (4096 * 6 /1000) - (4096 * 3)
    # Calculate local positions and CCD_n
    local_pos = np.array([
        loc2glob.glob2loc_img_coord(_x_glob, _y_glob) for _x_glob, _y_glob in global_pos
    ])
    # CCD list
    ccd_n_list = local_pos[:,0].astype(int)
    # Local positions
    local_pos = local_pos[:,1:]


    catalog_ids = [200, 500, 1000, 2000]  # [200]
    ccd_tot = 36  # 3

    metrics_list = []

    # Iterate over the catalogs
    for catalog_id in catalog_ids:
        # Create lists for exposure metrics
        psf_interp_list = []
        star_list = []
        psf_SR_list = []

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

            # Extract test catalog
            # Build CCD mask
            ccd_mask = (ccd_n_list == ccd_it)

            # Extract elements from selected CCD
            x_loc = local_pos[ccd_mask, 0].reshape(-1, 1)
            y_loc = local_pos[ccd_mask, 1].reshape(-1, 1)

            ccd_test_pos = np.concatenate((x_loc, y_loc), axis=1)
            ccd_test_stars = GT_predictions[ccd_mask, :, :]


            # Recover test PSFs
            interp_psfs = rca_inst.validation_stars(
                rca.utils.rca_format(ccd_test_stars),
                ccd_test_pos,
                upfact=upfact
            )
            interp_psfs_SR = rca_inst.estimate_psf(ccd_test_pos, upfact=1)

            # Match SR psfs to SR stars
            # Crop to output dim
            interp_psfs_SR = np.array([
                comp_utils.crop_at_center(_psf_SR, output_dim) for _psf_SR in interp_psfs_SR
            ])
            # Match to SR stars
            matched_SR_psfs = comp_utils.match_psfs(
                interp_psfs_SR,
                ccd_test_stars,
                psf_size=1.25 * upfact
            )

            # Save validation PSFs and Ground truth stars
            val_dict = {
                'PSF_VIGNETS': interp_psfs,
                'PSF_VIGNETS_SR': matched_SR_psfs,
                'GT_VIGNET': ccd_test_stars,
                'POS': ccd_test_pos,
            }
            val_save_name = run_id + '_validation-%07d-%02d.npy'%(catalog_id, ccd_it)
            np.save(val_save_dir_path + val_save_name, val_dict, allow_pickle=True)

            # Add images to lists
            psf_interp_list.append(np.copy(interp_psfs))
            psf_SR_list.append(np.copy(matched_SR_psfs))
            star_list.append(np.copy(ccd_test_stars))

        # Calculate RMSE metric on all the CCDs

        # Concatenate all the test stars in one array
        exp_psfs = np.concatenate(psf_SR_list)
        exp_stars = np.concatenate(star_list)
        # Calcualte pixel and shape errors
        metrics_dic = comp_utils.shape_pix_metrics(exp_psfs, exp_stars)
        # Add dictionary to saving list
        metrics_list.append(metrics_dic)


    # Save the metrics
    metrics_dict = {
        'metrics_dics': metrics_list,
        'catalog_ids': catalog_ids
    }
    metrics_save_name = run_id + '_metrics.npy'
    np.save(metrics_save_dir_path + metrics_save_name, metrics_dict, allow_pickle=True)

    print('Good bye!')


if __name__ == "__main__":
  main()
