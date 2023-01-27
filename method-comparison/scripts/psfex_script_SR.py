#!/bin/python

import os
import subprocess
import platform
import traceback

import numpy as np
import mccd
from wf_psf import method_comp_utils as comp_utils
from wf_psf.metrics import metrics as metrics
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--psfvar_degrees",
    default=2,
    type=int,
    help="Polynomial degree for each group. PSFEx variable.")
@click.option(
    "--psf_sampling",
    default=.33,
    type=float,
    help="Sampling step in pixel units. PSFEx variable.")
@click.option(
    "--psf_size",
    default=64,
    type=int,
    help="Image size of the PSF model. PSFEx variable.")
@click.option(
    "--run_id",
    default="psfex_SR_model",
    type=str,
    help="Id used for the saved models and validations.")
@click.option(
    "--exec_path",
    default="psfex",
    type=str,
    help="Executable path of psfex.")
@click.option(
    "--repo_base_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/",
    type=str,
    help="Path of the wf-psf repository.")
@click.option(
    "--dataset_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/psfex/",
    type=str,
    help="Path to the directory with the PSFEx datasets. Should include the directory /train.")
@click.option(
    "--saving_dir",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/comparison-PSF-methods/outputs/psfex_SR/",
    type=str,
    help="Path to the saving directory.")
@click.option(
    "--verbose",
    default=1,
    type=int,
    help="Verbose parameter. Bigger than 0 means verbose.")


def main(**args):
    print(args)
    # PSFEx SMP can be controlled directly from its config file
    psfex_SR_procedure(**args)


def psfex_SR_procedure(**args):
    # Python version
    python_var_tuple = platform.python_version_tuple()

    # Model parameters
    run_id = args['run_id']
    psfex_config_path = args['repo_base_path'] + 'method-comparison/config_files/new_default.psfex'
    psfex_exec_path = args['exec_path']

    # Load data
    saving_base_path = args['saving_dir']

    input_train_dir = args['dataset_path'] + 'train/'

    model_save_dir_path = saving_base_path + 'models/'
    val_save_dir_path = saving_base_path + 'validation/'
    metrics_save_dir_path = saving_base_path + 'metrics/'

    if args['psf_sampling'] != 0.33:
        raise NotImplementedError("The psf_sampling should be 0.33, meaning x3 Euclid resolution.")

    test_wf_file_path = args['repo_base_path'] + 'data/coherent_euclid_dataset/test_Euclid_res_id_001.npy'
    GT_predictions, wf_test_pos, _ = metrics.gen_GT_wf_model(
        test_wf_file_path,
        pred_output_Q=1,
        pred_output_dim=args['psf_size']
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


    # Iteration parameters
    catalog_ids = [200, 500, 1000, 2000]  # [200]
    
    metrics_list = []

    # Iterate over the catalogs
    for catalog_id in catalog_ids:
        # Create lists for exposure metrics
        psf_interp_list = []
        star_list = []

        # Iterate over the ccds
        for ccd_it in range(ccd_tot):

            ## Training
            # Train data
            train_file_path = input_train_dir + 'train_stars_psfex-%07d-%02d.fits'%(catalog_id, ccd_it)
            outcat_name = model_save_dir_path + '%s-%07d-%02d.psf'%(args['run_id'], catalog_id, ccd_it)

            # Define psfex command line
            command_line_base = (
                '{0} {1} -c {2} -PSF_DIR {3} -OUTCAT_NAME {4} -PSFVAR_DEGREES {5} -PSF_SAMPLING {6} -PSF_SIZE {7}'.format(
                    psfex_exec_path,
                    train_file_path,
                    psfex_config_path,
                    model_save_dir_path,
                    outcat_name,
                    args['psfvar_degrees'],
                    args['psf_sampling'],
                    '%2d,%2d'%(args['psf_size'], args['psf_size'])
                )
            )

            try:
                # For >=python3.7
                if int(python_var_tuple[0]) == 3 and int(python_var_tuple[1]) >= 7:
                    process_output = subprocess.run(command_line_base, shell=True, check=False, capture_output=True)
                    if args['verbose'] > 0:
                        print('STDERR:\n', process_output.stderr.decode("utf-8"))
                        print('STDOUT:\n', process_output.stdout.decode("utf-8"))
                # For python3.6
                elif int(python_var_tuple[0]) == 3 and int(python_var_tuple[1]) < 7:
                    process_output = subprocess.run(
                        command_line_base,
                        shell=True,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT
                    )
                    if args['verbose'] > 0:
                        print('STDOUT:\n', process_output.stdout.decode("utf-8"))

                # Extract test catalog
                # Build CCD mask
                ccd_mask = (ccd_n_list == ccd_it)

                # Extract elements from selected CCD
                x_loc = local_pos[ccd_mask, 0].reshape(-1, 1)
                y_loc = local_pos[ccd_mask, 1].reshape(-1, 1)

                ccd_test_pos = np.concatenate((x_loc, y_loc), axis=1)
                ccd_test_stars = GT_predictions[ccd_mask, :, :]

                # PSFEx default model name
                psf_model_path = model_save_dir_path + 'train_stars_psfex-%07d-%02d.psf'%(catalog_id, ccd_it)

                # Interpolate PSFs
                interp_psfs = comp_utils.interpsfex(psf_model_path, ccd_test_pos)

                # Match PSFs, flux and intrapixel shifts
                matched_psfs = comp_utils.match_psfs(
                    interp_psfs,
                    test_stars=ccd_test_stars,
                    psf_size=1.25 / args['psf_sampling']
                )

                # Save validation PSFs and Ground truth stars
                val_dict = {
                    'PSF_VIGNETS': matched_psfs,
                    'GT_VIGNET': ccd_test_stars,
                    'POS': ccd_test_pos,
                }

                val_save_name = run_id + '_validation-%07d-%02d.npy'%(catalog_id, ccd_it)
                np.save(val_save_dir_path + val_save_name, val_dict, allow_pickle=True)

                # Add images to lists
                psf_interp_list.append(np.copy(matched_psfs))
                star_list.append(np.copy(ccd_test_stars))
            except Exception:
                traceback.print_exc()
                print('Problem with catalog: train_stars_psfex-%07d-%02d.psf'%(catalog_id, ccd_it))


        # Concatenate all the test stars in one array
        exp_psfs = np.concatenate(psf_interp_list)
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

