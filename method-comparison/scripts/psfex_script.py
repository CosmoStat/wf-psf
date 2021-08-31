#!/bin/python

import os
import subprocess
import platform
import traceback
import numpy as np
from astropy.io import fits
import mccd
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--config_file",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/config_files/new_default.psfex",
    type=str,
    help="PSFEx configuration file path.")
@click.option(
    "--run_id",
    default="psfex_model",
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
    "--saving_dir",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/comparison-PSF-methods/outputs/psfex/",
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
    psfex_procedure(**args)

def interpsfex(dotpsfpath, pos):
    """Use PSFEx generated model to perform spatial PSF interpolation.
        Parameters
        ----------
        dotpsfpath : string
            Path to .psf file (PSFEx output).
        pos : np.ndaray
            Positions where the PSF model should be evaluated.
        Returns
        -------
        PSFs : np.ndarray
            Each row is the PSF imagette at the corresponding asked position.
    """

    if not os.path.exists(dotpsfpath):
        return None

    # read PSF model and extract basis and polynomial degree and scale position
    PSF_model = fits.open(dotpsfpath)[1]

    PSF_basis = np.array(PSF_model.data)[0][0]
    try:
        deg = PSF_model.header['POLDEG1']
    except KeyError:
        # constant PSF model
        return PSF_basis[0, :, :]

    # scale coordinates
    x_interp, x_scale = (PSF_model.header['POLZERO1'],
                         PSF_model.header['POLSCAL1'])
    y_interp, y_scale = (PSF_model.header['POLZERO2'],
                         PSF_model.header['POLSCAL2'])
    xs, ys = (pos[:, 0] - x_interp) / x_scale, (pos[:, 1] - y_interp) / y_scale

    # compute polynomial coefficients
    coeffs = np.array([[x**i for i in range(deg+1)] for x in xs])
    cross_coeffs = np.array([np.concatenate([[(x ** j) * (y ** i)
                                              for j in range(deg - i + 1)]
                                             for i in range(1, deg + 1)])
                             for x, y in zip(xs, ys)])
    coeffs = np.hstack((coeffs, cross_coeffs))

    # compute interpolated PSF
    PSFs = np.array([np.sum([coeff * atom for coeff, atom in
                     zip(coeffs_posi, PSF_basis)], axis=0)
                     for coeffs_posi in coeffs])

    return PSFs

def validation_stars(dotpsfpath, test_stars, test_pos, psf_size=1.25):
    """ Match PSF model to stars - in flux, shift and pixel sampling - for validation tests.
    Returns both the matched PSFs' stamps.

    Parameters
    ----------
    dotpsfpath: str
        Path to the .psf model output of PSFEx.
    test_stars: np.ndarray
        Star stamps to be used for comparison with the PSF model.
    test_pos: np.ndarray
        Their corresponding positions.
    psf_size: float
        PSF size in sigma format.
    """
    
    sigmas = np.ones((test_pos.shape[0],)) * psf_size
        
    cents = [
        mccd.utils.CentroidEstimator(test_stars[it, :, :], sig=sigmas[it])
        for it in range(test_stars.shape[0])
    ]
    # Calculate shifts
    test_shifts = np.array([ce.return_shifts() for ce in cents])
    # Calculate fluxes
    test_fluxes = mccd.utils.flux_estimate_stack(mccd.utils.rca_format(test_stars), sigmas=sigmas)


    # Interpolate PSFs
    interp_psfs = interpsfex(dotpsfpath, test_pos)
    
    # Estimate shift kernels
    lanc_rad = np.ceil(3. * np.max(sigmas)).astype(int)
    shift_kernels, _ = mccd.utils.shift_ker_stack(
        test_shifts,
        upfact=1,
        lanc_rad=lanc_rad
    )
    
    # Degrade PSFs
    interp_psfs = np.array([
        mccd.utils.degradation_op(
            interp_psfs[j, :, :],
            shift_kernels[:, :, j],
            D=1
        )
        for j in range(test_pos.shape[0])
    ])

    # Optimised fulx matching
    norm_factor = np.array([
        np.sum(_star * _psf) / np.sum(_psf * _psf)
        for _star, _psf in zip(test_stars, interp_psfs)
    ]).reshape(-1, 1, 1)
    interp_psfs *= norm_factor
            
    return interp_psfs


def psfex_procedure(**args):
    # Python version
    python_var_tuple = platform.python_version_tuple()

    # Model parameters
    run_id = args['run_id']
    psfex_config_path = args['config_file']
    psfex_exec_path = args['exec_path']

    # Load data
    saving_base_path = args['saving_dir']

    input_train_dir = args['repo_base_path'] + 'method-comparison/compatible-datasets/psfex/train/'
    input_test_dir = args['repo_base_path'] + 'method-comparison/compatible-datasets/psfex/test/'

    model_save_dir_path = saving_base_path + 'models/'
    val_save_dir_path = saving_base_path + 'validation/'
    metrics_save_dir_path = saving_base_path + 'metrics/'

    # Need to change positions to local coordinates
    loc2glob = mccd.mccd_utils.Loc2Glob_EUCLID_sim()

    catalog_ids = [200, 500, 1000, 2000]  # [200]
    ccd_tot = loc2glob.ccd_tot  # 36

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
            train_file_path = input_train_dir + 'train_stars_psfex-%07d-%02d.fits'%(catalog_id, ccd_it)
            outcat_name = model_save_dir_path + '%s-%07d-%02d.psf'%(args['run_id'], catalog_id, ccd_it)

            # Define psfex command line
            command_line_base = (
                '{0} {1} -c {2} -PSF_DIR {3} -OUTCAT_NAME {4}'.format(
                    psfex_exec_path,
                    train_file_path,
                    psfex_config_path,
                    model_save_dir_path,
                    outcat_name
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

                # PSFEx default model name
                psf_model_path = model_save_dir_path + 'train_stars_psfex-%07d-%02d.psf'%(catalog_id, ccd_it)

                # Test catalog
                test_file_path = input_test_dir + 'test_stars_psfex-%07d-%02d.fits'%(0, ccd_it)
                test_catalog = fits.open(test_file_path)

                test_stars = test_catalog[2].data['VIGNET']
                test_pos = np.array([
                    test_catalog[2].data['XWIN_IMAGE'],
                    test_catalog[2].data['YWIN_IMAGE']
                ]).T

                matched_psfs = validation_stars(
                    psf_model_path,
                    test_stars=test_stars,
                    test_pos=test_pos
                )


                # Save validation PSFs and Ground truth stars
                val_dict = {
                    'PSF_VIGNETS': matched_psfs,
                    'GT_VIGNET': test_stars,
                    'POS': test_pos,
                }

                val_save_name = run_id + '_validation-%07d-%02d.npy'%(catalog_id, ccd_it)
                np.save(val_save_dir_path + val_save_name, val_dict, allow_pickle=True)

                # Add images to lists
                psf_interp_list.append(np.copy(matched_psfs))
                star_list.append(np.copy(test_stars))
            except Exception:
                traceback.print_exc()
                print('Problem with catalog: train_stars_psfex-%07d-%02d.psf'%(catalog_id, ccd_it))

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

    print('Good bye!')


if __name__ == "__main__":
  main()

