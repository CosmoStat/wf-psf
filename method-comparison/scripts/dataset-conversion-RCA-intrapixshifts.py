#!/usr/bin/env python
# coding: utf-8
"""
## Create RCA catalogs with intra pixel shifts

This script generates RCA-compatible datasets with random
intra-pixel shifts. The inputs are the wf_psf coherent datasets.

The intra pixel is added in this script.

Fits required table names
'VIGNET' : observed vignets
'XWIN_IMAGE' :  X local coordinate
'YWIN_IMAGE' :  Y local coordinte

The following is optional (optional)
'SNR_WIN' : SNR estimation 
'XWIN_WORLD' : X WCS global coordinate (RA)
'XWIN_WORLD' : Y WCS global coordinate (DEC)

Name example:
An filename example would be: ``star_selection-1234567-04.fits``
    Where the exposure ID is: ``1234567``
    Where the CCD ID is: ``04`` (using 2 digits)
"""

import numpy as np
import mccd
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--base_repo_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/",
    type=str,
    help="Base wf-psf repository path.")
@click.option(
    "--rca_saving_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/rca/",
    type=str,
    help="RCA dataset saving path. Should have /test and /train folders inside.")
@click.option(
    "--psf_size",
    default=1.25,
    type=float,
    help="PSF/star size in sigma.")
@click.option(
    "--shift_rand_seed",
    default=10,
    type=int,
    help="Seed for the random shifts.")


def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1)(
            delayed(rca_dataset_conversion)(**args) for i in range(1)
        )



def rca_dataset_conversion(**args):

    # Seed for random generator
    seed = args['shift_rand_seed']
    np.random.seed(seed)

    # Loading dir
    dataset_path = args['base_repo_path'] + 'data/coherent_euclid_dataset/'

    # Saving parameters
    rca_base_path = args['rca_saving_path']

    psf_size = args['psf_size']

    separator = '-'
    save_extension = '.fits'


    catalog_ids = [200, 500, 1000, 2000]
    train_cat = [True, False]

    for catalog_id in catalog_ids:
        for train_bool in train_cat:

            if train_bool:
                save_name = 'train_stars'
                folder_path = 'train/' 
                load_file = 'train_Euclid_res_%d_TrainStars_id_001.npy'%(catalog_id)
            else:
                save_name = 'test_stars'
                folder_path = 'test/'
                load_file = 'test_Euclid_res_id_001.npy'

            # Load dataset
            dataset = np.load(dataset_path + load_file, allow_pickle=True)[()]
            data_stars = dataset['stars']
            data_pos = dataset['positions']
            if save_name == 'train_stars':
                noisy_train_stars = dataset['noisy_stars']
                noise_realisations = noisy_train_stars - data_stars
            
            # Let's add intra-pixel shifts
            # First we calculate the shift needed to center the stars to the 
            # postage stamp centroid
            cents = [mccd.utils.CentroidEstimator(star, sig=psf_size) for star in data_stars]
            req_shifts = np.array([ce.return_shifts() for ce in cents])

            # This is required as we shift the star to the postage stamp centroid
            # and not the other way around
            req_shifts *= -1

            shift_kernels, _ = mccd.utils.shift_ker_stack(
                shifts=req_shifts,
                upfact=1,
                lanc_rad=4
            )
            shift_kernels = mccd.utils.reg_format(shift_kernels)
            shifted_stars = np.array([
                mccd.utils.degradation_op(star, ker, D=1)
                for star, ker in zip(data_stars, shift_kernels)
            ])
            # Now we have the centred stars, we can proced to add a random intra-pixel shift.
            # We do it in two step to avoid having a possible large shift

            # We first calculate random shift uniformly distributed in [-0.4, 0.4]
            rand_shifts = (np.random.rand(data_stars.shape[0], 2) - 0.5) * 0.8
            # We generate the shift kernels
            shift_kernels, _ = mccd.utils.shift_ker_stack(
                shifts=rand_shifts,
                upfact=1,
                lanc_rad=4
            )
            # Change to regular format (first dimension is the batch)
            shift_kernels = mccd.utils.reg_format(shift_kernels)
            # Shift the noiseless images and add the noise realisation
            shifted_stars = np.array([
                mccd.utils.degradation_op(star, ker, D=1)
                for star, ker in zip(shifted_stars, shift_kernels)
            ])
            # If there are the training stars we add the noise realisation
            if save_name == 'train_stars':
                shifted_stars += noise_realisations

            # Convert positions and calculate CCD_n
            # Need to switch these coordinates to the global coordinates
            # Each CCD is 4096x4096 and the origin is bottom left corner of the CCD 15.
            loc2glob = mccd.mccd_utils.Loc2Glob_EUCLID_sim()
            euclid_CCD_n = loc2glob.ccd_tot

            # Shift origin to the global positions
            global_pos = data_pos * (4096*6 /1000) - (4096 * 3)
            # Calculate local positions and CCD_n
            local_pos = np.array([
                loc2glob.glob2loc_img_coord(_x_glob, _y_glob) for _x_glob, _y_glob in global_pos
            ])

            ccd_n_list = local_pos[:,0].astype(int)
            local_pos = local_pos[:,1:]

            # Save the local CCDs on fits catalogs
            for it in range(euclid_CCD_n):
                # Select the CCD
                ccd_it = it

                # Build CCD mask
                ccd_mask = (ccd_n_list == ccd_it)
                
                # Extract elements from selected CCD
                x_local = local_pos[ccd_mask, 0]
                y_local = local_pos[ccd_mask, 1]
                GT_stars = data_stars[ccd_mask, :, :]
                obs_stars = shifted_stars[ccd_mask, :, :]
                ccd_rand_shifts = rand_shifts[ccd_mask, :]

                saving_dic = {
                    'VIGNET': obs_stars,
                    'GT_VIGNET': GT_stars,
                    'XWIN_IMAGE': x_local,
                    'YWIN_IMAGE': y_local,
                    'SHIFTS': ccd_rand_shifts,
                }

                ccd_str = '%02d'%ccd_it
                catalog_str = '%07d'%catalog_id

                saving_path = rca_base_path + folder_path + save_name + separator \
                                + catalog_str + separator + ccd_str + save_extension

                mccd.mccd_utils.save_to_fits(saving_dic, saving_path)


if __name__ == "__main__":
  main()
