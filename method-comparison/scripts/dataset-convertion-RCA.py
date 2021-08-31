#!/usr/bin/env python
# coding: utf-8

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


### Create RCA catalogs

# Fits required table names
# 'VIGNET' : observed vignets
# 'XWIN_IMAGE' :  X local coordinate
# 'YWIN_IMAGE' :  Y local coordinte

# The following is optional (optional)
# 'SNR_WIN' : SNR estimation 
# 'XWIN_WORLD' : X WCS global coordinate (RA)
# 'XWIN_WORLD' : Y WCS global coordinate (DEC)

# Name example:
# An filename example would be: ``star_selection-1234567-04.fits``
#     Where the exposure ID is: ``1234567``
#     Where the CCD ID is: ``04`` (using 2 digits)


def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1)(
            delayed(rca_dataset_conversion)(**args) for i in range(1)
        )



def rca_dataset_conversion(**args):

    # Loading dir
    dataset_path = args['base_repo_path'] + 'data/coherent_euclid_dataset/'

    # Saving parameters
    rca_base_path = args['rca_saving_path']

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
            if save_name == 'train_stars':
                noisy_train_stars = dataset['noisy_stars']
            data_pos = dataset['positions']

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

                if save_name == 'train_stars':
                    obs_stars = noisy_train_stars[ccd_mask, :, :]
                elif save_name == 'test_stars':
                    obs_stars = GT_stars

                saving_dic = {
                    'VIGNET': obs_stars,
                    'GT_VIGNET': GT_stars,
                    'XWIN_IMAGE': x_local,
                    'YWIN_IMAGE': y_local,   
                }

                ccd_str = '%02d'%ccd_it
                catalog_str = '%07d'%catalog_id

                saving_path = rca_base_path + folder_path + save_name + separator \
                                + catalog_str + separator + ccd_str + save_extension

                mccd.mccd_utils.save_to_fits(saving_dic, saving_path)


if __name__ == "__main__":
  main()
