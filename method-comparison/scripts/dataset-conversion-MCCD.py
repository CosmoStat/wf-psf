#!/usr/bin/env python
# coding: utf-8

import numpy as np
import mccd
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--rca_data_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/rca/",
    type=str,
    help="RCA dataset directory path. Should have /test and /train folders inside.")
@click.option(
    "--mccd_saving_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/mccd/",
    type=str,
    help="MCCD dataset directory saving path. Should have /test and /train folders inside.")


def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1)(
            delayed(mccd_dataset_conversion)(**args) for i in range(1)
        )



def mccd_dataset_conversion(**args):
    ### Create MCCD catalogs
    # This should be done after the RCA catalogs have been built
    # as we will be tranforming those into MCCD format.

    loc2glob = mccd.mccd_utils.Loc2Glob_EUCLID_sim()
    euclid_CCD_n = loc2glob.ccd_tot

    base_input_path = args['rca_data_path']
    base_output_path = args['mccd_saving_path']


    # Train dataset
    input_folder_path = base_input_path + 'train/'
    output_path = base_output_path + 'train/'

    mccd.auxiliary_fun.mccd_preprocessing(
        input_folder_path,
        output_path,
        min_n_stars=1,
        file_pattern='train_stars-*-*.fits',
        separator='-',
        CCD_id_filter_list=np.arange(euclid_CCD_n),
        outlier_std_max=100.,
        save_masks=False,
        save_name='train_star_selection',
        save_extension='.fits',
        verbose=True,
        loc2glob=mccd.mccd_utils.Loc2Glob_EUCLID_sim(),
        fits_tb_pos=1)


    # Test dataset
    input_folder_path = base_input_path + 'test/'
    output_path = base_output_path + 'test/'

    mccd.auxiliary_fun.mccd_preprocessing(
        input_folder_path,
        output_path,
        min_n_stars=1,
        file_pattern='test_stars-*-*.fits',
        separator='-',
        CCD_id_filter_list=np.arange(euclid_CCD_n),
        outlier_std_max=100.,
        save_masks=False,
        save_name='test_star_selection',
        save_extension='.fits',
        verbose=True,
        loc2glob=mccd.mccd_utils.Loc2Glob_EUCLID_sim(),
        fits_tb_pos=1)


if __name__ == "__main__":
  main()
