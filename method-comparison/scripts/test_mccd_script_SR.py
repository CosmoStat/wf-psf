#!/bin/python

import traceback
import numpy as np
import mccd
from wf_psf import method_comp_utils as comp_utils
from wf_psf.metrics import metrics as metrics
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--config_file",
    default="/home/tliaudat/github/mccd_develop/mccd/wf_exps/config_files/config_MCCD_wf_exp.ini",
    type=str,
    help="MCCD configuration file path.")
@click.option(
    "--psf_out_dim",
    default=64,
    type=int,
    help="Image dimension of the PSF vignet. Like the PSFEx variable (psf_size).")
@click.option(
    "--run_id",
    default="mccd_SR",
    type=str,
    help="Id used for the saved SR validations and metrics.")
@click.option(
    "--repo_base_path",
    default="/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/",
    type=str,
    help="Path of the wf-psf repository.")


def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1)(
            delayed(mccd_procedure)(**args) for i in range(1)
        )

def mccd_procedure(**args):

    output_dim = [args['psf_out_dim'], args['psf_out_dim']]
    # Validation upfact
    upfact = 1
    run_id = args['run_id']

    # Generate instance of MCCD runner
    run_mccd = mccd.auxiliary_fun.RunMCCD(
        args['config_file'],
        fits_table_pos=1,
        verbose=True
    )
    # Train/fit the models
    run_mccd.parse_config_file()
    run_mccd.preprocess_inputs()
    # run_mccd.fit_MCCD_models()
    run_mccd.preprocess_val_inputs()

    # Prepare validation

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


    # Fit model input dir
    fit_model_input_dir = run_mccd.param_parser.get_extra_kw(
        'val_model_input_dir'
    )
    # Validation output dir
    val_output_dir = run_mccd.param_parser.get_extra_kw(
        'val_model_output_dir'
    )

    catalog_ids = [200, 500, 1000, 2000]
    metrics_list = []

    for catalog_id in catalog_ids:

        # Fitted model path
        fit_model_path = fit_model_input_dir + \
            run_mccd.fitting_model_saving_name + run_mccd.separator + \
            '%07d'%catalog_id + '.npy'

        mccd_inst = mccd.mccd_quickload(fit_model_path)
        fit_upfact = mccd_inst.upfact

        # Create lists for exposure metrics
        star_list = []
        psf_SR_list = []

        # Iterate over the ccds
        for ccd_it in range(ccd_tot):
            try:
                # Extract test catalog
                # Build CCD mask
                ccd_mask = (ccd_n_list == ccd_it)

                # Extract elements from selected CCD
                x_loc = local_pos[ccd_mask, 0].reshape(-1, 1)
                y_loc = local_pos[ccd_mask, 1].reshape(-1, 1)

                ccd_test_pos = np.concatenate((x_loc, y_loc), axis=1)
                ccd_test_stars = GT_predictions[ccd_mask, :, :]

                # Recover the PSFs from the model
                interp_psfs_SR = run_mccd.recover_MCCD_PSFs(
                    fit_model_path,
                    positions=ccd_test_pos,
                    ccd_id=ccd_it,
                    local_pos=True,
                    upfact=upfact
                )

                # Match SR psfs to SR stars
                # Crop to output dim
                interp_psfs_SR = np.array([
                    comp_utils.crop_at_center(_psf_SR, output_dim) for _psf_SR in interp_psfs_SR
                ])

                # Match to SR stars
                matched_SR_psfs = comp_utils.match_psfs(
                    interp_psfs_SR,
                    ccd_test_stars,
                    psf_size=1.25 * (fit_upfact / upfact)
                )

                # Save validation PSFs and Ground truth stars
                val_dict = {
                    'PSF_VIGNETS_SR': matched_SR_psfs,
                    'GT_VIGNET': ccd_test_stars,
                    'POS_LOCAL': ccd_test_pos,
                    'CCD_ID': np.ones(ccd_test_pos.shape[0]) * ccd_it,
                }
                val_save_name = run_id + '_validation-%07d-%02d.npy'%(catalog_id, ccd_it)
                np.save(val_output_dir + val_save_name, val_dict, allow_pickle=True)

                # Add images to lists
                psf_SR_list.append(np.copy(matched_SR_psfs))
                star_list.append(np.copy(ccd_test_stars))

            except Exception:
                traceback.print_exc()
                print('Problem with catalog: train_stars_psfex-%07d-%02d.psf'%(catalog_id, ccd_it))

        
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
    np.save(val_output_dir + metrics_save_name, metrics_dict, allow_pickle=True)

    print('Good bye!')

if __name__ == "__main__":
  main()
