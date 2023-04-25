"""Metrics.

A module which defines the classes and methods
to manage metrics evaluation of the trained psf model.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
from wf_psf.utils.read_config import read_conf
import wf_psf.data.training_preprocessing as training_preprocessing
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
import os
import logging
import wf_psf.utils.io as io

logger = logging.getLogger(__name__)


class Saved_Model_Type:
    # Define saved model to use

    #ids = ('checkpoint', )
    def __init__(self, metrics_params, model_dir, id_name="-coherent_euclid_200stars"):
        self.run_id_name = model_dir + '/' + metrics_params.model + id_name
        self.weights_paths = model_dir + '/' + metrics_params.chkp_save_path + 'chkp_callback_' + self.run_id_name + '_' + metrics_params.saved_training_cycle

    #ids = ('final', ) 
    def __init__(self, metrics_params, model_dir, id_name="-coherent_euclid_200stars"):
        self.run_id_name = model_dir + '/' + metrics_params.model + id_name
        self.weights_paths = model_dir + '/' + metrics_params.chkp_save_path + 'chkp_' + self.run_id_name + '_' + metrics_params.saved_training_cycle

    def _get_psf_model(self, metrics_params, model_dir):
        self.weights_paths = model_dir + '/' + metrics_params.chkp_save_path
        return self.weights_paths

    def _get_psf_model_weights(self, metrics_params, model_dir):
        self.run_id_name = model_dir + '/' + metrics_params.model + id_name
        self.weights_paths = model_dir + '/' + metrics_params.chkp_save_path + 'chkp_callback_' + self.run_id_name + '_' + metrics_params.saved_training_cycle
        return self.weights_paths


class MetricsParamsHandler:
    """Metrics Parameters Handler.
    A class to handle metrics parameters accessed:
    Parameters
    ----------
    metrics_params: Recursive Namespace object
        Recursive Namespace object containing metrics input parameters
    id_name: str
        ID name
    output_dirs: FileIOHandler
        FileIOHandler instance
    """

    def __init__(
        self, metrics_params, output_dirs, id_name="-coherent_euclid_200stars"
    ):
        self.metrics_params = metrics_params
        self.id_name = id_name
        self.run_id_name = self.model_name + self.id_name


def evaluate(metrics_params):
    """Evaluate Metrics.
    
    Input
    -----
    metrics_params
    """
    pass

def evaluate_model(training_data, test_data, psf_model, weights_path):
    r""" Evaluate the trained model.
    
    For parameters check the training script click help.
    """
    # Start measuring elapsed time
    starting_time = time.time()

    try:
        ## Load datasets
         # -----------------------------------------------------
        # Get training data
        logger.info(f"Fetching and preprocessing training and test data...")
        #training_data = metrics_handler._get_training_data()

        #test_data = metrics_handler._get_test_data()


        #train_dataset = np.load(
        #    args['dataset_folder'] + args['train_dataset_file'], allow_pickle=True
        #)[()]
        #train_SEDs = train_dataset['SEDs']
      #  train_C_poly = train_dataset['C_poly']
      #  train_parameters = train_dataset['parameters']

      #  test_dataset = np.load(
      #      args['dataset_folder'] + args['test_dataset_file'], allow_pickle=True
      #  )[()]   
       #test_data = 
      #  test_SEDs = test_dataset['SEDs']  <-- pass this wherever it is used rather than assign to a var

       # if args['model_eval'] == 'poly':
       #     d_max_gt = test_dataset['parameters']['d_max']  <-- pass this whereever it's used rather than assign to a var

        # Convert to tensor
       # tf_noisy_train_stars = tf.convert_to_tensor(train_dataset['noisy_stars'], dtype=tf.float32) <-- training_preprocessing.py
       # tf_train_pos = tf.convert_to_tensor(train_dataset['positions'], dtype=tf.float32) <-- training_preprocessing.py
        # tf_test_pos = tf.convert_to_tensor(test_dataset['positions'], dtype=tf.float32) <-- training_preprocessing.py

      #  if args['model'] == 'poly_physical':
      #      # Concatenate the Zernike and the positions from train and test datasets
      #      all_zernike_prior = np.concatenate(
      #          (train_dataset['zernike_prior'], test_dataset['zernike_prior']), axis=0
       #     )
       #     all_pos = np.concatenate((train_dataset['positions'], test_dataset['positions']), axis=0)
       #     # Convert to tensor
       #     tf_zernike_prior_all = tf.convert_to_tensor(all_zernike_prior, dtype=tf.float32)
       #     tf_pos_all = tf.convert_to_tensor(all_pos, dtype=tf.float32)

       # if args['model_eval'] == 'physical': <---- Move to a class
       #     # Concatenate both datasets
       #     all_zernike_GT = np.concatenate(
       #         (train_dataset['zernike_GT'], test_dataset['zernike_GT']),
       #         axis=0
       #     )
       #     all_pos = np.concatenate((train_dataset['positions'], test_dataset['positions']), axis=0)
       #     # Convert to tensor
       #     tf_zernike_GT_all = tf.convert_to_tensor(all_zernike_GT, dtype=tf.float32)
       #     tf_pos_all = tf.convert_to_tensor(all_pos, dtype=tf.float32)

        print('Dataset parameters:')
        print(training_data.train_parameters)

        ## Prepare models
        # Generate Zernike maps
        #zernikes = wf_utils.zernike_generator(
        #    n_zernikes=args['n_zernikes'], wfe_dim=args['pupil_diameter']
        #)
        # Now as cubes
        #np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

        #for it in range(len(zernikes)):
        #    np_zernike_cube[it, :, :] = zernikes[it]

        #np_zernike_cube[np.isnan(np_zernike_cube)] = 0
        #tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

        # New interp features backwards compatibility <--- Moved to metrics config
       # if 'interp_pts_per_bin' not in args:
       #     args['interp_pts_per_bin'] = 0
       #     args['extrapolate'] = True
       #     args['sed_interp_kind'] = 'linear'

        # Prepare np input
        simPSF_np = training_data.simPSF
        
       # SimPSFToolkit(
       #     zernikes,
       #     max_order=args['n_zernikes'],
       #     pupil_diameter=args['pupil_diameter'],
       #     output_dim=args['output_dim'],
       #     oversampling_rate=args['oversampling_rate'],
       #     output_Q=args['output_q'],
       #     interp_pts_per_bin=args['interp_pts_per_bin'],
       #     extrapolate=args['extrapolate'],
       #     SED_interp_kind=args['sed_interp_kind'],
       #     SED_sigma=args['sed_sigma']
       # )
       # simPSF_np.gen_random_Z_coeffs(max_order=args['n_zernikes'])
       # z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
       # simPSF_np.set_z_coeffs(z_coeffs)
       # simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

        # Obscurations
        #obscurations = simPSF_np.generate_pupil_obscurations(N_pix=args['pupil_diameter'], N_filter=2)
        #tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)
        tf_obscurations = psf_model.obscurations

        # Outputs (needed for the MCCD model)
        outputs = training_data.train_dataset["noisy_stars"] #tf_noisy_train_stars

        ## Create the model
        ## Select the model
        #if args['model'] == 'poly':
         #   # Initialize the model
         #   tf_semiparam_field = tf_psf_field.TF_SemiParam_field(
         #       zernike_maps=tf_zernike_cube,
         #       obscurations=tf_obscurations,
         #       batch_size=args['batch_size'],
         #       output_Q=args['output_q'],
         #       d_max_nonparam=args['d_max_nonparam'],
         #       l2_param=args['l2_param'],
         #       output_dim=args['output_dim'],
         #       n_zernikes=args['n_zernikes'],
         #       d_max=args['d_max'],
         #       x_lims=args['x_lims'],
         #       y_lims=args['y_lims']
         #   )


        ## Load the model's weights
        psf_model.load_weights(weights_paths)

        # If eval_only_param is true we put non param model to zero.
        if 'eval_only_param' not in args:
            args['eval_only_param'] = False
        elif args['eval_only_param']:
            if args['project_dd_features']:
                psf_model.project_DD_features(tf_zernike_cube)
            psf_model.set_zero_nonparam()


        ## Prepare ground truth model
        # Generate Zernike maps
        zernikes = wf_utils.zernike_generator(
            n_zernikes=args['gt_n_zernikes'], wfe_dim=args['pupil_diameter']
        )
        # Now as cubes
        np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
        for it in range(len(zernikes)):
            np_zernike_cube[it, :, :] = zernikes[it]

        np_zernike_cube[np.isnan(np_zernike_cube)] = 0
        tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

        if args['model_eval'] == 'poly':
            # Initialize the model
            GT_tf_semiparam_field = tf_psf_field.TF_SemiParam_field(
                zernike_maps=tf_zernike_cube,
                obscurations=tf_obscurations,
                batch_size=args['batch_size'],
                output_Q=args['output_q'],
                d_max_nonparam=args['d_max_nonparam'],
                output_dim=args['output_dim'],
                n_zernikes=args['gt_n_zernikes'],
                # d_max_GT may differ from the current d_max of the parametric model
                # d_max=args['d_max'],
                d_max=d_max_gt,
                x_lims=args['x_lims'],
                y_lims=args['y_lims']
            )
            # For the Ground truth model
            GT_tf_semiparam_field.tf_poly_Z_field.assign_coeff_matrix(train_C_poly)
            GT_tf_semiparam_field.set_zero_nonparam()

        ## Metric evaluation on the test dataset
        print('\n***\nMetric evaluation on the test dataset\n***\n')

        if 'n_bins_gt' not in args:
            args['n_bins_gt'] = args['n_bins_lda']

        # Polychromatic star reconstructions
        print('Computing polychromatic metrics at low resolution.')
        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            tf_semiparam_field=tf_semiparam_field,
            GT_tf_semiparam_field=GT_tf_semiparam_field,
            simPSF_np=simPSF_np,
            tf_pos=tf_test_pos,
            tf_SEDs=test_SEDs,
            n_bins_lda=args['n_bins_lda'],
            n_bins_gt=args['n_bins_gt'],
            batch_size=args['eval_batch_size'],
            dataset_dict=test_dataset,
        )

        poly_metric = {
            'rmse': rmse,
            'rel_rmse': rel_rmse,
            'std_rmse': std_rmse,
            'std_rel_rmse': std_rel_rmse
        }

        # Monochromatic star reconstructions
        if args['eval_mono_metric_rmse'] is True or 'eval_mono_metric_rmse' not in args:
            print('Computing monochromatic metrics.')
            lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation
            rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda = wf_metrics.compute_mono_metric(
                tf_semiparam_field=tf_semiparam_field,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                simPSF_np=simPSF_np,
                tf_pos=tf_test_pos,
                lambda_list=lambda_list
            )

            mono_metric = {
                'rmse_lda': rmse_lda,
                'rel_rmse_lda': rel_rmse_lda,
                'std_rmse_lda': std_rmse_lda,
                'std_rel_rmse_lda': std_rel_rmse_lda
            }
        else:
            mono_metric = None

        # OPD metrics
        if args['eval_opd_metric_rmse'] is True or 'eval_opd_metric_rmse' not in args:
            print('Computing OPD metrics.')
            rmse_opd, rel_rmse_opd, rmse_std_opd, rel_rmse_std_opd = wf_metrics.compute_opd_metrics(
                tf_semiparam_field=tf_semiparam_field,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                pos=tf_test_pos,
                batch_size=args['eval_batch_size']
            )

            opd_metric = {
                'rmse_opd': rmse_opd,
                'rel_rmse_opd': rel_rmse_opd,
                'rmse_std_opd': rmse_std_opd,
                'rel_rmse_std_opd': rel_rmse_std_opd
            }
        else:
            opd_metric = None


        # Check if all stars SR pixel RMSE are needed
        if 'opt_stars_rel_pix_rmse' not in args:
            args['opt_stars_rel_pix_rmse'] = False

        # Shape metrics
        print('Computing polychromatic high-resolution metrics and shape metrics.')
        shape_results_dict = wf_metrics.compute_shape_metrics(
            tf_semiparam_field=tf_semiparam_field,
            GT_tf_semiparam_field=GT_tf_semiparam_field,
            simPSF_np=simPSF_np,
            SEDs=test_SEDs,
            tf_pos=tf_test_pos,
            n_bins_lda=args['n_bins_lda'],
            n_bins_gt=args['n_bins_gt'],
            output_Q=1,
            output_dim=64,
            batch_size=args['eval_batch_size'],
            opt_stars_rel_pix_rmse=args['opt_stars_rel_pix_rmse'],
            dataset_dict=test_dataset,
        )

        # Save metrics
        test_metrics = {
            'poly_metric': poly_metric,
            'mono_metric': mono_metric,
            'opd_metric': opd_metric,
            'shape_results_dict': shape_results_dict
        }

        ## Metric evaluation on the train dataset
        print('\n***\nMetric evaluation on the train dataset\n***\n')

        # Polychromatic star reconstructions
        print('Computing polychromatic metrics at low resolution.')
        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            tf_semiparam_field=tf_semiparam_field,
            GT_tf_semiparam_field=GT_tf_semiparam_field,
            simPSF_np=simPSF_np,
            tf_pos=tf_train_pos,
            tf_SEDs=train_SEDs,
            n_bins_lda=args['n_bins_lda'],
            n_bins_gt=args['n_bins_gt'],
            batch_size=args['eval_batch_size'],
            dataset_dict=train_dataset,
        )

        train_poly_metric = {
            'rmse': rmse,
            'rel_rmse': rel_rmse,
            'std_rmse': std_rmse,
            'std_rel_rmse': std_rel_rmse
        }

        # Monochromatic star reconstructions turn into a class
        if args['eval_mono_metric_rmse'] is True or 'eval_mono_metric_rmse' not in args:
            print('Computing monochromatic metrics.')
            lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation
            rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda = wf_metrics.compute_mono_metric(
                tf_semiparam_field=tf_semiparam_field,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                simPSF_np=simPSF_np,
                tf_pos=tf_train_pos,
                lambda_list=lambda_list
            )

            train_mono_metric = {
                'rmse_lda': rmse_lda,
                'rel_rmse_lda': rel_rmse_lda,
                'std_rmse_lda': std_rmse_lda,
                'std_rel_rmse_lda': std_rel_rmse_lda
            }
        else:
            train_mono_metric = None


        # OPD metrics turn into a class
        if args['eval_opd_metric_rmse'] is True or 'eval_opd_metric_rmse' not in args:
            print('Computing OPD metrics.')
            rmse_opd, rel_rmse_opd, rmse_std_opd, rel_rmse_std_opd = wf_metrics.compute_opd_metrics(
                tf_semiparam_field=tf_semiparam_field,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                pos=tf_train_pos,
                batch_size=args['eval_batch_size']
            )

            train_opd_metric = {
                'rmse_opd': rmse_opd,
                'rel_rmse_opd': rel_rmse_opd,
                'rmse_std_opd': rmse_std_opd,
                'rel_rmse_std_opd': rel_rmse_std_opd
            }
        else:
            train_opd_metric = None


        # Shape metrics  turn into a class
        if args['eval_train_shape_sr_metric_rmse'] is True or 'eval_train_shape_sr_metric_rmse' not in args:
            print('Computing polychromatic high-resolution metrics and shape metrics.')
            train_shape_results_dict = wf_metrics.compute_shape_metrics(
                tf_semiparam_field=tf_semiparam_field,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                simPSF_np=simPSF_np,
                SEDs=train_SEDs,
                tf_pos=tf_train_pos,
                n_bins_lda=args['n_bins_lda'],
                n_bins_gt=args['n_bins_gt'],
                output_Q=1,
                output_dim=64,
                batch_size=args['eval_batch_size'],
                dataset_dict=train_dataset,
            )

            # Save metrics into dictionary
            train_metrics = {
                'poly_metric': train_poly_metric,
                'mono_metric': train_mono_metric,
                'opd_metric': train_opd_metric,
                'shape_results_dict': train_shape_results_dict
            }
        else:
            train_metrics = None

        ## Save results
        metrics = {'test_metrics': test_metrics, 'train_metrics': train_metrics}
        output_path = args['metric_base_path'] + 'metrics-' + run_id_name
        np.save(output_path, metrics, allow_pickle=True)

        ## Print final time
        final_time = time.time()
        print('\nTotal elapsed time: %f' % (final_time - starting_time))

        ## Close log file
        print('\n Good bye..')
        sys.stdout = old_stdout
        log_file.close()

    except Exception as e:
        print('Error: %s' % e)
        sys.stdout = old_stdout
        log_file.close()
        raise 