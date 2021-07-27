#!/usr/bin/env python
# coding: utf-8

# PSF modelling and evaluation

# Import packages
import sys
import numpy as np
import time
import wf_psf as wf
import tensorflow as tf
import tensorflow_addons as tfa

import click

# from absl import app
# from absl import flags

@click.command()

## Training options
# Model definition
@click.option(
    "--model",
    default="poly",
    type=str,
    help="Model type. Options are: 'mccd', 'poly, 'param'.")
@click.option(
    "--id_name",
    default="-coherent_euclid_200stars",
    type=str,
    help="Model saving id.")
# Saving paths
@click.option(
    "--base_path",
    default="/gpfswork/rech/xdy/ulx23va/wf-outputs/",
    type=str,
    help="Base path for saving files.")
@click.option(
    "--log_folder",
    default="log-files/",
    type=str,
    help="Folder name to save log files.")
@click.option(
    "--model_folder",
    default="chkp/",
    type=str,
    help="Folder name to save trained models.")
@click.option(
    "--optim_hist_folder",
    default="optim-hist/",
    type=str,
    help="Folder name to save optimisation history files.")
@click.option(
    "--chkp_save_path",
    default="/gpfsscratch/rech/xdy/ulx23va/wf-outputs/chkp/",
    type=str,
    help="Path to save model checkpoints during training.")
# Input dataset paths
@click.option(
    "--dataset_folder",
    default="/gpfswork/rech/xdy/ulx23va/repo/wf-psf/data/coherent_euclid_dataset/",
    type=str,
    help="Folder path of datasets.")
@click.option(
    "--train_dataset_file",
    default="train_Euclid_res_200_TrainStars_id_001.npy",
    type=str,
    help="Train dataset file name.")
@click.option(
    "--test_dataset_file",
    default="test_Euclid_res_id_001.npy",
    type=str,
    help="Test dataset file name.")
# Model parameters
@click.option(
    "--n_zernikes",
    default=15,
    type=int,
    help="Zernike polynomial modes to use on the parametric part.")
@click.option(
    "--pupil_diameter",
    default=256,
    type=int,
    help="Dimension of the OPD/Wavefront space.")
@click.option(
    "--n_bins_lda",
    default=20,
    type=int,
    help="Number of wavelength bins to use to reconstruct polychromatic objects.")
@click.option(
    "--output_q",
    default=3.,
    type=float,
    help="Downsampling rate to match the specified telescope's sampling from the oversampling rate used in the model.")
@click.option(
    "--oversampling_rate",
    default=3.,
    type=float,
    help="Oversampling rate used for the OPD/WFE PSF model.")
@click.option(
    "--output_dim",
    default=32,
    type=int,
    help="Dimension of the pixel PSF postage stamp.")
@click.option(
    "--d_max",
    default=2,
    type=int,
    help="Max polynomial degree of the parametric part.")
@click.option(
    "--d_max_nonparam",
    default=3,
    type=int,
    help="Max polynomial degree of the non-parametric part.")
@click.option(
    "--x_lims",
    nargs=2,
    default=[0, 1e3],
    type=float,
    help="Limits of the PSF field coordinates for the x axis.")
@click.option(
    "--y_lims",
    nargs=2,
    default=[0, 1e3],
    type=float,
    help="Limits of the PSF field coordinates for the y axis.")
@click.option(
    "--graph_features",
    default=10,
    type=int,
    help="Number of graph-constrained features of the non-parametric part.")
@click.option(
    "--l1_rate",
    default=1e-8,
    type=float,
    help="L1 regularisation parameter for the non-parametric part.")
# Training parameters
@click.option(
    "--batch_size",
    default=32,
    type=int,
    help="Batch size used for the trainingin the stochastic gradient descend type of algorithm.")
@click.option(
    "--l_rate_param",
    nargs=2,
    default=[1e-2, 1e-2],
    type=float,
    help="Learning rates for the parametric parts.")
@click.option(
    "--l_rate_non_param",
    nargs=2,
    default=[1e-1, 1e-1],
    type=float,
    help="Learning rates for the non-parametric parts.")
@click.option(
    "--n_epochs_param",
    nargs=2,
    default=[20, 20],
    type=int,
    help="Number of training epochs of the parametric parts.")
@click.option(
    "--n_epochs_non_param",
    nargs=2,
    default=[100, 120],
    type=int,
    help="Number of training epochs of the non-parametric parts.")
@click.option(
    "--total_cycles",
    default=2,
    type=int,
    help="Total amount of cycles to perform. For the moment the only available options are '1' or '2'.")
## Evaluation flags
# Saving paths
@click.option(
    "--metric_base_path",
    default="/gpfswork/rech/xdy/ulx23va/wf-outputs/metrics/",
    type=str,
    help="Base path for saving metric files.")
@click.option(
    "--saved_model_type",
    default="final",
    type=str,
    help="Type of saved model to use for the evaluation. Can be 'final' or 'checkpoint'.")
@click.option(
    "--saved_cycle",
    default="cycle2",
    type=str,
    help="Saved cycle to use for the evaluation. Can be 'cycle1' or 'cycle2'.")
# Evaluation parameters
@click.option(
    "--GT_n_zernikes",
    default=45,
    type=int,
    help="Zernike polynomial modes to use on the ground truth model parametric part.")
@click.option(
    "--eval_batch_size",
    default=16,
    type=int,
    help="Batch size to use for the evaluation.")

def main(**args):
    train_model(**args)
    evaluate_model(**args)


def train_model(**args):
    """ Train the model defined in the  """
    # Start measuring elapsed time
    starting_time = time.time()

    # Define model run id
    run_id_name = args['model'] + args['id_name']

    # Define paths
    log_save_file = args['base_path'] + args['log_folder']
    model_save_file= args['base_path'] + args['model_folder']
    optim_hist_file = args['base_path']  + args['optim_hist_folder']
    saving_optim_hist = dict()


    # Save output prints to logfile
    old_stdout = sys.stdout
    log_file = open(log_save_file + run_id_name + '_output.log','w')
    sys.stdout = log_file
    print('Starting the log file.')

    # Print GPU and tensorflow info
    device_name = tf.test.gpu_device_name()
    print('Found GPU at: {}'.format(device_name))
    print('tf_version: ' + str(tf.__version__))

    ## Prepare the inputs
    # Generate Zernike maps
    zernikes = wf.utils.zernike_generator(n_zernikes=args['n_zernikes'], wfe_dim=args['pupil_diameter'])
    # Now as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)
    print('Zernike cube:')
    print(tf_zernike_cube.shape)


    ## Load the dictionaries
    train_dataset = np.load(args['dataset_folder'] + args['train_dataset_file'], allow_pickle=True)[()]
    # train_stars = train_dataset['stars']
    # noisy_train_stars = train_dataset['noisy_stars']
    # train_pos = train_dataset['positions']
    train_SEDs = train_dataset['SEDs']
    # train_zernike_coef = train_dataset['zernike_coef']
    train_C_poly = train_dataset['C_poly']
    train_parameters = train_dataset['parameters']

    test_dataset = np.load(args['dataset_folder'] + args['test_dataset_file'], allow_pickle=True)[()]
    # test_stars = test_dataset['stars']
    # test_pos = test_dataset['positions']
    test_SEDs = test_dataset['SEDs']
    # test_zernike_coef = test_dataset['zernike_coef']

    # Convert to tensor
    tf_noisy_train_stars = tf.convert_to_tensor(train_dataset['noisy_stars'], dtype=tf.float32)
    tf_train_stars = tf.convert_to_tensor(train_dataset['stars'], dtype=tf.float32)
    tf_train_pos = tf.convert_to_tensor(train_dataset['positions'], dtype=tf.float32)
    tf_test_stars = tf.convert_to_tensor(test_dataset['stars'], dtype=tf.float32)
    tf_test_pos = tf.convert_to_tensor(test_dataset['positions'], dtype=tf.float32)

    print('Dataset parameters:')
    print(train_parameters)


    ## Generate initializations
    # Prepare np input
    simPSF_np = wf.SimPSFToolkit(zernikes, max_order=args['n_zernikes'],
                                    pupil_diameter=args['pupil_diameter'], output_dim=args['output_dim'],
                                    oversampling_rate=args['oversampling_rate'], output_Q=args['output_q'])
    simPSF_np.gen_random_Z_coeffs(max_order=args['n_zernikes'])
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(N_pix=args['pupil_diameter'], N_filter=2)
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)
    # Initialize the SED data list
    packed_SED_data = [wf.utils.generate_packed_elems(_sed, simPSF_np, n_bins=args['n_bins_lda'])
                    for _sed in train_SEDs]


    # Prepare the inputs for the training
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])

    inputs = [tf_train_pos, tf_packed_SED_data]

    # Select the observed stars (noisy or noiseless)
    outputs = tf_noisy_train_stars
    # outputs = tf_train_stars


    ## Prepare validation data inputs
    val_SEDs = test_SEDs
    tf_val_pos = tf_test_pos
    tf_val_stars = tf_test_stars

    # Initialize the SED data list
    val_packed_SED_data = [wf.utils.generate_packed_elems(_sed, simPSF_np, n_bins=args['n_bins_lda'])
                    for _sed in val_SEDs]

    # Prepare the inputs for the validation
    tf_val_packed_SED_data = tf.convert_to_tensor(val_packed_SED_data, dtype=tf.float32)
    tf_val_packed_SED_data = tf.transpose(tf_val_packed_SED_data, perm=[0, 2, 1])
                    
    # Prepare input validation tuple
    val_x_inputs = [tf_val_pos, tf_val_packed_SED_data]
    val_y_inputs = tf_val_stars
    val_data = (val_x_inputs, val_y_inputs)


    ## Select the model
    if args['model'] == 'mccd':
        poly_dic, graph_dic = wf.tf_mccd_psf_field.build_mccd_spatial_dic_v2(obs_stars=outputs.numpy(),
                                            obs_pos=tf_train_pos.numpy(),
                                            x_lims=args['x_lims'],
                                            y_lims=args['y_lims'],
                                            d_max=args['d_max_nonparam'],
                                            graph_features=args['graph_features'])

        spatial_dic = [poly_dic, graph_dic]

        # Initialize the model
        tf_semiparam_field = wf.tf_mccd_psf_field.TF_SP_MCCD_field(zernike_maps=tf_zernike_cube,
                                                                    obscurations=tf_obscurations,
                                                                    batch_size=args['batch_size'],
                                                                    obs_pos=tf_train_pos,
                                                                    spatial_dic=spatial_dic,
                                                                    output_Q=args['output_q'],
                                                                    d_max_nonparam=args['d_max_nonparam'],
                                                                    graph_features=args['graph_features'],
                                                                    l1_rate=args['l1_rate'],
                                                                    output_dim=args['output_dim'],
                                                                    n_zernikes=args['n_zernikes'],
                                                                    d_max=args['d_max'],
                                                                    x_lims=args['x_lims'],
                                                                    y_lims=args['y_lims'])

    elif args['model'] == 'poly':
        # # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_SemiParam_field(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=args['batch_size'],
                                                output_Q=args['output_q'],
                                                d_max_nonparam=args['d_max_nonparam'],
                                                output_dim=args['output_dim'],
                                                n_zernikes=args['n_zernikes'],
                                                d_max=args['d_max'],
                                                x_lims=args['x_lims'],
                                                y_lims=args['y_lims'])

    elif args['model'] == 'param':
        # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_PSF_field_model(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=args['batch_size'],
                                                output_Q=args['output_q'],
                                                output_dim=args['output_dim'],
                                                n_zernikes=args['n_zernikes'],
                                                d_max=args['d_max'],
                                                x_lims=args['x_lims'],
                                                y_lims=args['y_lims'])


    # # Model Training
    # Prepare the saving callback
    # Prepare to save the model as a callback
    filepath_chkp_callback = args['chkp_save_path'] + 'chkp_callback_' + run_id_name + '_cycle1'
    model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath_chkp_callback,
        monitor='mean_squared_error', verbose=1, save_best_only=True,
        save_weights_only=False, mode='min', save_freq='epoch',
        options=None)

    # Prepare the optimisers
    param_optim = tfa.optimizers.RectifiedAdam(lr=args['l_rate_param'][0])
    non_param_optim = tfa.optimizers.RectifiedAdam(lr=args['l_rate_non_param'][0])

    print('Starting cycle 1..')
    start_cycle1 = time.time()

    if args['model'] == 'param':
        tf_semiparam_field, hist_param = wf.train_utils.param_train_cycle(
            tf_semiparam_field,
            inputs=inputs,
            outputs=outputs,
            val_data=val_data,
            batch_size=args['batch_size'],
            l_rate=args['l_rate_param'][0],
            n_epochs=args['n_epochs_param'][0], 
            param_optim=param_optim,
            param_loss=None, 
            param_metrics=None, 
            param_callback=None, 
            general_callback=[model_chkp_callback],
            verbose=2)

    else:
        tf_semiparam_field, hist_param, hist_non_param = wf.train_utils.general_train_cycle(
            tf_semiparam_field,
            inputs=inputs,
            outputs=outputs,
            val_data=val_data,
            batch_size=args['batch_size'],
            l_rate_param=args['l_rate_param'][0],
            l_rate_non_param=args['l_rate_non_param'][0],
            n_epochs_param=args['n_epochs_param'][0],
            n_epochs_non_param=args['n_epochs_non_param'][0],
            param_optim=param_optim,
            non_param_optim=non_param_optim,
            param_loss=None, non_param_loss=None,
            param_metrics=None, non_param_metrics=None,
            param_callback=None, non_param_callback=None,
            general_callback=[model_chkp_callback],
            first_run=True,
            verbose=2)

    # Save weights
    tf_semiparam_field.save_weights(model_save_file + 'chkp_' + run_id_name + '_cycle1')

    end_cycle1 = time.time()
    print('Cycle1 elapsed time: %f'%(end_cycle1-start_cycle1))

    # Save optimisation history in the saving dict
    saving_optim_hist['param_cycle1'] = hist_param.history
    if args['model'] != 'param':
        saving_optim_hist['nonparam_cycle1'] = hist_non_param.history

    if args['total_cycles'] >= 2:
        # Prepare to save the model as a callback
        filepath_chkp_callback = args['chkp_save_path'] + 'chkp_callback_' + run_id_name + '_cycle2'
        model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath_chkp_callback,
            monitor='mean_squared_error', verbose=1, save_best_only=True,
            save_weights_only=False, mode='min', save_freq='epoch',
            options=None)

        # Prepare the optimisers
        param_optim = tfa.optimizers.RectifiedAdam(lr=args['l_rate_param'][1])
        non_param_optim = tfa.optimizers.RectifiedAdam(lr=args['l_rate_non_param'][1])

        print('Starting cycle 2..')
        start_cycle2 = time.time()


        # Compute the next cycle
        if args['model'] == 'param':
            tf_semiparam_field, hist_param_2 = wf.train_utils.param_train_cycle(
                tf_semiparam_field,
                inputs=inputs,
                outputs=outputs,
                val_data=val_data,
                batch_size=args['batch_size'],
                l_rate=args['l_rate_param'][1],
                n_epochs=args['n_epochs_param'][1], 
                param_optim=param_optim,
                param_loss=None, 
                param_metrics=None, 
                param_callback=None, 
                general_callback=[model_chkp_callback],
                verbose=2)
        else:
            # Compute the next cycle
            tf_semiparam_field, hist_param_2, hist_non_param_2 = wf.train_utils.general_train_cycle(
                tf_semiparam_field,
                inputs=inputs,
                outputs=outputs,
                val_data=val_data,
                batch_size=args['batch_size'],
                l_rate_param=args['l_rate_param'][1],
                l_rate_non_param=args['l_rate_non_param'][1],
                n_epochs_param=args['n_epochs_param'][1],
                n_epochs_non_param=args['n_epochs_non_param'][1],
                param_optim=param_optim,
                non_param_optim=non_param_optim,
                param_loss=None, non_param_loss=None,
                param_metrics=None, non_param_metrics=None,
                param_callback=None, non_param_callback=None,
                general_callback=[model_chkp_callback],
                first_run=False,
                verbose=2)

        # Save the weights at the end of the second cycle
        tf_semiparam_field.save_weights(model_save_file + 'chkp_' + run_id_name + '_cycle2')

        end_cycle2 = time.time()
        print('Cycle2 elapsed time: %f'%(end_cycle2 - start_cycle2))

        # Save optimisation history in the saving dict
        saving_optim_hist['param_cycle2'] = hist_param_2.history
        if args['model'] != 'param':
            saving_optim_hist['nonparam_cycle2'] = hist_non_param_2.history

    # Save optimisation history dictionary
    np.save(optim_hist_file + 'optim_hist_' + run_id_name + '.npy', saving_optim_hist)

    ## Print final time
    final_time = time.time()
    print('\nTotal elapsed time: %f'%(final_time - starting_time))

    ## Close log file
    print('\n Good bye..')
    sys.stdout = old_stdout
    log_file.close()


def evaluate_model(**args):
    """ Evaluate the trained model."""
    # Start measuring elapsed time
    starting_time = time.time()

    # Define model run id
    run_id_name = args['model'] + args['id_name']
    # Define paths
    log_save_file = args['base_path'] + args['log_folder']

    # Define saved model to use
    if args['saved_model_type'] == 'checkpoint':
        weights_paths = args['chkp_save_path'] + 'chkp_callback_' + run_id_name + '_' + args['saved_cycle']

    elif args['saved_model_type'] == 'final':
        model_save_file= args['base_path'] + args['model_folder']
        weights_paths = model_save_file + 'chkp_' + run_id_name + '_' + args['saved_cycle']

    
    ## Save output prints to logfile
    old_stdout = sys.stdout
    log_file = open(log_save_file + run_id_name + '-metrics_output.log', 'w')
    sys.stdout = log_file
    print('Starting the log file.')

    ## Check GPU and tensorflow version
    device_name = tf.test.gpu_device_name()
    print('Found GPU at: {}'.format(device_name))
    print('tf_version: ' + str(tf.__version__))


    ## Load datasets
    train_dataset = np.load(args['dataset_folder'] + args['train_dataset_file'], allow_pickle=True)[()]
    # train_stars = train_dataset['stars']
    # noisy_train_stars = train_dataset['noisy_stars']
    # train_pos = train_dataset['positions']
    train_SEDs = train_dataset['SEDs']
    # train_zernike_coef = train_dataset['zernike_coef']
    train_C_poly = train_dataset['C_poly']
    train_parameters = train_dataset['parameters']

    test_dataset = np.load(args['dataset_folder'] + args['test_dataset_file'], allow_pickle=True)[()]
    # test_stars = test_dataset['stars']
    # test_pos = test_dataset['positions']
    test_SEDs = test_dataset['SEDs']
    # test_zernike_coef = test_dataset['zernike_coef']

    # Convert to tensor
    tf_noisy_train_stars = tf.convert_to_tensor(train_dataset['noisy_stars'], dtype=tf.float32)
    tf_train_pos = tf.convert_to_tensor(train_dataset['positions'], dtype=tf.float32)
    tf_test_pos = tf.convert_to_tensor(test_dataset['positions'], dtype=tf.float32)

    print('Dataset parameters:')
    print(train_parameters)


    ## Prepare models
    # Generate Zernike maps
    zernikes = wf.utils.zernike_generator(n_zernikes=args['n_zernikes'], wfe_dim=args['pupil_diameter'])
    # Now as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Prepare np input
    simPSF_np = wf.SimPSFToolkit(zernikes, max_order=args['n_zernikes'],
                                    pupil_diameter=args['pupil_diameter'], output_dim=args['output_dim'],
                                    oversampling_rate=args['oversampling_rate'], output_Q=args['output_q'])
    simPSF_np.gen_random_Z_coeffs(max_order=args['n_zernikes'])
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(N_pix=args['pupil_diameter'], N_filter=2)
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)

    # Outputs (needed for the MCCD model)
    outputs = tf_noisy_train_stars


    ## Create the model
    ## Select the model
    if args['model'] == 'mccd':
        poly_dic, graph_dic = wf.tf_mccd_psf_field.build_mccd_spatial_dic_v2(obs_stars=outputs.numpy(),
                                            obs_pos=tf_train_pos.numpy(),
                                            x_lims=args['x_lims'],
                                            y_lims=args['y_lims'],
                                            d_max=args['d_max_nonparam'],
                                            graph_features=args['graph_features'])

        spatial_dic = [poly_dic, graph_dic]

       # Initialize the model
        tf_semiparam_field = wf.tf_mccd_psf_field.TF_SP_MCCD_field(zernike_maps=tf_zernike_cube,
                                                                    obscurations=tf_obscurations,
                                                                    batch_size=args['batch_size'],
                                                                    obs_pos=tf_train_pos,
                                                                    spatial_dic=spatial_dic,
                                                                    output_Q=args['output_q'],
                                                                    d_max_nonparam=args['d_max_nonparam'],
                                                                    graph_features=args['graph_features'],
                                                                    l1_rate=args['l1_rate'],
                                                                    output_dim=args['output_dim'],
                                                                    n_zernikes=args['n_zernikes'],
                                                                    d_max=args['d_max'],
                                                                    x_lims=args['x_lims'],
                                                                    y_lims=args['y_lims'])

    elif args['model'] == 'poly':
        # # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_SemiParam_field(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=args['batch_size'],
                                                output_Q=args['output_q'],
                                                d_max_nonparam=args['d_max_nonparam'],
                                                output_dim=args['output_dim'],
                                                n_zernikes=args['n_zernikes'],
                                                d_max=args['d_max'],
                                                x_lims=args['x_lims'],
                                                y_lims=args['y_lims'])

    elif args['model'] == 'param':
        # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_PSF_field_model(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=args['batch_size'],
                                                output_Q=args['output_q'],
                                                output_dim=args['output_dim'],
                                                n_zernikes=args['n_zernikes'],
                                                d_max=args['d_max'],
                                                x_lims=args['x_lims'],
                                                y_lims=args['y_lims'])

    ## Load the model's weights
    tf_semiparam_field.load_weights(weights_paths)

    ## Prepare ground truth model
    # Generate Zernike maps
    zernikes = wf.utils.zernike_generator(n_zernikes=args['GT_n_zernikes'], wfe_dim=args['pupil_diameter'])
    # Now as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Initialize the model
    GT_tf_semiparam_field = wf.tf_psf_field.TF_SemiParam_field(
        zernike_maps=tf_zernike_cube,
        obscurations=tf_obscurations,
        batch_size=args['batch_size'],
        output_Q=args['output_q'],
        d_max_nonparam=args['d_max_nonparam'],
        output_dim=args['output_dim'],
        n_zernikes=args['GT_n_zernikes'],
        d_max=args['d_max'],
        x_lims=args['x_lims'],
        y_lims=args['y_lims'])

    # For the Ground truth model
    GT_tf_semiparam_field.tf_poly_Z_field.assign_coeff_matrix(train_C_poly)
    _ = GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat.assign(
        np.zeros_like(GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat))


    ## Metric evaluation on the test dataset
    print('\n***\nMetric evaluation on the test dataset\n***\n')

    # Polychromatic star reconstructions
    rmse, rel_rmse, std_rmse, std_rel_rmse = wf.metrics.compute_poly_metric(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        simPSF_np=simPSF_np,
        tf_pos=tf_test_pos,
        tf_SEDs=test_SEDs,
        n_bins_lda=args['n_bins_lda'],
        batch_size=args['eval_batch_size'])

    poly_metric = {'rmse': rmse,
                'rel_rmse': rel_rmse,
                'std_rmse': std_rmse,
                'std_rel_rmse': std_rel_rmse
                }

    # Monochromatic star reconstructions
    lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation
    rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda = wf.metrics.compute_mono_metric(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        simPSF_np=simPSF_np,
        tf_pos=tf_test_pos,
        lambda_list=lambda_list)

    mono_metric = {'rmse_lda': rmse_lda,
                'rel_rmse_lda': rel_rmse_lda,
                'std_rmse_lda': std_rmse_lda,
                'std_rel_rmse_lda': std_rel_rmse_lda
                }

    # OPD metrics
    rmse_opd, rel_rmse_opd, rmse_std_opd, rel_rmse_std_opd = wf.metrics.compute_opd_metrics(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        pos=tf_test_pos,
        batch_size=args['eval_batch_size'])

    opd_metric = { 'rmse_opd': rmse_opd,
                'rel_rmse_opd': rel_rmse_opd,
                'rmse_std_opd': rmse_std_opd,
                'rel_rmse_std_opd': rel_rmse_std_opd
                }

    # Shape metrics
    shape_results_dict = wf.metrics.compute_shape_metrics(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        simPSF_np=simPSF_np,
        SEDs=test_SEDs,
        tf_pos=tf_test_pos,
        n_bins_lda=args['n_bins_lda'],
        output_Q=1,
        output_dim=64,
        batch_size=args['eval_batch_size'])

    # Save metrics
    test_metrics = {'poly_metric': poly_metric,
                    'mono_metric': mono_metric,
                    'opd_metric': opd_metric,
                    'shape_results_dict': shape_results_dict
                }


    ## Metric evaluation on the train dataset
    print('\n***\nMetric evaluation on the train dataset\n***\n')

    # Polychromatic star reconstructions
    rmse, rel_rmse, std_rmse, std_rel_rmse = wf.metrics.compute_poly_metric(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        simPSF_np=simPSF_np,
        tf_pos=tf_train_pos,
        tf_SEDs=train_SEDs,
        n_bins_lda=args['n_bins_lda'],
        batch_size=args['eval_batch_size'])

    train_poly_metric = {'rmse': rmse,
                'rel_rmse': rel_rmse,
                'std_rmse': std_rmse,
                'std_rel_rmse': std_rel_rmse
                }

    # Monochromatic star reconstructions
    lambda_list = np.arange(0.55, 0.9, 0.01)    # 10nm separation
    rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda = wf.metrics.compute_mono_metric(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        simPSF_np=simPSF_np,
        tf_pos=tf_train_pos,
        lambda_list=lambda_list)

    train_mono_metric = {'rmse_lda': rmse_lda,
                'rel_rmse_lda': rel_rmse_lda,
                'std_rmse_lda': std_rmse_lda,
                'std_rel_rmse_lda': std_rel_rmse_lda
                }

    # OPD metrics
    rmse_opd, rel_rmse_opd, rmse_std_opd, rel_rmse_std_opd = wf.metrics.compute_opd_metrics(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        pos=tf_train_pos,
        batch_size=args['eval_batch_size'])

    train_opd_metric = { 'rmse_opd': rmse_opd,
                'rel_rmse_opd': rel_rmse_opd,
                'rmse_std_opd': rmse_std_opd,
                'rel_rmse_std_opd': rel_rmse_std_opd
                }

    # Shape metrics
    train_shape_results_dict = wf.metrics.compute_shape_metrics(
        tf_semiparam_field=tf_semiparam_field,
        GT_tf_semiparam_field=GT_tf_semiparam_field,
        simPSF_np=simPSF_np,
        SEDs=train_SEDs,
        tf_pos=tf_train_pos,
        n_bins_lda=args['n_bins_lda'],
        output_Q=1,
        output_dim=64,
        batch_size=args['eval_batch_size'])

    # Save metrics into dictionary
    train_metrics = {'poly_metric': train_poly_metric,
                    'mono_metric': train_mono_metric,
                    'opd_metric': train_opd_metric,
                    'shape_results_dict': train_shape_results_dict
                    }


    ## Save results
    metrics = {'test_metrics': test_metrics,
               'train_metrics': train_metrics
              }
    output_path = args['metric_base_path'] + 'metrics-' + run_id_name
    np.save(output_path, metrics, allow_pickle=True)

    ## Print final time
    final_time = time.time()
    print('\nTotal elapsed time: %f'%(final_time - starting_time))


    ## Close log file
    print('\n Good bye..')
    sys.stdout = old_stdout
    log_file.close()



if __name__ == "__main__":
  main()
