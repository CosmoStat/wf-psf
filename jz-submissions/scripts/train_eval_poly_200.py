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

from absl import app
from absl import flags


# Define flags
FLAGS = flags.FLAGS

## Training flags
# Model definition
flags.DEFINE_string("model", "poly", "Model type. Options are: 'mccd', 'poly, 'param'.")
flags.DEFINE_string("id_name", "_test-coherent_euclid_200stars", "Model saving id.")
# Saving paths
flags.DEFINE_string("base_path", "/gpfswork/rech/xdy/ulx23va/wf-outputs/", "Base path for saving files.")
flags.DEFINE_string("log_folder", "log-files/", "Folder name to save log files.")
flags.DEFINE_string("model_folder", "chkp/", "Folder name to save trained models.")
flags.DEFINE_string("optim_hist_folder", "optim-hist/", "Folder name to save optimisation history files.")
flags.DEFINE_string("chkp_save_path", "/gpfsscratch/rech/xdy/ulx23va/wf-outputs/chkp/", "Path to save model checkpoints during training.")
# Input dataset paths
flags.DEFINE_string("dataset_folder", "/gpfswork/rech/xdy/ulx23va/repo/wf-psf/data/coherent_euclid_dataset/", "Folder path of datasets.")
flags.DEFINE_string("train_dataset_file", "train_Euclid_res_200_TrainStars_id_001.npy", "Train dataset file name.")
flags.DEFINE_string("test_dataset_file", "test_Euclid_res_id_001.npy", "Test dataset file name.")
# Model parameters
flags.DEFINE_integer("n_zernikes", 15, "Zernike polynomial modes to use on the parametric part.")
flags.DEFINE_integer("pupil_diameter", 256, "Dimension of the OPD/Wavefront space.")
flags.DEFINE_integer("n_bins_lda", 20, "Number of wavelength bins to use to reconstruct polychromatic objects.")
flags.DEFINE_float("output_Q", 3., "Downsampling rate to match the specified telescope's sampling from the oversampling rate used in the model.")
flags.DEFINE_float("oversampling_rate", 3., "Oversampling rate used for the OPD/WFE PSF model.")
flags.DEFINE_integer("output_dim", 32, "Dimension of the pixel PSF postage stamp.")
flags.DEFINE_integer("d_max", 2, "Max polynomial degree of the parametric part.")
flags.DEFINE_integer("d_max_nonparam", 3, "Max polynomial degree of the non-parametric part.")
flags.DEFINE_list("x_lims", [0, 1e3], "Limits of the PSF field coordinates for the x axis.")
flags.DEFINE_list("y_lims", [0, 1e3], "Limits of the PSF field coordinates for the y axis.")
flags.DEFINE_integer("graph_features", 10, "Number of graph-constrained features of the non-parametric part.")
flags.DEFINE_float("l1_rate", 1e-8, "L1 regularisation parameter for the non-parametric part.")
# Training parameters
flags.DEFINE_integer("batch_size", 32, "Batch size used for the trainingin the stochastic gradient descend type of algorithm.")
flags.DEFINE_list("l_rate_param", [1e-2, 1e-2], "Learning rates for the parametric parts.")
flags.DEFINE_list("l_rate_non_param", [1e-1, 1e-1], "Learning rates for the non-parametric parts.")
flags.DEFINE_list("n_epochs_param", [2, 2], "Number of training epochs of the parametric parts.")
flags.DEFINE_list("n_epochs_non_param", [2, 2], "Number of training epochs of the non-parametric parts.")
flags.DEFINE_integer("total_cycles", 2, "Total amount of cycles to perform. For the moment the only available options are '1' or '2'.")

## Evaluation flags
# Saving paths
flags.DEFINE_string("metric_base_path", "/gpfswork/rech/xdy/ulx23va/wf-outputs/metrics/", "Base path for saving metric files.")
flags.DEFINE_string("saved_model_type", "final", "Type of saved model to use for the evaluation. Can be 'final' or 'checkpoint'.")
flags.DEFINE_string("saved_cycle", "cycle2", "Saved cycle to use for the evaluation. Can be 'cycle1' or 'cycle2'.")
# Evaluation parameters
flags.DEFINE_integer("GT_n_zernikes", 45, "Zernike polynomial modes to use on the ground truth model parametric part.")
flags.DEFINE_integer("eval_batch_size", 16, "Batch size to use for the evaluation.")



def train_model():
    """ Train the model defined in the flags. """
    # Start measuring elapsed time
    starting_time = time.time()

    # Define model run id
    run_id_name = FLAGS.model + FLAGS.id_name

    # Define paths
    log_save_file = FLAGS.base_path + FLAGS.log_folder
    model_save_file= FLAGS.base_path + FLAGS.model_folder
    optim_hist_file = FLAGS.base_path  + FLAGS.optim_hist_folder
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
    zernikes = wf.utils.zernike_generator(n_zernikes=FLAGS.n_zernikes, wfe_dim=FLAGS.pupil_diameter)
    # Now as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)
    print('Zernike cube:')
    print(tf_zernike_cube.shape)


    ## Load the dictionaries
    train_dataset = np.load(FLAGS.dataset_folder + FLAGS.train_dataset_file, allow_pickle=True)[()]
    # train_stars = train_dataset['stars']
    # noisy_train_stars = train_dataset['noisy_stars']
    # train_pos = train_dataset['positions']
    train_SEDs = train_dataset['SEDs']
    # train_zernike_coef = train_dataset['zernike_coef']
    train_C_poly = train_dataset['C_poly']
    train_parameters = train_dataset['parameters']

    test_dataset = np.load(FLAGS.dataset_folder + FLAGS.test_dataset_file, allow_pickle=True)[()]
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
    simPSF_np = wf.SimPSFToolkit(zernikes, max_order=FLAGS.n_zernikes,
                                    pupil_diameter=FLAGS.pupil_diameter, output_dim=FLAGS.output_dim,
                                    oversampling_rate=FLAGS.oversampling_rate, output_Q=FLAGS.output_Q)
    simPSF_np.gen_random_Z_coeffs(max_order=FLAGS.n_zernikes)
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(N_pix=FLAGS.pupil_diameter, N_filter=2)
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)
    # Initialize the SED data list
    packed_SED_data = [wf.utils.generate_packed_elems(_sed, simPSF_np, n_bins=FLAGS.n_bins_lda)
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
    val_packed_SED_data = [wf.utils.generate_packed_elems(_sed, simPSF_np, n_bins=FLAGS.n_bins_lda)
                    for _sed in val_SEDs]

    # Prepare the inputs for the validation
    tf_val_packed_SED_data = tf.convert_to_tensor(val_packed_SED_data, dtype=tf.float32)
    tf_val_packed_SED_data = tf.transpose(tf_val_packed_SED_data, perm=[0, 2, 1])
                    
    # Prepare input validation tuple
    val_x_inputs = [tf_val_pos, tf_val_packed_SED_data]
    val_y_inputs = tf_val_stars
    val_data = (val_x_inputs, val_y_inputs)


    ## Select the model
    if FLAGS.model == 'mccd':
        poly_dic, graph_dic = wf.tf_mccd_psf_field.build_mccd_spatial_dic_v2(obs_stars=outputs.numpy(),
                                            obs_pos=tf_train_pos.numpy(),
                                            x_lims=FLAGS.x_lims,
                                            y_lims=FLAGS.y_lims,
                                            d_max=FLAGS.d_max_nonparam,
                                            graph_features=FLAGS.graph_features)

        spatial_dic = [poly_dic, graph_dic]

        # Initialize the model
        tf_semiparam_field = wf.tf_mccd_psf_field.TF_SP_MCCD_field(zernike_maps=tf_zernike_cube,
                                                                    obscurations=tf_obscurations,
                                                                    batch_size=FLAGS.batch_size,
                                                                    obs_pos=tf_train_pos,
                                                                    spatial_dic=spatial_dic,
                                                                    output_Q=FLAGS.output_Q,
                                                                    d_max_nonparam=FLAGS.d_max_nonparam,
                                                                    graph_features=FLAGS.graph_features,
                                                                    l1_rate=FLAGS.l1_rate,
                                                                    output_dim=FLAGS.output_dim,
                                                                    n_zernikes=FLAGS.n_zernikes,
                                                                    d_max=FLAGS.d_max,
                                                                    x_lims=FLAGS.x_lims,
                                                                    y_lims=FLAGS.y_lims)

    elif FLAGS.model == 'poly':
        # # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_SemiParam_field(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=FLAGS.batch_size,
                                                output_Q=FLAGS.output_Q,
                                                d_max_nonparam=FLAGS.d_max_nonparam,
                                                output_dim=FLAGS.output_dim,
                                                n_zernikes=FLAGS.n_zernikes,
                                                d_max=FLAGS.d_max,
                                                x_lims=FLAGS.x_lims,
                                                y_lims=FLAGS.y_lims)

    elif FLAGS.model == 'param':
        # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_PSF_field_model(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=FLAGS.batch_size,
                                                output_Q=FLAGS.output_Q,
                                                output_dim=FLAGS.output_dim,
                                                n_zernikes=FLAGS.n_zernikes,
                                                d_max=FLAGS.d_max,
                                                x_lims=FLAGS.x_lims,
                                                y_lims=FLAGS.y_lims)


    # # Model Training
    # Prepare the saving callback
    # Prepare to save the model as a callback
    filepath_chkp_callback = FLAGS.chkp_save_path + 'chkp_callback_' + run_id_name + '_cycle1'
    model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath_chkp_callback,
        monitor='mean_squared_error', verbose=1, save_best_only=True,
        save_weights_only=False, mode='min', save_freq='epoch',
        options=None)

    # Prepare the optimisers
    param_optim = tfa.optimizers.RectifiedAdam(lr=FLAGS.l_rate_param[0])
    non_param_optim = tfa.optimizers.RectifiedAdam(lr=FLAGS.l_rate_non_param[0])

    print('Starting cycle 1..')
    start_cycle1 = time.time()

    if FLAGS.model == 'param':
        tf_semiparam_field, hist_param = wf.train_utils.param_train_cycle(
            tf_semiparam_field,
            inputs=inputs,
            outputs=outputs,
            val_data=val_data,
            batch_size=FLAGS.batch_size,
            l_rate=FLAGS.l_rate_param[0],
            n_epochs=FLAGS.n_epochs_param[0], 
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
            batch_size=FLAGS.batch_size,
            l_rate_param=FLAGS.l_rate_param[0],
            l_rate_non_param=FLAGS.l_rate_non_param[0],
            n_epochs_param=FLAGS.n_epochs_param[0],
            n_epochs_non_param=FLAGS.n_epochs_non_param[0],
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
    if FLAGS.model != 'param':
        saving_optim_hist['nonparam_cycle1'] = hist_non_param.history

    if FLAGS.total_cycles >= 2:
        # Prepare to save the model as a callback
        filepath_chkp_callback = FLAGS.chkp_save_file + 'chkp_callback_' + run_id_name + '_cycle2'
        model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath_chkp_callback,
            monitor='mean_squared_error', verbose=1, save_best_only=True,
            save_weights_only=False, mode='min', save_freq='epoch',
            options=None)

        # Prepare the optimisers
        param_optim = tfa.optimizers.RectifiedAdam(lr=FLAGS.l_rate_param[1])
        non_param_optim = tfa.optimizers.RectifiedAdam(lr=FLAGS.l_rate_non_param[1])

        print('Starting cycle 2..')
        start_cycle2 = time.time()


        # Compute the next cycle
        if FLAGS.model == 'param':
            tf_semiparam_field, hist_param_2 = wf.train_utils.param_train_cycle(
                tf_semiparam_field,
                inputs=inputs,
                outputs=outputs,
                val_data=val_data,
                batch_size=FLAGS.batch_size,
                l_rate=FLAGS.l_rate_param[1],
                n_epochs=FLAGS.n_epochs_param[1], 
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
                batch_size=FLAGS.batch_size,
                l_rate_param=FLAGS.l_rate_param[1],
                l_rate_non_param=FLAGS.l_rate_non_param[1],
                n_epochs_param=FLAGS.n_epochs_param[1],
                n_epochs_non_param=FLAGS.n_epochs_non_param[1],
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
        if FLAGS.model != 'param':
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


def evaluate_model():
    """ Evaluate the trained model."""
    # Start measuring elapsed time
    starting_time = time.time()

    # Define model run id
    run_id_name = FLAGS.model + FLAGS.id_name
    # Define paths
    log_save_file = FLAGS.base_path + FLAGS.log_folder

    # Define saved model to use
    if FLAGS.saved_model_type == 'checkpoint':
        weights_paths = FLAGS.chkp_save_path + 'chkp_callback_' + run_id_name + '_' + FLAGS.saved_cycle

    elif FLAGS.saved_model_type == 'final':
        model_save_file= FLAGS.base_path + FLAGS.model_folder
        weights_paths = model_save_file + 'chkp_' + run_id_name + '_' + FLAGS.saved_cycle

    
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
    train_dataset = np.load(FLAGS.dataset_folder + FLAGS.train_dataset_file, allow_pickle=True)[()]
    # train_stars = train_dataset['stars']
    # noisy_train_stars = train_dataset['noisy_stars']
    # train_pos = train_dataset['positions']
    train_SEDs = train_dataset['SEDs']
    # train_zernike_coef = train_dataset['zernike_coef']
    train_C_poly = train_dataset['C_poly']
    train_parameters = train_dataset['parameters']

    test_dataset = np.load(FLAGS.dataset_folder + FLAGS.test_dataset_file, allow_pickle=True)[()]
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
    zernikes = wf.utils.zernike_generator(n_zernikes=FLAGS.n_zernikes, wfe_dim=FLAGS.pupil_diameter)
    # Now as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Prepare np input
    simPSF_np = wf.SimPSFToolkit(zernikes, max_order=FLAGS.n_zernikes,
                                    pupil_diameter=FLAGS.pupil_diameter, output_dim=FLAGS.output_dim,
                                    oversampling_rate=FLAGS.oversampling_rate, output_Q=FLAGS.output_Q)
    simPSF_np.gen_random_Z_coeffs(max_order=FLAGS.n_zernikes)
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(N_pix=FLAGS.pupil_diameter, N_filter=2)
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)

    # Outputs (needed for the MCCD model)
    outputs = tf_noisy_train_stars


    ## Create the model
    ## Select the model
    if FLAGS.model == 'mccd':
        poly_dic, graph_dic = wf.tf_mccd_psf_field.build_mccd_spatial_dic_v2(obs_stars=outputs.numpy(),
                                            obs_pos=tf_train_pos.numpy(),
                                            x_lims=FLAGS.x_lims,
                                            y_lims=FLAGS.y_lims,
                                            d_max=FLAGS.d_max_nonparam,
                                            graph_features=FLAGS.graph_features)

        spatial_dic = [poly_dic, graph_dic]

        # Initialize the model
        tf_semiparam_field = wf.tf_mccd_psf_field.TF_SP_MCCD_field(zernike_maps=tf_zernike_cube,
                                                                    obscurations=tf_obscurations,
                                                                    batch_size=FLAGS.batch_size,
                                                                    obs_pos=tf_train_pos,
                                                                    spatial_dic=spatial_dic,
                                                                    output_Q=FLAGS.output_Q,
                                                                    d_max_nonparam=FLAGS.d_max_nonparam,
                                                                    graph_features=FLAGS.graph_features,
                                                                    l1_rate=FLAGS.l1_rate,
                                                                    output_dim=FLAGS.output_dim,
                                                                    n_zernikes=FLAGS.n_zernikes,
                                                                    d_max=FLAGS.d_max,
                                                                    x_lims=FLAGS.x_lims,
                                                                    y_lims=FLAGS.y_lims)

    elif FLAGS.model == 'poly':
        # # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_SemiParam_field(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=FLAGS.batch_size,
                                                output_Q=FLAGS.output_Q,
                                                d_max_nonparam=FLAGS.d_max_nonparam,
                                                output_dim=FLAGS.output_dim,
                                                n_zernikes=FLAGS.n_zernikes,
                                                d_max=FLAGS.d_max,
                                                x_lims=FLAGS.x_lims,
                                                y_lims=FLAGS.y_lims)

    elif FLAGS.model == 'param':
        # Initialize the model
        tf_semiparam_field = wf.tf_psf_field.TF_PSF_field_model(zernike_maps=tf_zernike_cube,
                                                obscurations=tf_obscurations,
                                                batch_size=FLAGS.batch_size,
                                                output_Q=FLAGS.output_Q,
                                                output_dim=FLAGS.output_dim,
                                                n_zernikes=FLAGS.n_zernikes,
                                                d_max=FLAGS.d_max,
                                                x_lims=FLAGS.x_lims,
                                                y_lims=FLAGS.y_lims)

    ## Load the model's weights
    tf_semiparam_field.load_weights(weights_paths)

    ## Prepare ground truth model
    # Generate Zernike maps
    zernikes = wf.utils.zernike_generator(n_zernikes=FLAGS.GT_n_zernikes, wfe_dim=FLAGS.pupil_diameter)
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
        batch_size=FLAGS.batch_size,
        output_Q=FLAGS.output_Q,
        d_max_nonparam=FLAGS.d_max_nonparam,
        output_dim=FLAGS.output_dim,
        n_zernikes=FLAGS.GT_n_zernikes,
        d_max=FLAGS.d_max,
        x_lims=FLAGS.x_lims,
        y_lims=FLAGS.y_lims)

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
        n_bins_lda=FLAGS.n_bins_lda,
        batch_size=FLAGS.eval_batch_size)

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
        batch_size=FLAGS.eval_batch_size)

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
        n_bins_lda=FLAGS.n_bins_lda,
        output_Q=1,
        output_dim=64,
        batch_size=FLAGS.eval_batch_size)

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
        n_bins_lda=FLAGS.n_bins_lda,
        batch_size=FLAGS.eval_batch_size)

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
        batch_size=FLAGS.eval_batch_size)

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
        n_bins_lda=FLAGS.n_bins_lda,
        output_Q=1,
        output_dim=64,
        batch_size=FLAGS.eval_batch_size)

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
    output_path = FLAGS.metric_base_path + 'metrics-' + run_id_name
    np.save(output_path, metrics, allow_pickle=True)

    ## Print final time
    final_time = time.time()
    print('\nTotal elapsed time: %f'%(final_time - starting_time))


    ## Close log file
    print('\n Good bye..')
    sys.stdout = old_stdout
    log_file.close()


    

def main(_):
    train_model()
    evaluate_model()

if __name__ == "__main__":
  app.run(main)
