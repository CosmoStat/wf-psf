#!/usr/bin/env python
# coding: utf-8

# # PSF modelling


#@title Import packages
import sys
import numpy as np
import time

# Import wavefront code
import wf_psf as wf
import tensorflow as tf
import tensorflow_addons as tfa

# Start measuring elapsed time
starting_time = time.time()

# # Define saving paths
model = 'mccd'
# model = 'poly'
# model = 'param'

id_name = '-coherent_euclid_2000stars'
run_id_name = model + id_name

# Saving paths
base_path = '/gpfswork/rech/xdy/ulx23va/wf-outputs/'
log_save_file = base_path + 'log-files/'
model_save_file= base_path + 'chkp/'
optim_hist_file = base_path  + 'optim-hist/'
saving_optim_hist = dict()

chkp_save_file = '/gpfsscratch/rech/xdy/ulx23va/wf-outputs/chkp/'

# Input paths
dataset_path = '/gpfswork/rech/xdy/ulx23va/repo/wf-psf/data/coherent_euclid_dataset/'
train_path = 'train_Euclid_res_2000_TrainStars_id_001.npy'
test_path = 'test_Euclid_res_id_001.npy'


# Save output prints to logfile
old_stdout = sys.stdout
log_file = open(log_save_file + run_id_name + '_output.log','w')
sys.stdout = log_file
print('Starting the log file.')

# Check GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
print('tf_version: ' + str(tf.__version__))

# # Define new model

# Decimation factor for Zernike polynomials
n_zernikes = 15

# Some parameters
pupil_diameter = 256
n_bins_lda = 20

output_Q = 3.
oversampling_rate = 3.

batch_size = 16
output_dim = 32
d_max = 2
d_max_nonparam = 3  # polynomial-constraint features
x_lims = [0, 1e3]
y_lims = [0, 1e3]
graph_features = 10  # Graph-constraint features
l1_rate = 1e-8  # L1 regularisation

# Learning rates and number of epochs
l_rate_param = [1e-2, 1e-2]
l_rate_non_param = [1e-1, 1e-1]

n_epochs_param = [20, 20]
n_epochs_non_param = [100, 120]


## Prepare the inputs

# Generate Zernike maps
zernikes = wf.utils.zernike_generator(n_zernikes=n_zernikes, wfe_dim=pupil_diameter)

# Now as cubes
np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

for it in range(len(zernikes)):
    np_zernike_cube[it,:,:] = zernikes[it]

np_zernike_cube[np.isnan(np_zernike_cube)] = 0
tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

print('Zernike cube:')
print(tf_zernike_cube.shape)


## Load the dictionaries
train_dataset = np.load(dataset_path + train_path, allow_pickle=True)[()]
# train_stars = train_dataset['stars']
# noisy_train_stars = train_dataset['noisy_stars']
# train_pos = train_dataset['positions']
train_SEDs = train_dataset['SEDs']
# train_zernike_coef = train_dataset['zernike_coef']
train_C_poly = train_dataset['C_poly']
train_parameters = train_dataset['parameters']


test_dataset = np.load(dataset_path + test_path, allow_pickle=True)[()]
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
simPSF_np = wf.SimPSFToolkit(zernikes, max_order=n_zernikes,
                                 pupil_diameter=pupil_diameter, output_dim=output_dim,
                                 oversampling_rate=oversampling_rate, output_Q=output_Q)
simPSF_np.gen_random_Z_coeffs(max_order=n_zernikes)
z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
simPSF_np.set_z_coeffs(z_coeffs)
simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

# Obscurations
obscurations = simPSF_np.generate_pupil_obscurations(N_pix=pupil_diameter, N_filter=2)
tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)

# Initialize the SED data list
packed_SED_data = [wf.utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                   for _sed in train_SEDs]


# Prepare the inputs for the training
tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])

inputs = [tf_train_pos, tf_packed_SED_data]

# Select the observed stars (noisy or noiseless)
outputs = tf_noisy_train_stars
# outputs = tf_train_stars


## Prepare validation data inputs

# Let's take a subset of the testing data for the validation
#  in order to test things faster
val_SEDs = test_SEDs  # [0:50, :, :]
tf_val_pos = tf_test_pos  # [0:50, :]
tf_val_stars = tf_test_stars  # [0:50, :, :]

# Initialize the SED data list
val_packed_SED_data = [wf.utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                   for _sed in val_SEDs]

# Prepare the inputs for the validation
tf_val_packed_SED_data = tf.convert_to_tensor(val_packed_SED_data, dtype=tf.float32)
tf_val_packed_SED_data = tf.transpose(tf_val_packed_SED_data, perm=[0, 2, 1])
                 
# Prepare input validation tuple
val_x_inputs = [tf_val_pos, tf_val_packed_SED_data]
val_y_inputs = tf_val_stars
val_data = (val_x_inputs, val_y_inputs)


## Select the model
if model == 'mccd':
    poly_dic, graph_dic = wf.tf_mccd_psf_field.build_mccd_spatial_dic_v2(obs_stars=outputs.numpy(),
                                         obs_pos=tf_train_pos.numpy(),
                                         x_lims=x_lims,
                                         y_lims=y_lims,
                                         d_max=d_max_nonparam,
                                         graph_features=graph_features)

    spatial_dic = [poly_dic, graph_dic]

    # Initialize the model
    tf_semiparam_field = wf.tf_mccd_psf_field.TF_SP_MCCD_field(zernike_maps=tf_zernike_cube,
                                                                obscurations=tf_obscurations,
                                                                batch_size=batch_size,
                                                                obs_pos=tf_train_pos,
                                                                spatial_dic=spatial_dic,
                                                                output_Q=output_Q,
                                                                d_max_nonparam=d_max_nonparam,
                                                                graph_features=graph_features,
                                                                l1_rate=l1_rate,
                                                                output_dim=output_dim,
                                                                n_zernikes=n_zernikes,
                                                                d_max=d_max,
                                                                x_lims=x_lims,
                                                                y_lims=y_lims)

elif model == 'poly':
    # # Initialize the model
    tf_semiparam_field = wf.tf_psf_field.TF_SemiParam_field(zernike_maps=tf_zernike_cube,
                                            obscurations=tf_obscurations,
                                            batch_size=batch_size,
                                            output_Q=output_Q,
                                            d_max_nonparam=d_max_nonparam,
                                            output_dim=output_dim,
                                            n_zernikes=n_zernikes,
                                            d_max=d_max,
                                            x_lims=x_lims,
                                            y_lims=y_lims)

elif model == 'param':
    # Initialize the model
    tf_semiparam_field = wf.tf_psf_field.TF_PSF_field_model(zernike_maps=tf_zernike_cube,
                                            obscurations=tf_obscurations,
                                            batch_size=batch_size,
                                            output_dim=output_dim,
                                            n_zernikes=n_zernikes,
                                            d_max=d_max,
                                            x_lims=x_lims,
                                            y_lims=y_lims)




# # Model Training

# Prepare the saving callback
# Prepare to save the model as a callback
filepath_chkp_callback = chkp_save_file + 'chkp_callback_' + run_id_name + '_cycle1'
model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath_chkp_callback,
    monitor='mean_squared_error', verbose=1, save_best_only=True,
    save_weights_only=False, mode='min', save_freq='epoch',
    options=None)

# Prepare the optimisers
param_optim = tfa.optimizers.RectifiedAdam(lr=l_rate_param[0])
non_param_optim = tfa.optimizers.RectifiedAdam(lr=l_rate_non_param[0])

print('Starting cycle 1..')
start_cycle1 = time.time()

tf_semiparam_field, hist_param, hist_non_param = wf.train_utils.general_train_cycle(
    tf_semiparam_field,
    inputs=inputs,
    outputs=outputs,
    val_data=val_data,
    batch_size=batch_size,
    l_rate_param=l_rate_param[0],
    l_rate_non_param=l_rate_non_param[0],
    n_epochs_param=n_epochs_param[0],
    n_epochs_non_param=n_epochs_non_param[0],
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
saving_optim_hist['nonparam_cycle1'] = hist_non_param.history




# Prepare to save the model as a callback
filepath_chkp_callback = chkp_save_file + 'chkp_callback_' + run_id_name + '_cycle2'
model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath_chkp_callback,
    monitor='mean_squared_error', verbose=1, save_best_only=True,
    save_weights_only=False, mode='min', save_freq='epoch',
    options=None)

# Prepare the optimisers
param_optim = tfa.optimizers.RectifiedAdam(lr=l_rate_param[1])
non_param_optim = tfa.optimizers.RectifiedAdam(lr=l_rate_non_param[1])

print('Starting cycle 2..')
start_cycle2 = time.time()

# Compute the next cycle
tf_semiparam_field, hist_param_2, hist_non_param_2 = wf.train_utils.general_train_cycle(
    tf_semiparam_field,
    inputs=inputs,
    outputs=outputs,
    val_data=val_data,
    batch_size=batch_size,
    l_rate_param=l_rate_param[1],
    l_rate_non_param=l_rate_non_param[1],
    n_epochs_param=n_epochs_param[1],
    n_epochs_non_param=n_epochs_non_param[1],
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
