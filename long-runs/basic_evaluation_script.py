#!/usr/bin/env python
# coding: utf-8

## PSF modelling evaluation

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import wf_psf as wf
import tensorflow as tf


## Start measuring elapsed time
starting_time = time.time()


## Define saving paths
model = 'mccd'
# model = 'poly'
# model = 'param'

id_name = '-coherent_euclid_200stars'
run_id_name = model + id_name

# Saving folder
log_save_file = '/local/home/tliaudat/checkpoints/coherent-dataset/log-files/'
model_folder = '/local/home/tliaudat/checkpoints/coherent-dataset/chkp/'

# Input paths
dataset_path = '/local/home/tliaudat/github/wf-psf/data/coherent_euclid_dataset/'
train_path = 'train_Euclid_res_200_TrainStars_id_001.npy'
test_path = 'test_Euclid_res_id_001.npy'


# Saving path
saving_path = '/local/home/tliaudat/checkpoints/coherent-dataset/metrics/'


## Model parameters
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


## Save output prints to logfile
old_stdout = sys.stdout
log_file = open(log_save_file + run_id_name + '-metrics_output.log', 'w')
sys.stdout = log_file
print('Starting the log file.')


## Check GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
print('tf_version: ' + str(tf.__version__))


## Load datasets
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


## Prepare models
# Generate Zernike maps
zernikes = wf.utils.zernike_generator(n_zernikes=n_zernikes, wfe_dim=pupil_diameter)
# Now as cubes
np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

for it in range(len(zernikes)):
    np_zernike_cube[it,:,:] = zernikes[it]

np_zernike_cube[np.isnan(np_zernike_cube)] = 0
tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

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

# Outputs (needed for the MCCD model)
outputs = tf_noisy_train_stars


## Create the model
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


## Load the model's weights
tf_semiparam_field.load_weights(model_folder + run_id_name)


## Prepare ground truth model
n_zernikes_bis = 45
# Generate Zernike maps
zernikes = wf.utils.zernike_generator(n_zernikes=n_zernikes_bis, wfe_dim=pupil_diameter)
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
    batch_size=batch_size,
    output_Q=output_Q,
    d_max_nonparam=d_max_nonparam,
    output_dim=output_dim,
    n_zernikes=n_zernikes_bis,
    d_max=d_max,
    x_lims=x_lims,
    y_lims=y_lims)

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
    n_bins_lda=n_bins_lda,
    batch_size=batch_size)

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
    batch_size=batch_size)

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
    n_bins_lda=n_bins_lda,
    output_Q=1,
    output_dim=64,
    batch_size=batch_size)

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
    n_bins_lda=n_bins_lda,
    batch_size=batch_size)

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
    batch_size=batch_size)

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
    n_bins_lda=n_bins_lda,
    output_Q=1,
    output_dim=64,
    batch_size=batch_size)

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
output_path = saving_path + 'metrics-' + run_id_name
np.save(output_path, metrics, allow_pickle=True)


## Print final time
final_time = time.time()
print('\nTotal elapsed time: %f'%(final_time - starting_time))


## Close log file
print('\n Good bye..')
sys.stdout = old_stdout
log_file.close()

