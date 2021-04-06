import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from wf_psf.utils import generate_packed_elems
from wf_psf.tf_psf_field import build_PSF_model

""" Place to temporarily save some useful functions.
"""

class L1ParamScheduler(tf.keras.callbacks.Callback):
    """L1 rate scheduler which sets the L1 rate according to schedule.

  Arguments:
      l1_schedule_rule: a function that takes an epoch index
          (integer, indexed from 0) and current l1_rate
          as inputs and returns a new l1_rate as output (float).
  """

    def __init__(self, l1_schedule_rule):
        super(L1ParamScheduler, self).__init__()
        self.l1_schedule_rule = l1_schedule_rule

    def on_epoch_begin(self, epoch, logs=None):
        # Get the current learning rate from model's optimizer.
        l1_rate = float(tf.keras.backend.get_value(self.model.l1_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_l1_rate = self.l1_schedule_rule(epoch, l1_rate)
        # Set the value back to the optimizer before this epoch starts
        self.model.set_l1_rate(scheduled_l1_rate)
        # tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

def l1_schedule_rule(epoch_n, l1_rate):
    if epoch_n!= 0 and epoch_n%10 == 0:
        scheduled_l1_rate = l1_rate/2
        print("\nEpoch %05d: L1 rate is %0.4e." % (epoch_n, scheduled_l1_rate))
        return scheduled_l1_rate
    else:
        return l1_rate


def first_train_cycle(tf_semiparam_field, inputs, outputs, batch_size,
                      l_rate_param, l_rate_non_param,
                      n_epochs_param, n_epochs_non_param,
                      param_callback=None, non_param_callback=None):

    ## First parametric train

    # Define the model optimisation
    # l_rate_param = 1e-2
    # n_epochs_param = 20

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=l_rate_param, beta_1=0.9, beta_2=0.999,
        epsilon=1e-07, amsgrad=False)
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Set the non-parametric model to zero
    # With alpha to zero its already enough
    tf_semiparam_field.set_zero_nonparam()

    # Set the non parametric layer to non trainable
    # And keep the parametric layer to trainable
    tf_semiparam_field.set_trainable_layers(param_bool=True, nonparam_bool=False)


    # Compile the model for the first optimisation
    tf_semiparam_field = build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                           batch_size=batch_size,
                                           epochs=n_epochs_param,
                                           callbacks=param_callback)

    # Plot losses
    plt.figure()
    plt.subplot(211)
    plt.plot(history_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.subplot(212)
    plt.loglog(history_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.show()


    ## Non parametric train
    # Set the non-parametric model to non-zero
    # With alpha to zero its already enough
    tf_semiparam_field.set_nonzero_nonparam()

    # Set the non parametric layer to non trainable
    # Set the parametric layer to non trainable
    tf_semiparam_field.set_trainable_layers(param_bool=False, nonparam_bool=True)


    # Non parametric parameters
    # l_rate_non_param = 1.0
    # n_epochs_non_param = 100

    # Define the model optimisation
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=l_rate_non_param, beta_1=0.9, beta_2=0.999,
        epsilon=1e-07, amsgrad=False)
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Compile the model again for the second optimisation
    tf_semiparam_field = build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_non_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                               batch_size=batch_size,
                                               epochs=n_epochs_non_param,
                                               callbacks=non_param_callback)

    # Plot losses
    plt.figure()
    plt.subplot(211)
    plt.plot(history_non_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.subplot(212)
    plt.loglog(history_non_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.show()

    return tf_semiparam_field


def train_cycle(tf_semiparam_field, inputs, outputs, batch_size,
                l_rate_param, l_rate_non_param,
                n_epochs_param, n_epochs_non_param,
                param_callback=None, non_param_callback=None):

    ## Parametric train

    # Define the model optimisation
    # l_rate_param = 1e-2
    # n_epochs_param = 20

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=l_rate_param, beta_1=0.9, beta_2=0.999,
        epsilon=1e-07, amsgrad=False)
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Set the trainable layer
    tf_semiparam_field.set_trainable_layers(param_bool=True, nonparam_bool=False)

    # Compile the model for the first optimisation
    tf_semiparam_field = build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                           batch_size=batch_size,
                                           epochs=n_epochs_param,
                                           callbacks=param_callback)

    # Plot losses
    plt.figure()
    plt.subplot(211)
    plt.plot(history_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.subplot(212)
    plt.loglog(history_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.show()


    ## Non parametric train
    # Set the non parametric layer to non trainable
    tf_semiparam_field.set_trainable_layers(param_bool=False, nonparam_bool=True)

    # Non parametric parameters
    # l_rate_non_param = 1.0
    # n_epochs_non_param = 100

    # Define the model optimisation
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=l_rate_non_param, beta_1=0.9, beta_2=0.999,
        epsilon=1e-07, amsgrad=False)
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Compile the model again for the second optimisation
    tf_semiparam_field = build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_non_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                               batch_size=batch_size,
                                               epochs=n_epochs_non_param,
                                               callbacks=non_param_callback)

    # Plot losses
    plt.figure()
    plt.subplot(211)
    plt.plot(history_non_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.subplot(212)
    plt.loglog(history_non_param.history['loss'])
    plt.xlabel('Number of iterations')
    plt.ylabel('Total loss');
    plt.show()

    return tf_semiparam_field


def compute_metrics(tf_semiparam_field, simPSF_np, test_SEDs, train_SEDs,
                    tf_test_pos, tf_train_pos, tf_test_stars, tf_train_stars,
                    n_bins_lda, batch_size=16):
    # Generate SED data list
    test_packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in test_SEDs]

    tf_test_packed_SED_data = tf.convert_to_tensor(test_packed_SED_data, dtype=tf.float32)
    tf_test_packed_SED_data = tf.transpose(tf_test_packed_SED_data, perm=[0, 2, 1])
    test_pred_inputs = [tf_test_pos , tf_test_packed_SED_data]
    test_predictions = tf_semiparam_field.predict(x=test_pred_inputs, batch_size=batch_size)


    # Initialize the SED data list
    packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                    for _sed in train_SEDs]
    # First estimate the stars for the observations
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    inputs = [tf_train_pos, tf_packed_SED_data]
    train_predictions = tf_semiparam_field.predict(x=inputs, batch_size=batch_size)

    # Calculate RMSE values
    test_res = np.sqrt(np.mean((tf_test_stars - test_predictions)**2))
    train_res = np.sqrt(np.mean((tf_train_stars - train_predictions)**2))

    # Pritn RMSE values
    print('Test stars RMSE:\t %.4e'%test_res)
    print('Training stars RMSE:\t %.4e'%train_res)


    return test_res, train_res

def compute_opd_metrics(tf_semiparam_field, GT_tf_semiparam_field, test_pos,
                        train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Test stars OPD RMSE:\t %.4e'%test_opd_rmse)


    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Train stars OPD RMSE:\t %.4e'%train_opd_rmse)

    return test_opd_rmse, train_opd_rmse


def compute_opd_metrics_polymodel(tf_semiparam_field, GT_tf_semiparam_field,
                                  test_pos, train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_np_poly_opd(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Test stars OPD RMSE:\t %.4e'%test_opd_rmse)


    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_np_poly_opd(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Train stars OPD RMSE:\t %.4e'%train_opd_rmse)

    return test_opd_rmse, train_opd_rmse



def compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, pos, is_poly=False):
    """ Compute the OPD map for one position!. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    tf_pos = tf.convert_to_tensor(pos, dtype=tf.float32)

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(tf_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    if is_poly == False:
        NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(tf_pos)
    else:
        NP_opd_pred =  tf_semiparam_field.tf_np_poly_opd(tf_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(tf_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    opd_rmse = np.sqrt(np.mean(res_opd**2))

    return opd_rmse

def plot_function(mesh_pos, residual, tf_train_pos, tf_test_pos, title='Error'):
    vmax = np.max(residual)
    vmin = np.min(residual)

    plt.figure(figsize=(12,8))
    plt.scatter(mesh_pos[:,0], mesh_pos[:,1], s=100, c=residual.reshape(-1,1), cmap='viridis', marker='s', vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.scatter(tf_train_pos[:,0], tf_train_pos[:,1], c='k', marker='*', s=10, label='Train stars')
    plt.scatter(tf_test_pos[:,0], tf_test_pos[:,1], c='r', marker='*', s=10, label='Test stars')
    plt.title(title)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()


def plot_residual_maps(GT_tf_semiparam_field, tf_semiparam_field, simPSF_np, train_SEDs,
                       tf_train_pos, tf_test_pos, n_bins_lda=20, n_points_per_dim=30,
                       is_poly=False):

    # Recover teh grid limits
    x_lims = tf_semiparam_field.x_lims
    y_lims = tf_semiparam_field.y_lims

    # Generate mesh of testing positions
    x = np.linspace(x_lims[0], x_lims[1], n_points_per_dim)
    y = np.linspace(y_lims[0], y_lims[1], n_points_per_dim)
    x_pos, y_pos = np.meshgrid(x, y)

    mesh_pos = np.concatenate((x_pos.flatten().reshape(-1,1), y_pos.flatten().reshape(-1,1)), axis=1)
    tf_mesh_pos = tf.convert_to_tensor(mesh_pos, dtype=tf.float32)

    # Testing the positions
    rec_x_pos = mesh_pos[:,0].reshape(x_pos.shape)
    rec_y_pos = mesh_pos[:,1].reshape(y_pos.shape)

    # Get random SED from the training catalog
    SED_random_integers = np.random.choice(np.arange(train_SEDs.shape[0]), size=mesh_pos.shape[0], replace=True)
    # Build the SED catalog for the testing mesh
    mesh_SEDs = np.array([train_SEDs[_id,:,:] for _id in SED_random_integers])


    # Generate SED data list
    mesh_packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in mesh_SEDs]

    # Generate inputs
    tf_mesh_packed_SED_data = tf.convert_to_tensor(mesh_packed_SED_data, dtype=tf.float32)
    tf_mesh_packed_SED_data = tf.transpose(tf_mesh_packed_SED_data, perm=[0, 2, 1])
    mesh_pred_inputs = [tf_mesh_pos , tf_mesh_packed_SED_data]


    # Predict mesh stars
    model_mesh_preds = tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)
    GT_mesh_preds = GT_tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)

    # Calculate pixel RMSE for each star
    pix_rmse = np.array([np.sqrt(np.mean((_GT_pred-_model_pred)**2))
                            for _GT_pred, _model_pred  in zip(GT_mesh_preds, model_mesh_preds)])

    relative_pix_rmse = np.array([np.sqrt(np.mean((_GT_pred-_model_pred)**2))/np.sqrt(np.mean((_GT_pred)**2))
                                    for _GT_pred, _model_pred  in zip(GT_mesh_preds, model_mesh_preds)])

    # Plot absolute pixel error
    plot_function(mesh_pos, pix_rmse, tf_train_pos, tf_test_pos, title='Absolute pixel error')
    # Plot relative pixel error
    plot_function(mesh_pos, relative_pix_rmse, tf_train_pos, tf_test_pos, title='Relative pixel error')

    # Compute OPD errors
    opd_rmse = np.array([compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, _pos.reshape(1,-1), is_poly) for _pos in mesh_pos])

    # Plot absolute pixel error
    plot_function(mesh_pos, opd_rmse, tf_train_pos, tf_test_pos, title='Absolute OPD error')


# Not to loose this lines of code..
def GT_model():
    # Preparate the GT model

    Zcube = sio.loadmat(Zcube_path)
    zernikes = []
    # Decimation factor for Zernike polynomials
    decim_f = 4  # Original shape (1024x1024)

    n_zernikes_bis = 45

    for it in range(n_zernikes_bis):
        zernike_map = wf_utils.decimate_im(Zcube['Zpols'][0,it][5], decim_f)
        zernikes.append(zernike_map)

    # Now as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0

    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    print('Zernike cube:')
    print(tf_zernike_cube.shape)


    # Initialize the model
    GT_tf_semiparam_field = wf_psf_field.TF_SemiParam_field(
                                            zernike_maps=tf_zernike_cube,
                                            obscurations=tf_obscurations,
                                            batch_size=batch_size,
                                            d_max_nonparam=d_max_nonparam,
                                            output_dim=output_dim,
                                            n_zernikes=n_zernikes_bis,
                                            d_max=d_max,
                                            x_lims=x_lims,
                                            y_lims=y_lims)


    # For the Ground truth model
    GT_tf_semiparam_field.tf_poly_Z_field.assign_coeff_matrix(train_C_poly)
    _ = GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat.assign(np.zeros_like(GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat))