import numpy as np
import tensorflow as tf

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
    tf_semiparam_field = wf_psf_field.build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                           batch_size=batch_size,
                                           epochs=n_epochs_param,
                                           callbacks=param_callback)

    # Plot losses
    figure()
    subplot(211)
    plot(history_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    subplot(212)
    loglog(history_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    show()


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
    tf_semiparam_field = wf_psf_field.build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_non_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                               batch_size=batch_size,
                                               epochs=n_epochs_non_param,
                                               callbacks=non_param_callback)

    # Plot losses
    figure()
    subplot(211)
    plot(history_non_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    subplot(212)
    loglog(history_non_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    show()

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
    tf_semiparam_field = wf_psf_field.build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                           batch_size=batch_size,
                                           epochs=n_epochs_param,
                                           callbacks=param_callback)

    # Plot losses
    figure()
    subplot(211)
    plot(history_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    subplot(212)
    loglog(history_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    show()


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
    tf_semiparam_field = wf_psf_field.build_PSF_model(tf_semiparam_field, optimizer=optimizer,
                                                    loss=loss, metrics=metrics)

    # Train the parametric part
    history_non_param = tf_semiparam_field.fit(x=inputs, y=outputs,
                                               batch_size=batch_size,
                                               epochs=n_epochs_non_param,
                                               callbacks=non_param_callback)

    # Plot losses
    figure()
    subplot(211)
    plot(history_non_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    subplot(212)
    loglog(history_non_param.history['loss'])
    xlabel('Number of iterations')
    ylabel('Total loss');
    show()

    return tf_semiparam_field

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
