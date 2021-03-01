import numpy as np
import tensorflow as tf

""" Place to temporarily save some useful functions.
"""

def first_train_cycle(tf_semiparam_field, inputs, outputs, batch_size,
                      l_rate_param, l_rate_non_param,
                      n_epochs_param, n_epochs_non_param):

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
    history_param = tf_semiparam_field.fit(x=inputs, y=outputs, batch_size=batch_size, epochs=n_epochs_param)

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
    history_non_param = tf_semiparam_field.fit(x=inputs, y=outputs, batch_size=batch_size, epochs=n_epochs_non_param)

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
                n_epochs_param, n_epochs_non_param):

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
    history_param = tf_semiparam_field.fit(x=inputs, y=outputs, batch_size=batch_size, epochs=n_epochs_param)

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
    history_non_param = tf_semiparam_field.fit(x=inputs, y=outputs, batch_size=batch_size, epochs=n_epochs_non_param)

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


def compute_metrics(tf_semiparam_field, simPSF_np, test_SEDs, train_SEDs, tf_test_pos, train_pos, n_bins_lda):
    # Generate SED data list
    test_packed_SED_data = [wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in test_SEDs]

    tf_test_packed_SED_data = tf.convert_to_tensor(test_packed_SED_data, dtype=tf.float32)
    tf_test_packed_SED_data = tf.transpose(tf_test_packed_SED_data, perm=[0, 2, 1])
    test_pred_inputs = [tf_test_pos , tf_test_packed_SED_data]
    expected_outputs = tf_test_stars

    test_predictions = tf_semiparam_field.predict(x=test_pred_inputs, batch_size=batch_size)


    # Initialize the SED data list
    packed_SED_data = [wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                    for _sed in train_SEDs]
    # First estimate the stars for the observations
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    inputs = [train_pos, tf_packed_SED_data]
    outputs = tf_train_stars

    train_predictions = tf_semiparam_field.predict(x=inputs, batch_size=batch_size)

    # Calculate RMSE values
    test_res = np.sqrt(np.mean((test_stars - test_predictions)**2))
    train_res = np.sqrt(np.mean((train_stars - train_predictions)**2))

    # Pritn RMSE values
    print('Training stars RMSE:\t %.4e'%train_res)
    print('Test stars RMSE:\t %.4e'%test_res)

    return train_res, test_res
