import numpy as np
import tensorflow as tf
from wf_psf.psf_models.tf_psf_field import build_PSF_model
from wf_psf.utils.utils import NoiseEstimator


class L1ParamScheduler(tf.keras.callbacks.Callback):
    """L1 rate scheduler which sets the L1 rate according to schedule.

    Parameters
    ----------
      l1_schedule_rule: function
        a function that takes an epoch index
        (integer, indexed from 0) and current l1_rate
          as inputs and returns a new l1_rate as output (float).
    """

    def __init__(self, l1_schedule_rule):
        super(L1ParamScheduler, self).__init__()
        breakpoint()
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
    if epoch_n != 0 and epoch_n % 10 == 0:
        scheduled_l1_rate = l1_rate / 2
        print("\nEpoch %05d: L1 rate is %0.4e." % (epoch_n, scheduled_l1_rate))
        return scheduled_l1_rate
    else:
        return l1_rate


def general_train_cycle(
    tf_semiparam_field,
    inputs,
    outputs,
    validation_data,
    batch_size,
    learning_rate_param,
    learning_rate_non_param,
    n_epochs_param,
    n_epochs_non_param,
    param_optim=None,
    non_param_optim=None,
    param_loss=None,
    non_param_loss=None,
    param_metrics=None,
    non_param_metrics=None,
    param_callback=None,
    non_param_callback=None,
    general_callback=None,
    first_run=False,
    cycle_def="complete",
    use_sample_weights=False,
    verbose=1,
):
    """Function to do a BCD iteration on the model.

    Define the model optimisation.

    For the parametric part we are using:
    ``learning_rate_param = 1e-2``, ``n_epochs_param = 20``.
    For the non-parametric part we are using:
    ``learning_rate_non_param = 1.0``, ``n_epochs_non_param = 100``.

    Parameters
    ----------
    tf_semiparam_field: tf.keras.Model
        The model to be trained.
    inputs: Tensor or list of tensors
        Inputs used for Model.fit()
    outputs: Tensor
        Outputs used for Model.fit()
    validation_data: Tuple
        Validation test data used for Model.fit().
        Tuple of input, output validation data
    batch_size: int
        Batch size for the training.
    learning_rate_param: float
        Learning rate for the parametric part
    learning_rate_non_param: float
        Learning rate for the non-parametric part
    n_epochs_param: int
        Number of epochs for the parametric part
    n_epochs_non_param: int
        Number of epochs for the non-parametric part
    param_optim: Tensorflow optimizer
        Optimizer for the parametric part.
        Optional, default is the Adam optimizer
    non_param_optim: Tensorflow optimizer
        Optimizer for the non-parametric part.
        Optional, default is the Adam optimizer
    param_loss: Tensorflow loss
        Loss function for the parametric part.
        Optional, default is the MeanSquaredError() loss
    non_param_loss: Tensorflow loss
        Loss function for the non-parametric part.
        Optional, default is the MeanSquaredError() loss
    param_metrics: Tensorflow metrics
        Metrics for the parametric part.
        Optional, default is the MeanSquaredError() metric
    non_param_metrics: Tensorflow metrics
        Metrics for the non-parametric part.
        Optional, default is the MeanSquaredError() metric
    param_callback: Tensorflow callback
        Callback for the parametric part only.
        Optional, default is no callback
    non_param_callback: Tensorflow callback
        Callback for the non-parametric part only.
        Optional, default is no callback
    general_callback: Tensorflow callback
        Callback shared for both the parametric and non-parametric parts.
        Optional, default is no callback
    first_run: bool
        If True, it is the first iteration of the model training.
        The Non-parametric part is not considered in the first parametric training.
    cycle_def: string
        Train cycle definition. It can be: `parametric`, `non-parametric`, `complete`.
        Default is `complete`.
    use_sample_weights: bool
        If True, the sample weights are used for the training.
        The sample weights are computed as the inverse noise estimated variance
    verbose: int
        Verbosity mode used for the training procedure.
        If a log of the training is being saved, `verbose=2` is recommended.

    Returns
    -------
    tf_semiparam_field: tf.keras.Model
        Trained Tensorflow model.
    hist_param: Tensorflow's History object
        History of the parametric training.
    hist_non_param: Tensorflow's History object
        History of the non-parametric training.

    """
    # Initialize return variables
    hist_param = None
    hist_non_param = None

    # Parametric train

    # Define Loss
    if param_loss is None:
        loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = param_loss

    # Define optimisers
    if param_optim is None:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_param,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
        )
    else:
        optimizer = param_optim

    # Define metrics
    if param_metrics is None:
        metrics = [tf.keras.metrics.MeanSquaredError()]
    else:
        metrics = param_metrics

    # Define callbacks
    if param_callback is None and general_callback is None:
        callbacks = None
    else:
        if general_callback is None:
            callbacks = param_callback
        elif param_callback is None:
            callbacks = general_callback
        else:
            callbacks = general_callback + param_callback

    # Calculate sample weights
    if use_sample_weights:
        # Generate standard deviation estimator
        img_dim = (outputs.shape[1], outputs.shape[2])
        win_rad = np.ceil(outputs.shape[1] / 3.33)
        std_est = NoiseEstimator(img_dim=img_dim, win_rad=win_rad)
        # Estimate noise std_dev
        imgs_std = np.array([std_est.estimate_noise(_im) for _im in outputs])
        # Calculate weights
        variances = imgs_std**2

        # Define sample weight strategy
        strategy_opt = 1

        if strategy_opt == 0:
            # Parameters
            max_w = 2.0
            min_w = 0.1
            # Epsilon is to avoid outliers
            epsilon = np.median(variances) * 0.1
            w = 1 / (variances + epsilon)
            scaled_w = (w - np.min(w)) / (np.max(w) - np.min(w))  # Transform to [0,1]
            scaled_w = scaled_w * (max_w - min_w) + min_w  # Transform to [min_w, max_w]
            scaled_w = scaled_w + (1 - np.mean(scaled_w))  # Adjust the mean to 1
            scaled_w[scaled_w < min_w] = min_w
            # Save the weights
            sample_weight = scaled_w

        elif strategy_opt == 1:
            # Use inverse variance for weights
            # Then scale the values by the median
            sample_weight = 1 / variances
            sample_weight /= np.median(sample_weight)
    else:
        sample_weight = None

    # Define the training cycle
    if (
        cycle_def == "parametric"
        or cycle_def == "complete"
        or cycle_def == "only-parametric"
    ):
        # If it is the first run
        if first_run:
            # Set the non-parametric model to zero
            # With alpha to zero its already enough
            tf_semiparam_field.set_zero_nonparam()
        if cycle_def == "only-parametric":
            # Set the non-parametric part to zero
            tf_semiparam_field.set_zero_nonparam()

        # Set the trainable layer
        tf_semiparam_field.set_trainable_layers(param_bool=True, nonparam_bool=False)

        # Compile the model for the first optimisation
        tf_semiparam_field = build_PSF_model(
            tf_semiparam_field,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        # Train the parametric part
        print("Starting parametric update..")
        hist_param = tf_semiparam_field.fit(
            x=inputs,
            y=outputs,
            batch_size=batch_size,
            epochs=n_epochs_param,
            validation_data=validation_data,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=verbose,
        )

    ## Non parametric train
    # Define the training cycle
    if (
        cycle_def == "non-parametric"
        or cycle_def == "complete"
        or cycle_def == "only-non-parametric"
    ):
        # If it is the first run
        if first_run:
            # Set the non-parametric model to non-zero
            # With alpha to zero its already enough
            tf_semiparam_field.set_nonzero_nonparam()
        if cycle_def == "only-non-parametric":
            # Set the parametric layer to zero
            coeff_mat = tf_semiparam_field.get_coeff_matrix()
            tf_semiparam_field.assign_coeff_matrix(tf.zeros_like(coeff_mat))

        # Set the non parametric layer to non trainable
        tf_semiparam_field.set_trainable_layers(param_bool=False, nonparam_bool=True)

        # Define Loss
        if non_param_loss is None:
            loss = tf.keras.losses.MeanSquaredError()
        else:
            loss = non_param_loss

        # Define optimiser
        if non_param_optim is None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_non_param,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
            )
        else:
            optimizer = non_param_optim

        # Define metric
        if non_param_metrics is None:
            metrics = [tf.keras.metrics.MeanSquaredError()]
        else:
            metrics = non_param_metrics

        # Define callbacks
        if non_param_callback is None and general_callback is None:
            callbacks = None
        else:
            if general_callback is None:
                callbacks = non_param_callback
            elif non_param_callback is None:
                callbacks = general_callback
            else:
                callbacks = general_callback + non_param_callback

        # Compile the model again for the second optimisation
        tf_semiparam_field = build_PSF_model(
            tf_semiparam_field,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        # Train the nonparametric part
        print("Starting non-parametric update..")
        hist_non_param = tf_semiparam_field.fit(
            x=inputs,
            y=outputs,
            batch_size=batch_size,
            epochs=n_epochs_non_param,
            validation_data=validation_data,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=verbose,
        )

    return tf_semiparam_field, hist_param, hist_non_param


def param_train_cycle(
    tf_semiparam_field,
    inputs,
    outputs,
    validation_data,
    batch_size,
    learning_rate,
    n_epochs,
    param_optim=None,
    param_loss=None,
    param_metrics=None,
    param_callback=None,
    general_callback=None,
    use_sample_weights=False,
    verbose=1,
):
    """Training cycle for parametric model."""
    # Define Loss
    if param_loss is None:
        loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = param_loss

    # Define optimiser
    if param_optim is None:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        )
    else:
        optimizer = param_optim

    # Define metrics
    if param_metrics is None:
        metrics = [tf.keras.metrics.MeanSquaredError()]
    else:
        metrics = param_metrics

    # Define callbacks
    if param_callback is None and general_callback is None:
        callbacks = None
    else:
        if general_callback is None:
            callbacks = param_callback
        elif param_callback is None:
            callbacks = general_callback
        else:
            callbacks = general_callback + param_callback

        # Calculate sample weights
    if use_sample_weights:
        # Generate standard deviation estimator
        img_dim = (outputs.shape[1], outputs.shape[2])
        win_rad = np.ceil(outputs.shape[1] / 3.33)
        std_est = NoiseEstimator(img_dim=img_dim, win_rad=win_rad)
        # Estimate noise std_dev
        imgs_std = np.array([std_est.estimate_noise(_im) for _im in outputs])
        # Calculate weights
        variances = imgs_std**2

        strategy_opt = 1

        if strategy_opt == 0:
            # Parameters
            max_w = 2.0
            min_w = 0.1
            # Epsilon is to avoid outliers
            epsilon = np.median(variances) * 0.1
            w = 1 / (variances + epsilon)
            scaled_w = (w - np.min(w)) / (np.max(w) - np.min(w))  # Transform to [0,1]
            scaled_w = scaled_w * (max_w - min_w) + min_w  # Transform to [min_w, max_w]
            scaled_w = scaled_w + (1 - np.mean(scaled_w))  # Adjust the mean to 1
            scaled_w[scaled_w < min_w] = min_w
            # Save the weights
            sample_weight = scaled_w

        elif strategy_opt == 1:
            # Use inverse variance for weights
            # Then scale the values by the median
            sample_weight = 1 / variances
            sample_weight /= np.median(sample_weight)

    else:
        sample_weight = None

    # Compile the model for the first optimisation
    tf_semiparam_field = build_PSF_model(
        tf_semiparam_field, optimizer=optimizer, loss=loss, metrics=metrics
    )

    # Train the parametric part
    print("Starting parametric update..")
    hist_param = tf_semiparam_field.fit(
        x=inputs,
        y=outputs,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        sample_weight=sample_weight,
        verbose=verbose,
    )

    return tf_semiparam_field, hist_param
