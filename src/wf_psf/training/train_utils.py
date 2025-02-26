"""
Training utilities for the PSF model.

This module contains helper functions and utilities related to the training 
process for the PSF model. These functions help with managing training cycles, 
callbacks, and related operations.

Author: Tobias Liaudat <tobias.liaudat@cea.fr>
"""


import numpy as np
import tensorflow as tf
from typing import Optional, Callable
from wf_psf.psf_models.psf_models import build_PSF_model
from wf_psf.utils.utils import NoiseEstimator
import logging

# from keras.src import backend
# import tf.keras.ops.convert_to_tensor as convert_to_tensor
# from keras.src import tree
# from keras.src.losses import reduce_weighted_values

logger = logging.getLogger(__name__)


class L1ParamScheduler(tf.keras.callbacks.Callback):
    """L1 rate scheduler that adjusts the L1 rate during training according to a specified schedule.

    This callback modifies the L1 regularization rate at each epoch based on the given scheduling function. 
    The function takes the epoch index and the current L1 rate as inputs, and it outputs the updated L1 rate.

    Parameters
    ----------
    l1_schedule_rule: function
        A function that defines how to update the L1 rate. The function should take two arguments:
        - `epoch` (int): The current epoch index, starting from 0.
        - `current_l1_rate` (float): The L1 rate at the current epoch.
        
        The function should return a new L1 rate (float) to be applied at the next epoch.

    Example
    -------
    def schedule_fn(epoch, current_l1_rate):
        # Example schedule function
        return current_l1_rate * 0.95  # Decaying the rate by 5% every epoch

    l1_scheduler = L1ParamScheduler(l1_schedule_rule=schedule_fn)
    """

    def __init__(self, l1_schedule_rule):
        """
        Initialize the L1ParamScheduler.

        Parameters
        ----------
        l1_schedule_rule : function
            A function that defines how to update the L1 rate at each epoch. See class docstring for details.
        """
        super().__init__()
        self.l1_schedule_rule = l1_schedule_rule

    def on_epoch_begin(self, epoch, logs=None):
        """
        Execute callback function at the beginning of each epoch to adjust the L1 rate.

        This function gets the current L1 rate from the model's optimizer, computes the new scheduled
        L1 rate using the `l1_schedule_rule` function, and sets it back to the model's optimizer.

        Parameters
        ----------
        epoch: int
            The current epoch index, starting from 0.
        logs: dict, optional
            A dictionary containing logs for the current epoch (default is None).
        """
        # Get the current learning rate from model's optimizer.
        l1_rate = float(tf.keras.backend.get_value(self.model.l1_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_l1_rate = self.l1_schedule_rule(epoch, l1_rate)
        # Set the value back to the optimizer before this epoch starts
        self.model.set_l1_rate(scheduled_l1_rate)
        # tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

def masked_mse(y_true, y_pred, mask):
    """Masked Mean Squared Error.

    Parameters
    ----------
    y_true: Tensor
        True values
    y_pred: Tensor
        Predicted values
    mask: Tensor
        Mask to be applied

    Returns
    -------
    Tensor
        Masked Mean Squared Error
    """
    # Calculate the MSE
    error = tf.square(y_true - y_pred)
    masked_error = error * mask
    return tf.reduce_mean(masked_error)

class MaskedMeanSquaredError(tf.keras.losses.Loss):
    """Masked Mean Squared Error.""" 
    def __init__(self, name="masked_mean_squared_error", **kwargs):
        super().__init__(name=name, **kwargs)

    # def __call__(self, y_true, y_pred, sample_weight=None):
    #     in_mask = backend.get_keras_mask(y_pred)
    #     print('y_true', y_true.shape)
    #     with ops.name_scope(self.name):
    #         y_pred = tree.map_structure(
    #             lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred
    #         )
    #         y_true = tree.map_structure(
    #             lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true
    #         )

    #         losses = self.call(y_true, y_pred)
    #         out_mask = backend.get_keras_mask(losses)

    #         if in_mask is not None and out_mask is not None:
    #             print("in+out")
    #             mask = in_mask & out_mask
    #         elif in_mask is not None:
    #             print("in")
    #             mask = in_mask
    #         elif out_mask is not None:
    #             print("out")
    #             mask = out_mask
    #         else:
    #             print("none")
    #             mask = None

    #         return reduce_weighted_values(
    #             losses,
    #             sample_weight=sample_weight,
    #             mask=mask,
    #             reduction=self.reduction,
    #             dtype=self.dtype,
    #         )

    def call(self, y_true, y_pred, sample_weight=None):
        # if sample_weight is None:
        #     raise ValueError("Sample weights are required for MaskedMeanSquaredError")
        
        # return masked_mse(y_true, y_pred, sample_weight)
        print("y_true", y_true.shape)
        print("y_pred", y_pred.shape)
        y_target = y_true[:,:, :y_true.shape[1]]
        mask = y_true[:,:, y_true.shape[1]:]
        print("y_target", y_target.shape)
        print("mask", mask.shape)
        return masked_mse(y_target, y_pred, mask)
    
class MaskedMeanSquaredErrorMetric(tf.keras.metrics.Metric):
    def __init__(self, name="masked_mean_squared_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # if sample_weight is None:
        #     raise ValueError("Sample weights are required for MaskedMeanSquaredErrorMetric")
        
        # loss = masked_mse(y_true, y_pred, sample_weight)
        y_target = y_true[:,:, :y_true.shape[1]]
        mask = y_true[:,:, y_true.shape[1]:]
        loss = masked_mse(y_target, y_pred, mask)
        self.total_loss.assign_add(loss)
        self.batch_count.assign_add(1.0)

    def result(self):
        return self.total_loss / self.batch_count
    
    def reset_state(self):
        self.total_loss.assign(0.0)
        self.batch_count.assign(0.0)


def l1_schedule_rule(epoch_n: int, l1_rate: float) -> float:
    """
    Schedules the L1 rate based on the epoch number.

    If the current epoch is a multiple of 10 (except for the first epoch), 
    the L1 rate is halved. Otherwise, the L1 rate remains unchanged.

    Parameters
    ----------
    epoch_n: int
        The current epoch number, where the epoch index starts from 0.
    l1_rate: float
        The current L1 regularization rate.

    Returns
    -------
    float
        The updated L1 rate for the given epoch.
        
    Example
    -------
    For `epoch_n = 10` and `l1_rate = 0.01`, the function returns `0.005`.
    """
    if epoch_n != 0 and epoch_n % 10 == 0:
        scheduled_l1_rate = l1_rate / 2
        logger.info(f"Epoch {epoch_n:05d}: L1 rate is {scheduled_l1_rate:0.4e}.")
        return scheduled_l1_rate
    return l1_rate

def configure_optimizer_and_loss(
    learning_rate: float,
    optimizer: Optional[Callable] = None,
    loss: Optional[Callable] = None,
    metrics: Optional[list[Callable]] = None,
    is_parametric: bool = True
) -> tuple[Callable, Callable, list[Callable]]:
    """
    Configure and return the optimizer, loss function, and metrics for model training.

    This function configures the optimizer, loss function, and metrics for either the parametric 
    or non-parametric model components. If no optimizer, loss, or metrics are provided, 
    default values are used.

    Parameters
    ----------
    learning_rate: float
        The learning rate to be used by the optimizer.
    
    optimizer: callable, optional
        A function or object used to initialize the optimizer (e.g., `tf.keras.optimizers.Adam`).
        If None, the default Adam optimizer with the specified learning rate is used.
    
    loss: callable, optional
        The loss function to be used during training (e.g., `tf.keras.losses.MeanSquaredError`).
        If None, the default Mean Squared Error loss is used.
    
    metrics: list of callable, optional
        A list of metric functions to evaluate during training (e.g., `tf.keras.metrics.MeanSquaredError`).
        If None, the default metric `MeanSquaredError` is used.

    is_parametric: bool, optional
        Flag to indicate whether the configuration is for the parametric part of the model.
        Default is True, indicating that this configuration is for the parametric component.

    Returns
    -------
    optimizer: callable
        The optimizer function or object configured for training.
    
    loss: callable
        The loss function configured for training.
    
    metrics: list of callable
        The list of metrics to be used for evaluating the model.
    """
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError()
    
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
        )

    if metrics is None:
        metrics = [tf.keras.metrics.MeanSquaredError()]

    return optimizer, loss, metrics

def calculate_sample_weights(outputs: np.ndarray, use_sample_weights: bool) -> np.ndarray or None:
    """
    Calculate sample weights based on image noise standard deviation.

    The function computes sample weights by estimating the noise standard deviation for each image, calculating the inverse variance, 
    and then normalizing the weights by dividing by the median.

    Parameters
    ----------
    outputs: np.ndarray
        A 3D array of shape (batch_size, height, width) representing images, where the first dimension is the batch size 
        and the next two dimensions are the image height and width.
    use_sample_weights: bool
        Flag indicating whether to compute sample weights. If True, sample weights will be computed based on the image noise.

    Returns
    -------
    np.ndarray or None
        An array of sample weights, or None if `use_sample_weights` is False.
    """
    if use_sample_weights:
        img_dim = (outputs.shape[1], outputs.shape[2])
        win_rad = np.ceil(outputs.shape[1] / 3.33)
        std_est = NoiseEstimator(img_dim=img_dim, win_rad=win_rad)
        
        # Estimate noise standard deviation
        imgs_std = np.array([std_est.estimate_noise(_im) for _im in outputs])
        variances = imgs_std ** 2
  
        # Use inverse variance for weights and scale by median
        sample_weight = 1 / variances
        sample_weight /= np.median(sample_weight)

    else:
        sample_weight = None

    return sample_weight

def train_cycle_part(
    psf_model: tf.keras.Model,
    inputs: tf.Tensor,
    outputs: tf.Tensor,
    batch_size: int,
    epochs: int,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: Callable,
    metrics: list[Callable],
    validation_data: Optional[tuple[tf.Tensor, tf.Tensor]] = None,
    callbacks: Optional[list[Callable]] = None,
    sample_weight: Optional[tf.Tensor] = None,
    verbose: int = 1,
    first_run: bool = False,
    cycle_part: str = "parametric"
) -> tf.keras.Model:
    """
    Train either the parametric or non-parametric part of the PSF model using the specified parameters. This function trains a single component of the model (either parametric or non-parametric) based on the provided configuration.

    Parameters
    ----------
    psf_model: tf.keras.Model
        A TensorFlow model representing the PSF (Point Spread Function), which consists of either a parametric or a non-parametric component.

    inputs: tf.Tensor
        Input data for training the model. Expected to be a tensor with the shape of the input batch.
    outputs: tf.Tensor
        Target output data for training the model. Expected to match the shape of `inputs`.
    batch_size: int
        The number of samples per batch during training.
    epochs: int
        The number of epochs to train the model.
    optimizer: tf.keras.optimizers.Optimizer
        The optimizer used for training the model (e.g., Adam, SGD).
    loss: Callable
        The loss function used for training the model. Typically a callable like `tf.keras.losses.MeanSquaredError()`.
    metrics: list of Callable
        List of metrics to monitor during training. Each element should be a callable metric (e.g., accuracy, precision).
    validation_data: tuple of (tf.Tensor, tf.Tensor), optional
        Tuple of input and output tensors to evaluate the model during training. Default is None.
    callbacks: list of Callable, optional
        List of callbacks to apply during training, such as `tf.keras.callbacks.EarlyStopping`. Default is None.
    sample_weight: tf.Tensor, optional
        Weights for the samples during training. Default is None.
    verbose: int, optional
        Verbosity mode (0, 1, or 2). Default is 1.
    first_run: bool, optional
        Flag indicating if this is the first run (affects how the model is built). Default is False.
    cycle_part: str, optional
        Specifies which part of the model to train ("parametric" or "non-parametric"). Default is "parametric".

    Returns
    -------
    tf.keras.Model
        The trained TensorFlow model after completing the specified number of epochs.
    
    Notes
    -----
    This function trains the model based on the provided `cycle_part`. If `cycle_part` is set to
    "parametric", the function assumes the model is being trained in a parametric setting, while
    "non-parametric" indicates the training of a non-parametric part. The model is built using the 
    `build_PSF_model` function before fitting.

    Examples
    --------
    model = train_cycle_part(
        psf_model=model, 
        inputs=train_inputs, 
        outputs=train_outputs, 
        batch_size=32, 
        epochs=10, 
        optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.MeanSquaredError(), 
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
        validation_data=(val_inputs, val_outputs),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
        sample_weight=None, 
        verbose=1
    )
    """
    logger.info(f"Starting {cycle_part} update..")

    psf_model = build_PSF_model(
        psf_model, optimizer=optimizer, loss=loss, metrics=metrics
    )

    return psf_model.fit(
        x=inputs,
        y=outputs,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        sample_weight=sample_weight,
        verbose=verbose,
    )


def get_callbacks(callback1, callback2):
    """
    Combine two callback lists into one.

    If both are None, returns None. If one is None, returns the other.
    Otherwise, combines both lists.
    
    Parameters
    ----------
    callback1: list of tf.keras.callbacks.Callback or None
        The first callback list (e.g., parametric or non-parametric).
    callback2: list of tf.keras.callbacks.Callback or None
        The second callback list (e.g., general callback).

    Returns
    -------
    list of tf.keras.callbacks.Callback or None
        The combined list of callbacks or None.
    """
    if callback1 is None and callback2 is None:
        return None

    return (callback1 or []) + (callback2 or [])


def general_train_cycle(
    psf_model,
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
    """
    Perform a Bi-Cycle Descent (BCD) training iteration on a semi-parametric model.

    The function alternates between optimizing the parametric and/or non-parametric parts of the model
    across specified training cycles. Each part of the model can be trained individually or together 
    depending on the `cycle_def` parameter.

    For the parametric part:
    - Default learning rate: `learning_rate_param = 1e-2`
    - Default epochs: `n_epochs_param = 20`
    
    For the non-parametric part:
    - Default learning rate: `learning_rate_non_param = 1.0`
    - Default epochs: `n_epochs_non_param = 100`

    Parameters
    ----------
    psf_model: tf.keras.Model
        A TensorFlow model representing the PSF (Point Spread Function), which may consist of both parametric and non-parametric components, or an individual component. These components are partitioned for training, with each part addressing different aspects of the PSF.

    inputs: Tensor or list of tensors
        Input data for training (`Model.fit()`).
    outputs: Tensor
        Output data for training (`Model.fit()`).
    validation_data: Tuple
        Validation data used for model evaluation during training.
        (input_data, output_data).
    batch_size: int
        The batch size for the training.
    learning_rate_param: float
        Learning rate for the parametric part of the PSF model.
    learning_rate_non_param: float
        Learning rate for the non-parametric part of the PSF model.
    n_epochs_param: int
        Number of epochs to train the parametric part.
    n_epochs_non_param: int
        Number of epochs to train the non-parametric part.
    param_optim: tf.keras.optimizers.Optimizer, optional
        Optimizer for the parametric part. Defaults to Adam if not provided.
    non_param_optim: tf.keras.optimizers.Optimizer, optional
        Optimizer for the non-parametric part. Defaults to Adam if not provided.
    param_loss: tf.keras.losses.Loss, optional
        Loss function for the parametric part. Defaults to the MeanSquaredError().
    non_param_loss: tf.keras.losses.Loss, optional
        Loss function for the non-parametric part. Defaults to MeanSquaredError().
    param_metrics: list of tf.keras.metrics.Metric, optional
        List of metrics for the parametric part. Defaults to MeanSquaredError().
    non_param_metrics: list of tf.keras.metrics.Metric, optional
        List of metrics for the non-parametric part. Defaults to MeanSquaredError().
    param_callback: list of tf.keras.callbacks.Callback, optional
        Callback for the parametric part only. Defaults to no callback.
    non_param_callback: list of tf.keras.callbacks.Callback, optional
        Callback for the non-parametric part only. Defaults to no callback.
    general_callback: list of tf.keras.callbacks.Callback, optional
        Callback shared between both the parametric and non-parametric parts. Defaults to no callback.
    first_run: bool, optional
        If True, the first iteration of training is assumed, and the non-parametric part 
        is not considered during the parametric training. Default is False.
    cycle_def: str, optional
        Defines the training cycle: `parametric`, `non-parametric`, `complete`, `only-parametric`, or `only-non-parametric`.
        The `complete` cycle trains both parts, while the others train only the specified part (both parametric and non-parametric). Default is `complete`.
    use_sample_weights: bool, optional
        If True, sample weights are used in training. Sample weights are computed 
        based on estimated noise variance. Default is False.
    verbose: int, optional
        Verbosity mode. `0` = silent, `1` = progress bar, `2` = one line per epoch.
        Default is 1.

    Returns
    -------
    psf_model: tf.keras.Model
        The trained PSF model.
    hist_param: tf.keras.callbacks.History
        History object for the parametric training.
    hist_non_param: tf.keras.callbacks.History
        History object for the non-parametric training.

    """
    # Initialize return variables
    hist_param, hist_non_param = None, None

    # Parametric  part
    optimizer, loss, metrics = configure_optimizer_and_loss(
        learning_rate_param, param_optim, param_loss, param_metrics
    )

    # Calculate sample weights
    if use_sample_weights and loss.name != 'masked_mean_squared_error':
         sample_weight = calculate_sample_weights(outputs, use_sample_weights)
        

    # Define the training cycle
    if cycle_def in ("parametric", "complete", "only-parametric"):
        # If it is the first run
        if first_run:
            # Set the non-parametric model to zero
            # With alpha to zero its already enough
            psf_model.set_zero_nonparam()
        if cycle_def == "only-parametric":
            # Set the non-parametric part to zero
            psf_model.set_zero_nonparam()
        
        # Define callbacks for parametric part
        # If both are None, set callbacks to None
        callbacks = get_callbacks(param_callback, general_callback)

        # Set the trainable layer
        psf_model.set_trainable_layers(param_bool=True, nonparam_bool=False)
        hist_param = train_cycle_part(
            psf_model=psf_model,
            inputs=inputs,
            outputs=outputs,
            batch_size=batch_size,
            epochs=n_epochs_param,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            validation_data=validation_data,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=verbose,
            first_run=first_run,
            cycle_part="parametric"
        )
       
    # Non-parametric part
    optimizer, loss, metrics = configure_optimizer_and_loss(
        learning_rate_non_param, non_param_optim, non_param_loss, non_param_metrics, is_parametric=False
    )

    if cycle_def in ("non-parametric", "complete", "only-non-parametric"):
        if first_run:
            # Set the non-parametric model to non-zero
            # With alpha to zero its already enough
            psf_model.set_nonzero_nonparam()
        if cycle_def == "only-non-parametric":
            # Set the parametric layer to zero
            coeff_mat = psf_model.get_coeff_matrix()
            psf_model.assign_coeff_matrix(tf.zeros_like(coeff_mat))

        # Define callbacks for non-parametric part
        # If both are None, set callbacks to None
        callbacks = get_callbacks(non_param_callback, general_callback)
        
        psf_model.set_trainable_layers(param_bool=False, nonparam_bool=True)
        hist_non_param = train_cycle_part(
            psf_model=psf_model,
            inputs=inputs,
            outputs=outputs,
            batch_size=batch_size,
            epochs=n_epochs_non_param,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            validation_data=validation_data,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=verbose,
            first_run=first_run,
            cycle_part="non-parametric"
        )

    return psf_model, hist_param, hist_non_param
