"""Train.

A module which defines the classes and methods
to manage training of the psf model.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>, Tobias Liaudat <tobias.liaudat@cea.fr>, Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
import logging
from wf_psf.psf_models import psf_models
import wf_psf.training.train_utils as train_utils

logger = logging.getLogger(__name__)


def get_gpu_info():
    """Get GPU Information.

    A function to return GPU
    device name.

    Returns
    -------
    device_name: str
        Name of GPU device

    """
    device_name = tf.test.gpu_device_name()
    return device_name


def setup_training():
    """Set up Training.

    A function to setup training.

    """
    device_name = get_gpu_info()
    logger.info(f"Found GPU at: {device_name}")


def filepath_chkp_callback(
    checkpoint_dir: str, model_name: str, id_name: str, current_cycle: int
) -> str:
    """
    Generate a file path for a checkpoint callback.

    Parameters
    ----------
    checkpoint_dir : str
        The directory where the checkpoint will be saved.
    model_name : str
        The name of the model.
    id_name : str
        The unique identifier for the model instance.
    current_cycle : int
        The current cycle number.

    Returns
    -------
    str
        A string representing the full file path for the checkpoint callback.
    """
    return (
        checkpoint_dir
        + "/checkpoint_callback_"
        + model_name
        + id_name
        + "_cycle"
        + str(current_cycle)
    )


class TrainingParamsHandler:
    """Training Parameters Handler.

    A class to handle training parameters accessed:

    Parameters
    ----------
    training_params: Recursive Namespace object
        Recursive Namespace object containing training input parameters

    """

    def __init__(self, training_params):
        self.training_params = training_params
        self.run_id_name = self.model_name + self.id_name
        self.optimizer_params = {}

    @property
    def id_name(self):
        """ID Name.

        Set unique ID name.

        Returns
        -------
        id_name: str
            A unique ID.
        """
        return self.training_params.id_name

    @property
    def model_name(self):
        """PSF Model Name.

        Set model_name.

        Returns
        -------
        model_name: str
            Name of PSF model

        """
        return self.training_params.model_params.model_name

    @property
    def model_params(self):
        """PSF Model Params.

        Set PSF model training parameters

        Returns
        -------
        model_params: Recursive Namespace object
            Recursive Namespace object storing PSF model parameters

        """
        return self.training_params.model_params

    @property
    def training_hparams(self):
        """Training Hyperparameters.

        Set training hyperparameters

        Returns
        -------
        training_hparams: Recursive Namespace object
            Recursive Namespace object storing training hyper parameters

        """
        return self.training_params.training_hparams

    @property
    def multi_cycle_params(self):
        """Training Multi Cycle Parameters.

        Set training multi cycle parameters

        Returns
        -------
        multi_cycle_params: Recursive Namespace object
            Recursive Namespace object storing training multi-cycle parameters

        """
        return self.training_hparams.multi_cycle_params

    @property
    def total_cycles(self):
        """Total Number of Cycles.

        Set total number of cycles for
        training.

        Returns
        -------
        total_cycles: int
            Total number of cycles for training
        """
        return self.multi_cycle_params.total_cycles

    @property
    def n_epochs_params(self):
        """Number of Epochs for Parametric PSF model.

        Set the number of epochs for
        training parametric PSF model.

        Returns
        -------
        n_epochs_params: list
            List of number of epochs for training parametric PSF model.

        """
        return self.multi_cycle_params.n_epochs_params

    @property
    def n_epochs_non_params(self):
        """Number of Epochs for Non-parametric PSF model.

        Set the number of epochs for
        training non-parametric PSF model.

        Returns
        -------
        n_epochs_non_params: list
            List of number of epochs for training non-parametric PSF model.

        """
        return self.multi_cycle_params.n_epochs_non_params

    @property
    def learning_rate_params(self):
        """Parametric Model Learning Rate.

        Set learning rate for parametric
        PSF model.

        Returns
        -------
        learning_rate_params: list
            List containing learning rate for parametric PSF model

        """
        return self.multi_cycle_params.learning_rate_params

    @property
    def learning_rate_non_params(self):
        """Non-parametric Model Learning Rate.

        Set learning rate for non-parametric
        PSF model.

        Returns
        -------
        learning_rate_non_params: list
            List containing learning rate for non-parametric PSF model

        """
        return self.multi_cycle_params.learning_rate_non_params

    def _prepare_callbacks(
        self, checkpoint_dir, current_cycle, monitor="mean_squared_error"
    ):
        """Prepare Callbacks.

        A function to prepare to save the model as a callback.

        Parameters
        ----------
        checkpoint_dir: str
            Checkpoint directory
        current_cycle: int
            Integer representing the current cycle

        Returns
        -------
            keras.callbacks.ModelCheckpoint class
                Class to save the Keras model or model weights at some frequency

        """
        # -----------------------------------------------------
        logger.info("Preparing Keras model callback...")
        return tf.keras.callbacks.ModelCheckpoint(
            filepath_chkp_callback(
                checkpoint_dir, self.model_name, self.id_name, current_cycle
            ),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
            options=None,
        )


def get_loss_metrics_monitor_and_outputs(training_handler, data_conf):
    """Generate factory for loss, metrics, monitor, and outputs.

    A function to generate loss, metrics, monitor, and outputs
    for training.

    Parameters
    ----------
    training_handler: TrainingParamsHandler
        TrainingParamsHandler object containing training parameters
    data_conf: object
        Data configuration object containing training and test data

    Returns
    -------
    loss: tf.keras.losses.Loss
        Loss function to be used for training
    param_metrics: list
        List of metrics for the parametric model
    non_param_metrics: list
        List of metrics for the non-parametric model
    monitor: str
        Metric to monitor for saving the model
    outputs: tf.Tensor
        Tensor containing the outputs for training
    output_val: tf.Tensor
        Tensor containing the outputs for validation

    """
    if training_handler.training_hparams.loss == "mask_mse":
        loss = train_utils.MaskedMeanSquaredError()
        monitor = "loss"
        param_metrics = [train_utils.MaskedMeanSquaredErrorMetric()]
        non_param_metrics = [train_utils.MaskedMeanSquaredErrorMetric()]
        outputs = tf.stack(
            [
                data_conf.training_data.dataset["noisy_stars"],
                data_conf.training_data.dataset["masks"],
            ],
            axis=-1,
        )
        output_val = tf.stack(
            [
                data_conf.test_data.dataset["stars"],
                data_conf.test_data.dataset["masks"],
            ],
            axis=-1,
        )
    else:
        loss = tf.keras.losses.MeanSquaredError()
        monitor = "mean_squared_error"
        param_metrics = [tf.keras.metrics.MeanSquaredError()]
        non_param_metrics = [tf.keras.metrics.MeanSquaredError()]
        outputs = data_conf.training_data.dataset["noisy_stars"]
        output_val = data_conf.test_data.dataset["stars"]

    return loss, param_metrics, non_param_metrics, monitor, outputs, output_val


def train(
    training_params,
    data_conf,
    checkpoint_dir,
    optimizer_dir,
    psf_model_dir,
):
    """
    Train the PSF model over one or more parametric and non-parametric training cycles.

    This function manages multi-cycle training of a parametric + non-parametric PSF model,
    including initialization, loss/metric configuration, optimizer setup, model checkpointing,
    and optional projection or resetting of non-parametric features. Each cycle can include
    both parametric and non-parametric training stages, and training history is saved for each.

    Parameters
    ----------
    training_params : RecursiveNamespace
        Contains all training configuration parameters, including:
        - learning rates per cycle
        - number of epochs per component per cycle
        - model type and training behavior flags
        - multi-cycle definitions and callbacks

    data_conf : object
        Contains training and validation datasets via attributes:
        - data_conf.training_data: TrainingDataHandler instance with SEDs and positions
        - data_conf.test_data: TestDataHandler instance with validation SEDs and positions

    checkpoint_dir : str
        Directory where model checkpoints will be saved during training.

    optimizer_dir : str
        Directory where the optimizer history (as a NumPy .npy file) will be stored.

    psf_model_dir : str
        Directory where the final trained PSF model weights will be saved per cycle.

    Notes
    -----
    - Utilizes TensorFlow and TensorFlow Addons for model training and optimization.
    - Supports masked mean squared error loss for training with masked data.
    - Allows for projection of data-driven features onto parametric models between cycles.
    - Supports resetting of non-parametric features to initial states.
    - Saves model weights to `psf_model_dir` per training cycle (or final one if not all saved)
    - Saves optimizer histories to `optimizer_dir`
    - Logs cycle information and time durations
    """
    # Start measuring elapsed time
    starting_time = time.time()

    training_handler = TrainingParamsHandler(training_params)

    psf_model = psf_models.get_psf_model(
        training_handler.model_params,
        training_handler.training_hparams,
        data_conf,
    )

    logger.info(f"PSF Model class: `{training_handler.model_name}` initialized...")
    # Model Training
    # -----------------------------------------------------
    # Save optimisation history in the saving dict
    saving_optim_hist = {}

    # Perform all the necessary cycles
    current_cycle = 0
    while training_handler.total_cycles > current_cycle:
        current_cycle += 1

        # Instantiate fresh loss, monitor, and independent metric objects per training phase (param / non-param)
        loss, param_metrics, non_param_metrics, monitor, outputs, output_val = (
            get_loss_metrics_monitor_and_outputs(training_handler, data_conf)
        )

        # If projected learning is enabled project DD_features.
        if hasattr(psf_model, "project_dd_features") and psf_model.project_dd_features:
            if current_cycle > 1:
                psf_model.project_DD_features(
                    psf_model.zernike_maps
                )  # make this a callable function
                logger.info(
                    "Projected non-parametric DD features onto the parametric model."
                )

        if hasattr(psf_model, "reset_dd_features") and psf_model.reset_dd_features:
            psf_model.tf_np_poly_opd.init_vars()
            logger.info("DataDriven features were reset to random initialisation.")

        # Prepare the saving callback
        # Prepare to save the model as a callback
        # -----------------------------------------------------
        model_chkp_callback = training_handler._prepare_callbacks(
            checkpoint_dir, current_cycle, monitor=monitor
        )

        # Prepare the optimizers
        param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=training_handler.learning_rate_params[current_cycle - 1]
        )
        non_param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=training_handler.learning_rate_non_params[current_cycle - 1]
        )
        logger.info(f"Starting cycle {current_cycle}..")

        start_cycle = time.time()

        # Compute training per cycle
        (
            psf_model,
            hist_param,
            hist_non_param,
        ) = train_utils.general_train_cycle(
            psf_model,
            inputs=[
                data_conf.training_data.dataset["positions"],
                data_conf.training_data.sed_data,
            ],
            outputs=outputs,
            validation_data=(
                [
                    data_conf.test_data.dataset["positions"],
                    data_conf.test_data.sed_data,
                ],
                output_val,
            ),
            batch_size=training_handler.training_hparams.batch_size,
            learning_rate_param=training_handler.learning_rate_params[
                current_cycle - 1
            ],
            learning_rate_non_param=training_handler.learning_rate_non_params[
                current_cycle - 1
            ],
            n_epochs_param=training_handler.n_epochs_params[current_cycle - 1],
            n_epochs_non_param=training_handler.n_epochs_non_params[current_cycle - 1],
            param_optim=param_optim,
            non_param_optim=non_param_optim,
            param_loss=loss,
            non_param_loss=loss,
            param_metrics=param_metrics,
            non_param_metrics=non_param_metrics,
            param_callback=None,
            non_param_callback=None,
            general_callback=[model_chkp_callback],
            first_run=True if current_cycle == 1 else False,
            cycle_def=training_handler.multi_cycle_params.cycle_def,
            use_sample_weights=training_handler.model_params.use_sample_weights,
            apply_sigmoid=training_handler.model_params.sample_weights_sigmoid.apply_sigmoid,
            sigmoid_max_val=training_handler.model_params.sample_weights_sigmoid.sigmoid_max_val,
            sigmoid_power_k=training_handler.model_params.sample_weights_sigmoid.sigmoid_power_k,
            verbose=2,
        )

        # Save the weights at the end of the nth cycle
        if training_handler.multi_cycle_params.save_all_cycles:
            psf_model.save_weights(
                psf_model_dir
                + "/psf_model_"
                + training_handler.model_name
                + training_handler.id_name
                + "_cycle"
                + str(current_cycle)
            )

        end_cycle = time.time()
        logger.info(f"Cycle{current_cycle} elapsed time: {end_cycle - start_cycle}")

        # Save optimisation history in the saving dict
        if (
            hasattr(psf_model, "save_optim_history_param")
            and psf_model.save_optim_history_param
        ):
            saving_optim_hist[f"param_cycle{current_cycle}"] = hist_param.history

        if (
            hasattr(psf_model, "save_optim_history_nonparam")
            and psf_model.save_optim_history_nonparam
        ):
            saving_optim_hist[f"nonparam_cycle{current_cycle}"] = hist_non_param.history

    # Save last cycle if no cycles were saved
    if not training_handler.multi_cycle_params.save_all_cycles:
        psf_model.save_weights(
            psf_model_dir
            + "/psf_model_"
            + training_handler.model_name
            + training_handler.id_name
            + "_cycle"
            + str(current_cycle)
        )

    # Save optimisation history dictionary
    np.save(
        optimizer_dir
        + "/optim_hist_"
        + training_handler.model_name
        + training_handler.id_name
        + ".npy",
        saving_optim_hist,
    )

    # Print final time
    final_time = time.time()
    logger.info("\nTotal elapsed time: %f" % (final_time - starting_time))
    logger.info("\n Training complete..")
