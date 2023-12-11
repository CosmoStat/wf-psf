"""Train.

A module which defines the classes and methods
to manage training of the psf model.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
from wf_psf.utils.read_config import read_conf
import os
import logging
import wf_psf.utils.io as io
from wf_psf.psf_models import psf_models, psf_model_semiparametric
import wf_psf.training.train_utils as train_utils
import wf_psf.data.training_preprocessing as training_preprocessing

logger = logging.getLogger(__name__)


def setup_training():
    """Setup Training.

    A function to setup training.


    """
    device_name = get_gpu_info()
    logger.info("Found GPU at: {}".format(device_name))


def filepath_chkp_callback(checkpoint_dir, model_name, id_name, current_cycle):
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

    def _prepare_callbacks(self, checkpoint_dir, current_cycle):
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
        logger.info(f"Preparing Keras model callback...")
        return tf.keras.callbacks.ModelCheckpoint(
            filepath_chkp_callback(
                checkpoint_dir, self.model_name, self.id_name, current_cycle
            ),
            monitor="mean_squared_error",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
            options=None,
        )


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


def train(
    training_params,
    training_data,
    test_data,
    checkpoint_dir,
    optimizer_dir,
    psf_model_dir,
):
    """Train.

    A function to train the psf model.

    Parameters
    ----------
    training_params: Recursive Namespace object
        Recursive Namespace object containing the training parameters
    training_data: obj
        TrainingDataHandler object containing the training data parameters
    test_data: object
        TestDataHandler object containing the test data parameters
    checkpoint_dir: str
        Absolute path to checkpoint directory
    optimizer_dir: str
        Absolute path to optimizer history directory
    psf_model_dir: str
        Absolute path to psf model directory

    """
    # Start measuring elapsed time
    starting_time = time.time()

    training_handler = TrainingParamsHandler(training_params)

    psf_model = psf_models.get_psf_model(
        training_handler.model_params,
        training_handler.training_hparams,
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

        # If projected learning is enabled project DD_features.
        if psf_model.project_dd_features:  # need to change this
            psf_model.project_DD_features(
                psf_model.zernike_maps
            )  # make this a callable function
            logger.info("Project non-param DD features onto param model: done!")
            if psf_model.reset_dd_features:
                psf_model.tf_np_poly_opd.init_vars()
                logger.info("DD features reset to random initialisation.")

        # Prepare the saving callback
        # Prepare to save the model as a callback
        # -----------------------------------------------------
        logger.info(f"Preparing Keras model callback...")

        model_chkp_callback = training_handler._prepare_callbacks(
            checkpoint_dir, current_cycle
        )

        # Prepare the optimizers
        param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=training_handler.learning_rate_params[current_cycle - 1]
        )
        non_param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=training_handler.learning_rate_non_params[current_cycle - 1]
        )
        logger.info("Starting cycle {}..".format(current_cycle))
        start_cycle = time.time()

        # Compute training per cycle
        (
            psf_model,
            hist_param,
            hist_non_param,
        ) = train_utils.general_train_cycle(
            psf_model,
            # training data
            inputs=[
                training_data.train_dataset["positions"],
                training_data.sed_data,
            ],
            outputs=training_data.train_dataset["noisy_stars"],
            validation_data=(
                [
                    test_data.test_dataset["positions"],
                    test_data.sed_data,
                ],
                test_data.test_dataset["stars"],
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
            param_loss=None,
            non_param_loss=None,
            param_metrics=None,
            non_param_metrics=None,
            param_callback=None,
            non_param_callback=None,
            general_callback=[model_chkp_callback],
            first_run=True if current_cycle == 1 else False,
            cycle_def=training_handler.multi_cycle_params.cycle_def,
            use_sample_weights=training_handler.model_params.use_sample_weights,
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
        logger.info(
            "Cycle{} elapsed time: {}".format(current_cycle, end_cycle - start_cycle)
        )

        # Save optimisation history in the saving dict
        if psf_model.save_optim_history_param:
            saving_optim_hist[
                "param_cycle{}".format(current_cycle)
            ] = hist_param.history
        if psf_model.save_optim_history_nonparam:
            saving_optim_hist[
                "nonparam_cycle{}".format(current_cycle)
            ] = hist_non_param.history

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
