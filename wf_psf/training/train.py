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
from wf_psf.psf_models import *
import training.train_utils as train_utils
import wf_psf.data.training_preprocessing as training_preprocessing
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler

logger = logging.getLogger(__name__)


def setup_training():
    """Setup Training.

    A function to setup training.


    """
    device_name = get_gpu_info()
    logger.info("Found GPU at: {}".format(device_name))


class TrainingParamsHandler:
    """Training Parameters Handler.

    A class to handle training parameters accessed:

    Parameters
    ----------
    training_params: type
        Type containing training input parameters
    id_name: str
        ID name
    output_dirs: FileIOHandler
        FileIOHandler instance


    """

    def __init__(
        self, training_params, output_dirs, id_name="-coherent_euclid_200stars"
    ):
        self.training_params = training_params
        self.id_name = id_name
        self.run_id_name = self.model_name + self.id_name
        self.checkpoint_dir = output_dirs.get_checkpoint_dir()
        self.optimizer_params = {}

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
        model_params: type
            Recursive Namespace object

        """
        return self.training_params.model_params

    @property
    def training_hparams(self):
        """Training Hyperparameters.

        Set training hyperparameters

        Returns
        -------
        training_hparams: type
            Recursive Namespace object

        """
        return self.training_params.training_hparams

    @property
    def training_multi_cycle_params(self):
        """Training Multi Cycle Parameters.

        Set training multi cycle parameters

        Returns
        -------
        training_multi_cycle_params: type
            Recursive Namespace object

        """
        return self.training_params.training_hparams.multi_cycle_params

    @property
    def training_data_params(self):
        """Training Data Params.

        Set training data parameters

        Returns
        -------
        training_data_params: type
            Recursive Namespace object

        """
        return self.training_params.data.training

    @property
    def test_data_params(self):
        """Test Data Params.

        Set test data parameters

        Returns
        -------
        test_data_params: type
            Recursive Namespace object


        """
        return self.training_params.data.test


    def _filepath_chkp_callback(self, current_cycle):
        return (self.checkpoint_dir
            + "/"
            + "chkp_callback_"
            + self.model_name
            + self.id_name
            + "_cycle"
            + str(current_cycle)
        )
        
        
    def _prepare_callbacks(self, current_cycle):
        # Prepare to save the model as a callback
        # -----------------------------------------------------
        logger.info(f"Preparing Keras model callback...")
        return tf.keras.callbacks.ModelCheckpoint(
            self._filepath_chkp_callback(current_cycle),
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



def train(training_params, output_dirs):
    """Train.

    A function to train the psf model.

    Parameters
    ----------
    training_params: type
        Recursive Namespace object
    output_dirs: str
        Absolute paths to training output directories

    """
    # Start measuring elapsed time
    starting_time = time.time()

    training_handler = TrainingParamsHandler(training_params, output_dirs)

    psf_model = psf_models.get_psf_model(
        training_handler.model_name,
        training_handler.model_params,
        training_handler.training_hparams,
    )

    logger.info(f"PSF Model class: `{training_handler.model_name}` initialized...")
    # Model Training
    # -----------------------------------------------------
    # Instantiate Simulated PSF Toolkit
    logger.info(f"Instantiating simPSF toolkit...")
    simPSF = psf_models.simPSF(training_handler.model_params)

    # -----------------------------------------------------
    # Get training data
    logger.info(f"Fetching and preprocessing training and test data...")
    training_data = TrainingDataHandler(
        training_handler.training_data_params,
        simPSF,
        training_handler.model_params.n_bins_lda,
    )
    test_data = TestDataHandler(
        training_handler.test_data_params,
        simPSF,
        training_handler.model_params.n_bins_lda,
    )

    # Save optimisation history in the saving dict
    saving_optim_hist = {}

    # Perform all the necessary cycles
    current_cycle = 0

    while (
        training_handler.training_hparams.multi_cycle_params.total_cycles
        > current_cycle
    ):
        current_cycle += 1

        # If projected learning is enabled project DD_features.
        if psf_model.project_dd_features:  #need to change this
            psf_model.project_DD_features(psf_model.zernike_maps) #make this a callable function
            logger.info("Project non-param DD features onto param model: done!")
            if psf_model.reset_dd_features:
                psf_model.tf_np_poly_opd.init_vars()
                logger.info("DD features reset to random initialisation.")

        # Prepare the saving callback
        # Prepare to save the model as a callback
        # -----------------------------------------------------
        logger.info(f"Preparing Keras model callback...")
  
        model_chkp_callback = training_handler._prepare_callbacks(current_cycle)  

        # Prepare the optimisers
        param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=training_handler.training_multi_cycle_params.learning_rate_param_multi_cycle[
                current_cycle-1
            ]
        )
        non_param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=training_handler.training_multi_cycle_params.learning_rate_non_param_multi_cycle[
                current_cycle-1
            ]
        )
        logger.info("Starting cycle {}..".format(current_cycle))
        start_cycle = time.time()

        # Compute the next cycle
        (
            psf_model,
            hist_param,
            hist_non_param,
        ) = train_utils.general_train_cycle(
            psf_model,
            # training data
            inputs=[training_data.train_dataset["positions"], training_data.sed_data],
            outputs=training_data.train_dataset["noisy_stars"],
            validation_data=(
                [test_data.test_dataset["positions"], test_data.sed_data],
                test_data.test_dataset["stars"],
            ),
            batch_size=training_handler.training_hparams.batch_size,
            learning_rate_param=training_handler.training_multi_cycle_params.learning_rate_param_multi_cycle[
            current_cycle-1],
            learning_rate_non_param=training_handler.training_multi_cycle_params.learning_rate_non_param_multi_cycle[
            current_cycle-1],
            n_epochs_param=training_handler.training_hparams.n_epochs_param[
                current_cycle-1
            ],
            n_epochs_non_param=training_handler.training_hparams.n_epochs_non_param[current_cycle-1],
            param_optim=param_optim,
            non_param_optim=non_param_optim,
            param_loss=None,
            non_param_loss=None,
            param_metrics=None,
            non_param_metrics=None,
            param_callback=None,
            non_param_callback=None,
            general_callback=[model_chkp_callback],
            first_run=False,
            cycle_def=training_handler.training_multi_cycle_params.cycle_def,
            use_sample_weights=training_handler.model_params.use_sample_weights,
            verbose=2,
        )

        # Save the weights at the end of the second cycle
        if training_handler.training_hparams.multi_cycle_params.save_all_cycles:
            psf_model.save_weights(
                model_save_file + "chkp_" + training_handler.model_name + training_handler.id_name + "_cycle" + str(current_cycle)
            )

        end_cycle = time.time()
        logger.info("Cycle{} elapsed time: {}".format(current_cycle, end_cycle - start_cycle))

        # Save optimisation history in the saving dict
        if psf_model.save_optim_history_param:
            saving_optim_hist[
                "param_cycle{}".format(current_cycle)
            ] = hist_param.history
        if psf_model.save_optim_history_non_param:
            saving_optim_hist[
                "nonparam_cycle{}".format(current_cycle)
            ] = hist_non_param.history

    # Save last cycle if no cycles were saved
    if not training_handler.training_hparams.multi_cycle_params.save_all_cycles:
        psf_model.save_weights(output_dirs.get_checkpoint_dir()
            + "/chkp_" + training_handler.model_name + training_handler.id_name + "_cycle" + str(current_cycle)
        )

    # Save optimisation history dictionary
    np.save(output_dirs.get_optimizer_dir() + "/optim_hist_" + training_handler.model_name + training_handler.id_name + ".npy", saving_optim_hist)

    # Print final time
    final_time = time.time()
    logger.info("\nTotal elapsed time: %f" % (final_time - starting_time))

    # Close log file
    logger.info("\n Good bye..")
    # log_file.close()
