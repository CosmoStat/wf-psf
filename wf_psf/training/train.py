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

    training_handler = TrainingParamsHandler(training_params, output_dirs)

    psf_model = psf_models.get_psf_model(
        training_handler.model_name,
        training_handler.model_params,
        training_handler.training_hparams,
    )

    logger.info(f"PSF Model class: `{training_handler.model_name}` initialized...")
    # Model Training
    # Prepare the saving callback
    # Prepare to save the model as a callback
    # -----------------------------------------------------
    logger.info(f"Preparing Keras model callback...")
    filepath_chkp_callback = (
        training_handler.checkpoint_dir
        + "/"
        + "chkp_callback_"
        + training_handler.model_name
        + training_handler.id_name
        + "_cycle1"
    )
    model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath_chkp_callback,
        monitor="mean_squared_error",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
        options=None,
    )
    # -----------------------------------------------------
    # Instantiate Simulated PSF Toolkit
    logger.info(f"Instantiating simPSF toolkit...")
    simPSF = psf_models.simPSF(training_handler.model_params)

    # Prepare the optimisers
    param_optim = tfa.optimizers.RectifiedAdam(
        learning_rate=training_handler.training_multi_cycle_params.learning_rate_param_multi_cycle[
            0
        ]
    )
    non_param_optim = tfa.optimizers.RectifiedAdam(
        learning_rate=training_handler.training_multi_cycle_params.learning_rate_non_param_multi_cycle[
            0
        ]
    )
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

    print("Starting training cycle 1..")
    start_cycle1 = time.time()

    tf_semiparam_field, hist_param, hist_non_param = train_utils.general_train_cycle(
        # poly model
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
            0
        ],
        learning_rate_non_param=training_handler.training_multi_cycle_params.learning_rate_non_param_multi_cycle[
            0
        ],
        n_epochs_param=training_handler.training_hparams.n_epochs_param[0],
        n_epochs_non_param=training_handler.training_hparams.n_epochs_non_param[0],
        param_optim=param_optim,
        non_param_optim=non_param_optim,
        param_loss=None,
        non_param_loss=None,
        param_metrics=None,
        non_param_metrics=None,
        param_callback=None,
        non_param_callback=None,
        general_callback=[model_chkp_callback],
        first_run=True,
        cycle_def=training_handler.training_multi_cycle_params.cycle_def,
        use_sample_weights=training_handler.model_params.use_sample_weights,
        verbose=2,
    )

    end_cycle1 = time.time()
    print("Cycle1 elapsed time: %f" % (end_cycle1 - start_cycle1))

    # Save optimisation history in the saving dict
    saving_optim_hist={}
    if hist_param is not None:
        saving_optim_hist["param_cycle1"] = hist_param.history
    if  psf_model.ids != "param" and hist_non_param is not None:
        saving_optim_hist["nonparam_cycle1"] = hist_non_param.history

    # Perform all the necessary cycles
    current_cycle = 1

    while training_handler.training_hparams.multi_cycle_params.total_cycles > current_cycle:
        current_cycle += 1

        # If projected learning is enabled project DD_features.
        if args["project_dd_features"] and psf_model.ids == "poly":
            tf_semiparam_field.project_DD_features(tf_zernike_cube)
            print("Project non-param DD features onto param model: done!")
            if args["reset_dd_features"]:
                psf_model.tf_np_poly_opd.init_vars()
                print("DD features reseted to random initialisation.")

            # Prepare the saving callback
        # Prepare to save the model as a callback
        # -----------------------------------------------------
        logger.info(f"Preparing Keras model callback...")
        filepath_chkp_callback = (
            training_handler.checkpoint_dir
            + "/"
            + "chkp_callback_"
            + training_handler.model_name
            + training_handler.id_name
            + "_cycle"
            + str(current_cycle)
        )
        model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath_chkp_callback,
            monitor="mean_squared_error",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
            options=None,
        )
  
        # Prepare the optimisers
        param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=args["learning_rate_param"][current_cycle - 1]
        )
        non_param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=args["learning_rate_non_param"][current_cycle - 1]
        )

        print("Starting cycle {}..".format(current_cycle))
        start_cycle = time.time()

        # Compute the next cycle
        (   psf_model,
            hist_param_2,
            hist_non_param_2,
        ) = wf_train_utils.general_train_cycle(
            psf_model,
            # training data
            inputs=[training_data.train_dataset["positions"], training_data.sed_data],
            outputs=training_data.train_dataset["noisy_stars"],
            validation_data=(
            [test_data.test_dataset["positions"], test_data.sed_data],
            test_data.test_dataset["stars"],
        ),
            batch_size=training_handler.training_hparams.batch_size,
            learning_rate_param=training_handler.training_multi_cycle_params.learning_rate_param_multi_cycle[current_cycle - 1],
            learning_rate_non_param=training_handler.training_multi_cycle_params.learning_rate_non_param[current_cycle - 1],
            n_epochs_param=training_handler.training_multi_cycle_params.n_epochs_param[current_cycle - 1],
            n_epochs_non_param=training_handler.training_multi_cycle_params.learning_rate_non_param_multi_cycle[current_cycle - 1],
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
                model_save_file + "chkp_" + run_id_name + "_cycle" + str(current_cycle)
            )

        end_cycle = time.time()
        print("Cycle{} elapsed time: {}".format(current_cycle, end_cycle - start_cycle))

        # Save optimisation history in the saving dict
        if hist_param_2 is not None:
            saving_optim_hist[
                "param_cycle{}".format(current_cycle)
            ] = hist_param_2.history
        if psf_model.ids != "param" and hist_non_param_2 is not None:
            saving_optim_hist[
                "nonparam_cycle{}".format(current_cycle)
            ] = hist_non_param_2.history

    # Save last cycle if no cycles were saved
    if not training_handler.training_hparams.multi_cycle_params.save_all_cycles:
        tf_semiparam_field.save_weights(
            model_save_file + "chkp_" + run_id_name + "_cycle" + str(current_cycle)
        )

    # Save optimisation history dictionary
    np.save(optim_hist_file + "optim_hist_" + run_id_name + ".npy", saving_optim_hist)

    # Print final time
    final_time = time.time()
    print("\nTotal elapsed time: %f" % (final_time - starting_time))

    # Close log file
    print("\n Good bye..")
    sys.stdout = old_stdout
    log_file.close()
