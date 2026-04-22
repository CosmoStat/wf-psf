"""Training Config Handler.

A module which provides a class to manage the parameters of the training config file.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import os
import tensorflow as tf
from wf_psf.data.data_adapter import StructureState, RepresentationState, DataAdapter
from wf_psf.data.data_config_handler import DataConfigHandler
from wf_psf.data.factory import DataAdapterFactory
from wf_psf.metrics.metrics_config_handler import MetricsConfigHandler
from wf_psf.psf_models import psf_models
from wf_psf.utils.configs_handler import ConfigHandler, register_configclass
from wf_psf.utils.read_config import read_conf
from wf_psf.training import train
from wf_psf.training.training_data_adapter import TrainingDataAdapter
import logging


logger = logging.getLogger(__name__)


@register_configclass
class TrainingConfigHandler(ConfigHandler):
    """TrainingConfigHandler.

    A class to handle training configuration
    parameters.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    training_conf: str
        Path of the training configuration file
    file_handler: object
        A instance of the FileIOHandler class

    """

    ids = ("training_conf",)

    def __init__(self, training_conf, file_handler):
        self.training_conf = read_conf(training_conf).training
        self.file_handler = file_handler
        self.data_params = DataConfigHandler(
            os.path.join(file_handler.config_path, self.training_conf.data_config),
        )
        self.n_bins_lambda = self.training_conf.model_params.n_bins_lambda
        self.simPSF = psf_models.simPSF(self.training_conf.model_params)
        self.file_handler.copy_conffile_to_output_dir(self.training_conf.data_config)
        self.checkpoint_dir = file_handler.get_checkpoint_dir(
            self.file_handler._run_output_dir
        )
        self.optimizer_dir = file_handler.get_optimizer_dir(
            self.file_handler._run_output_dir
        )
        self.psf_model_dir = file_handler.get_psf_model_dir(
            self.file_handler._run_output_dir
        )

    def run(self):
        """Run.

        A function to run wavediff according to the
        input configuration.

        """
        training_adapter, psf_model = prepare_training_inputs(
            self.data_params,
            self.simPSF,
            self.n_bins_lambda,
            self.training_conf.training_hparams.loss,
            self.training_conf.model_params,
            self.training_conf.training_hparams,
        )

        train.train(
            self.training_conf,
            training_adapter,
            psf_model,
            self.checkpoint_dir,
            self.optimizer_dir,
            self.psf_model_dir,
        )

        if self.training_conf.metrics_config is not None:
            self.file_handler.copy_conffile_to_output_dir(
                self.training_conf.metrics_config
            )

            metrics = MetricsConfigHandler(
                os.path.join(
                    self.file_handler.config_path,
                    self.training_conf.metrics_config,
                ),
                self.file_handler,
                self.training_conf,
            )

            metrics.run()


def prepare_training_inputs(
    data_params,
    simPSF,
    n_bins_lambda,
    loss,
    model_params,
    training_hparams,
) -> tuple[TrainingDataAdapter, tf.keras.Model]:
    """Build a training-ready data adapter and PSF model.

    The sequence is order-dependent: the dataset must be joined into
    complete form before PSF model initialisation (certain models require
    the full dataset), then split and converted to tensors afterward.

    Parameters
    ----------
    data_params : RecursiveNamespace or SHEPSFDataset
        Data configuration parameters or a pre-loaded in-memory dataset.
    simPSF : PSFSimulator
        PSF simulator instance used for SED encoding during conversion.
    n_bins_lambda : int
        Number of wavelength bins for SED discretisation.
    loss : str
        Loss function identifier, determines whether masks are packed
        with target images in the training adapter.
    model_params : RecursiveNamespace
        PSF model configuration parameters.
    training_hparams : RecursiveNamespace
        Training hyperparameters passed to PSF model initialisation.

    Returns
    -------
    tuple[TrainingDataAdapter, PSFModel]
        A fully prepared training adapter and initialised PSF model,
        ready to be passed to the training loop.
    """
    adapter = DataAdapterFactory.build(data_params)

    if adapter.structure_state == StructureState.SPLIT:
        logger.info("Joining split datasets...")
        adapter.join_data()

    # PSF model initialisation requires complete data
    logger.info(f"Initialising PSF model {model_params.model_name}...")
    psf_model = psf_models.get_psf_model(
        model_params, training_hparams, adapter.complete_data
    )

    # Update StructureState to SPLIT
    if adapter.structure_state == StructureState.COMPLETE:
        logger.info("Generating split datasets...")
        adapter.split_data()

    # -------------------------------
    # REPRESENTATION STATE MACHINE
    # -------------------------------
    if adapter.representation_state == RepresentationState.NUMPY:
        logger.info("Converting dataset to tensors...")
        adapter.convert_to_tensorflow(simPSF, n_bins_lambda)

    # -------------------------------
    # DATA_PREPARED BOUNDARY
    # -------------------------------
    logger.info("Validating data preparation state...")
    _assert_data_prepared(adapter)

    logger.info("Data preparation complete. Freezing dataset snapshot...")

    return TrainingDataAdapter(adapter, loss), psf_model


def _assert_data_prepared(adapter: DataAdapter):
    if adapter.structure_state != StructureState.SPLIT:
        raise RuntimeError(f"Expected SPLIT data, got {adapter.structure_state}")

    if adapter.representation_state != RepresentationState.TENSORFLOW:
        raise RuntimeError(
            f"Expected TensorFlow data, got {adapter.representation_state}"
        )
