"""DataConfigHandler.

A module which provides a class to manage the parameters of the data config file.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from wf_psf.utils.configs_handler import ConfigHandler
from wf_psf.data.data_handler import DataHandler
from wf_psf.utils.read_config import read_conf
from wf_psf.psf_models import psf_models
import logging


logger = logging.getLogger(__name__)


class DataConfigHandler(ConfigHandler):
    """DataConfigHandler.

    A class to handle data configuration
    parameters.

    Parameters
    ----------
    data_conf : str
        Path of the data configuration file
    training_model_params : Recursive Namespace object
        Recursive Namespace object containing the training model parameters
    batch_size : int
       Training hyperparameter used for batched pre-processing of data.

    """

    ids = ("data_conf",)

    def __init__(self, data_conf, training_model_params, batch_size=16, load_data=True):
        try:
            self.data_conf = read_conf(data_conf)
        except (FileNotFoundError, TypeError) as e:
            logger.exception(e)
            exit()

        self.simPSF = psf_models.simPSF(training_model_params)

        # Extract sub-configs early
        train_params = self.data_conf.data.training
        test_params = self.data_conf.data.test

        self.training_data = DataHandler(
            dataset_type="training",
            data_params=train_params,
            simPSF=self.simPSF,
            n_bins_lambda=training_model_params.n_bins_lda,
            load_data=load_data,
        )
        self.test_data = DataHandler(
            dataset_type="test",
            data_params=test_params,
            simPSF=self.simPSF,
            n_bins_lambda=training_model_params.n_bins_lda,
            load_data=load_data,
        )

        self.batch_size = batch_size

    def run(self):
        """Run DataConfigHandler.

        A function to run the data configuration handler.

        """
        raise NotImplementedError
