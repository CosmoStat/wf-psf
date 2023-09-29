"""Training Data Processing.

A module to load and preprocess training and validation test data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr> and Tobias Liaudat <tobiasliaudat@gmail.com>

"""
import numpy as np
import wf_psf.utils.utils as utils
import tensorflow as tf
import tensorflow_addons as tfa
import wf_psf.sims.SimPSFToolkit as SimPSFToolkit
import os


class TrainingDataHandler:
    """Training Data Handler.

    A class to manage training data.

    Parameters
    ----------
    training_data_params: Recursive Namespace object
        Recursive Namespace object containing training data parameters
    simPSF: object
        SimPSFToolkit instance
    n_bins_lambda: int
        Number of bins in wavelength

    """

    def __init__(self, training_data_params, simPSF, n_bins_lambda):
        self.training_data_params = training_data_params
        self.train_dataset = np.load(
            os.path.join(
                self.training_data_params.data_dir, self.training_data_params.file
            ),
            allow_pickle=True,
        )[()]
        self.train_dataset["positions"] = tf.convert_to_tensor(
            self.train_dataset["positions"], dtype=tf.float32
        )
        self.train_dataset["noisy_stars"] = tf.convert_to_tensor(
            self.train_dataset["noisy_stars"], dtype=tf.float32
        )
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(
                _sed, self.simPSF, n_bins=self.n_bins_lambda, tf_dtype=tf.float64
            )
            for _sed in self.train_dataset["SEDs"]
        ]
        self.sed_data = tf.convert_to_tensor(self.sed_data, dtype=tf.float32)
        self.sed_data = tf.transpose(self.sed_data, perm=[0, 2, 1])


class TestDataHandler:
    """Test Data Handler.

    A class to handle test data for model validation.

    Parameters
    ----------
    test_data_params: Recursive Namespace object
        Recursive Namespace object containing test data parameters
    simPSF: object
        SimPSFToolkit instance
    n_bins_lambda: int
        Number of bins in wavelength

    """

    def __init__(self, test_data_params, simPSF, n_bins_lambda):
        self.test_data_params = test_data_params
        self.test_dataset = np.load(
            os.path.join(self.test_data_params.data_dir, self.test_data_params.file),
            allow_pickle=True,
        )[()]
        self.test_dataset["stars"] = tf.convert_to_tensor(
            self.test_dataset["stars"], dtype=tf.float32
        )
        self.test_dataset["positions"] = tf.convert_to_tensor(
            self.test_dataset["positions"], dtype=tf.float32
        )

        # Prepare validation data inputs
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda

        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(
                _sed, self.simPSF, n_bins=self.n_bins_lambda, tf_dtype=tf.float64
            )
            for _sed in self.test_dataset["SEDs"]
        ]
        self.sed_data = tf.convert_to_tensor(self.sed_data, dtype=tf.float32)
        self.sed_data = tf.transpose(self.sed_data, perm=[0, 2, 1])
