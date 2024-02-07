"""Training Data Processing.

A module to load and preprocess training and validation test data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr> and Tobias Liaudat <tobiasliaudat@gmail.com>

"""

import numpy as np
import wf_psf.utils.utils as utils
import tensorflow as tf
import tensorflow_addons as tfa
import os


class DataHandler:
    """Training Data Handler.

    A class to manage training data.

    Parameters
    ----------
    training_data_params: Recursive Namespace object
        Recursive Namespace object containing training data parameters
    simPSF: object
        PSFSimulator instance
    n_bins_lambda: int
        Number of bins in wavelength

    """

    def __init__(self, data_type, data_params, simPSF, n_bins_lambda):
        self.data_params = data_params.__dict__[data_type]
        self.dataset = np.load(
            os.path.join(
                self.data_params.data_dir, self.data_params.file
            ),
            allow_pickle=True,
        )[()]
        self.dataset["positions"] = tf.convert_to_tensor(
            self.dataset["positions"], dtype=tf.float32
        )
        
        if "train" in self.data_params.file:
            self.dataset["noisy_stars"] = tf.convert_to_tensor(
                self.dataset["noisy_stars"], dtype=tf.float32
            )
        elif "test" in self.data_params.file:
            self.dataset["stars"] = tf.convert_to_tensor(
            self.dataset["stars"], dtype=tf.float32
            )
            
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(
                _sed, self.simPSF, n_bins=self.n_bins_lambda, tf_dtype=tf.float64
            )
            for _sed in self.dataset["SEDs"]
        ]
        self.sed_data = tf.convert_to_tensor(self.sed_data, dtype=tf.float32)
        self.sed_data = tf.transpose(self.sed_data, perm=[0, 2, 1])
