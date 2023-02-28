"""Training Data Processing.
Rename module to training data processing.
A module to load and preprocess training and test data.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import numpy as np
import wf_psf.utils.utils as utils
import tensorflow as tf
import tensorflow_addons as tfa

import wf_psf.SimPSFToolkit as SimPSFToolkit


def load_dataset_dict(filename,allow_pickle_flag=True):
    """Load Numpy Dataset Dictionary.

    A function to load dataset dictionary.

    Parameters
    ----------
    filename: str
        Name of file 
    allow_pickle_flag: bool
        Boolean flag to set when loading numpy files
    
    """
    dataset = np.load(filename, allow_pickle=allow_pickle_flag)[()]
    return dataset

    # Convert to tensor
   # tf_noisy_train_stars = tf.convert_to_tensor(
  #      train_dataset["noisy_stars"], dtype=tf.float32)



class TrainingDataHandler:
    """Training Data Handler.

    maybe rename to TensorFlow Handler
    or TrainingDataSetup

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
        self.train_dataset = load_dataset_dict(self.training_data_params.file)
        self.train_dataset["positions"] = tf.convert_to_tensor(self.train_dataset["positions"], dtype=tf.float32)
        self.train_dataset["noisy_stars"] = tf.convert_to_tensor(self.train_dataset["noisy_stars"], dtype=tf.float32)
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
       # self.sed_data = self._convert_SED_format_to_tensor_flow()
        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(_sed, self.simPSF, n_bins=self.n_bins_lambda)
            for _sed in self.train_dataset["SEDs"]
        ]
        self.sed_data = tf.transpose(self.sed_data, perm=[0, 2, 1])
        breakpoint()

    @property
    def training_data_filename(self):
        return self.training_data_params.file

    def _convert_SED_format_to_tensor_flow(self):
        """Temporary Function"""
        # Initialize the SED data list
        packed_SED_data = [
            utils.alternative_generate_SED_elems(_sed, self.simPSF, n_bins=self.n_bins_lambda)
            for _sed in self.train_dataset["SEDs"]
        ]

        # Prepare the inputs for the training
       # tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)

        tf_packed_SED_data = tf.transpose(packed_SED_data, perm=[0, 2, 1])
        return tf_packed_SED_data



class TestDataHandler:
    """Test Data.

    A class to handle test data.
    
    """
    def __init__(self, test_data_params):
        self.test_data_params = test_data_params
        self.test_dataset = load_dataset_dict(self.test_data_params.file)
        self.test_dataset["stars"] = tf.convert_to_tensor(self.test_dataset["stars"], dtype=tf.float32)
        self.test_dataset["positions"] = tf.convert_to_tensor(self.test_dataset["positions"], dtype=tf.float32)
        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(_sed, self.simPSF, n_bins=self.n_bins_lambda)
            for _sed in self.test_dataset["SEDs"]
        ]
    # Prepare validation data inputs
    # JP: not necessary -- just making a duplicate copy in memory
   # validation_SEDs = test_SEDs
   # tf_validation_pos = tf_test_pos
   # tf_validation_stars = tf_test_stars



        # Prepare the inputs for the validation
        tf_validation_packed_SED_data = tf.convert_to_tensor(validation_packed_SED_data, dtype=tf.float32)
        tf_validation_packed_SED_data = tf.transpose(tf_validation_packed_SED_data, perm=[0, 2, 1])



def get_SED_validation_data(simPSF, test_SEDs, nbins_lambda):
    """Get Validation SED Data.

    A function to get validation SED data.

    Parameters
    ----------
    simPSF: simPSF object
        SIM PSF object
    test_SEDS:
        SED test data
    nbins_lambda:
        Number of wavelength bins

    Returns
    -------
    tf_validation_SED_data: 
        Validation SED data in Tensor Flow format
    """
    # Initialize the SED data list
    validation_packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF, n_bins=nbins_lambda)
        for _sed in test_SEDs
    ]

    # Prepare the inputs for the validation
    tf_validation_SED_data = tf.convert_to_tensor(validation_packed_SED_data, dtype=tf.float32)
    tf_validation_SED_data = tf.transpose(tf_validation_SED_data, perm=[0, 2, 1])

    return  tf_validation_SED_data