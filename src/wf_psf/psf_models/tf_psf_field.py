"""Ground-Truth TensorFlow PSF Field Model.

A module with classes for the Ground-Truth TensorFlow-based PSF field models.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models.tf_layers import (
    TFZernikeOPD,
    TFBatchPolychromaticPSF,
    TFBatchMonochromaticPSF,
    TFPhysicalLayer,
)
from wf_psf.psf_models.psf_model_semiparametric import TFSemiParametricField
from wf_psf.data.training_preprocessing import get_obs_positions
from wf_psf.psf_models import psf_models as psfm
import logging

logger = logging.getLogger(__name__)


@psfm.register_psfclass
class GroundTruthSemiParamFieldFactory(psfm.PSFModelBaseFactory):
    """Factory class for the TensorFlow Ground Truth Physical PSF Field Model.

    This factory class is responsible for instantiating instances
    of the TensorFlow Ground Truth SemiParametric PSF Field Model.
    It is registered with the PSF model factory registry.

    Parameters
    ----------
    ids : tuple
        A tuple containing identifiers for the factory class.

    Methods
    -------
    get_model_instance(model_params, training_params, data, coeff_mat=None)
        Instantiates an instance of the TensorFlow Ground Truth SemiParametric Field class with the provided parameters.
    """

    ids = ("ground_truth_poly",)

    def get_model_instance(self, model_params, training_params, data, coeff_mat=None):
        """Get Model Instance.

        This method creates an instance of the TensorFlow Ground Truth SemiParametric PSF Field Model using the provided parameters.

        Parameters
        ----------
        model_params: Recursive Namespace
            A Recursive Namespace object containing parameters for this PSF model class.
        training_params: Recursive Namespace
            A Recursive Namespace object containing training hyperparameters for this PSF model class.
        data: DataConfigHandler
            A DataConfigHandler object that provides access to training and test datasets, as well as prior knowledge like Zernike coefficients.
            **Note**: This parameter is not used in this method.
        coeff_mat: Tensor or None, optional
            Coefficient matrix defining the parametric PSF field model.


        Returns
        -------
        TFGroundTruthSemiParametricField
            An instance of the TensorFlow Ground Truth SemiParametric PSF Field model.
        """
        return TFGroundTruthSemiParametricField(
            model_params, training_params, coeff_mat
        )


@psfm.register_psfclass
class GroundTruthPhysicalFieldFactory(psfm.PSFModelBaseFactory):
    """Factory class for the Tensor Flow Ground Truth Physical PSF Field Model.

    This factory class is responsible for instantiating instances of the TensorFlow Ground Truth Physical PSF Field Model.
    It is registered with the PSF model factory registry.

    Parameters
    ----------
    ids : tuple
        A tuple containing identifiers for the factory class.

    Methods
    -------
    get_model_instance(model_params, training_params, data, coeff_mat=None)
        Instantiates an instance of the TensorFlow Physical Polychromatic Field class with the provided parameters.
    """

    ids = ("ground_truth_physical_poly",)

    def get_model_instance(self, model_params, training_params, data, coeff_mat):
        """Get Model Instance.

        This method creates an instance of the TensorFlow Ground Truth SemiParametric PSF Field Model using the provided parameters.

        Parameters
        ----------
        model_params: Recursive Namespace
            A Recursive Namespace object containing parameters for this PSF model class.
        training_params: Recursive Namespace
            A Recursive Namespace object containing training hyperparameters for this PSF model class.
        data: DataConfigHandler
            A DataConfigHandler object that provides access to training and test datasets, as well as prior knowledge like Zernike coefficients.
        coeff_mat: Tensor or None, optional
            Coefficient matrix defining the parametric PSF field model.


        Returns
        -------
        TFGroundTruthPhysicalField
            An instance of the TensorFlow Ground Truth SemiParametric PSF Field model.
        """
        return TFGroundTruthPhysicalField(
            model_params, training_params, data, coeff_mat
        )


class TFGroundTruthSemiParametricField(TFSemiParametricField):
    """A TensorFlow-based Ground Truth Semi-Parametric PSF Field Model.

    This class represents a ground truth semi-parametric PSF (Point Spread Function)
    field model implemented using TensorFlow. The model generates a ground truth PSF
    field based on the provided Zernike coefficient matrix. If the coefficient matrix
    (`coeff_mat`) is provided, it will be used to produce the ground truth PSF model,
    which serves as the reference for the previously simulated PSF fields. If no
    coefficient matrix is provided, the model proceeds without it.

    Parameters
    ----------
    model_params : RecursiveNamespace
        A RecursiveNamespace object containing parameters for configuring the PSF model.
    training_params : RecursiveNamespace
        A RecursiveNamespace object containing training hyperparameters for the PSF model.
    coeff_mat : Tensor or None
        The Zernike coefficient matrix used in generating simulations of the PSF model. This
        matrix defines the Zernike polynomials up to a given order used to simulate the PSF
        field. It is only used by this model if present to produce the ground truth PSF model.
        If not provided, the model will proceed without it.

    Attributes
    ----------
    GT_tf_semiparam_field : TFSemiParametricField
        An instance of the TensorFlow-based Semi-Parametric PSF Field Model used for
        generating ground truth PSF fields.

    Methods
    -------
    __init__(model_params, training_params, data, coeff_mat)
        Initializes the TFGroundTruthSemiParametricField instance.
    """

    def __init__(self, model_params, training_params, coeff_mat):
        super().__init__(model_params, training_params, coeff_mat)

        # For the Ground truth model
        self.set_zero_nonparam()


def get_ground_truth_zernike(data):
    """Get Ground Truth Zernikes from the provided dataset.

    This method concatenates the Ground Truth Zernike from both the training
    and test datasets.

    Parameters
    ----------
    data : DataConfigHandler
        A DataConfigHandler object that provides access to training and test datasets, as well as prior knowledge like Zernike coefficients.

    Returns
    -------
    tf.Tensor
        Tensor containing the observed positions of the stars.

    Notes
    -----
    The Zernike Ground Truth are obtained by concatenating the
    Ground Truth Zernikes from both the training and test datasets
    along the 0th axis.

    """
    zernike_ground_truth = np.concatenate(
        (
            data.training_data.dataset["zernike_GT"],
            data.test_data.dataset["zernike_GT"],
        ),
        axis=0,
    )
    return tf.convert_to_tensor(zernike_ground_truth, dtype=tf.float32)


class TFGroundTruthPhysicalField(tf.keras.Model):
    """Ground Truth PSF field forward model with a physical layer.

    Ground truth PSF field used for evaluation purposes.

    Parameters
    ----------
    ids : tuple
        A tuple storing the string attribute of the PSF model class
    model_params : Recursive Namespace
        A Recursive Namespace object containing parameters for this PSF model class.
    training_params : Recursive Namespace
        A Recursive Namespace object containing training hyperparameters for this PSF model class.
    data : DataConfigHandler object
        A DataConfigHandler object containing training and tests datasets, as well as prior knowledge like Zernike coefficients.
    coeff_mat : Tensor or None
        Initialization of the coefficient matrix defining the parametric PSF field model.

    """

    def __init__(self, model_params, training_params, data, coeff_mat):
        super().__init__()

        logger.info("Initialising TFGroundTruthPhysicalField class...")
        # Inputs: oversampling used
        self.output_Q = model_params.output_Q

        # Inputs: TF_physical_layer
        self.obs_pos = get_obs_positions(data)
        self.zks_prior = get_ground_truth_zernike(data)
        self.n_zks_prior = tf.shape(self.zks_prior)[1].numpy()

        self.n_zks_total = max(
            model_params.param_hparams.n_zernikes,
            tf.cast(tf.shape(self.zks_prior)[1], tf.int32),
        )
        self.zernike_maps = psfm.generate_zernike_maps_3d(
            self.n_zks_total, model_params.pupil_diameter
        )

        # Check if the Zernike maps are enough
        if self.n_zks_prior > self.n_zks_total:
            raise ValueError("The number of Zernike maps is not enough.")

        # Inputs: TF_zernike_OPD
        # They are not stored as they are memory-intensive
        # zernike_maps =[]

        # Inputs: TF_batch_poly_PSF
        self.batch_size = training_params.batch_size
        self.obscurations = psfm.tf_obscurations(
            pupil_diam=model_params.pupil_diameter,
            N_filter=model_params.LP_filter_length,
            rotation_angle=model_params.obscuration_rotation_angle,
        )
        self.output_dim = model_params.output_dim

        # Initialize the physical layer
        self.tf_physical_layer = TFPhysicalLayer(
            self.obs_pos,
            self.zks_prior,
            interpolation_type=None,
        )
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TFZernikeOPD(zernike_maps=self.zernike_maps)

        # Initialize the batch OPD to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TFBatchPolychromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    def set_output_Q(self, output_Q: float, output_dim: int = None) -> None:
        """Set the value of the output_Q parameter.

        This method is useful for generating or predicting Point Spread Functions (PSFs) at a different
        sampling rate compared to the observation sampling. It allows for adjusting the resolution
        of the generated PSFs.

        Parameters
        ----------
        output_Q : float
            The output sampling rate factor, which determines the resolution of the PSFs to be generated.
        output_dim : int, optional
            The output dimension of the generated PSFs. If not provided, the existing dimension will be used.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        After setting the `output_Q` and optionally the `output_dim`, the PSF batch polynomial generator
        is reinitialized with the new parameters.
        """
        self.output_Q = output_Q
        if output_dim is not None:
            self.output_dim = output_dim

        # Reinitialize the PSF batch poly generator
        self.tf_batch_poly_PSF = TFBatchPolychromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    def predict_step(self, data: tuple, evaluate_step: bool = False) -> tf.Tensor:
        """ "Apply custom prediction (inference) step for the PSF model.

        This method applies a specialized interpolation required by the physical layer, distinct from the training process. It processes the input data, computes the Zernike coefficients, propagates them to obtain the Optical Path Difference (OPD), and generates the corresponding polychromatic PSFs.

        Parameters
        ----------
        data : tuple
            Input data for prediction. Expected to be a tuple containing positions (tf.Tensor) and Spectral Energy Distributions (SEDs) (tf.Tensor), or a batch of data.
        evaluate_step : bool, optional
            If True, `data` is used as-is. Otherwise, it is formatted and unpacked using `data_adapter.expand_1d` and `data_adapter.unpack_x_y_sample_weight`. Default is False.

        Returns
        -------
        poly_psfs : tf.Tensor
            A tensor of shape `[batch_size, output_dim, output_dim]` containing the computed polychromatic PSFs.

        Notes
        -----
        - The method assumes `data_adapter.expand_1d` and `data_adapter.unpack_x_y_sample_weight`
        are used to properly format and extract `input_positions` and `packed_SEDs`.
        - Unlike standard Keras `predict_step`, this implementation does not expect `sample_weight`.
        """
        if evaluate_step:
            input_data = data
        else:
            # Format input data
            data = data_adapter.expand_1d(data)
            input_data, _, _ = data_adapter.unpack_x_y_sample_weight(data)

        # Unpack inputs
        input_positions = input_data[0]
        packed_SEDs = input_data[1]

        # Predict Zernike coefficients from the parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)

        # Propagate to obtain the Optical Path Difference (OPD)
        opd_maps = self.tf_zernike_OPD(zks_coeffs)

        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs

    def predict_mono_psfs(
        self, input_positions: tf.Tensor, lambda_obs: float, phase_N: int
    ):
        """Predict a batch of monochromatic PSFs at the specified positions.

        This method calculates the monochromatic Point Spread Functions (PSFs) for
        a given set of input positions, using the observed wavelength and wavefront
        dimension. The PSFs are computed using a parametric model and a physical layer.

        Parameters
        ----------
        input_positions : tf.Tensor [batch_dim, 2]
            A tensor containing the positions (in 2D space) at which to compute the PSF.
            Shape should be `[batch_dim, 2]`, where `batch_dim` is the batch size.
        lambda_obs : float
            The observed wavelength (in micrometers) for which the monochromatic PSFs are to be computed.
        phase_N : int
            The required wavefront dimension. This should be calculated using the following:
            ```
            simPSF_np = wf_psf.sims.psf_simulator.PSFSimulator(...)
            phase_N = simPSF_np.feasible_N(lambda_obs)
            ```

        Returns
        -------
        mono_psf_batch : tf.Tensor
            A tensor containing the computed batch of monochromatic PSFs for the input positions.
            The shape of the tensor depends on the model's implementation.

        Notes
        -----
        The method uses the `TFBatchMonochromaticPSF` class to handle the computation of
        monochromatic PSFs. The `set_lambda_phaseN` method is called to set the observed
        wavelength and wavefront dimension before the PSFs are computed.
        """
        # Initialise the monochromatic PSF batch calculator
        tf_batch_mono_psf = TFBatchMonochromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

        # Set the observed wavelength and wavefront dimension
        tf_batch_mono_psf.set_lambda_phaseN(phase_N, lambda_obs)

        # Predict Zernike coefficients from the parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)

        # Propagate to obtain the Optical Path Difference (OPD)
        opd_maps = self.tf_zernike_OPD(zks_coeffs)

        # Compute the monochromatic PSFs
        mono_psf_batch = tf_batch_mono_psf(opd_maps)

        return mono_psf_batch

    def predict_opd(self, input_positions: tf.Tensor):
        """Predict the Optical Path Difference (OPD) at the specified positions.

        This method uses the `predict_zernikes` method to compute Zernike coefficients.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions to predict the OPD.

        Returns
        -------
        opd_maps : Tensor [batch, opd_dim, opd_dim]
            OPD at requested positions.

        """
        # Predict Zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)

        # Propagate to obtain the OPD
        opd_maps = self.tf_zernike_OPD(zks_coeffs)

        return opd_maps

    def compute_zernikes(self, input_positions):
        """Compute Zernike coefficients at a batch of positions.

        This only includes the physical layer

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions to compute the Zernikes.

        Returns
        -------
        zks_coeffs : Tensor [batch, n_zks_total, 1, 1]
            Zernikes at requested positions

        """
        # Calculate the physical layer
        return self.tf_physical_layer.call(input_positions)

    def predict_zernikes(self, input_positions):
        """Predict Zernike coefficients at a batch of positions.

        This only includes the physical layer.
        For the moment, it is the same as the `compute_zernikes`.
        No interpolation done to avoid interpolation error in the metrics.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions to compute the Zernikes.

        Returns
        -------
        zks_coeffs : Tensor [batch, n_zks_total, 1, 1]
            Zernikes at requested positions

        """
        # Calculate the physical layer
        return self.tf_physical_layer.predict(input_positions)

    def call(self, inputs, training=True):
        """Define the PSF field forward model.

        [1] From positions to Zernike coefficients
        [2] From Zernike coefficients to OPD maps
        [3] From OPD maps and SED info to polychromatic PSFs

        OPD: Optical Path Differences
        """
        # Unpack inputs
        input_positions = inputs[0]
        packed_SEDs = inputs[1]

        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.compute_zernikes(input_positions)

        # Propagate to obtain the OPD
        opd_maps = self.tf_zernike_OPD(zks_coeffs)

        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs
