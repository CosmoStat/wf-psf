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
from wf_psf.data.training_preprocessing import get_obs_positions, get_zernike_prior
from wf_psf.psf_models import psf_models as psfm
import logging

logger = logging.getLogger(__name__)


@psfm.register_psfclass
class GroundTruthSemiParamFieldFactory(psfm.PSFModelBaseFactory):
    """Factory class for the Tensor Flow Ground Truth Physical PSF Field Model.

    This factory class is responsible for instantiating instances
    of the TensorFlow Ground Truth SemiParametric PSF Field Model.
    It is registered with the PSF model factory registry.

    Parameters
    ----------
    ids : tuple
        A tuple containing identifiers for the factory class.

    Methods
    -------
    get_model_instance(model_params, training_params, data=None, coeff_mat=None)
        Instantiates an instance of the TensorFlow SemiParametric `poly` Field class with the provided parameters.
    """

    ids = ("ground_truth_poly",)

    def get_model_instance(
        self, model_params, training_params, data, dataset, coeff_mat=None
    ):
        return TFGroundTruthSemiParametricField(
            model_params, training_params, dataset, coeff_mat
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
    get_model_instance(model_params, training_params, data=None, coeff_mat=None)
        Instantiates an instance of the TensorFlow Physical Polychromatic Field class with the provided parameters.
    """

    ids = ("ground_truth_physical_poly",)

    def get_model_instance(
        self, model_params, training_params, data, dataset, coeff_mat=None
    ):
        return TFGroundTruthPhysicalField(
            model_params, training_params, data, coeff_mat
        )


def get_ground_truth_zernike(data):
    """Get Ground Truth Zernikes from the provided dataset.

    This method concatenates the Ground Truth Zernike from both the training
    and test datasets.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

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


class TFGroundTruthSemiParametricField(TFSemiParametricField):
    """A TensorFlow-based Ground Truth Semi-Parametric PSF Field Model.

    This class represents a ground truth semi-parametric PSF (Point Spread Function)
    field model implemented using TensorFlow. The model is designed to generate
    ground truth PSF fields based on provided parameters and coefficient matrices.

    Parameters
    ----------
    model_params : RecursiveNamespace
        A RecursiveNamespace object containing parameters for configuring the PSF model.
    training_params : RecursiveNamespace
        A RecursiveNamespace object containing training hyperparameters for the PSF model.
    dataset : dict or None
        A dictionary containing dataset [test, training] required for model initialization, including the
        coefficient matrix ('C_poly') for the semi-parametric PSF field.
    coeff_mat : Tensor or None, optional
        The coefficient matrix defining the semi-parametric PSF field model. This matrix
        specifies the coefficients for the Zernike polynomials used in the PSF field.
        If None, the model will be initialized without a coefficient matrix.

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

    def __init__(self, model_params, training_params, dataset, coeff_mat=None):
        super().__init__(model_params, training_params, coeff_mat)

        # For the Ground truth model
        self.tf_poly_Z_field.assign_coeff_matrix(dataset["C_poly"])
        self.set_zero_nonparam()


class TFGroundTruthPhysicalField(tf.keras.Model):
    """Ground Truth PSF field forward model with a physical layer.

    Ground truth PSF field used for evaluation purposes.

    Parameters
    ----------
    ids : tuple
        A tuple storing the string attribute of the PSF model class
    model_params : Recursive Namespace
        A Recursive Namespace object containing parameters for this PSF model class
    training_params : Recursive Namespace
        A Recursive Namespace object containing training hyperparameters for this PSF model class
    data : DataConfigHandler object
        A DataConfigHandler object containing training and tests datasets
    coeff_mat : Tensor or None
        Initialization of the coefficient matrix defining the parametric psf field model

    """

    def __init__(self, model_params, training_params, data, coeff_mat):
        super(TFGroundTruthPhysicalField, self).__init__()

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
        self.obscurations = psfm.tf_obscurations(model_params.pupil_diameter)
        self.output_dim = model_params.output_dim

        # Initialize the physical layer
        self.tf_physical_layer = TFPhysicalLayer(
            self.obs_pos,
            self.zks_prior,
            interpolation_type=None,
        )
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TFZernikeOPD(zernike_maps=self.zernike_maps)

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TFBatchPolychromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    def set_output_Q(self, output_Q, output_dim=None):
        """Set the value of the output_Q parameter.
        Useful for generating/predicting PSFs at a different sampling wrt the
        observation sampling.
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

    def predict_step(self, data, evaluate_step=False):
        r"""Custom predict (inference) step.

        It is needed as the physical layer requires a special
        interpolation (different from training).

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

        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)
        # Propagate to obtain the OPD
        opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        """Predict a set of monochromatic PSF at desired positions.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions at which to compute the PSF
        lambda_obs: float
            Observed wavelength in um.
        phase_N: int
            Required wavefront dimension. Should be calculated with as:
            ``simPSF_np = wf_psf.sims.psf_simulator.PSFSimulator(...)``
            ``phase_N = simPSF_np.feasible_N(lambda_obs)``

        """

        # Initialise the monochromatic PSF batch calculator
        tf_batch_mono_psf = TFBatchMonochromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )
        # Set the lambda_obs and the phase_N parameters
        tf_batch_mono_psf.set_lambda_phaseN(phase_N, lambda_obs)

        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)
        # Propagate to obtain the OPD
        opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Compute the monochromatic PSFs
        mono_psf_batch = tf_batch_mono_psf(opd_maps)

        return mono_psf_batch

    def predict_opd(self, input_positions):
        """Predict the OPD at some positions.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions to predict the OPD.

        Returns
        -------
        opd_maps : Tensor [batch, opd_dim, opd_dim]
            OPD at requested positions.

        """
        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)
        # Propagate to obtain the OPD
        opd_maps = self.tf_zernike_OPD(zks_coeffs)

        return opd_maps

    def compute_zernikes(self, input_positions):
        """Compute Zernike coefficients at a batch of positions

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
        """Predict Zernike coefficients at a batch of positions

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
