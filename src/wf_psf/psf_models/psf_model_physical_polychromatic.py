"""PSF Model Physical Semi-Parametric Polychromatic.

A module which defines the classes and methods
to manage the parameters of the psf physical polychromatic model.

:Authors: Tobias Liaudat <tobias.liaudat@cea.fr> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models import psf_models as psfm
from wf_psf.psf_models import tf_layers as tfl
from wf_psf.utils.utils import zernike_generator
from wf_psf.data.training_preprocessing import get_obs_positions, get_zernike_prior
from wf_psf.psf_models.tf_layers import (
    TFPolynomialZernikeField,
    TFZernikeOPD,
    TFBatchPolychromaticPSF,
    TFBatchMonochromaticPSF,
    TFNonParametricPolynomialVariationsOPD,
    TFPhysicalLayer,
)
import logging


logger = logging.getLogger(__name__)


@psfm.register_psfclass
class PhysicalPolychromaticFieldFactory(psfm.PSFModelBaseFactory):
    """Factory class for the Tensor Flow Physical Polychromatic PSF Field Model.

    This factory class is responsible for instantiating instances of the Tensor Flow Physical Polychromatic PSF Field Model.
    It is registered with the PSF model factory registry.

    Parameters
    ----------
    ids : tuple
        A tuple containing identifiers for the factory class.

    Methods
    -------
    get_model_instance(model_params, training_params, data=None, coeff_mat=None)
        Instantiates an instance of the Tensor Flow Physical Polychromatic Field class with the provided parameters.
    """

    ids = ("physical_poly",)

    def get_model_instance(self, model_params, training_params, data, coeff_mat=None):
        return TFPhysicalPolychromaticField(
            model_params, training_params, data, coeff_mat
        )


class TFPhysicalPolychromaticField(tf.keras.Model):
    """Tensor Flow Physical Polychromatic PSF Field class.

    This class represents a polychromatic PSF field model with a physical layer,
    which is part of a larger PSF modeling framework.

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

    Returns
    -------
    PSF model instance
        An instance of the Physical Polychromatic PSF Field Model.

    """

    ids = ("physical_poly",)

    def __init__(self, model_params, training_params, data, coeff_mat=None):
        """Initialize the TFPhysicalPolychromaticField instance.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.
        training_params : RecursiveNamespace
            Object containing training hyperparameters for this PSF model class.
        data : DataConfigHandler
            Object containing training and test datasets and zernike prior.

        coeff_mat : Tensor or None
            Coefficient matrix defining the parametric PSF field model.

        Returns
        -------
        TFPhysicalPolychromaticField
            Initialized instance of the TFPhysicalPolychromaticField class.
        """
        super().__init__(model_params, training_params, coeff_mat)
        self._initialize_parameters_and_layers(
            model_params, training_params, data, coeff_mat
        )

    def _initialize_parameters_and_layers(
        self, model_params, training_params, data, coeff_mat=None
    ):
        """Initialize Parameters of the PSF model.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.
        training_params : RecursiveNamespace
            Object containing training hyperparameters for this PSF model class.
        data : DataConfigHandler
            Object containing training and test datasets.
        coeff_mat : Tensor or None
            Coefficient matrix defining the parametric PSF field model.
        """
        self.output_Q = model_params.output_Q
        self.obs_pos = get_obs_positions(data)
        self.l2_param = model_params.param_hparams.l2_param

        self._initialize_zernike_parameters(model_params, data)
        self._initialize_layers(model_params, training_params)

        # Initialize the model parameters with non-default value
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

    def _initialize_zernike_parameters(self, model_params, data):
        """Initialize the Zernike parameters.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parametrs for this PSF model class.

        """
        self.zks_prior = get_zernike_prior(data)
        self.n_zks_total = max(
            model_params.param_hparams.n_zernikes,
            tf.cast(tf.shape(self.zks_prior)[1], tf.int32),
        )
        self.zernike_maps = psfm.generate_zernike_maps_3d(
            self.n_zks_total, model_params.pupil_diameter
        )

    def _initialize_layers(self, model_params, training_params):
        """Initialize the layers of the PSF model.

        This method initializes the layers of the PSF model, including the physical
        layer, polynomial Zernike field, batch polychromatic layer, and non-parametric
        OPD layer.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.
        training_params : RecursiveNamespace
            Object containing training hyperparameters for this PSF model class.
        coeff_mat : Tensor or None
            Initialization of the coefficient matrix defining the parametric PSF field model.

        """
        self._initialize_physical_layer(model_params)
        self._initialize_polynomial_Z_field(model_params)
        self._initialize_Zernike_OPD(model_params)
        self._initialize_batch_polychromatic_layer(model_params, training_params)
        self._initialize_nonparametric_opd_layer(model_params, training_params)

    def _initialize_physical_layer(self, model_params):
        """Initialize the physical layer of the PSF model.

        This method initializes the physical layer of the PSF model using parameters
        specified in the `model_params` object.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.

        """
        self.tf_physical_layer = TFPhysicalLayer(
            self.obs_pos,
            self.zks_prior,
            interpolation_type=model_params.interpolation_type,
            interpolation_args=model_params.interpolation_args,
        )

    def _initialize_polynomial_Z_field(self, model_params):
        """Initialize the polynomial Zernike field of the PSF model.

        This method initializes the polynomial Zernike field of the PSF model using
        parameters specified in the `model_params` object.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.

        """
        self.tf_poly_Z_field = TFPolynomialZernikeField(
            x_lims=model_params.x_lims,
            y_lims=model_params.y_lims,
            random_seed=model_params.param_hparams.random_seed,
            n_zernikes=model_params.param_hparams.n_zernikes,
            d_max=model_params.param_hparams.d_max,
        )

    def _initialize_Zernike_OPD(self, model_params):
        """Initialize the Zernike OPD field of the PSF model.

        This method initializes the Zernike Optical Path Difference
        field of the PSF model using parameters specified in the `model_params` object.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.

        """
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TFZernikeOPD(zernike_maps=self.zernike_maps)

    def _initialize_batch_polychromatic_layer(self, model_params, training_params):
        """Initialize the batch polychromatic PSF layer.

        This method initializes the batch opd to batch polychromatic PSF layer
        using the provided `model_params` and `training_params`.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.
        training_params : RecursiveNamespace
            Object containing training hyperparameters for this PSF model class.


        """
        self.batch_size = training_params.batch_size
        self.obscurations = psfm.tf_obscurations(model_params.pupil_diameter)
        self.output_dim = model_params.output_dim

        self.tf_batch_poly_PSF = TFBatchPolychromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    def _initialize_nonparametric_opd_layer(self, model_params, training_params):
        """Initialize the non-parametric OPD layer.

        This method initializes the non-parametric OPD layer using the provided
        `model_params` and `training_params`.

        Parameters
        ----------
        model_params : RecursiveNamespace
            Object containing parameters for this PSF model class.
        training_params : RecursiveNamespace
            Object containing training hyperparameters for this PSF model class.

        """
        # self.d_max_nonparam = model_params.nonparam_hparams.d_max_nonparam
        # self.opd_dim = tf.shape(self.zernike_maps)[1].numpy()

        self.tf_np_poly_opd = TFNonParametricPolynomialVariationsOPD(
            x_lims=model_params.x_lims,
            y_lims=model_params.y_lims,
            d_max=model_params.nonparam_hparams.d_max_nonparam,
            opd_dim=tf.shape(self.zernike_maps)[1].numpy(),
        )

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """Assigns the coefficient matrix defining the parametric PSF field model.

        This method assigns the coefficient matrix to the parametric PSF field model,
        allowing for customization and modification of the PSF field.

        Parameters
        ----------
        coeff_mat : Tensor or None
            The coefficient matrix defining the parametric PSF field model.
            If None, the default coefficient matrix will be used.


        """
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def set_zero_nonparam(self):
        """Set the non-parametric part of the OPD (Optical Path Difference) to zero.

        This method sets the non-parametric component of the Optical Path Difference (OPD)
        to zero, effectively removing its contribution from the overall PSF (Point Spread Function).

        """
        self.tf_np_poly_opd.set_alpha_zero()

    def set_nonzero_nonparam(self):
        """Set the non-parametric part to non-zero values.

        This method sets the non-parametric component of the Optical Path Difference (OPD)
        to non-zero values, allowing it to contribute to the overall PSF (Point Spread Function).

        """
        self.tf_np_poly_opd.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """Set the layers to be trainable.

        A method to set layers to be trainable.

        Parameters
        ----------
        param_bool: bool
            Boolean flag for parametric model layers

        nonparam_bool: bool
            Boolean flag for non-parametric model layers

        """
        self.tf_np_poly_opd.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def pad_zernikes(self, zk_param, zk_prior):
        """Pad the Zernike coefficients to match the maximum length.

        Pad the input Zernike coefficient tensors to match the length of the
        maximum number of Zernike coefficients among the parametric and prior parts.

        Parameters
        ----------
        zk_param: tf.Tensor
            Zernike coefficients for the parametric part. Shape [batch, n_zks_param, 1, 1].
        zk_prior: tf.Tensor
            Zernike coefficients for the prior part. Shape [batch, n_zks_prior, 1, 1].

        Returns
        -------
        padded_zk_param: tf.Tensor
            Padded Zernike coefficients for the parametric part. Shape [batch, n_zks_total, 1, 1].
        padded_zk_prior: tf.Tensor
            Padded Zernike coefficients for the prior part. Shape [batch, n_zks_total, 1, 1].
        """
        # Calculate the number of Zernikes to pad for parametric and prior parts
        pad_num_param = self.n_zks_total - tf.shape(zk_param)[1]
        pad_num_prior = self.n_zks_total - tf.shape(zk_prior)[1]

        if pad_num_param != 0:
            # Pad the Zernike coefficients for parametric and prior parts
            padding_param = [(0, 0), (0, pad_num_param), (0, 0), (0, 0)]
            padded_zk_param = tf.pad(zk_param, padding_param)
        else:
            padded_zk_param = zk_param

        if pad_num_prior != 0:
            padding_prior = [(0, 0), (0, pad_num_prior), (0, 0), (0, 0)]
            padded_zk_prior = tf.pad(zk_prior, padding_prior)
        else:
            padded_zk_prior = zk_prior

        # Assert that the shapes are correct
        if padded_zk_param.shape != padded_zk_prior.shape:
            raise ValueError(
                "Shapes of padded tensors {zk_param.shape} and {zk_prior.shape} do not match."
            )

        return padded_zk_param, padded_zk_prior

    def predict_step(self, data, evaluate_step=False):
        """Predict (inference) step.

        A method to enable a special type of
        interpolation (different from training) for
        the physical layer.

        Parameters
        ----------
        data : NOT SURE

        evaluate_step : bool
            Boolean flag to evaluate step

        Returns
        -------
        poly_psfs TFBatchPolychromaticPSF
            Instance of TFBatchPolychromaticPSF class containing computed polychromatic PSFs.

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
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        """Predict a set of monochromatic Point Spread Functions (PSFs) at desired positions.

        This method calculates monochromatic PSFs based on the provided input positions,
        observed wavelength, and required wavefront dimension.

        Parameters
        ----------
        input_positions : Tensor [batch_dim, 2]
            Positions at which to compute the PSFs.
        lambda_obs : float
            Observed wavelength in micrometers (um).
        phase_N : int
            Required wavefront dimension. This should be calculated using a SimPSFToolkit
            instance. Example:
            ```
            simPSF_np = wf.SimPSFToolkit(...)
            phase_N = simPSF_np.feasible_N(lambda_obs)
            ```

        Returns
        -------
        mono_psf_batch : Tensor
            Batch of monochromatic PSFs.

        """

        # Initialise the monochromatic PSF batch calculator
        tf_batch_mono_psf = TF_batch_mono_PSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )
        # Set the lambda_obs and the phase_N parameters
        tf_batch_mono_psf.set_lambda_phaseN(phase_N, lambda_obs)

        # Predict zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)
        # Propagate to obtain the OPD
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

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
        # Predict zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)
        # Propagate to obtain the OPD
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        return opd_maps

    def compute_zernikes(self, input_positions):
        """Compute Zernike coefficients at a batch of positions.

        This method computes the Zernike coefficients for a batch of input positions
        using both the parametric model and the physical layer.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions for which to compute the Zernike coefficients.

        Returns
        -------
        zernike_coefficients : Tensor [batch, n_zks_total, 1, 1]
            Computed Zernike coefficients for the input positions.

        Notes
        -----
        This method combines the predictions from both the parametric model and
        the physical layer to obtain the final Zernike coefficients.

        """
        # Calculate parametric part
        zernike_params = self.tf_poly_Z_field(input_positions)

        # Calculate the physical layer
        zernike_prior = self.tf_physical_layer.call(input_positions)

        # Pad and sum the zernike coefficients
        padded_zernike_params, padded_zernike_prior = self.pad_zernikes(
            zernike_params, zernike_prior
        )
        zernike_coeffs = tf.math.add(padded_zernike_params, padded_zernike_prior)

        return zernike_coeffs

    def predict_zernikes(self, input_positions):
        """Predict Zernike coefficients at a batch of positions.

        This method predicts the Zernike coefficients for a batch of input positions
        using both the parametric model and the physical layer. During training,
        the prediction from the physical layer is typically not used.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions for which to predict the Zernike coefficients.

        Returns
        -------
        zernike_coeffs : Tensor [batch, n_zks_total, 1, 1]
            Predicted Zernike coefficients for the input positions.

        Notes
        -----
        At training time, the prediction from the physical layer may not be utilized,
        as the model might be trained to rely solely on the parametric part.

        """
        # Calculate parametric part
        zernike_params = self.tf_poly_Z_field(input_positions)

        # Calculate the prediction from the physical layer
        physical_layer_prediction = self.tf_physical_layer.predict(input_positions)

        # Pad and sum the Zernike coefficients
        padded_zernike_params, padded_physical_layer_prediction = self.pad_zernikes(
            zernike_params, physical_layer_prediction
        )
        zernike_coeffs = tf.math.add(
            padded_zernike_params, padded_physical_layer_prediction
        )

        return zernike_coeffs

    def call(self, inputs, training=True):
        """Define the PSF (Point Spread Function) field forward model.

        This method defines the forward model of the PSF field, which involves several steps:
        1. Transforming input positions into Zernike coefficients.
        2. Converting Zernike coefficients into Optical Path Difference (OPD) maps.
        3. Combining OPD maps with Spectral Energy Distribution (SED) information to generate
           polychromatic PSFs.

        Parameters
        ----------
        inputs : list
            List containing input data required for PSF computation. It should contain two
            elements:
            - input_positions: Tensor [batch_dim, 2]
                Positions at which to compute the PSFs.
            - packed_SEDs: Tensor [batch_dim, ...]
                Packed Spectral Energy Distributions (SEDs) for the corresponding positions.
        training : bool, optional
            Indicates whether the model is being trained or used for inference. Defaults to True.

        Returns
        -------
        poly_psfs : Tensor
            Polychromatic PSFs generated by the forward model.

        Notes
        -----
        - The `input_positions` tensor should have a shape of [batch_dim, 2], where each row
          represents the x and y coordinates of a position.
        - The `packed_SEDs` tensor should have a shape of [batch_dim, ...], containing the SED
          information for each position.
        - During training, this method computes the Zernike coefficients from the input positions
          and calculates the corresponding OPD maps. Additionally, it adds an L2 loss term based on
          the parametric OPD maps.
        - During inference, this method generates predictions using precomputed OPD maps or by
          propagating through the forward model.

        Examples
        --------
        # Usage during training
        inputs = [input_positions, packed_SEDs]
        poly_psfs = psf_model(inputs)

        # Usage during inference
        inputs = [input_positions, packed_SEDs]
        poly_psfs = psf_model(inputs, training=False)
        """
        # Unpack inputs
        input_positions = inputs[0]
        packed_SEDs = inputs[1]

        # For the training
        if training:
            # Compute zernikes from parametric model and physical layer
            zks_coeffs = self.compute_zernikes(input_positions)

            # Propagate to obtain the OPD
            param_opd_maps = self.tf_zernike_OPD(zks_coeffs)

            # Add l2 loss on the parametric OPD
            self.add_loss(
                self.l2_param * tf.math.reduce_sum(tf.math.square(param_opd_maps))
            )

            # Calculate the non parametric part
            nonparam_opd_maps = self.tf_np_poly_opd(input_positions)

            # Add the estimations
            opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

            # Compute the polychromatic PSFs
            poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])
        # For the inference
        else:
            # Compute predictions
            poly_psfs = self.predict_step(inputs, evaluate_step=True)

        return poly_psfs
