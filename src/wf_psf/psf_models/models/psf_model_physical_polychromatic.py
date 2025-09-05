"""PSF Model Physical Semi-Parametric Polychromatic.

A module which defines the classes and methods
to manage the parameters of the psf physical polychromatic model.

:Authors: Tobias Liaudat <tobias.liaudat@cea.fr> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from typing import Optional
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.data.data_handler import get_data_array
from wf_psf.data.data_zernike_utils import (
    ZernikeInputsFactory,
    assemble_zernike_contributions,
    pad_tf_zernikes,
)
from wf_psf.psf_models import psf_models as psfm
from wf_psf.psf_models.tf_modules.tf_layers import (
    TFPolynomialZernikeField,
    TFZernikeOPD,
    TFBatchPolychromaticPSF,
    TFBatchMonochromaticPSF,
    TFNonParametricPolynomialVariationsOPD,
    TFPhysicalLayer,
)
from wf_psf.psf_models.tf_modules.tf_utils import ensure_tensor
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.configs_handler import DataConfigHandler
import logging


logger = logging.getLogger(__name__)


@psfm.register_psfclass
class PhysicalPolychromaticFieldFactory(psfm.PSFModelBaseFactory):
    """Factory class for the TensorFlow Physical Polychromatic PSF Field Model.

    This factory class is responsible for instantiating instances of the TensorFlow Physical Polychromatic PSF Field Model. It is registered with the PSF model factory registry.

    Parameters
    ----------
    ids : tuple
        A tuple containing identifiers for the factory class.

    Methods
    -------
    get_model_instance(model_params, training_params, data=None, coeff_mat=None)
        Instantiates an instance of the TensorFlow Physical Polychromatic Field class with the provided parameters.
    """

    ids = ("physical_poly",)

    def get_model_instance(self, model_params, training_params, data, coeff_mat=None):
        """Create an instance of the TensorFlow Physical Polychromatic Field model.

        This method instantiates a `TFPhysicalPolychromaticField` object with the given model and training parameters, and data containing prior information like Zernike coefficients. Optionally, a coefficient matrix can be provided.

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
        TFPhysicalPolychromaticField
            An instance of the TensorFlow Physical Polychromatic Field model.
        """
        return TFPhysicalPolychromaticField(
            model_params, training_params, data, coeff_mat
        )


class TFPhysicalPolychromaticField(tf.keras.Model):
    """TensorFlow Physical Polychromatic PSF Field class.

    This class represents a polychromatic PSF field model with a physical layer.
    It incorporates parametric and non-parametric modeling approaches to accurately reconstruct the point spread function (PSF) across multiple wavelengths.

    The model provides functionalities for:
    - Initializing model parameters and defining the physical PSF layer.
    - Performing forward passes and computing wavefront transformations.
    - Handling Zernike parameterization and coefficient matrices.
    - Evaluating model performance and saving optimization history.

    See individual method docstrings for more details.
    """

    def __init__(self, model_params, training_params, data, coeff_mat=None):
        """Initialize the TFPhysicalPolychromaticField instance.

        Parameters
        ----------
        model_params: Recursive Namespace
            A Recursive Namespace object containing parameters for this PSF model class.
        training_params: Recursive Namespace
            A Recursive Namespace object containing training hyperparameters for this PSF model class.
        data: DataConfigHandler or dict
            A DataConfigHandler object or dict that provides access to single or multiple datasets (e.g. train and test), as well as prior knowledge like Zernike coefficients.
        coeff_mat: Tensor or None, optional
            Coefficient matrix defining the parametric PSF field model.

        Returns
        -------
        TFPhysicalPolychromaticField
            Initialized instance of the TFPhysicalPolychromaticField class.
        """
        super().__init__(model_params, training_params, coeff_mat)
        self.model_params = model_params
        self.training_params = training_params
        self.data = data
        self.run_type = self._get_run_type(data)
        self.obs_pos = self.get_obs_pos()

        # Initialize the model parameters
        self.output_Q = model_params.output_Q
        self.l2_param = model_params.param_hparams.l2_param
        self.output_dim = model_params.output_dim

        # Initialise lazy loading of external Zernike prior
        self._external_prior = None

        # Set Zernike Polynomial Coefficient Matrix if not None
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

        # Compute contributions once eagerly (outside graph)
        zks_total_contribution_np = self._assemble_zernike_contributions().numpy()
        self._zks_total_contribution = tf.convert_to_tensor(
            zks_total_contribution_np, dtype=tf.float32
        )

        # Compute n_zks_total as int
        self._n_zks_total = max(
            self.model_params.param_hparams.n_zernikes,
            zks_total_contribution_np.shape[1],
        )

        # Precompute zernike maps as tf.float32
        self._zernike_maps = psfm.generate_zernike_maps_3d(
            n_zernikes=self._n_zks_total, pupil_diam=self.model_params.pupil_diameter
        )

        # Precompute OPD dimension
        self._opd_dim = self._zernike_maps.shape[1]

        # Precompute obscurations as tf.complex64
        self._obscurations = psfm.tf_obscurations(
            pupil_diam=self.model_params.pupil_diameter,
            N_filter=self.model_params.LP_filter_length,
            rotation_angle=self.model_params.obscuration_rotation_angle,
        )

        # Eagerly initialise model layers
        self.tf_batch_poly_PSF = self._build_tf_batch_poly_PSF()
        _ = self.tf_poly_Z_field
        _ = self.tf_np_poly_opd

    def _get_run_type(self, data):
        if hasattr(data, "run_type"):
            run_type = data.run_type
        elif isinstance(data, dict) and "run_type" in data:
            run_type = data["run_type"]
        else:
            raise ValueError("data must have a 'run_type' attribute or key")

        if run_type not in {"training", "simulation", "metrics", "inference"}:
            raise ValueError(f"Unknown run_type: {run_type}")
        return run_type

    def _assemble_zernike_contributions(self):
        zks_inputs = ZernikeInputsFactory.build(
            data=self.data,
            run_type=self.run_type,
            model_params=self.model_params,
            prior=self._external_prior if hasattr(self, "_external_prior") else None,
        )
        return assemble_zernike_contributions(
            model_params=self.model_params,
            zernike_prior=zks_inputs.zernike_prior,
            centroid_dataset=zks_inputs.centroid_dataset,
            positions=zks_inputs.misalignment_positions,
            batch_size=self.training_params.batch_size,
        )

    @property
    def save_param_history(self) -> bool:
        """Check if the model should save the optimization history for parametric features."""
        return getattr(
            self.model_params.param_hparams, "save_optim_history_param", False
        )

    @property
    def save_nonparam_history(self) -> bool:
        """Check if the model should save the optimization history for non-parametric features."""
        return getattr(
            self.model_params.nonparam_hparams, "save_optim_history_nonparam", False
        )

    def get_obs_pos(self):
        assert self.run_type in {
            "training",
            "simulation",
            "metrics",
            "inference",
        }, f"Unknown run_type: {self.run_type}"

        raw_pos = get_data_array(
            data=self.data, run_type=self.run_type, key="positions"
        )

        obs_pos = ensure_tensor(raw_pos, dtype=tf.float32)

        return obs_pos

    # === Lazy properties ===.
    @property
    def zks_total_contribution(self):
        return self._zks_total_contribution

    @property
    def n_zks_total(self):
        """Get the total number of Zernike coefficients."""
        return self._n_zks_total

    @property
    def zernike_maps(self):
        """Get Zernike maps."""
        return self._zernike_maps

    @property
    def opd_dim(self):
        return self._opd_dim

    @property
    def obscurations(self):
        return self._obscurations

    @property
    def tf_poly_Z_field(self):
        """Lazy loading of the polynomial Zernike field layer."""
        if not hasattr(self, "_tf_poly_Z_field"):
            self._tf_poly_Z_field = TFPolynomialZernikeField(
                x_lims=self.model_params.x_lims,
                y_lims=self.model_params.y_lims,
                random_seed=self.model_params.param_hparams.random_seed,
                n_zernikes=self.model_params.param_hparams.n_zernikes,
                d_max=self.model_params.param_hparams.d_max,
            )
        return self._tf_poly_Z_field

    @tf_poly_Z_field.deleter
    def tf_poly_Z_field(self):
        del self._tf_poly_Z_field

    @property
    def tf_physical_layer(self):
        """Lazy loading of the physical layer of the PSF model."""
        if not hasattr(self, "_tf_physical_layer"):
            self._tf_physical_layer = TFPhysicalLayer(
                self.obs_pos,
                self.zks_total_contribution,
                interpolation_type=self.model_params.interpolation_type,
                interpolation_args=self.model_params.interpolation_args,
            )
        return self._tf_physical_layer

    @property
    def tf_zernike_OPD(self):
        """Lazy loading of the Zernike Optical Path Difference (OPD) layer."""
        if not hasattr(self, "_tf_zernike_OPD"):
            self._tf_zernike_OPD = TFZernikeOPD(zernike_maps=self.zernike_maps)
        return self._tf_zernike_OPD

    def _build_tf_batch_poly_PSF(self):
        """Eagerly build the TFBatchPolychromaticPSF layer with numpy-based obscurations."""

        return TFBatchPolychromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    @property
    def tf_np_poly_opd(self):
        """Lazy loading of the non-parametric polynomial variations OPD layer."""
        if not hasattr(self, "_tf_np_poly_opd"):
            self._tf_np_poly_opd = TFNonParametricPolynomialVariationsOPD(
                x_lims=self.model_params.x_lims,
                y_lims=self.model_params.y_lims,
                random_seed=self.model_params.param_hparams.random_seed,
                d_max=self.model_params.nonparam_hparams.d_max_nonparam,
                opd_dim=self.opd_dim,
            )
        return self._tf_np_poly_opd

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat: Optional[tf.Tensor]) -> None:
        """Assign a coefficient matrix to the parametric PSF field model.

        This method updates the coefficient matrix used by the parametric PSF field model,
        allowing for customization or modification of the model's parameters.
        If `coeff_mat` is `None`, the model will revert to using its default coefficient matrix.

        Parameters
        ----------
        coeff_mat : Optional[tf.Tensor]
            A TensorFlow tensor representing the coefficient matrix for the PSF field model.
            If `None`, the model will use the default coefficient matrix.

        Returns
        -------
        None
        """
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def set_output_Q(self, output_Q: float, output_dim: Optional[int] = None) -> None:
        """Set the output sampling rate (output_Q) for PSF generation.

        This method updates the `output_Q` parameter, which defines the
        resampling factor for generating PSFs at different resolutions
        relative to the telescope's native sampling. It also allows optionally updating `output_dim`, which sets the output resolution of the PSF model.

        If `output_dim` is provided, the PSF model's output resolution is updated.
        The method then reinitializes the batch polychromatic PSF generator to reflect the updated parameters.

        Parameters
        ----------
        output_Q : float
            The resampling factor that determines the output PSF resolution relative to the telescope's native sampling.
        output_dim : Optional[int], default=None
            The new output dimension for the PSF model. If `None`, the output
            dimension remains unchanged.

        Returns
        -------
        None
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

        # Pad the Zernike coefficients for parametric and prior parts
        padded_zk_param = tf.cond(
            tf.not_equal(pad_num_param, 0),
            lambda: tf.pad(zk_param, [(0, 0), (0, pad_num_param), (0, 0), (0, 0)]),
            lambda: zk_param,
        )

        padded_zk_prior = tf.cond(
            tf.not_equal(pad_num_prior, 0),
            lambda: tf.pad(zk_prior, [(0, 0), (0, pad_num_prior), (0, 0), (0, 0)]),
            lambda: zk_prior,
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
        tf_batch_mono_psf = TFBatchMonochromaticPSF(
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
        padded_zernike_params, padded_zernike_prior = pad_tf_zernikes(
            zernike_params, zernike_prior, self.n_zks_total
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
        padded_zernike_params, padded_physical_layer_prediction = pad_tf_zernikes(
            zernike_params, physical_layer_prediction, self.n_zks_total
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

            # Parametric OPD maps from Zernikes
            param_opd_maps = self.tf_zernike_OPD(zks_coeffs)

            # Add L2 regularization loss on parametric OPD maps
            self.add_loss(self.l2_param * tf.reduce_sum(tf.square(param_opd_maps)))

            # Non-parametric correction
            nonparam_opd_maps = self.tf_np_poly_opd(input_positions)

            # Combine both contributions
            opd_maps = tf.add(param_opd_maps, nonparam_opd_maps)

            # Compute the polychromatic PSFs
            poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        # For the inference
        else:
            # Compute predictions
            poly_psfs = self.predict_step(inputs, evaluate_step=True)

        return poly_psfs
