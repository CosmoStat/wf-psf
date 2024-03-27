import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models.tf_layers import (
    TFZernikeOPD,
    TFBatchPolychromaticPSF,
    TFBatchMonochromaticPSF,
    TFPhysicalLayer,
)


class TFGroundTruthPhysicalField(tf.keras.Model):
    """Ground Truth PSF field forward model with a physical layer

    Ground truth PSF field used for evaluation purposes.

    Parameters
    ----------
    zernike_maps: Tensor(n_batch, opd_dim, opd_dim)
        Zernike polynomial maps.
    obscurations: Tensor(opd_dim, opd_dim)
        Predefined obscurations of the phase.
    batch_size: int
        Batch size
    obs_pos: Tensor(n_stars, 2)
        The positions of all the stars
    zks_prior: Tensor(n_stars, n_zks)
        The Zernike coeffients of the prior for all the stars
    output_Q: float
        Oversampling used. This should match the oversampling Q used to generate
        the diffraction zero padding that is found in the input `packed_SEDs`.
        We call this other Q the `input_Q`.
        In that case, we replicate the original sampling of the model used to
        calculate the input `packed_SEDs`.
        The final oversampling of the generated PSFs with respect to the
        original instrument sampling depend on the division `input_Q/output_Q`.
        It is not recommended to use `output_Q < 1`.
        Although it works with float values it is better to use integer values.
    output_dim: int
        Output dimension of the PSF stamps.


    """

    def __init__(
        self,
        zernike_maps,
        obscurations,
        batch_size,
        obs_pos,
        zks_prior,
        output_Q,
        output_dim=64,
        name="TF_GT_physical_field",
    ):
        super(TFGroundTruthPhysicalField, self).__init__()

        # Inputs: oversampling used
        self.output_Q = output_Q
        self.n_zks_total = tf.shape(zernike_maps)[0].numpy()

        # Inputs: TF_physical_layer
        self.obs_pos = obs_pos
        self.zks_prior = zks_prior
        self.n_zks_prior = tf.shape(zks_prior)[1].numpy()

        # Check if the Zernike maps are enough
        if self.n_zks_prior > self.n_zks_total:
            raise ValueError("The number of Zernike maps is not enough.")

        # Inputs: TF_zernike_OPD
        # They are not stored as they are memory-intensive
        # zernike_maps =[]

        # Inputs: TF_batch_poly_PSF
        self.batch_size = batch_size
        self.obscurations = obscurations
        self.output_dim = output_dim

        # Initialize the physical layer
        self.tf_physical_layer = TFPhysicalLayer(
            self.obs_pos,
            self.zks_prior,
            interpolation_type="none",
        )
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TFZernikeOPD(zernike_maps=zernike_maps)

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
