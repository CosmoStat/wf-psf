"""PSF Model Parametric.

A module which defines the classes and methods
to manage the parameters of the psf parametric model.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models.psf_models import register_psfclass
from wf_psf.psf_models.tf_layers import (
    TFPolynomialZernikeField,
    TFZernikeOPD,
    TFBatchPolychromaticPSF,
    TFBatchMonochromaticPSF,
    TFNonParametricPolynomialVariationsOPD,
    TFPhysicalLayer,
)


@register_psfclass
class TFParametricPSFFieldModel(tf.keras.Model):
    """Parametric PSF field model!

    Fully parametric model based on the Zernike polynomial basis.

    Parameters
    ----------
    ids: tuple
        A tuple storing the string attribute of the PSF model class
    zernike_maps: Tensor(n_batch, opd_dim, opd_dim)
        Zernike polynomial maps.
    obscurations: Tensor(opd_dim, opd_dim)
        Predefined obscurations of the phase.
    batch_size: int
        Batch size.
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
    l2_param: float
        Parameter going with the l2 loss on the opd. If it is `0.` the loss
        is not added. Default is `0.`.
    output_dim: int
        Output dimension of the PSF stamps.
    n_zernikes: int
        Order of the Zernike polynomial for the parametric model.
    d_max: int
        Maximum degree of the polynomial for the Zernike coefficient variations.
    x_lims: [float, float]
        Limits for the x coordinate of the PSF field.
    y_lims: [float, float]
        Limits for the x coordinate of the PSF field.
    coeff_mat: Tensor or None
        Initialization of the coefficient matrix defining the parametric psf
        field model.

    """

    ids = ("parametric",)

    def __init__(
        self,
        zernike_maps,
        obscurations,
        batch_size,
        output_Q,
        l2_param=0.0,
        output_dim=64,
        n_zernikes=45,
        d_max=2,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        coeff_mat=None,
        name="TF_PSF_field_model",
    ):
        super(TF_PSF_field_model, self).__init__()

        self.output_Q = output_Q

        # Inputs: TF_poly_Z_field
        self.n_zernikes = n_zernikes
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_zernike_OPD
        # They are not stored as they are memory-heavy
        # zernike_maps =[]

        # Inputs: TF_batch_poly_PSF
        self.batch_size = batch_size
        self.obscurations = obscurations
        self.output_dim = output_dim

        # Inputs: Loss
        self.l2_param = l2_param

        # Initialize the first layer
        self.tf_poly_Z_field = TFPolynomialZernikeField(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            n_zernikes=self.n_zernikes,
            d_max=self.d_max,
        )

        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TFZernikeOPD(zernike_maps=zernike_maps)

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TFBatchPolychromaticPSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

        # Initialize the model parameters with non-default value
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

        # # Depending on the parameter we define the forward model
        # # This is, we add or not the L2 loss to the OPD.
        # if self.l2_param == 0.:
        #     self.call = self.call_basic
        # else:
        #     self.call = self.call_l2_opd_loss

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """Assign coefficient matrix."""
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

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

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        """Predict a set of monochromatic PSF at desired positions.

        input_positions: Tensor(batch_dim x 2)

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

        # Continue the OPD maps
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        opd_maps = self.tf_zernike_OPD(zernike_coeffs)

        # Compute the monochromatic PSFs
        mono_psf_batch = tf_batch_mono_psf(opd_maps)

        return mono_psf_batch

    def predict_opd(self, input_positions):
        """Predict the OPD at some positions.

        Parameters
        ----------
        input_positions: Tensor(batch_dim x 2)
            Positions to predict the OPD.

        Returns
        -------
        opd_maps : Tensor [batch x opd_dim x opd_dim]
            OPD at requested positions.

        """
        # Continue the OPD maps
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        opd_maps = self.tf_zernike_OPD(zernike_coeffs)

        return opd_maps

    def call(self, inputs):
        """Define the PSF field forward model.

        [1] From positions to Zernike coefficients
        [2] From Zernike coefficients to OPD maps
        [3] From OPD maps and SED info to polychromatic PSFs

        OPD: Optical Path Differences
        """
        # Unpack inputs
        input_positions = inputs[0]
        packed_SEDs = inputs[1]

        # Continue the forward model
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        opd_maps = self.tf_zernike_OPD(zernike_coeffs)
        # Add l2 loss on the OPD
        self.add_loss(self.l2_param * tf.math.reduce_sum(tf.math.square(opd_maps)))
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs
