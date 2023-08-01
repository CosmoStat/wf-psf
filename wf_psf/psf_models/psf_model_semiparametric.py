"""PSF Model Semi-Parametric.

A module which defines the classes and methods
to manage the parameters of the psf semi-parametric model.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models import psf_models as psfm
from wf_psf.psf_models import tf_layers as tfl
from wf_psf.utils.utils import PI_zernikes, zernike_generator
from wf_psf.psf_models.tf_layers import (
    TF_batch_poly_PSF,
    TF_batch_mono_PSF,
)


@psfm.register_psfclass
class TF_SemiParam_field(tf.keras.Model):
    """PSF field forward model.

    Semi parametric model based on the Zernike polynomial basis.

    Parameters
    ----------
    model_params: Recursive Namespace
        Recursive Namespace object containing parameters for this PSF model class
    training_params: Recursive Namespace
        Recursive Namespace object containing training hyperparameters for this PSF model class
    coeff_mat: Tensor or None
        Initialization of the coefficient matrix defining the parametric psf field model

    """

    ids = ("poly",)

    def __init__(self, model_params, batch_size, coeff_mat=None):
        super().__init__()

        # Inputs: pupil diameter
        self.pupil_diam = model_params.pupil_diameter

        # Inputs: oversampling used
        self.output_Q = model_params.output_Q

        # Inputs: TF_poly_Z_field
        self.n_zernikes = model_params.param_hparams.n_zernikes
        self.d_max = model_params.param_hparams.d_max
        self.x_lims = model_params.x_lims
        self.y_lims = model_params.y_lims

        # Inputs: TF_NP_poly_OPD
        self.d_max_nonparam = model_params.nonparam_hparams.d_max_nonparam
        self.zernike_maps = psfm.tf_zernike_cube(self.n_zernikes, self.pupil_diam)
        self.opd_dim = tf.shape(self.zernike_maps)[1].numpy()

        # Inputs: TF_batch_poly_PSF
        self.batch_size = batch_size
        self.obscurations = psfm.tf_obscurations(self.pupil_diam, model_params)
        self.output_dim = model_params.output_dim

        # Inputs: Loss
        self.l2_param = model_params.param_hparams.l2_param

        # Inputs: Project DD model features
        self.project_dd_features = model_params.nonparam_hparams.project_dd_features

        # Inputs: Reset DD model features
        self.reset_dd_features = model_params.nonparam_hparams.reset_dd_features

        # Inputs: Save optimiser history Parametric model features
        self.save_optim_history_param = (
            model_params.param_hparams.save_optim_history_param
        )

        # Inputs: Save optimiser history NonParameteric model features
        self.save_optim_history_nonparam = (
            model_params.nonparam_hparams.save_optim_history_nonparam
        )

        # Initialize the first layer
        self.tf_poly_Z_field = tfl.TF_poly_Z_field(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            n_zernikes=self.n_zernikes,
            d_max=self.d_max,
        )

        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = tfl.TF_zernike_OPD(zernike_maps=self.zernike_maps)

        # Initialize the non-parametric (np) layer
        self.tf_np_poly_opd = tfl.TF_NP_poly_OPD(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            d_max=self.d_max_nonparam,
            opd_dim=self.opd_dim,
        )

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = tfl.TF_batch_poly_PSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

        # Initialize the model parameters with non-default value
        # self._coeff_mat = coeff_mat
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

    def get_coeff_matrix(self):
        """Get coefficient matrix.

        A function to get the coefficient matrix
        for parametric model.

        Returns
        -------
        coefficient matrix: float
            Tensor Flow coefficient matrix for the parametric PSF field model

        """
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """Assign coefficient matrix.

        A function to set the coefficient matrix.

        Parameters
        ----------
        coeff_mat: float
            Tensor Flow coefficient matrix for the parametric PSF field model

        """
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def set_zero_nonparam(self):
        """Set to zero the non-parametric part.

        A function to set non-parametric alpha parameters
        equal to zero.

        """
        self.tf_np_poly_opd.set_alpha_zero()

    def set_nonzero_nonparam(self):
        """Set to non-zero the non-parametric part.

        A function to set non-parametric alpha parameters
        equal to non-zero values.

        """
        self.tf_np_poly_opd.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """Set Trainable Layers.

        A function to set the layers to be trainable or not.

        Parameters
        ----------
        param_bool: bool
            Boolean flag for the parametric layers
        nonparam_bool: bool
            Boolean flag for the non-parametric layers

        """
        self.tf_np_poly_opd.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def set_output_Q(self, output_Q, output_dim=None):
        """Set the value of the output_Q parameter.

        Useful for generating/predicting PSFs at a different sampling wrt the
        observation sampling.

        Parameters
        ----------
        output_Q: float
            Oversampling factor
        output_dim: int
            Output dimension

        """
        self.output_Q = output_Q
        if output_dim is not None:
            self.output_dim = output_dim

        # Reinitialize the PSF batch poly generator
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        # TO Do Clean up
        """Predict a set of monochromatic PSF at desired positions.

        Parameters
        ----------
        input_positions: Tensor(batch_dim x 2)
        lambda_obs: float
            Observed wavelength in um.

        phase_N: int
            Required wavefront dimension. Should be calculated with as:
            ``simPSF_np = wf.SimPSFToolkit(...)``
            ``phase_N = simPSF_np.feasible_N(lambda_obs)``

        """
        # Initialise the monochromatic PSF batch calculator
        tf_batch_mono_psf = TF_batch_mono_PSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )
        # Set the lambda_obs and the phase_N parameters
        tf_batch_mono_psf.set_lambda_phaseN(phase_N, lambda_obs)

        # Calculate parametric part
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)

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
        input_positions: Tensor(batch_dim x 2)
            Positions to predict the OPD.

        Returns
        -------
        opd_maps : Tensor [batch x opd_dim x opd_dim]
            OPD at requested positions.

        """
        # Calculate parametric part
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        return opd_maps

    def assign_S_mat(self, S_mat):
        """Assign DD features matrix."""
        self.tf_np_poly_opd.assign_S_mat(S_mat)

    def project_DD_features(self, tf_zernike_cube):
        """
        Project non-parametric wavefront onto first n_z Zernikes and transfer
        their parameters to the parametric model.

        """
        # Compute Zernike norm for projections
        n_pix_zernike = PI_zernikes(tf_zernike_cube[0, :, :], tf_zernike_cube[0, :, :])
        # Multiply Alpha matrix with DD features matrix S
        inter_res_v2 = tf.tensordot(
            self.tf_np_poly_opd.alpha_mat[: self.tf_poly_Z_field.coeff_mat.shape[1], :],
            self.tf_np_poly_opd.S_mat,
            axes=1,
        )
        # Project over first n_z Zernikes
        # TO DO: Clean up
        delta_C_poly = tf.constant(
            np.array(
                [
                    [
                        PI_zernikes(
                            tf_zernike_cube[i, :, :],
                            inter_res_v2[j, :, :],
                            n_pix_zernike,
                        )
                        for j in range(self.tf_poly_Z_field.coeff_mat.shape[1])
                    ]
                    for i in range(self.n_zernikes)
                ]
            ),
            dtype=tf.float32,
        )
        old_C_poly = self.tf_poly_Z_field.coeff_mat
        # Corrected parametric coeff matrix
        new_C_poly = old_C_poly + delta_C_poly
        self.assign_coeff_matrix(new_C_poly)

        # Remove extracted features from non-parametric model
        # Mix DD features with matrix alpha
        S_tilde = tf.tensordot(
            self.tf_np_poly_opd.alpha_mat, self.tf_np_poly_opd.S_mat, axes=1
        )
        # TO DO: Clean Up
        # Get beta tilde as the protection of the first n_param_poly_terms (6 for d_max=2) onto the first n_zernikes.
        beta_tilde_inner = np.array(
            [
                [
                    PI_zernikes(tf_zernike_cube[j, :, :], S_tilde_slice, n_pix_zernike)
                    for j in range(self.n_zernikes)
                ]
                for S_tilde_slice in S_tilde[
                    : self.tf_poly_Z_field.coeff_mat.shape[1], :, :
                ]
            ]
        )

        # Only pad in the first dimension so we get a matrix of size (d_max_nonparam_terms)x(n_zernikes)  --> 21x15 or 21x45.
        beta_tilde = np.pad(
            beta_tilde_inner,
            [(0, S_tilde.shape[0] - beta_tilde_inner.shape[0]), (0, 0)],
            mode="constant",
        )

        # Unmix beta tilde with the inverse of alpha
        beta = tf.constant(
            np.linalg.inv(self.tf_np_poly_opd.alpha_mat) @ beta_tilde, dtype=tf.float32
        )
        # To do: Clarify comment or delete.
        # Get the projection for the unmixed features

        # Now since beta.shape[1]=n_zernikes we can take the whole beta matrix.
        S_mat_projected = tf.tensordot(beta, tf_zernike_cube, axes=[1, 0])

        # Subtract the projection from the DD features
        S_new = self.tf_np_poly_opd.S_mat - S_mat_projected
        self.assign_S_mat(S_new)

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

        # Forward model
        # Calculate parametric part
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)
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

        return poly_psfs
