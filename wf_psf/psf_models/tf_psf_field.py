import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models.tf_layers import TF_poly_Z_field, TF_zernike_OPD, TF_batch_poly_PSF
from wf_psf.psf_models.tf_layers import TF_NP_poly_OPD, TF_batch_mono_PSF, TF_physical_layer
from wf_psf.utils.utils import PI_zernikes


class TF_PSF_field_model(tf.keras.Model):
    """Parametric PSF field model!

    Fully parametric model based on the Zernike polynomial basis.

    Parameters
    ----------
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
        self.tf_poly_Z_field = TF_poly_Z_field(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            n_zernikes=self.n_zernikes,
            d_max=self.d_max,
        )

        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TF_zernike_OPD(zernike_maps=zernike_maps)

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
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
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
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


class TF_SemiParam_field(tf.keras.Model):
    """PSF field forward model!

    Semi parametric model based on the Zernike polynomial basis. The

    Parameters
    ----------
    zernike_maps: Tensor(n_batch, opd_dim, opd_dim)
        Zernike polynomial maps.
    obscurations: Tensor(opd_dim, opd_dim)
        Predefined obscurations of the phase.
    batch_size: int
        Batch sizet
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
    d_max_nonparam: int
        Maximum degree of the polynomial for the non-parametric variations.
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

    def __init__(
        self,
        zernike_maps,
        obscurations,
        batch_size,
        output_Q,
        d_max_nonparam=3,
        l2_param=0.0,
        output_dim=64,
        n_zernikes=45,
        d_max=2,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        coeff_mat=None,
        name="TF_SemiParam_field",
    ):
        super(TF_SemiParam_field, self).__init__()

        # Inputs: oversampling used
        self.output_Q = output_Q

        # Inputs: TF_poly_Z_field
        self.n_zernikes = n_zernikes
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_NP_poly_OPD
        self.d_max_nonparam = d_max_nonparam
        self.opd_dim = tf.shape(zernike_maps)[1].numpy()

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
        self.tf_poly_Z_field = TF_poly_Z_field(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            n_zernikes=self.n_zernikes,
            d_max=self.d_max,
        )

        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TF_zernike_OPD(zernike_maps=zernike_maps)

        # Initialize the non-parametric layer
        self.tf_np_poly_opd = TF_NP_poly_OPD(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            d_max=self.d_max_nonparam,
            opd_dim=self.opd_dim,
        )

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
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

    def set_zero_nonparam(self):
        """Set to zero the non-parametric part."""
        self.tf_np_poly_opd.set_alpha_zero()

    def set_nonzero_nonparam(self):
        """Set to non-zero the non-parametric part."""
        self.tf_np_poly_opd.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """Set the layers to be trainable or not."""
        self.tf_np_poly_opd.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def set_output_Q(self, output_Q, output_dim=None):
        """Set the value of the output_Q parameter.
        Useful for generating/predicting PSFs at a different sampling wrt the
        observation sampling.
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
        """Predict a set of monochromatic PSF at desired positions.

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
        # Get beta tilde as the proyection of the first n_param_poly_terms (6 for d_max=2) onto the first n_zernikes.
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

        # Only pad in the firs dimention so we get a matrix of size (d_max_nonparam_terms)x(n_zernikes)  --> 21x15 or 21x45.
        beta_tilde = np.pad(
            beta_tilde_inner,
            [(0, S_tilde.shape[0] - beta_tilde_inner.shape[0]), (0, 0)],
            mode="constant",
        )

        # Unmix beta tilde with the inverse of alpha
        beta = tf.constant(
            np.linalg.inv(self.tf_np_poly_opd.alpha_mat) @ beta_tilde, dtype=tf.float32
        )
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


class TF_physical_poly_field(tf.keras.Model):
    """PSF field forward model with a physical layer

    WaveDiff-original with a physical layer

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
    d_max_nonparam: int
        Maximum degree of the polynomial for the non-parametric variations.
    l2_param: float
        Parameter going with the l2 loss on the opd. If it is `0.` the loss
        is not added. Default is `0.`.
    output_dim: int
        Output dimension of the PSF stamps.
    n_zks_param: int
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
    interpolation_type: str
        Option for the interpolation type of the physical layer.
        Default is no interpolation.
    interpolation_args: dict
        Additional arguments for the interpolation.

    """

    def __init__(
        self,
        zernike_maps,
        obscurations,
        batch_size,
        obs_pos,
        zks_prior,
        output_Q,
        d_max_nonparam=3,
        l2_param=0.0,
        output_dim=64,
        n_zks_param=45,
        d_max=2,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        coeff_mat=None,
        interpolation_type="none",
        interpolation_args=None,
        name="TF_physical_poly_field",
    ):
        super(TF_physical_poly_field, self).__init__(name=name)

        # Inputs: oversampling used
        self.output_Q = output_Q
        self.n_zks_total = tf.shape(zernike_maps)[0].numpy()

        # Inputs: TF_poly_Z_field
        self.n_zks_param = n_zks_param
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_physical_layer
        self.obs_pos = obs_pos
        self.zks_prior = zks_prior
        self.n_zks_prior = tf.shape(zks_prior)[1].numpy()
        self.interpolation_type = interpolation_type
        self.interpolation_args = interpolation_args

        # Inputs: TF_NP_poly_OPD
        self.d_max_nonparam = d_max_nonparam
        self.opd_dim = tf.shape(zernike_maps)[1].numpy()

        # Check if the Zernike maps are enough
        if (self.n_zks_prior > self.n_zks_total) or (
            self.n_zks_param > self.n_zks_total
        ):
            raise ValueError("The number of Zernike maps is not enough.")

        # Inputs: TF_zernike_OPD
        # They are not stored as they are memory-intensive
        # zernike_maps =[]

        # Inputs: TF_batch_poly_PSF
        self.batch_size = batch_size
        self.obscurations = obscurations
        self.output_dim = output_dim

        # Inputs: Loss
        self.l2_param = l2_param

        # Initialize the first layer
        self.tf_poly_Z_field = TF_poly_Z_field(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            n_zernikes=self.n_zks_param,
            d_max=self.d_max,
        )
        # Initialize the physical layer
        self.tf_physical_layer = TF_physical_layer(
            self.obs_pos,
            self.zks_prior,
            interpolation_type=self.interpolation_type,
            interpolation_args=self.interpolation_args,
        )
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TF_zernike_OPD(zernike_maps=zernike_maps)

        # Initialize the non-parametric layer
        self.tf_np_poly_opd = TF_NP_poly_OPD(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            d_max=self.d_max_nonparam,
            opd_dim=self.opd_dim,
        )
        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )
        # Initialize the model parameters with non-default value
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """Assign coefficient matrix."""
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def set_zero_nonparam(self):
        """Set to zero the non-parametric part."""
        self.tf_np_poly_opd.set_alpha_zero()

    def set_nonzero_nonparam(self):
        """Set to non-zero the non-parametric part."""
        self.tf_np_poly_opd.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """Set the layers to be trainable or not."""
        self.tf_np_poly_opd.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def set_output_Q(self, output_Q, output_dim=None):
        """Set the value of the output_Q parameter.
        Useful for generating/predicting PSFs at a different sampling wrt the
        observation sampling.
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

    def zks_pad(self, zk_param, zk_prior):
        """Pad the zernike coefficients with zeros to have the same length.

        Pad them to have `n_zks_total` length.

        Parameters
        ----------
        zk_param: Tensor [batch, n_zks_param, 1, 1]
            Zernike coefficients for the parametric part
        zk_prior: Tensor [batch, n_zks_prior, 1, 1]
            Zernike coefficients for the prior part

        Returns
        -------
        zk_param: Tensor [batch, n_zks_total, 1, 1]
            Zernike coefficients for the parametric part
        zk_prior: Tensor [batch, n_zks_total, 1, 1]
            Zernike coefficients for the prior part

        """
        # Calculate the number of zernikes to pad
        pad_num = tf.cast(self.n_zks_total - self.n_zks_param, dtype=tf.int32)
        # Pad the zernike coefficients
        padding = [
            (0, 0),
            (0, pad_num),
            (0, 0),
            (0, 0),
        ]
        padded_zk_param = tf.pad(zk_param, padding)

        # Calculate the number of zernikes to pad
        pad_num = tf.cast(self.n_zks_total - self.n_zks_prior, dtype=tf.int32)
        # Pad the zernike coefficients
        padding = [
            (0, 0),
            (0, pad_num),
            (0, 0),
            (0, 0),
        ]
        padded_zk_prior = tf.pad(zk_prior, padding)

        return padded_zk_param, padded_zk_prior

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
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
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

        # Compute zernikes from parametric model and physical layer
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
        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.predict_zernikes(input_positions)
        # Propagate to obtain the OPD
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        return opd_maps

    def compute_zernikes(self, input_positions):
        """Compute Zernike coefficients at a batch of positions

        This includes the parametric model and the physical layer

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions to compute the Zernikes.

        Returns
        -------
        zks_coeffs : Tensor [batch, n_zks_total, 1, 1]
            Zernikes at requested positions

        """
        # Calculate parametric part
        zks_params = self.tf_poly_Z_field(input_positions)
        # Calculate the physical layer
        zks_prior = self.tf_physical_layer.call(input_positions)
        # Pad and sum the zernike coefficients
        padded_zk_param, padded_zk_prior = self.zks_pad(zks_params, zks_prior)
        zks_coeffs = tf.math.add(padded_zk_param, padded_zk_prior)

        return zks_coeffs

    def predict_zernikes(self, input_positions):
        """Predict Zernike coefficients at a batch of positions

        This includes the parametric model and the physical layer.
        The prediction of the physical layer to positions is not used
        at training time.

        Parameters
        ----------
        input_positions: Tensor [batch_dim, 2]
            Positions to compute the Zernikes.

        Returns
        -------
        zks_coeffs : Tensor [batch, n_zks_total, 1, 1]
            Zernikes at requested positions

        """
        # Calculate parametric part
        zks_params = self.tf_poly_Z_field(input_positions)
        # Calculate the physical layer
        zks_prior = self.tf_physical_layer.predict(input_positions)
        # Pad and sum the zernike coefficients
        padded_zk_param, padded_zk_prior = self.zks_pad(zks_params, zks_prior)
        zks_coeffs = tf.math.add(padded_zk_param, padded_zk_prior)

        return zks_coeffs

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


class TF_GT_physical_field(tf.keras.Model):
    """Ground truth PSF field forward model with a physical layer

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
        super(TF_GT_physical_field, self).__init__()

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
        self.tf_physical_layer = TF_physical_layer(
            self.obs_pos,
            self.zks_prior,
            interpolation_type="none",
        )
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TF_zernike_OPD(zernike_maps=zernike_maps)

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
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
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
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


def build_PSF_model(model_inst, optimizer=None, loss=None, metrics=None):
    """Define the model-compilation parameters.

    Specially the loss function, the optimizer and the metrics.
    """
    # Define model loss function
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError()

    # Define optimizer function
    if optimizer is None:
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        )

    # Define metric functions
    if metrics is None:
        metrics = [tf.keras.metrics.MeanSquaredError()]

    # Compile the model
    model_inst.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=False,
    )

    return model_inst
