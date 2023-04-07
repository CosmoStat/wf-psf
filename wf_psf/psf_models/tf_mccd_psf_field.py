import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.psf_models.tf_layers import TF_poly_Z_field, TF_zernike_OPD, TF_batch_poly_PSF
from wf_psf.psf_models.tf_layers import TF_NP_MCCD_OPD_v2, TF_NP_GRAPH_OPD
from wf_psf.psf_models.tf_layers import TF_batch_mono_PSF
from wf_psf.utils.graph_utils import GraphBuilder
from wf_psf.utils.utils import calc_poly_position_mat


class TF_SP_MCCD_field(tf.keras.Model):
    r"""Semi-parametric MCCD PSF field model!

    Semi parametric model based on the hybrid-MCCD matrix factorization scheme.

    The forward model is different for the training procedure and for the
    inference procedure. This makes things more complicated and requires several
    custom functions.

    The prediction step is the forward model for the inference while the
    ``call(inputs, trainable=True)`` is the forward model for the training
    procedure. When calling ``call(inputs, trainable=False)`` we are falling
    back to the predict function for the inference forward model. This is needed
    in order to handle the calculation of validation metrics on the validation
    dataset.


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
    spatial_dic:
        TODO ...
    output_Q: int
        Downsampling rate to match the specified telescope's sampling. The value
        of `output_Q` should be equal to `oversampling_rate` in order to have
        the right pixel sampling corresponding to the telescope characteristics
        `pix_sampling`, `tel_diameter`, `tel_focal_length`. The final
        oversampling obtained is `oversampling_rate/output_Q`.
        Default is `1`, so the output psf will be super-resolved by a factor of
        `oversampling_rate`.
    l2_param: float
        Parameter going with the l2 loss on the opd. If it is `0.` the loss
        is not added. Default is `0.`.
    d_max_nonparam: int
        Maximum degree of the polynomial for the non-parametric variations.
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
        obs_pos,
        spatial_dic,
        output_Q,
        l2_param=0.0,
        d_max_nonparam=3,
        graph_features=6,
        l1_rate=1e-3,
        output_dim=64,
        n_zernikes=45,
        d_max=2,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        coeff_mat=None,
        name="TF_SP_MCCD_field",
    ):
        super(TF_SP_MCCD_field, self).__init__()

        # Inputs: oversampling used
        self.output_Q = output_Q

        # Inputs: TF_poly_Z_field
        self.n_zernikes = n_zernikes
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_NP_MCCD_OPD
        self.d_max_nonparam = d_max_nonparam
        self.opd_dim = tf.shape(zernike_maps)[1].numpy()
        self.graph_features = graph_features
        self.l1_rate = l1_rate

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
        self.tf_NP_mccd_OPD = TF_NP_MCCD_OPD_v2(
            obs_pos=obs_pos,
            spatial_dic=spatial_dic,
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            d_max=self.d_max_nonparam,
            graph_features=self.graph_features,
            l1_rate=self.l1_rate,
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

    def set_zero_nonparam(self):
        r"""Set to zero the non-parametric part."""
        self.tf_NP_mccd_OPD.set_alpha_zero()

    def set_output_Q(self, output_Q, output_dim=None):
        r"""Set the value of the output_Q parameter.
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

    def set_l1_rate(self, new_l1_rate):
        r"""Set l1 rate the non-parametric part."""
        self.l1_rate = new_l1_rate
        self.tf_NP_mccd_OPD.l1_rate = new_l1_rate

    def set_nonzero_nonparam(self):
        r"""Set to non-zero the non-parametric part."""
        self.tf_NP_mccd_OPD.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        r"""Set the layers to be trainable or not."""
        self.tf_NP_mccd_OPD.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        r"""Assign coefficient matrix."""
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def predict_step(self, data, evaluate_step=False):
        r"""Custom predict (inference) step.

        It is needed as the non-parametric MCCD part requires a special
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

        # Calculate parametric part
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)

        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_NP_mccd_OPD.predict(input_positions)

        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        r"""Predict a set of monochromatic PSF at desired positions.

        Parameters
        ----------
        input_positions: Tensor(batch x 2)
            Positions to predict the monochromatic PSFs.
        lambda_obs: float
            Observed wavelength in um.
        phase_N: int
            Required wavefront dimension. Should be calculated with as:
            ``simPSF_np = wf.SimPSFToolkit(...)``
            ``phase_N = simPSF_np.feasible_N(lambda_obs)``

        Returns
        -------
        mono_psf_batch: Tensor [batch x output_dim x output_dim]
            Batch of monochromatic PSFs at requested positions and
            wavelength.
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
        nonparam_opd_maps = self.tf_NP_mccd_OPD.predict(input_positions)

        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        # Compute the monochromatic PSFs
        mono_psf_batch = tf_batch_mono_psf(opd_maps)

        return mono_psf_batch

    def predict_opd(self, input_positions):
        r"""Predict the OPD at some positions.

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
        nonparam_opd_maps = self.tf_NP_mccd_OPD.predict(input_positions)

        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        return opd_maps

    def call(self, inputs, training=True):
        r"""Define the PSF field forward model.

        [1] From positions to Zernike coefficients
        [2] From Zernike coefficients to OPD maps
        [3] From OPD maps and SED info to polychromatic PSFs

        OPD: Optical Path Differences
        """
        # Unpack inputs
        input_positions = inputs[0]
        packed_SEDs = inputs[1]

        # Forward model
        # For the training
        if training:
            # Calculate parametric part
            zernike_coeffs = self.tf_poly_Z_field(input_positions)
            param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)
            # Calculate the non parametric part
            nonparam_opd_maps = self.tf_NP_mccd_OPD(input_positions)
            # Add l2 loss on the parmetric OPD
            self.add_loss(
                self.l2_param * tf.math.reduce_sum(tf.math.square(nonparam_opd_maps))
            )
            # Add the estimations
            opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
            # Compute the polychromatic PSFs
            poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        # For the inference
        # This is absolutely needed to compute the metrics on the
        # validation data.
        else:
            # Compute predictions
            poly_psfs = self.predict_step(inputs, evaluate_step=True)

        return poly_psfs


def build_mccd_spatial_dic(
    obs_stars, obs_pos, x_lims, y_lims, d_max=2, graph_features=6, verbose=0
):
    """Build the spatial-constraint dictionary.

    Based on the hybrid approach from the MCCD model.
    """
    # The obs_data needs to be in RCA format (with the batch dim at the end)

    # Graph parameters
    graph_kwargs = {
        "obs_data": obs_stars.swapaxes(0, 1).swapaxes(1, 2),
        "obs_pos": obs_pos,
        "obs_weights": np.ones_like(obs_stars),
        "n_comp": graph_features,
        "n_eigenvects": 5,
        "n_iter": 3,
        "ea_gridsize": 10,
        "distances": None,
        "auto_run": True,
        "verbose": verbose,
    }

    # Compute graph-spatial constraint matrix
    VT = GraphBuilder(**graph_kwargs).VT

    # Compute polynomial-spatial constaint matrix
    tf_Pi = calc_poly_position_mat(
        pos=obs_pos, x_lims=x_lims, y_lims=y_lims, d_max=d_max
    )

    # Need to translate to have the batch dimension first
    spatial_dic = np.concatenate((tf_Pi.numpy(), VT), axis=0).T

    # Return the tf spatial dictionary
    return tf.convert_to_tensor(spatial_dic, dtype=tf.float32)


def build_mccd_spatial_dic_v2(
    obs_stars, obs_pos, x_lims, y_lims, d_max=2, graph_features=6, verbose=0
):
    """Build the spatial-constraint dictionaries.

    Based on the hybrid approach from the MCCD model.
    Returns the polynomial dict and the graph dict.
    """
    # The obs_data needs to be in RCA format (with the batch dim at the end)

    # Graph parameters
    graph_kwargs = {
        "obs_data": obs_stars.swapaxes(0, 1).swapaxes(1, 2),
        "obs_pos": obs_pos,
        "obs_weights": np.ones_like(obs_stars),
        "n_comp": graph_features,
        "n_eigenvects": 5,
        "n_iter": 3,
        "ea_gridsize": 10,
        "distances": None,
        "auto_run": True,
        "verbose": verbose,
    }

    # Compute graph-spatial constraint matrix
    VT = GraphBuilder(**graph_kwargs).VT

    # Compute polynomial-spatial constaint matrix
    tf_Pi = calc_poly_position_mat(
        pos=obs_pos, x_lims=x_lims, y_lims=y_lims, d_max=d_max
    )

    # Return the poly dictionary and the graph dictionary
    return tf.transpose(tf_Pi, perm=[1, 0]), tf.convert_to_tensor(
        VT.T, dtype=tf.float32
    )




class TF_SP_graph_field(tf.keras.Model):
    #   ids=("graph",)
    r"""Semi-parametric graph-constraint-only PSF field model!

    Semi parametric model based on the graph-constraint-only matrix factorization scheme.

    Parameters
    ----------
    zernike_maps: Tensor(n_batch, opd_dim, opd_dim)
        Zernike polynomial maps.
    obscurations: Tensor(opd_dim, opd_dim)
        Predefined obscurations of the phase.
    batch_size: int
        Batch size
    d_max_nonparam: int
        Maximum degree of the polynomial for the non-parametric variations.
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
        obs_pos,
        spatial_dic,
        output_Q,
        l2_param=0.0,
        graph_features=6,
        l1_rate=1e-3,
        output_dim=64,
        n_zernikes=45,
        d_max=2,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        coeff_mat=None,
        name="TF_SP_graph_field",
    ):
        super(TF_SP_graph_field, self).__init__()

        # Inputs: oversampling used
        self.output_Q = output_Q

        # Inputs: TF_poly_Z_field
        self.n_zernikes = n_zernikes
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_NP_GRAPH_OPD
        self.opd_dim = tf.shape(zernike_maps)[1].numpy()
        self.graph_features = graph_features
        self.l1_rate = l1_rate

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
        self.tf_NP_graph_OPD = TF_NP_GRAPH_OPD(
            obs_pos=obs_pos,
            spatial_dic=spatial_dic,
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            graph_features=self.graph_features,
            l1_rate=self.l1_rate,
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

    def set_zero_nonparam(self):
        """Set to zero the non-parametric part."""
        self.tf_NP_graph_OPD.set_alpha_zero()

    def set_output_Q(self, output_Q, output_dim=None):
        r"""Set the value of the output_Q parameter.
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

    def set_l1_rate(self, new_l1_rate):
        """Set l1 rate the non-parametric part."""
        self.l1_rate = new_l1_rate
        self.tf_NP_graph_OPD.l1_rate = new_l1_rate

    def set_nonzero_nonparam(self):
        """Set to non-zero the non-parametric part."""
        self.tf_NP_graph_OPD.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """Set the layers to be trainable or not."""
        self.tf_NP_graph_OPD.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """Assign coefficient matrix."""
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def predict_step(self, data, evaluate_step=False):
        """Custom predict (inference) step.

        It is needed as the non-parametric MCCD part requires a special
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

        # Calculate parametric part
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)

        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_NP_graph_OPD.predict(input_positions)

        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        r"""Predict a set of monochromatic PSF at desired positions.

        Parameters
        ----------
        input_positions: Tensor(batch x 2)
            Positions to predict the monochromatic PSFs.
        lambda_obs: float
            Observed wavelength in um.
        phase_N: int
            Required wavefront dimension. Should be calculated with as:
            ``simPSF_np = wf.SimPSFToolkit(...)``
            ``phase_N = simPSF_np.feasible_N(lambda_obs)``

        Returns
        -------
        mono_psf_batch: Tensor [batch x output_dim x output_dim]
            Batch of monochromatic PSFs at requested positions and
            wavelength.
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
        nonparam_opd_maps = self.tf_NP_graph_OPD.predict(input_positions)

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
        nonparam_opd_maps = self.tf_NP_graph_OPD.predict(input_positions)

        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        return opd_maps

    def call(self, inputs, training=True):
        r"""Define the PSF field forward model.

        [1] From positions to Zernike coefficients
        [2] From Zernike coefficients to OPD maps
        [3] From OPD maps and SED info to polychromatic PSFs

        OPD: Optical Path Differences
        """
        # Unpack inputs
        input_positions = inputs[0]
        packed_SEDs = inputs[1]

        # Forward model
        # For the training
        if training:
            # Calculate parametric part
            zernike_coeffs = self.tf_poly_Z_field(input_positions)
            param_opd_maps = self.tf_zernike_OPD(zernike_coeffs)
            # Calculate the non parametric part
            nonparam_opd_maps = self.tf_NP_graph_OPD(input_positions)
            # Add l2 loss on the parmetric OPD
            self.add_loss(
                self.l2_param * tf.math.reduce_sum(tf.math.square(nonparam_opd_maps))
            )
            # Add the estimations
            opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
            # Compute the polychromatic PSFs
            poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        # For the inference
        # This is absolutely needed to compute the metrics on the
        # validation data.
        else:
            # Compute predictions
            poly_psfs = self.predict_step(inputs, evaluate_step=True)

        return poly_psfs
