import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from wf_psf.psf_models.tf_modules import TF_mono_PSF
from wf_psf.utils.utils import calc_poly_position_mat
import wf_psf.utils.utils as utils
import logging

logger = logging.getLogger(__name__)


class TF_poly_Z_field(tf.keras.layers.Layer):
    """Calculate the zernike coefficients for a given position.

    This module implements a polynomial model of Zernike
    coefficient variation.

    Parameters
    ----------
    n_zernikes: int
        Number of Zernike polynomials to consider
    d_max: int
        Max degree of polynomial determining the FoV variations.

    """

    def __init__(
        self,
        x_lims,
        y_lims,
        random_seed=None,
        n_zernikes=45,
        d_max=2,
        name="TF_poly_Z_field",
    ):
        super().__init__(name=name)

        self.n_zernikes = n_zernikes
        self.d_max = d_max

        self.coeff_mat = None
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.random_seed = random_seed

        self.init_coeff_matrix()

    def get_poly_coefficients_shape(self):
        """Return the shape of the coefficient matrix."""
        return (self.n_zernikes, int((self.d_max + 1) * (self.d_max + 2) / 2))

    def assign_coeff_matrix(self, coeff_mat):
        """Assign coefficient matrix."""
        self.coeff_mat.assign(coeff_mat)

    def get_coeff_matrix(self):
        """Get coefficient matrix."""
        return self.coeff_mat

    def init_coeff_matrix(self):
        """Initialize coefficient matrix."""
        tf.random.set_seed(self.random_seed)
        coef_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        self.coeff_mat = tf.Variable(
            initial_value=coef_init(self.get_poly_coefficients_shape()),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, positions):
        """Calculate the zernike coefficients for a given position.

        The position polynomial matrix and the coefficients should be
        set before calling this function.

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        zernikes_coeffs: Tensor(batch, n_zernikes, 1, 1)
        """
        poly_mat = calc_poly_position_mat(
            positions, self.x_lims, self.y_lims, self.d_max
        )
        zernikes_coeffs = tf.transpose(tf.linalg.matmul(self.coeff_mat, poly_mat))

        return zernikes_coeffs[:, :, tf.newaxis, tf.newaxis]


class TF_zernike_OPD(tf.keras.layers.Layer):
    """Turn zernike coefficients into an OPD.

    Will use all of the Zernike maps provided.
    Both the Zernike maps and the Zernike coefficients must be provided.

    Parameters
    ----------
    zernike_maps: Tensor (Num_coeffs, x_dim, y_dim)
    z_coeffs: Tensor (batch_size, n_zernikes, 1, 1)

    Returns
    -------
    opd: Tensor (batch_size, x_dim, y_dim)

    """

    def __init__(self, zernike_maps, name="TF_zernike_OPD"):
        super().__init__(name=name)

        self.zernike_maps = zernike_maps

    def call(self, z_coeffs):
        """Perform the weighted sum of Zernikes coeffs and maps.

        Returns
        -------
        opd: Tensor (batch_size, x_dim, y_dim)
        """
        return tf.math.reduce_sum(tf.math.multiply(self.zernike_maps, z_coeffs), axis=1)


class TF_batch_poly_PSF(tf.keras.layers.Layer):
    """Calculate a polychromatic PSF from an OPD and stored SED values.

    The calculation of the packed values with the respective SED is done
    with the SimPSFToolkit class but outside the TF class.

    Parameters
    ----------
    obscurations: Tensor [opd_dim, opd_dim]
        Obscurations to apply to the wavefront.
    packed_SED_data: Tensor [batch_size, 3, n_bins_lda]
        Comes from: tf.convert_to_tensor(list(list(Tensor,Tensor,Tensor)))
        Where each inner list consist of a packed_elem:

            packed_elems: Tuple of tensors
            Contains three 1D tensors with the parameters needed for
            the calculation of one monochromatic PSF.

            packed_elems[0]: phase_N
            packed_elems[1]: lambda_obs
            packed_elems[2]: SED_norm_val
        The SED data is constant in a FoV.
    psf_batch: Tensor [batch_size, output_dim, output_dim]
        Tensor containing the psfs that will be updated each
        time a calculation is required. REMOVED!

    """

    def __init__(self, obscurations, output_Q, output_dim=64, name="TF_batch_poly_PSF"):
        super().__init__(name=name)

        self.output_Q = output_Q
        self.obscurations = obscurations
        self.output_dim = output_dim

        self.current_opd = None

    def calculate_mono_PSF(self, packed_elems):
        """Calculate monochromatic PSF from packed elements.

        packed_elems[0]: phase_N
        packed_elems[1]: lambda_obs
        packed_elems[2]: SED_norm_val
        """
        # Unpack elements
        phase_N = packed_elems[0]
        lambda_obs = packed_elems[1]
        SED_norm_val = packed_elems[2]

        # Build the monochromatic PSF generator
        tf_mono_psf_gen = TF_mono_PSF(
            phase_N,
            lambda_obs,
            self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

        # Calculate the PSF
        mono_psf = tf_mono_psf_gen.__call__(self.current_opd)
        mono_psf = tf.squeeze(mono_psf, axis=0)
        # mono_psf = tf.reshape(mono_psf, shape=(mono_psf.shape[1],mono_psf.shape[2]))

        # Multiply with the respective normalized SED and return
        return tf.math.scalar_mul(SED_norm_val, mono_psf)

    def calculate_poly_PSF(self, packed_elems):
        """Calculate a polychromatic PSF."""

        self.current_opd = packed_elems[0][tf.newaxis, :, :]
        SED_pack_data = packed_elems[1]

        def _calculate_poly_PSF(elems_to_unpack):
            return tf.map_fn(
                self.calculate_mono_PSF,
                elems_to_unpack,
                parallel_iterations=10,
                fn_output_signature=tf.float32,
                swap_memory=True,
            )

        # Readability
        # stacked_psfs = _calculate_poly_PSF(packed_elems)
        # poly_psf = tf.math.reduce_sum(stacked_psfs, axis=0)
        # return poly_psf

        stack_psf = _calculate_poly_PSF(SED_pack_data)
        poly_psf = tf.math.reduce_sum(stack_psf, axis=0)

        return poly_psf

    def call(self, inputs):
        """Calculate the batch poly PSFs."""

        # Unpack Inputs
        opd_batch = inputs[0]
        packed_SED_data = inputs[1]

        def _calculate_PSF_batch(elems_to_unpack):
            return tf.map_fn(
                self.calculate_poly_PSF,
                elems_to_unpack,
                parallel_iterations=10,
                fn_output_signature=tf.float32,
                swap_memory=True,
            )

        psf_poly_batch = _calculate_PSF_batch((opd_batch, packed_SED_data))

        return psf_poly_batch


class TF_batch_mono_PSF(tf.keras.layers.Layer):
    """Calculate a monochromatic PSF from a batch of OPDs.

    The calculation of the ``phase_N`` variable is done
    with the SimPSFToolkit class but outside the TF class.

    Parameters
    ----------
    obscurations: Tensor [opd_dim, opd_dim]
        Obscurations to apply to the wavefront.
    psf_batch: Tensor [batch_size, output_dim, output_dim]
        Tensor containing the psfs that will be updated each
        time a calculation is required.
        Can be started with zeros.
    output_Q: int
        Output oversampling value.
    output_dim: int
        Output PSF stamp dimension.

    """

    def __init__(self, obscurations, output_Q, output_dim=64, name="TF_batch_mono_PSF"):
        super().__init__(name=name)

        self.output_Q = output_Q
        self.obscurations = obscurations
        self.output_dim = output_dim

        self.phase_N = None
        self.lambda_obs = None
        self.tf_mono_psf_gen = None

        self.current_opd = None

    def calculate_mono_PSF(self, current_opd):
        """Calculate monochromatic PSF from OPD info."""
        # Calculate the PSF
        mono_psf = self.tf_mono_psf_gen.__call__(current_opd[tf.newaxis, :, :])
        mono_psf = tf.squeeze(mono_psf, axis=0)

        return mono_psf

    def init_mono_PSF(self):
        """Initialise or restart the PSF generator."""
        self.tf_mono_psf_gen = TF_mono_PSF(
            self.phase_N,
            self.lambda_obs,
            self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )

    def set_lambda_phaseN(self, phase_N=914, lambda_obs=0.7):
        """Set the lambda value for monochromatic PSFs and the phaseN."""
        self.phase_N = phase_N
        self.lambda_obs = lambda_obs
        self.init_mono_PSF()

    def set_output_params(self, output_Q, output_dim):
        """Set output patams, Q and dimension."""
        self.output_Q = output_Q
        self.output_dim = output_dim
        self.init_mono_PSF()

    def call(self, opd_batch):
        """Calculate the batch poly PSFs."""

        if self.phase_N is None:
            self.set_lambda_phaseN()

        def _calculate_PSF_batch(elems_to_unpack):
            return tf.map_fn(
                self.calculate_mono_PSF,
                elems_to_unpack,
                parallel_iterations=10,
                fn_output_signature=tf.float32,
                swap_memory=True,
            )

        mono_psf_batch = _calculate_PSF_batch((opd_batch))

        return mono_psf_batch


class TF_NP_poly_OPD(tf.keras.layers.Layer):
    """Non-parametric OPD generation with polynomial variations.


    Parameters
    ----------
    x_lims: [int, int]
        Limits of the x axis.
    y_lims: [int, int]
        Limits of the y axis.
    random_seed: int
        Random seed initialization for Tensor Flow
    d_max: int
        Max degree of polynomial determining the FoV variations.
    opd_dim: int
        Dimension of the OPD maps. Same as pupil diameter.

    """

    def __init__(
        self,
        x_lims,
        y_lims,
        random_seed=None,
        d_max=3,
        opd_dim=256,
        name="TF_NP_poly_OPD",
    ):
        super().__init__(name=name)
        # Parameters
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.random_seed = random_seed
        self.d_max = d_max
        self.opd_dim = opd_dim

        self.n_poly = int((self.d_max + 1) * (self.d_max + 2) / 2)

        # Variables
        self.S_mat = None
        self.alpha_mat = None
        self.init_vars()

    def init_vars(self):
        """Initialize trainable variables.

        Basic initialization. Random uniform for S and identity for alpha.
        """
        # S initialization
        tf.random.set_seed(self.random_seed)
        random_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
        self.S_mat = tf.Variable(
            initial_value=random_init(shape=[self.n_poly, self.opd_dim, self.opd_dim]),
            trainable=True,
            dtype=tf.float32,
        )

        # Alpha initialization
        self.alpha_mat = tf.Variable(
            initial_value=tf.eye(self.n_poly), trainable=True, dtype=tf.float32
        )

    def set_alpha_zero(self):
        """Set alpha matrix to zero."""
        self.alpha_mat.assign(tf.zeros_like(self.alpha_mat, dtype=tf.float32))

    def set_alpha_identity(self):
        """Set alpha matrix to the identity."""
        self.alpha_mat.assign(tf.eye(self.n_poly, dtype=tf.float32))

    def assign_S_mat(self, S_mat):
        """Assign DD features matrix."""
        self.S_mat.assign(S_mat)

    def call(self, positions):
        """Calculate the OPD maps for the given positions.

        Calculating: Pi(pos) x alpha x S

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        opd_maps: Tensor(batch, opd_dim, opd_dim)
        """
        # Calculate the Pi matrix
        poly_mat = calc_poly_position_mat(
            positions, self.x_lims, self.y_lims, self.d_max
        )
        # We need to transpose it here to have the batch dimension at first
        poly_mat = tf.transpose(poly_mat, perm=[1, 0])

        inter_res = tf.linalg.matmul(poly_mat, self.alpha_mat)

        return tf.tensordot(inter_res, self.S_mat, axes=1)


class TF_NP_MCCD_OPD_v2(tf.keras.layers.Layer):
    """Non-parametric OPD generation with hybrid-MCCD variations.


    Parameters
    ----------
    obs_pos: tensor(n_stars, 2)
        Observed positions of the `n_stars` in the dataset. The indexing of the
        positions has to correspond to the indexing in the `spatial_dic`.
    spatial_dic: tensor(n_stars, n_dic_elems)
        Dictionary containing the spatial-constraint dictionary. `n_stars`
        corresponds to the total number of stars in the dataset. `n_dic_elems`
        corresponds to the number of elements of the dictionary, not to be
        confounded with `n_comp`, the total number of non-parametric features
        of the wavefront-PSF.
    x_lims: [int, int]
        Limits of the x axis.
    y_lims: [int, int]
        Limits of the y axis.
    graph_comps: int
        Number of wavefront-PSF features correspondign to the graph constraint.
    d_max: int
        Max degree of polynomial determining the FoV variations. The number of
        wavefront-PSF features of the polynomial part is
        computed `(d_max+1)*(d_max+2)/2`.
    opd_dim: int
        Dimension of the OPD maps. Same as pupil diameter.

    """

    def __init__(
        self,
        obs_pos,
        spatial_dic,
        x_lims,
        y_lims,
        random_seed=None,
        d_max=2,
        graph_features=6,
        l1_rate=1e-5,
        opd_dim=256,
        name="TF_NP_MCCD_OPD_v2",
    ):
        super().__init__(name=name)
        # Parameters
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.random_seed = random_seed
        logger.info(type(self.random_seed))
        self.d_max = d_max
        self.opd_dim = opd_dim

        # L1 regularisation parameter
        self.l1_rate = l1_rate

        self.obs_pos = obs_pos
        self.poly_dic = spatial_dic[0]
        self.graph_dic = spatial_dic[1]

        self.n_stars = self.poly_dic.shape[0]
        self.n_graph_elems = self.graph_dic.shape[1]
        self.poly_features = int((self.d_max + 1) * (self.d_max + 2) / 2)
        self.graph_features = graph_features

        # Variables
        self.S_poly = None
        self.S_graph = None
        self.alpha_poly = None
        self.alpha_graph = None
        self.init_vars()

    def init_vars(self):
        """Initialize trainable variables.

        Basic initialization. Random uniform for S and identity for alpha.
        """
        # S initialization
        tf.random.set_seed(self.random_seed)
        random_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
        self.S_poly = tf.Variable(
            initial_value=random_init(
                shape=[self.poly_features, self.opd_dim, self.opd_dim]
            ),
            trainable=True,
            dtype=tf.float32,
        )
        self.S_graph = tf.Variable(
            initial_value=random_init(
                shape=[self.graph_features, self.opd_dim, self.opd_dim]
            ),
            trainable=True,
            dtype=tf.float32,
        )

        # Alpha initialization
        self.alpha_poly = tf.Variable(
            initial_value=tf.eye(
                num_rows=self.poly_features, num_columns=self.poly_features
            ),
            trainable=True,
            dtype=tf.float32,
        )
        self.alpha_graph = tf.Variable(
            initial_value=tf.eye(
                num_rows=self.n_graph_elems, num_columns=self.graph_features
            ),
            trainable=True,
            dtype=tf.float32,
        )

    def set_alpha_zero(self):
        """Set alpha matrix to zero."""
        self.alpha_poly.assign(tf.zeros_like(self.alpha_poly, dtype=tf.float32))
        self.alpha_graph.assign(tf.zeros_like(self.alpha_graph, dtype=tf.float32))

    def set_alpha_identity(self):
        """Set alpha matrix to the identity."""
        self.alpha_poly.assign(
            tf.eye(
                num_rows=self.poly_features,
                num_columns=self.poly_features,
                dtype=tf.float32,
            )
        )
        self.alpha_graph.assign(
            tf.eye(
                num_rows=self.n_graph_elems,
                num_columns=self.graph_features,
                dtype=tf.float32,
            )
        )

    def predict(self, positions):
        """Prediction step."""
        ## Polynomial part
        # Calculate the Pi matrix
        poly_mat = calc_poly_position_mat(
            positions, self.x_lims, self.y_lims, self.d_max
        )
        # We need to transpose it here to have the batch dimension at first
        A_poly = tf.linalg.matmul(tf.transpose(poly_mat, perm=[1, 0]), self.alpha_poly)
        interp_poly_opd = tf.tensordot(A_poly, self.S_poly, axes=1)

        ## Graph part
        A_graph_train = tf.linalg.matmul(self.graph_dic, self.alpha_graph)
        # RBF interpolation
        # Order 2 means a thin_plate RBF interpolation
        # All tensors need to expand one dimension to fulfil requirement in
        # the tfa's interpolate_spline function
        A_interp_graph = tfa.image.interpolate_spline(
            train_points=tf.expand_dims(self.obs_pos, axis=0),
            train_values=tf.expand_dims(A_graph_train, axis=0),
            query_points=tf.expand_dims(positions, axis=0),
            order=2,
            regularization_weight=0.0,
        )

        # Remove extra dimension required by tfa's interpolate_spline
        A_interp_graph = tf.squeeze(A_interp_graph, axis=0)
        interp_graph_opd = tf.tensordot(A_interp_graph, self.S_graph, axes=1)

        return tf.math.add(interp_poly_opd, interp_graph_opd)

    def call(self, positions):
        """Calculate the OPD maps for the given positions.

        Calculating: batch(spatial_dict) x alpha x S

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        opd_maps: Tensor(batch, opd_dim, opd_dim)
        """
        # Add L1 loss of the graph alpha matrix
        # self.add_loss(self.l1_rate * tf.math.reduce_sum(tf.math.abs(self.alpha_graph)))
        # Try Lp norm with p=1.1
        p = 1.1
        self.add_loss(
            self.l1_rate
            * tf.math.pow(
                tf.math.reduce_sum(tf.math.pow(tf.math.abs(self.alpha_graph), p)), 1 / p
            )
        )

        def calc_index(idx_pos):
            return tf.where(tf.equal(self.obs_pos, idx_pos))[0, 0]

        # Calculate the indices of the input batch
        indices = tf.map_fn(calc_index, positions, fn_output_signature=tf.int64)

        # Recover the spatial dict from the batch indexes
        # Matrix multiplication dict*alpha
        # Tensor product to calculate the contribution

        # Polynomial contribution
        batch_poly_dict = tf.gather(
            self.poly_dic, indices=indices, axis=0, batch_dims=0
        )
        intermediate_poly = tf.linalg.matmul(batch_poly_dict, self.alpha_poly)
        contribution_poly = tf.tensordot(intermediate_poly, self.S_poly, axes=1)
        # Graph contribution
        batch_graph_dict = tf.gather(
            self.graph_dic, indices=indices, axis=0, batch_dims=0
        )
        intermediate_graph = tf.linalg.matmul(batch_graph_dict, self.alpha_graph)
        contribution_graph = tf.tensordot(intermediate_graph, self.S_graph, axes=1)

        return tf.math.add(contribution_poly, contribution_graph)


class TF_NP_GRAPH_OPD(tf.keras.layers.Layer):
    """Non-parametric OPD generation with only graph-cosntraint variations.


    Parameters
    ----------
    obs_pos: tensor(n_stars, 2)
        Observed positions of the `n_stars` in the dataset. The indexing of the
        positions has to correspond to the indexing in the `spatial_dic`.
    spatial_dic: tensor(n_stars, n_dic_elems)
        Dictionary containing the spatial-constraint dictionary. `n_stars`
        corresponds to the total number of stars in the dataset. `n_dic_elems`
        corresponds to the number of elements of the dictionary, not to be
        confounded with `n_comp`, the total number of non-parametric features
        of the wavefront-PSF.
    x_lims: [int, int]
        Limits of the x axis.
    y_lims: [int, int]
        Limits of the y axis.
    graph_comps: int
        Number of wavefront-PSF features correspondign to the graph constraint.
    d_max: int
        Max degree of polynomial determining the FoV variations. The number of
        wavefront-PSF features of the polynomial part is
        computed `(d_max+1)*(d_max+2)/2`.
    opd_dim: int
        Dimension of the OPD maps. Same as pupil diameter.

    """

    def __init__(
        self,
        obs_pos,
        spatial_dic,
        x_lims,
        y_lims,
        random_seed=None,
        graph_features=6,
        l1_rate=1e-5,
        opd_dim=256,
        name="TF_NP_GRAPH_OPD",
    ):
        super().__init__(name=name)
        # Parameters
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.random_seed = random_seed
        self.opd_dim = opd_dim

        # L1 regularisation parameter
        self.l1_rate = l1_rate

        self.obs_pos = obs_pos
        self.poly_dic = spatial_dic[0]
        self.graph_dic = spatial_dic[1]

        self.n_stars = self.poly_dic.shape[0]
        self.n_graph_elems = self.graph_dic.shape[1]
        self.graph_features = graph_features

        # Variables
        self.S_graph = None
        self.alpha_graph = None
        self.init_vars()

    def init_vars(self):
        """Initialize trainable variables.

        Basic initialization. Random uniform for S and identity for alpha.
        """
        # S initialization
        tf.random.set_seed(self.random_seed)
        random_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)

        self.S_graph = tf.Variable(
            initial_value=random_init(
                shape=[self.graph_features, self.opd_dim, self.opd_dim]
            ),
            trainable=True,
            dtype=tf.float32,
        )

        # Alpha initialization
        self.alpha_graph = tf.Variable(
            initial_value=tf.eye(
                num_rows=self.n_graph_elems, num_columns=self.graph_features
            ),
            trainable=True,
            dtype=tf.float32,
        )

    def set_alpha_zero(self):
        """Set alpha matrix to zero."""
        self.alpha_graph.assign(tf.zeros_like(self.alpha_graph, dtype=tf.float32))

    def set_alpha_identity(self):
        """Set alpha matrix to the identity."""
        self.alpha_graph.assign(
            tf.eye(
                num_rows=self.n_graph_elems,
                num_columns=self.graph_features,
                dtype=tf.float32,
            )
        )

    def predict(self, positions):
        """Prediction step."""

        ## Graph part
        A_graph_train = tf.linalg.matmul(self.graph_dic, self.alpha_graph)
        # RBF interpolation
        # Order 2 means a thin_plate RBF interpolation
        # All tensors need to expand one dimension to fulfil requirement in
        # the tfa's interpolate_spline function
        A_interp_graph = tfa.image.interpolate_spline(
            train_points=tf.expand_dims(self.obs_pos, axis=0),
            train_values=tf.expand_dims(A_graph_train, axis=0),
            query_points=tf.expand_dims(positions, axis=0),
            order=2,
            regularization_weight=0.0,
        )

        # Remove extra dimension required by tfa's interpolate_spline
        A_interp_graph = tf.squeeze(A_interp_graph, axis=0)
        interp_graph_opd = tf.tensordot(A_interp_graph, self.S_graph, axes=1)

        return interp_graph_opd

    def call(self, positions):
        """Calculate the OPD maps for the given positions.

        Calculating: batch(spatial_dict) x alpha x S

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        opd_maps: Tensor(batch, opd_dim, opd_dim)
        """
        # Add L1 loss of the graph alpha matrix
        # self.add_loss(
        #     self.l1_rate * tf.math.reduce_sum(tf.math.abs(self.alpha_graph))
        # )
        # Try Lp norm with p=1.1
        p = 1.1
        self.add_loss(
            self.l1_rate
            * tf.math.pow(
                tf.math.reduce_sum(tf.math.pow(tf.math.abs(self.alpha_graph), p)), 1 / p
            )
        )

        def calc_index(idx_pos):
            return tf.where(tf.equal(self.obs_pos, idx_pos))[0, 0]

        # Calculate the indices of the input batch
        indices = tf.map_fn(calc_index, positions, fn_output_signature=tf.int64)

        # Recover the spatial dict from the batch indexes
        # Matrix multiplication dict*alpha
        # Tensor product to calculate the contribution

        # Graph contribution
        batch_graph_dict = tf.gather(
            self.graph_dic, indices=indices, axis=0, batch_dims=0
        )
        intermediate_graph = tf.linalg.matmul(batch_graph_dict, self.alpha_graph)
        contribution_graph = tf.tensordot(intermediate_graph, self.S_graph, axes=1)

        return contribution_graph


class TF_physical_layer(tf.keras.layers.Layer):
    """Store and calculate the zernike coefficients for a given position.

    This layer gives the Zernike contribution of the physical layer.
    It is fixed and not trainable.

    Parameters
    ----------
    obs_pos: Tensor (n_stars, 2)
        Observed positions of the `n_stars` in the dataset. The indexing of the
        positions has to correspond to the indexing in the `zks_prior`.
    n_zernikes: int
        Number of Zernike polynomials
    zks_prior: Tensor (n_stars, n_zernikes)
        Zernike coefficients for each position
    interpolation_type: str
        Type of interpolation to be used.
        Options are: 'none', 'all', 'top_K', 'independent_Zk'.
        Default is 'none'.
    interpolation_args: dict
        Interpolation hyper-parameters. The order of the RBF interpolation,
        and the K elements in the `top_K` interpolation.

    """

    def __init__(
        self,
        obs_pos,
        zks_prior,
        interpolation_type="none",
        interpolation_args=None,
        name="TF_physical_layer",
    ):
        super().__init__(name=name)
        self.obs_pos = obs_pos
        self.zks_prior = zks_prior

        if interpolation_args is None:
            interpolation_args = {"order": 2, "K": 50}
        # Define the prediction routine
        if interpolation_type == "none":
            self.predict = self.call
        elif interpolation_type == "all":
            self.predict = self.interpolate_all
        elif interpolation_type == "top_K":
            self.predict = self.interpolate_top_K
        elif interpolation_type == "independent_Zk":
            self.predict = self.interpolate_independent_Zk

    def interpolate_all(self, positions):
        """Zernike interpolation

        Right now all the input elements are used to build the RBF interpolant
        that is going to be used for the interpolation.

        """
        # RBF interpolation of prior Zernikes
        # Order 2 means a thin_plate RBF interpolation
        # All tensors need to expand one dimension to fulfil requirement in
        # the tfa's interpolate_spline function
        interp_zks = tfa.image.interpolate_spline(
            train_points=tf.expand_dims(self.obs_pos, axis=0),
            train_values=tf.expand_dims(self.zks_prior, axis=0),
            query_points=tf.expand_dims(positions, axis=0),
            order=self.interpolation_args["order"],
            regularization_weight=0.0,
        )
        # Remove extra dimension required by tfa's interpolate_spline
        interp_zks = tf.squeeze(interp_zks, axis=0)

        return interp_zks[:, :, tf.newaxis, tf.newaxis]

    def interpolate_top_K(self, positions):
        """Zernike interpolation

        The class wf.utils.ZernikeInterpolation allows to use only the K closest
        elements for the interpolation. Even though, the interpolation error is smaller
        the computing time is bigger.

        """
        zk_interpolator = utils.ZernikeInterpolation(
            self.obs_pos,
            self.zks_prior,
            k=self.interpolation_args["K"],
            order=self.interpolation_args["order"],
        )
        interp_zks = zk_interpolator.interpolate_zks(positions)

        return interp_zks[:, :, tf.newaxis, tf.newaxis]

    def interpolate_independent_Zk(self, positions):
        """Zernike interpolation

        The class wf.utils.IndependentZernikeInterpolation allows to interpolate each
        order of the Zernike polynomials independently using all the points avaialble to build
        the interpolant.

        """
        zk_interpolator = utils.IndependentZernikeInterpolation(
            self.obs_pos, self.zks_prior, order=self.interpolation_args["order"]
        )
        interp_zks = zk_interpolator.interpolate_zks(positions)

        return interp_zks[:, :, tf.newaxis, tf.newaxis]

    def call(self, positions):
        """Calculate the prior zernike coefficients for a given position.

        The position polynomial matrix and the coefficients should be
        set before calling this function.

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        zernikes_coeffs: Tensor(batch, n_zernikes, 1, 1)
        """

        def calc_index(idx_pos):
            return tf.where(tf.equal(self.obs_pos, idx_pos))[0, 0]

        # Calculate the indices of the input batch
        indices = tf.map_fn(calc_index, positions, fn_output_signature=tf.int64)
        # Recover the prior zernikes from the batch indexes
        batch_zks = tf.gather(self.zks_prior, indices=indices, axis=0, batch_dims=0)

        return batch_zks[:, :, tf.newaxis, tf.newaxis]


# --- #
# Deprecated #
class OLD_TF_batch_poly_PSF(tf.keras.layers.Layer):
    """Calculate a polychromatic PSF from an OPD and stored SED values.

    The calculation of the packed values with the respective SED is done
    with the SimPSFToolkit class but outside the TF class.



    obscurations: Tensor(pupil_len, pupil_len)
        Obscurations to apply to the wavefront.

    packed_SED_data: Tensor(batch_size, 3, n_bins_lda)

    Comes from: tf.convert_to_tensor(list(list(Tensor,Tensor,Tensor)))
        Where each inner list consist of a packed_elem:

            packed_elems: Tuple of tensors
            Contains three 1D tensors with the parameters needed for
            the calculation of one monochromatic PSF.

            packed_elems[0]: phase_N
            packed_elems[1]: lambda_obs
            packed_elems[2]: SED_norm_val
        The SED data is constant in a FoV.

    psf_batch: Tensor(batch_size, output_dim, output_dim)
        Tensor containing the psfs that will be updated each
        time a calculation is required.

    """

    def __init__(
        self, obscurations, psf_batch, output_dim=64, name="TF_batch_poly_PSF"
    ):
        super().__init__(name=name)

        self.obscurations = obscurations
        self.output_dim = output_dim
        self.psf_batch = psf_batch

        self.current_opd = None

    def set_psf_batch(self, psf_batch):
        """Set poly PSF batch."""
        self.psf_batch = psf_batch

    def calculate_mono_PSF(self, packed_elems):
        """Calculate monochromatic PSF from packed elements.

        packed_elems[0]: phase_N
        packed_elems[1]: lambda_obs
        packed_elems[2]: SED_norm_val
        """
        # Unpack elements
        phase_N = packed_elems[0]
        lambda_obs = packed_elems[1]
        SED_norm_val = packed_elems[2]

        # Build the monochromatic PSF generator
        tf_mono_psf_gen = TF_mono_PSF(
            phase_N, lambda_obs, self.obscurations, output_dim=self.output_dim
        )

        # Calculate the PSF
        mono_psf = tf_mono_psf_gen.__call__(self.current_opd)

        # Multiply with the respective normalized SED and return
        return tf.math.scalar_mul(SED_norm_val, mono_psf)

    def calculate_poly_PSF(self, packed_elems):
        """Calculate a polychromatic PSF."""

        logger.info("TF_batch_poly_PSF: calculate_poly_PSF: packed_elems.type")
        logger.info(packed_elems.dtype)

        def _calculate_poly_PSF(elems_to_unpack):
            return tf.map_fn(
                self.calculate_mono_PSF,
                elems_to_unpack,
                parallel_iterations=10,
                fn_output_signature=tf.float32,
                swap_memory=True,
            )

        # Readability
        # stacked_psfs = _calculate_poly_PSF(packed_elems)
        # poly_psf = tf.math.reduce_sum(stacked_psfs, axis=0)
        # return poly_psf

        return tf.math.reduce_sum(_calculate_poly_PSF(packed_elems), axis=0)

    def call(self, inputs):
        """Calculate the batch poly PSFs."""

        # Unpack Inputs
        opd_batch = inputs[0]
        packed_SED_data = inputs[1]

        batch_num = opd_batch.shape[0]

        it = tf.constant(0)
        while_condition = lambda it: tf.less(it, batch_num)

        def while_body(it):
            # Extract the required data of _it_
            packed_elems = packed_SED_data[it]
            self.current_opd = opd_batch[it][tf.newaxis, :, :]

            # Calculate the _it_ poly PSF
            poly_psf = self.calculate_poly_PSF(packed_elems)

            # Update the poly PSF tensor with the result
            # Slice update of a tensor
            # See tf doc of _tensor_scatter_nd_update_ to understand
            indices = tf.reshape(it, shape=(1, 1))
            # self.psf_batch = tf.tensor_scatter_nd_update(self.psf_batch, indices, poly_psf)

            # increment i
            return [tf.add(it, 1)]

        # Loop over the PSF batches
        r = tf.while_loop(
            while_condition, while_body, [it], swap_memory=True, parallel_iterations=10
        )

        return self.psf_batch
