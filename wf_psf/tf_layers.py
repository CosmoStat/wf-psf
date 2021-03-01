import numpy as np
import tensorflow as tf
from wf_psf.tf_modules import TF_mono_PSF
from wf_psf.utils import calc_poly_position_mat


class TF_poly_Z_field(tf.keras.layers.Layer):
    """ Calculate the zernike coefficients for a given position.

    This module implements a polynomial model of Zernike
    coefficient variation.

    Parameters
    ----------
    n_zernikes: int
        Number of Zernike polynomials to consider
    d_max: int
        Max degree of polynomial determining the FoV variations.

    """
    def __init__(self, x_lims, y_lims, n_zernikes=45, d_max=2, name='TF_poly_Z_field'):
        super().__init__(name=name)

        self.n_zernikes = n_zernikes
        self.d_max = d_max

        self.coeff_mat = None
        self.x_lims = x_lims
        self.y_lims = y_lims

        self.init_coeff_matrix()


    def get_poly_coefficients_shape(self):
        """ Return the shape of the coefficient matrix."""
        return (self.n_zernikes, int((self.d_max+1)*(self.d_max+2)/2))

    def assign_coeff_matrix(self, coeff_mat):
        """ Assign coefficient matrix."""
        self.coeff_mat.assign(coeff_mat)

    def get_coeff_matrix(self):
        """ Get coefficient matrix."""
        return self.coeff_mat

    def init_coeff_matrix(self):
        """ Initialize coefficient matrix."""
        coef_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        self.coeff_mat = tf.Variable(
            initial_value=coef_init(self.get_poly_coefficients_shape()),
            trainable=True,
            dtype=tf.float32)

    # def calc_poly_position_mat(self, pos):
    #     """ Calculate a matrix with position polynomials.
    #
    #     Scale positions to the square:
    #     [self.x_lims[0], self.x_lims[1]] x [self.y_lims[0], self.y_lims[1]]
    #     to the square [-1,1] x [-1,1]
    #     """
    #     # Scale positions
    #     scaled_pos_x = (pos[:,0] - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0])
    #     scaled_pos_x = (scaled_pos_x - 0.5) * 2
    #     scaled_pos_y = (pos[:,1] - self.y_lims[0]) / (self.y_lims[1] - self.y_lims[0])
    #     scaled_pos_y = (scaled_pos_y - 0.5) * 2
    #
    #     poly_list = []
    #
    #     for d in range(self.d_max + 1):
    #         row_idx = d * (d + 1) // 2
    #         for p in range(d + 1):
    #             poly_list.append(scaled_pos_x ** (d - p) * scaled_pos_y ** p)
    #
    #     poly_mat = tf.convert_to_tensor(poly_list, dtype=tf.float32)
    #
    #     return poly_mat


    def call(self, positions):
        """ Calculate the zernike coefficients for a given position.

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
        poly_mat = calc_poly_position_mat(positions, self.x_lims, self.y_lims, self.d_max)
        # poly_mat = self.calc_poly_position_mat(positions)
        zernikes_coeffs = tf.transpose(tf.linalg.matmul(self.coeff_mat, poly_mat))

        return zernikes_coeffs[:, :, tf.newaxis, tf.newaxis]


class TF_zernike_OPD(tf.keras.layers.Layer):
    """ Turn zernike coefficients into an OPD.

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
    def __init__(self, zernike_maps, name='TF_zernike_OPD'):
        super().__init__(name=name)

        self.zernike_maps = zernike_maps

    def call(self, z_coeffs):
        """ Perform the weighted sum of Zernikes coeffs and maps.

        Returns
        -------
        opd: Tensor (batch_size, x_dim, y_dim)
        """
        return tf.math.reduce_sum(tf.math.multiply(self.zernike_maps, z_coeffs), axis=1)


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
    def __init__(self, obscurations, psf_batch,
        output_dim=64, name='TF_batch_poly_PSF'):
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
        tf_mono_psf_gen = TF_mono_PSF(phase_N,
                                      lambda_obs,
                                      self.obscurations,
                                      output_dim=self.output_dim)

        # Calculate the PSF
        mono_psf = tf_mono_psf_gen.__call__(self.current_opd)

        # Multiply with the respective normalized SED and return
        return tf.math.scalar_mul(SED_norm_val, mono_psf)


    def calculate_poly_PSF(self, packed_elems):
        """Calculate a polychromatic PSF."""

        print('TF_batch_poly_PSF: calculate_poly_PSF: packed_elems.type')
        print(packed_elems.dtype)

        def _calculate_poly_PSF(elems_to_unpack):
            return tf.map_fn(self.calculate_mono_PSF,
                             elems_to_unpack,
                             parallel_iterations=10,
                             fn_output_signature=tf.float32,
                             swap_memory=True)

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
            self.current_opd = opd_batch[it][tf.newaxis,:,:]

            # Calculate the _it_ poly PSF
            poly_psf = self.calculate_poly_PSF(packed_elems)

            # Update the poly PSF tensor with the result
            # Slice update of a tensor
            # See tf doc of _tensor_scatter_nd_update_ to understand
            indices = tf.reshape(it, shape=(1,1))
            # self.psf_batch = tf.tensor_scatter_nd_update(self.psf_batch, indices, poly_psf)

            # increment i
            return [tf.add(it, 1)]

        # Loop over the PSF batches
        r = tf.while_loop(while_condition, while_body, [it],
                          swap_memory=True, parallel_iterations=10)

        return self.psf_batch


class TF_batch_poly_PSF(tf.keras.layers.Layer):
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
    def __init__(self, obscurations, psf_batch,
        output_dim=64, name='TF_batch_poly_PSF'):
        super().__init__(name=name)

        self.obscurations = obscurations
        self.output_dim = output_dim
        self.psf_batch = psf_batch

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
        tf_mono_psf_gen = TF_mono_PSF(phase_N,
                                      lambda_obs,
                                      self.obscurations,
                                      output_dim=self.output_dim)

        # Calculate the PSF
        mono_psf = tf_mono_psf_gen.__call__(self.current_opd)
        mono_psf = tf.squeeze(mono_psf, axis=0)
        # mono_psf = tf.reshape(mono_psf, shape=(mono_psf.shape[1],mono_psf.shape[2]))

        # Multiply with the respective normalized SED and return
        return tf.math.scalar_mul(SED_norm_val, mono_psf)


    def calculate_poly_PSF(self, packed_elems):
        """Calculate a polychromatic PSF."""

        self.current_opd = packed_elems[0][tf.newaxis,:,:]
        SED_pack_data = packed_elems[1]

        def _calculate_poly_PSF(elems_to_unpack):
            return tf.map_fn(self.calculate_mono_PSF,
                             elems_to_unpack,
                             parallel_iterations=10,
                             fn_output_signature=tf.float32,
                             swap_memory=True)

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
            return tf.map_fn(self.calculate_poly_PSF,
                             elems_to_unpack,
                             parallel_iterations=10,
                             fn_output_signature=tf.float32,
                             swap_memory=True)

        poly_psf_batch = _calculate_PSF_batch((opd_batch, packed_SED_data))

        return poly_psf_batch


class TF_NP_poly_OPD(tf.keras.layers.Layer):
    """ Non-parametric OPD generation with polynomial variations.


    Parameters
    ----------
    x_lims: [int, int]
        Limits of the x axis.
    y_lims: [int, int]
        Limits of the y axis.
    d_max: int
        Max degree of polynomial determining the FoV variations.
    opd_dim: int
        Dimension of the OPD maps. Same as pupil diameter.

    """
    def __init__(self, x_lims, y_lims, d_max=3, opd_dim=256, name='TF_NP_poly_OPD'):
        super().__init__(name=name)
        # Parameters
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.d_max = d_max
        self.opd_dim = opd_dim

        # Variables
        self.S_mat = None
        self.alpha_mat = None
        self.init_vars()


    def init_vars(self):
        """ Initialize trainable variables.

        Basic initialization. Random uniform for S and identity for alpha.
        """
        n_poly = int((self.d_max+1)*(self.d_max+2)/2)
        # S initialization
        random_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
        self.S_mat = tf.Variable(
            initial_value=random_init(shape=[n_poly, self.opd_dim, self.opd_dim]),
            trainable=True,
            dtype=tf.float32)

        # Alpha initialization
        self.alpha_mat = tf.Variable(
            initial_value=tf.eye(n_poly),
            trainable=True,
            dtype=tf.float32)


    def call(self, positions):
        """ Calculate the OPD maps for the given positions.

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
        poly_mat = calc_poly_position_mat(positions, self.x_lims, self.y_lims, self.d_max)
        # We need to transpose it here to have the batch dimension at first
        poly_mat = tf.transpose(poly_mat, perm=[1,0])

        inter_res = tf.linalg.matmul(poly_mat, self.alpha_mat)

        return tf.tensordot(inter_res, self.S_mat, axes=1)


class TF_NP_MCCD_OPD(tf.keras.layers.Layer):
    """ Non-parametric OPD generation with hybrid-MCCD variations.


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
    def __init__(self, obs_pos, spatial_dic, x_lims, y_lims, d_max=3,
                graph_features=6, opd_dim=256, name='TF_NP_MCCD_OPD'):
        super().__init__(name=name)
        # Parameters
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.d_max = d_max
        self.opd_dim = opd_dim

        self.obs_pos = obs_pos
        self.spatial_dic = spatial_dic

        self.n_stars = spatial_dic.shape[0]
        self.n_dic_elems = spatial_dic.shape[1]
        self.n_features = int((self.d_max+1)*(self.d_max+2)/2) + graph_features

        # Variables
        self.S_mat = None
        self.alpha_mat = None
        self.init_vars()


    def init_vars(self):
        """ Initialize trainable variables.

        Basic initialization. Random uniform for S and identity for alpha.
        """
        # S initialization
        random_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
        self.S_mat = tf.Variable(
            initial_value=random_init(shape=[self.n_features,
                                             self.opd_dim,
                                             self.opd_dim]),
            trainable=True,
            dtype=tf.float32)

        # Alpha initialization
        self.alpha_mat = tf.Variable(
            initial_value=tf.eye(num_rows=self.n_dic_elems,
                                 num_columns=self.n_features),
            trainable=True,
            dtype=tf.float32)

    def set_alpha_zero(self):
        """ Set alpha matrix to zero."""
        _ = self.alpha_mat.assign(tf.zeros_like(self.alpha_mat,
                                                dtype=tf.float32))

    def set_alpha_identity(self):
        """ Set alpha matrix to the identity."""
        _ = self.alpha_mat.assign(tf.eye(num_rows=self.n_dic_elems,
                                         num_columns=self.n_features,
                                         dtype=tf.float32))

    def call(self, positions):
        """ Calculate the OPD maps for the given positions.

        Calculating: batch(spatial_dict) x alpha x S

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        opd_maps: Tensor(batch, opd_dim, opd_dim)
        """

        def calc_index(idx_pos):
            return tf.where(tf.equal(self.obs_pos, idx_pos))[0,0]

        # Calculate the indices of the input batch
        indices = tf.map_fn(calc_index, positions, fn_output_signature=tf.int64)

        # # Calculate the indices of the input batch
        # indices = [tf.where(tf.equal(self.obs_pos, _pos))[0,0] for _pos in positions]
        # indices = tf.convert_to_tensor(indices, dtype=tf.int32)

        # Recover the spatial dict from the batch indexes
        batch_dict = tf.gather(self.spatial_dic, indices=indices, axis=0, batch_dims=0)

        inter_res = tf.linalg.matmul(batch_dict, self.alpha_mat)

        return tf.tensordot(inter_res, self.S_mat, axes=1)
