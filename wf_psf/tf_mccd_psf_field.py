import numpy as np
import tensorflow as tf
from wf_psf.tf_layers import TF_poly_Z_field, TF_zernike_OPD, TF_batch_poly_PSF
from wf_psf.tf_layers import TF_NP_MCCD_OPD
from wf_psf.graph_utils import GraphBuilder
from wf_psf.utils import calc_poly_position_mat

class TF_SP_MCCD_field(tf.keras.Model):
    """ Semi-parametric MCCD PSF field model!

    Semi parametric model based on the hybrid-MCCD matrix factorization scheme.

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
    def __init__(self, zernike_maps, obscurations, batch_size, obs_pos,
        spatial_dic, d_max_nonparam=3, graph_features=6,
        output_dim=64, n_zernikes=45, d_max=2, x_lims=[0, 1e3], y_lims=[0, 1e3],
        coeff_mat=None, name='TF_SP_MCCD_field'):
        super(TF_SP_MCCD_field, self).__init__()

        # Inputs: TF_poly_Z_field
        self.n_zernikes = n_zernikes
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_NP_MCCD_OPD
        self.d_max_nonparam = d_max_nonparam
        self.opd_dim = tf.shape(zernike_maps)[1].numpy()
        self.graph_features = graph_features

        # Inputs: TF_zernike_OPD
        # They are not stored as they are memory-heavy
        # zernike_maps =[]

        # Inputs: TF_batch_poly_PSF
        self.batch_size = batch_size
        self.obscurations = obscurations
        self.output_dim = output_dim
        self.psf_batch = tf.zeros(
            (self.batch_size, self.output_dim, self.output_dim),
            dtype=tf.float32)


        # Initialize the first layer
        self.tf_poly_Z_field = TF_poly_Z_field(x_lims=self.x_lims,
                                                y_lims=self.y_lims,
                                                n_zernikes=self.n_zernikes,
                                                d_max=self.d_max)

        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TF_zernike_OPD(zernike_maps=zernike_maps)

        # Initialize the non-parametric layer
        self.tf_NP_mccd_OPD = TF_NP_MCCD_OPD(obs_pos=obs_pos,
                                             spatial_dic=spatial_dic,
                                             x_lims=self.x_lims,
                                             y_lims=self.y_lims,
                                             d_max=self.d_max_nonparam,
                                             graph_features=self.graph_features,
                                             opd_dim=self.opd_dim)


        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(obscurations=self.obscurations,
                                                    psf_batch=self.psf_batch,
                                                    output_dim=self.output_dim)

        # Initialize the model parameters with non-default value
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

    def set_zero_nonparam(self):
        """ Set to zero the non-parametric part."""
        self.tf_NP_mccd_OPD.set_alpha_zero()

    def set_nonzero_nonparam(self):
        """ Set to non-zero the non-parametric part."""
        self.tf_NP_mccd_OPD.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """ Set the layers to be trainable or not."""
        self.tf_NP_mccd_OPD.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def get_coeff_matrix(self):
        """ Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """ Assign coefficient matrix."""
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

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
        # Calculate the non parametric part
        nonparam_opd_maps =  self.tf_NP_mccd_OPD(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs




def build_mccd_spatial_dic(obs_stars, obs_pos, x_lims, y_lims,
    d_max = 2, graph_features=6, verbose=0):
    """Build the spatial-constraint dictionary.

    Based on the hybrid approach from the MCCD model.
    """
    # The obs_data needs to be in RCA format (with the batch dim at the end)

    # Graph parameters
    graph_kwargs = {
    'obs_data': obs_stars.swapaxes(0, 1).swapaxes(1, 2),
    'obs_pos': obs_pos,
    'obs_weights': np.ones_like(obs_stars),
    'n_comp': graph_features,
    'n_eigenvects': 5,
    'n_iter': 3,
    'ea_gridsize': 10,
    'distances': None,
    'auto_run': True,
    'verbose': verbose
    }

    # Compute graph-spatial constraint matrix
    VT = GraphBuilder(**graph_kwargs).VT

    # Compute polynomial-spatial constaint matrix
    tf_Pi = calc_poly_position_mat(pos=obs_pos, x_lims=x_lims, y_lims=y_lims, d_max=d_max)

    # Need to translate to have the batch dimension first
    spatial_dic = np.concatenate((tf_Pi.numpy(), VT), axis=0).T

    # Return the tf spatial dictionary
    return tf.convert_to_tensor(spatial_dic, dtype=tf.float32)
