import numpy as np
import tensorflow as tf
from wf_psf.tf_layers import TF_poly_Z_field, TF_zernike_OPD, TF_batch_poly_PSF


class TF_PSF_field_model(tf.keras.Model):
    """ PSF field forward model!

    Fully parametric model based on the Zernike polynomial basis. The

    Parameters
    ----------

    """
    def __init__(self, zernike_maps, obscurations, batch_size,
        output_dim=64, n_zernikes=45, d_max=2, x_lims=[0, 1e3], y_lims=[0, 1e3],
        coeff_mat=None, name='TF_PSF_field_model'):
        super(TF_PSF_field_model, self).__init__()

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

        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(obscurations=self.obscurations,
                                                    psf_batch=self.psf_batch,
                                                    output_dim=self.output_dim)

        # Initialize the model parameters with non-default value
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)

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

        # Continue the forward model
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        opd_maps = self.tf_zernike_OPD(zernike_coeffs)
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs

def build_PSF_model(model_inst, optimizer=None, loss=None,
    metrics=None):
    """ Define the model-compilation parameters.

    Specially the loss function, the optimizer and the metrics.
    """
    # Define model loss function
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError()

    # Define optimizer function
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-2, beta_1=0.9, beta_2=0.999,
            epsilon=1e-07, amsgrad=False)

    # Define metric functions
    if metrics is None:
        metrics = [tf.keras.metrics.MeanSquaredError()]

    # Compile the model
    model_inst.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics,
                       loss_weights=None,
                       weighted_metrics=None,
                       run_eagerly=False)

    return model_inst
