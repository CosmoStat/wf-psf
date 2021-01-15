import numpy as np
import scipy.signal as spsig
import scipy.interpolate as sinterp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL
import time
from tqdm.notebook import tqdm

%pylab inline

import tensorflow as tf

class TF_fft_diffract(tf.Module):
    def __init__(self, output_dim=64, name=None):
        super().__init__(name=name)
        self.output_dim = output_dim

    def crop_img(self, image):
        # Crop the image
        start = int(image.shape[0]//2-self.output_dim//2)
        stop = int(image.shape[0]//2+self.output_dim//2)

        return image[start:stop, start:stop]

    def tf_crop_img(self, image):
        """Crop images with tf methods.

        It handles a batch of 2D images: [batch, width, height]
        """
        # print('TF_fft_diffract: image.shape')
        # print(image.shape)
        # Define shape at runtime as we don't know it yet
        im_shape = tf.shape(image)
        # print(im_shape)
        # start
        offset_height = int(im_shape[2]//2 - self.output_dim//2)
        offset_width = int(im_shape[1]//2 - self.output_dim//2)
        # stop
        target_height = int(self.output_dim)
        target_width = int(self.output_dim)

        # Crop image
        cropped_image = tf.image.crop_to_bounding_box(
            tf.transpose(image, perm=[1,2,0]),
            # tf.reshape(image, shape=(image.shape[0], image.shape[1], -1)),
            offset_height,
            offset_width,
            target_height,
            target_width)

        return tf.transpose(cropped_image, perm=[2,0,1])
        # return tf.reshape(cropped_image, shape=(cropped_image.shape[0], cropped_image.shape[1]))

    def normalize_psf(self, psf):
        # Sum over all the dimensions
        # norm_factor = tf.math.reduce_sum(psf)
        norm_factor = tf.math.reduce_sum(psf, axis=[1,2], keepdims=True)

        return psf/norm_factor


    def __call__(self, input_phase):
        """ Calculate the normalized PSF from the padded phase array.
        """
        # print('TF_fft_diffract: input_phase.shape')
        # print(input_phase.shape)

        # Perform the FFT-based diffraction operation
        # fft_phase = tf.signal.fftshift(tf.signal.fft2d(input_phase))
        fft_phase = tf.signal.fftshift(tf.signal.fft2d(input_phase[:,...]), axes=[1, 2])
        psf = tf.math.pow(tf.cast(tf.math.abs(fft_phase), dtype=tf.float64), 2)
        # Crop the image
        # cropped_psf = self.crop_img(psf)
        cropped_psf = self.tf_crop_img(psf)
        # Normalize the PSF
        norm_psf = self.normalize_psf(cropped_psf)

        return norm_psf


class TF_build_phase(tf.Module):
    def __init__(self, phase_N, lambda_obs, obscurations, name=None):
        super().__init__(name=name)

        self.phase_N = phase_N
        self.lambda_obs = lambda_obs
        self.obscurations = obscurations

    def zero_padding_diffraction(self, no_pad_phase):
        """ Pad with zeros corresponding to the required lambda. """
        # pad_num = int(self.phase_N//2 - no_pad_phase.shape[0]//2)
        # pure tensorflow
        start = tf.math.floordiv(tf.cast(self.phase_N, dtype=tf.int32), tf.cast(2, dtype=tf.int32))
        stop = tf.math.floordiv(tf.cast(no_pad_phase.shape[0], dtype=tf.int32), tf.cast(2, dtype=tf.int32))
        pad_num = tf.math.subtract(start, stop) # start - stop

        padding = [
                   (0, 0),
                   (pad_num, pad_num),
                   (pad_num, pad_num)
                   ]
        # print('TF_build_phase: no_pad_phase.shape')
        # print(no_pad_phase.shape)

        padded_phase = tf.pad(no_pad_phase, padding)
        # print('TF_build_phase: padded_phase.shape')
        # print(padded_phase.shape)

        return padded_phase
        # return tf.pad(no_pad_phase, padding)

    def apply_obscurations(self, phase):
        """Multiply element-wise with the obscurations. """
        return tf.math.multiply(phase, tf.cast(self.obscurations, phase.dtype))


    def opd_to_phase(self, opd):
        """Convert from opd to phase."""
        pre_phase = tf.math.multiply(tf.cast((2*np.pi)/self.lambda_obs, opd.dtype), opd)
        phase = tf.math.exp(tf.dtypes.complex(tf.cast(0, pre_phase.dtype), pre_phase))
        # return tf.cast(phase, dtype=tf.complex64)
        return phase

    def __call__(self, opd):
        """Build the phase from the opd."""
        # print('TF_build_phase: opd.shape')
        # print(opd.shape)
        phase = self.opd_to_phase(opd)
        # print('TF_build_phase: phase.shape')
        # print(phase.shape)
        obsc_phase = self.apply_obscurations(phase)
        # print('TF_build_phase: obsc_phase.shape')
        # print(obsc_phase.shape)
        padded_phase = self.zero_padding_diffraction(obsc_phase)
        # print('TF_build_phase: padded_phase.shape')
        # print(padded_phase.shape)

        return padded_phase


class TF_zernike_OPD(tf.Module):
    """ Turn zernike coefficients into an OPD.

    Will use all of the Zernike maps provided.
    Both the Zernike maps and the Zernike coefficients must be provided.

    Parameters
    ----------
    zernike_maps: Tensor (Num_coeffs, x_dim, y_dim)
    z_coeffs: Tensor (num_star, num_coeffs, 1, 1)

    Returns
    -------
    opd: Tensor (num_star, x_dim, y_dim)

    """
    def __init__(self, zernike_maps, name=None):
        super().__init__(name=name)

        self.zernike_maps = zernike_maps

    def __call__(self, z_coeffs):
        # Perform the weighted sum of Zernikes coeffs and maps
        opd = tf.math.reduce_sum(tf.math.multiply(self.zernike_maps, z_coeffs), axis=1)
        return opd


class TF_Zernike_mono_PSF(tf.Module):
    """Build a monochromatic PSF from zernike coefficients.

    Following a Zernike model.
    """
    def __init__(self, phase_N, lambda_obs, obscurations, zernike_maps, output_dim=64, name=None):
        super().__init__(name=name)

        self.tf_build_opd_zernike = TF_zernike_OPD(zernike_maps)
        self.tf_build_phase = TF_build_phase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TF_fft_diffract(output_dim)

    def __call__(self, z_coeffs):
        opd = self.tf_build_opd_zernike.__call__(z_coeffs)
        phase = self.tf_build_phase.__call__(opd)
        psf = self.tf_fft_diffract.__call__(phase)

        return psf


class TF_mono_PSF(tf.Module):
    def __init__(self, phase_N, lambda_obs, obscurations, output_dim=64, name=None):
        super().__init__(name=name)

        self.tf_build_phase = TF_build_phase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TF_fft_diffract(output_dim)

    def __call__(self, opd):
        # print('TF_mono_PSF: opd.dtype')
        # print(opd.dtype)
        phase = self.tf_build_phase.__call__(opd)
        # print('TF_mono_PSF: phase.dtype')
        # print(phase.dtype)
        psf = self.tf_fft_diffract.__call__(phase)
        # print('TF_mono_PSF: psf.dtype')
        # print(psf.dtype)

        return tf.cast(psf, dtype=opd.dtype)


def generate_SED_elems(SED, sim_psf_toolkit, n_bins=20):
    """Generate the SED elements needed for using the TF_poly_PSF.

    sim_psf_toolkit: An instance of the SimPSFToolkit class with the correct
    initialization values.
    """

    feasible_wv, SED_norm = sim_psf_toolkit.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([sim_psf_toolkit.feasible_N(_wv)  for _wv in feasible_wv])

    return feasible_N, feasible_wv, SED_norm

def generate_packed_elems(SED, sim_psf_toolkit, n_bins=20):
    """Generate the packed values for using the TF_poly_PSF."""
    feasible_N, feasible_wv, SED_norm = generate_SED_elems(SED, sim_psf_toolkit, n_bins=n_bins)

    tf_feasible_N = tf.convert_to_tensor(feasible_N, dtype=tf.float32)
    tf_feasible_wv = tf.convert_to_tensor(feasible_wv, dtype=tf.float32)
    tf_SED_norm = tf.convert_to_tensor(SED_norm, dtype=tf.float32)

    # returnes the packed tensors
    return [tf_feasible_N, tf_feasible_wv, tf_SED_norm]


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

    # def build(self):
    #     """ Build the model paramters."""
    #     self.init_coeff_matrix()

    def get_poly_coefficients_shape(self):
        """ Return the shape of the coefficient matrix."""
        return (self.n_zernikes, int((self.d_max+1)*(self.d_max+2)/2))

    def assign_coeff_matrix(self, coeff_mat):
        """ Assign coefficient matrix."""
        self.coeff_mat.assign(coeff_mat)

    def init_coeff_matrix(self):
        """ Initialize coefficient matrix."""
        coef_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        self.coeff_mat = tf.Variable(
            initial_value=coef_init(self.get_poly_coefficients_shape()),
            trainable=True,
            dtype=tf.float32)

    def calc_poly_position_mat(self, pos, normalize=True):
        """ Calculate a matrix with position polynomials.

        Scale positions to the square:
        [self.x_lims[0], self.x_lims[1]] x [self.y_lims[0], self.y_lims[1]]
        to the square [0,1] x [0,1]
        """
        # _poly_mat = np.zeros((int((self.d_max+1)*(self.d_max+2)/2), pos.shape[0]))

        # pos[:,0] = (pos[:,0] - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0])
        # pos[:,1] = (pos[:,1] - self.y_lims[0]) / (self.y_lims[1] - self.y_lims[0])
        scaled_pos_x = (pos[:,0] - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0])
        scaled_pos_y = (pos[:,1] - self.y_lims[0]) / (self.y_lims[1] - self.y_lims[0])

        # print('scaled_pos_x')
        # print(scaled_pos_x.shape)

        # print('_poly_mat')
        # print(_poly_mat.shape)
        # print('type(_poly_mat)')
        # print(type(_poly_mat))
        # _poly_mat[0,1] = 0.5

        poly_list = []

        for d in range(self.d_max + 1):
            row_idx = d * (d + 1) // 2
            for p in range(d + 1):
                # print('row_idx + p')
                # print(row_idx + p)
                # print('(scaled_pos_x[:] ** (d - p) * scaled_pos_y[:] ** p)')
                # print((scaled_pos_x[:] ** (d - p) * scaled_pos_y[:] ** p).shape)
                # print('_poly_mat[row_idx + p, :]=tf.zeros_like(scaled_pos_x)')
                # _poly_mat[row_idx + p, :]=tf.zeros_like(scaled_pos_x)
                # _poly_mat[row_idx + p, :] = scaled_pos_x[:] ** (d - p) * scaled_pos_y[:] ** p
                poly_list.append(scaled_pos_x ** (d - p) * scaled_pos_y ** p)

        poly_mat = tf.convert_to_tensor(poly_list, dtype=tf.float32)

        poly_mat, _ = tf.linalg.normalize(poly_mat, ord='euclidean', axis=0)

        # print('TF_poly_Z_field: poly_mat.shape')
        # print(poly_mat.shape)

        return poly_mat

        # if normalize:
        #     weight_norms = np.sqrt(np.sum(_poly_mat ** 2, axis=0))
        #     _poly_mat /= weight_norms.reshape(1, -1)

        # return tf.convert_to_tensor(_poly_mat, dtype=tf.float32)

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
        poly_mat = self.calc_poly_position_mat(positions, normalize=True)
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

        print('TF_batch_poly_PSF: calculate_mono_PSF: packed_elems.shape')
        print(packed_elems.shape)
        # print('TF_batch_poly_PSF: lambda_obs')
        # print(lambda_obs)
        # print('TF_batch_poly_PSF: SED_norm_val')
        # print(SED_norm_val)
        # print('TF_batch_poly_PSF: self.current_opd.shape')
        # print(self.current_opd.shape)
        # print('TF_batch_poly_PSF: self.obscurations.shape')
        # print(self.obscurations.shape)

        # Build the monochromatic PSF generator
        tf_mono_psf_gen = TF_mono_PSF(phase_N,
                                      lambda_obs,
                                      self.obscurations,
                                      output_dim=self.output_dim)

        # Calculate the PSF
        mono_psf = tf_mono_psf_gen.__call__(self.current_opd)

        # print('TF_batch_poly_PSF: mono_psf.shape')
        # print(mono_psf.shape)
        # print('TF_batch_poly_PSF: SED_norm_val.type')
        # print(SED_norm_val.dtype)
        # print('TF_batch_poly_PSF: mono_psf.type')
        # print(mono_psf.dtype)

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

        # print('TF_batch_poly_PSF: opd_batch.shape')
        # print(opd_batch.shape)

        # print('TF_batch_poly_PSF: self.obscurations.shape')
        # print(self.obscurations.shape)


        it = tf.constant(0)
        while_condition = lambda it: tf.less(it, batch_num)

        def while_body(it):
            # Extract the required data of _it_
            packed_elems = packed_SED_data[it]
            self.current_opd = opd_batch[it][tf.newaxis,:,:]

            print('TF_batch_poly_PSF: self.current_opd.shape')
            print(self.current_opd.shape)
            print('TF_batch_poly_PSF: packed_elems.shape')
            print(packed_elems.shape)

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
                          swap_memory=True, parallel_iterations=1)

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

        # print('calculate_mono_PSF: mono_psf.shape')
        # print(mono_psf.shape)

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

        # poly_psf = tf.math.reduce_sum(_calculate_poly_PSF(SED_pack_data), axis=0)

        stack_psf = _calculate_poly_PSF(SED_pack_data)
        # print('calculate_poly_PSF: stack_psf.shape')
        # print(stack_psf.shape)

        poly_psf = tf.math.reduce_sum(stack_psf, axis=0)
        # print('calculate_poly_PSF: poly_psf.shape')
        # print(poly_psf.shape)

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

        # print('TF_batch_poly_PSF: poly_psf_batch.shape')
        # print(poly_psf_batch.shape)

        return poly_psf_batch

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
        # print('TF_PSF_field_model: input_positions.shape')
        # print(input_positions.shape)
        zernike_coeffs = self.tf_poly_Z_field(input_positions)
        # print('TF_PSF_field_model: zernike_coeffs.shape')
        # print(zernike_coeffs.shape)
        opd_maps = self.tf_zernike_OPD(zernike_coeffs)
        # print('TF_PSF_field_model: opd_maps.shape')
        # print(opd_maps.shape)
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])
        # print('TF_PSF_field_model: poly_psfs.shape')
        # print(poly_psfs.shape)

        return poly_psfs

def build_PSF_model(model_inst, l_rate):
    """ Define the model-compilation parameters.

    Specially the loss function, the optimizer and the metrics.
    """
    # Define model loss function
    loss = tf.keras.losses.MeanSquaredError()
    # Define optimizer function
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=l_rate, beta_1=0.9, beta_2=0.999,
        epsilon=1e-07, amsgrad=False)
    # Define metric functions
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Compile the model
    model_inst.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics,
                       loss_weights=None,
                       weighted_metrics=None,
                       run_eagerly=False)

    return model_inst
