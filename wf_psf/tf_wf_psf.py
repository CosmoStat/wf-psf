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

import tensorflow as tf

%pylab inline

plt.rcParams['figure.figsize'] = (16, 8)


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

        Right now it is implemented to handle 2D images: [width, height]
        """
        # start
        offset_height = int(image.shape[1]//2 - self.output_dim//2)
        offset_width = int(image.shape[0]//2 - self.output_dim//2)
        # stop
        target_height = int(self.output_dim)
        target_width = int(self.output_dim)

        # Crop image
        cropped_image = tf.image.crop_to_bounding_box(
            tf.reshape(image, shape=(image.shape[0], image.shape[1], -1)),
            offset_height,
            offset_width,
            target_height,
            target_width)

        return tf.reshape(cropped_image, shape=(cropped_image.shape[0], cropped_image.shape[1]))

    def normalize_psf(self, psf):
        # Sum over all the dimensions
        norm_factor = tf.math.reduce_sum(psf)

        return psf/norm_factor


    def __call__(self, input_phase):
        """ Calculate the normalized PSF from the padded phase array.
        """
        # Perform the FFT-based diffraction operation
        fft_phase = tf.signal.fftshift(tf.signal.fft2d(input_phase))
        psf = tf.math.pow(tf.math.abs(fft_phase), 2)

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

    @staticmethod
    def zero_padding_diffraction(no_pad_phase, phase_N):
        """ Pad with zeros corresponding to the required lambda. """
        pad_num = int(phase_N//2 - no_pad_phase.shape[0]//2)

        padded_phase = tf.pad(
            no_pad_phase,
            [
                (pad_num, pad_num),
                (pad_num, pad_num)
            ])

        return padded_phase

    def apply_obscurations(self, phase):
        """Multiply element-wise with the obscurations. """
        return tf.math.multiply(phase, tf.cast(self.obscurations, phase.dtype))


    def opd_to_phase(self, opd):
        """Convert from opd to phase."""
        pre_phase = tf.math.multiply(tf.cast((2*np.pi)/self.lambda_obs, opd.dtype), opd)
        phase = tf.math.exp(tf.dtypes.complex(tf.cast(0, pre_phase.dtype), pre_phase))
        return phase

    def __call__(self, opd):
        """Build the phase from the opd."""

        phase = self.opd_to_phase(opd)
        obsc_phase = self.apply_obscurations(phase)
        padded_phase = self.zero_padding_diffraction(obsc_phase, self.phase_N)

        return padded_phase


class TF_mono_PSF(tf.Module):
    def __init__(self, phase_N, lambda_obs, obscurations, output_dim=64, name=None):
        super().__init__(name=name)

        self.tf_build_phase = TF_build_phase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TF_fft_diffract(output_dim)


    def __call__(self, opd):
        phase = self.tf_build_phase.__call__(opd)
        psf = self.tf_fft_diffract.__call__(phase)

        return psf

class TF_zernike_OPD(tf.Module):
    """ Turn zernike coefficients into an OPD.

    Will use all of the Zernike maps provided.
    Both the Zernike maps and the Zernike coefficients must be provided.
    Parameters
    ----------
    zernike_maps: Tensor (Num_coeffs, x_dim, y_dim)
    z_coeffs: Tensor (Num_coeffs, 1, 1)

    """
    def __init__(self, zernike_maps, name=None):
        super().__init__(name=name)

        self.zernike_maps = zernike_maps

    def __call__(self, z_coeffs):
        # Perform the weighted sum of Zernikes coeffs and maps
        opd = tf.math.reduce_sum(tf.math.multiply(self.zernike_maps, z_coeffs), axis=0)
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


# sim_psf_toolkit = SimPSFToolkit.SimPSFToolkit(zernikes)

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
    feasible_N, feasible_wv, SED_norm = generate_SED_elems(SED, sim_psf_toolkit, n_bins=20)

    tf_feasible_N = tf.convert_to_tensor(feasible_N, dtype=tf.float64)
    tf_feasible_wv = tf.convert_to_tensor(feasible_wv, dtype=tf.float64)
    tf_SED_norm = tf.convert_to_tensor(SED_norm, dtype=tf.float64)

    # returnes the packed tensors
    return [tf_feasible_N, tf_feasible_wv, tf_SED_norm]


class TF_poly_PSF(tf.Module):
    """Calculate a polychromatic PSF from an OPD and stored SED values.

    The calculation of the packed values with the respective SED is done
    with the SimPSFToolkit class but outside the TF class.


    packed_elems: Tuple of tensors
        Contains three 1D tensors with the parameters needed for
        the calculation of each monochromatic PSF.

        packed_elems[0]: phase_N
        packed_elems[1]: lambda_obs
        packed_elems[2]: SED_norm_val
    """
    def __init__(self, obscurations, packed_elems, output_dim=64, name=None):
        super().__init__(name=name)

        self.obscurations = obscurations
        self.output_dim = output_dim
        self.packed_elems = packed_elems

        self.opd = None


    def update_packed_elems(self, new_packed_elems):
        """Update packed elements."""
        self.packed_elems = new_packed_elems


    def __call__(self, opd):

        # Save the OPD that will be shared by all the monochromatic PSFs
        self.opd = opd
        obscurations = self.obscurations
        output_dim = self.output_dim
        packed_elems = self.packed_elems

        def calculate_mono_PSF(packed_elems):
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
            tf_mono_psf = TF_mono_PSF(phase_N, lambda_obs, obscurations, output_dim=output_dim)

            # Calculate the PSF
            mono_psf = tf_mono_psf.__call__(opd)

            # Multiply with the respective normalized SED and return
            return tf.math.scalar_mul(SED_norm_val, mono_psf)

        # Use tf.function for parallelization over GPU
        # Not allowed since the dynamic padding for the diffraction does not
        # work in the @tf.function context
        # @tf.function
        def calculate_poly_PSF(elems):
            return tf.map_fn(calculate_mono_PSF, elems, parallel_iterations=10, fn_output_signature=tf.float64)

        stacked_psfs = calculate_poly_PSF(packed_elems)
        poly_psf = tf.math.reduce_sum(stacked_psfs, axis=0)

        return poly_psf
