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
        cropped_psf = self.crop_img(psf)

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
        pad_num = int(self.phase_N//2 - no_pad_phase.shape[0]//2)

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
        padded_phase = self.zero_padding_diffraction(obsc_phase)

        return padded_phase


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


class TF_mono_PSF(tf.Module):
    """Build a monochromatic PSF from an OPD.
    """
    def __init__(self, phase_N, lambda_obs, obscurations, output_dim=64, name=None):
        super().__init__(name=name)

        self.tf_build_phase = TF_build_phase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TF_fft_diffract(output_dim)


    def __call__(self, opd):
        phase = self.tf_build_phase.__call__(opd)
        psf = self.tf_fft_diffract.__call__(phase)

        return psf


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
