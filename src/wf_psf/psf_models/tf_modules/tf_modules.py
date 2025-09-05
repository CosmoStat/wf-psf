"""TensorFlow-Based PSF Modeling.

A module containing TensorFlow implementations for modeling monochromatic PSFs using Zernike polynomials and Fourier optics.

:Author: Tobias Liaudat <tobiasliaudat@gmail.com>

"""

import numpy as np
import tensorflow as tf
from typing import Optional


class TFFftDiffract(tf.Module):
    """Diffract the wavefront into a monochromatic PSF.

    Attributes
    ----------
    output_dim : int
        Dimension of the output square postage stamp
    output_Q : int
        Downsampling factor. Must be integer.
    """

    def __init__(
        self, output_dim: int = 64, output_Q: int = 2, name: Optional[str] = None
    ) -> None:
        """Initialize the TFFftDiffract class.

        Parameters
        ----------
        output_dim : int, optional
            The dimension of the output square postage stamp. The default is 64.
        output_Q : int, optional
            The downsampling factor. Must be an integer. The default is 2.
        name : str, optional
            The name for the TensorFlow module.
        """
        super().__init__(name=name)
        self.output_dim = output_dim
        self.output_Q = int(output_Q)

        self.downsample_layer = tf.keras.layers.AveragePooling2D(
            pool_size=(self.output_Q, self.output_Q),
            strides=None,
            padding="valid",
            data_format="channels_last",
        )

    def tf_crop_img(self, image, output_crop_dim):
        """Crop images using TensorFlow methods.

        This method handles a batch of 2D images and crops them to the specified dimension.
        The images are expected to have the shape [batch, width, height], and the method
        uses TensorFlow's `crop_to_bounding_box` to crop each image in the batch.

        Parameters
        ----------
        image : tf.Tensor
            A batch of 2D images with shape [batch, height, width]. The images are expected
            to be 3D tensors where the second and third dimensions represent the height and width.
        output_crop_dim : int
            The dimension of the square crop. The image will be cropped to this dimension.

        Returns
        -------
        tf.Tensor
            The cropped images with shape [batch, output_crop_dim, output_crop_dim].
        """
        # Define shape at runtime as we don't know it yet
        im_shape = tf.shape(image)
        # start
        offset_height = int(im_shape[2] // 2 - output_crop_dim // 2)
        offset_width = int(im_shape[1] // 2 - output_crop_dim // 2)
        # stop
        target_height = int(output_crop_dim)
        target_width = int(output_crop_dim)

        # Crop image
        cropped_image = tf.image.crop_to_bounding_box(
            tf.transpose(image, perm=[1, 2, 0]),
            offset_height,
            offset_width,
            target_height,
            target_width,
        )

        return tf.transpose(cropped_image, perm=[2, 0, 1])

    def normalize_psf(self, psf):
        """Normalize the Point Spread Function (PSF).

        This function normalizes a given Point Spread Function (PSF) by summing over the spatial dimensions and dividing the PSF by the resulting sum. The PSF is expected to have at least 3 dimensions, with the first dimension representing the batch size and the remaining two dimensions representing the spatial dimensions (height and width).

        Parameters
        ----------
        psf : tf.Tensor
            A tensor representing the Point Spread Function (PSF) with shape [batch, height, width].
            The PSF is expected to be a 3D tensor, where the first dimension corresponds to the batch size, and the other two dimensions represent the spatial dimensions of the PSF.

        Returns
        -------
        tf.Tensor
            The normalized PSF with the same shape as the input, [batch, height, width], where each PSF has been normalized by the sum of the PSF over the spatial dimensions.
        """
        # Sum over all the dimensions
        norm_factor = tf.math.reduce_sum(psf, axis=[1, 2], keepdims=True)

        return psf / norm_factor

    def __call__(self, input_phase):
        """Calculate the normalized Point Spread Function (PSF) from a phase array.

        This method takes a 2D input phase array, applies a 2D FFT-based diffraction operation,
        crops the resulting PSF, and downscales it by a factor of Q if necessary. Finally, the PSF
        is normalized by summing over its spatial dimensions.

        Parameters
        ----------
        input_phase : tf.Tensor
            A tensor of shape [batch, height, width] representing the input phase array.

        Returns
        -------
        tf.Tensor
            The normalized PSF tensor with shape [batch, height, width], where each PSF is normalized
            by its sum over the spatial dimensions.
        """
        # Perform the FFT-based diffraction operation
        fft_phase = tf.signal.fftshift(
            tf.signal.fft2d(input_phase[:, ...]), axes=[1, 2]
        )
        psf = tf.math.pow(tf.cast(tf.math.abs(fft_phase), dtype=tf.float64), 2)

        # Crop the image
        cropped_psf = self.tf_crop_img(
            psf, output_crop_dim=int(self.output_dim * self.output_Q)
        )

        # Downsample image
        if self.output_Q != 1:
            cropped_psf = self.downsample_layer(cropped_psf[..., tf.newaxis])

            # # Alternative solution but tf.image.resize does not have the
            # # gradients implemented in tensorflow
            # cropped_psf = tf.image.resize(
            #     cropped_psf[ ..., tf.newaxis],
            #     size=[self.output_dim, self.output_dim],
            #     method=tf.image.ResizeMethod.AREA,
            #     preserve_aspect_ratio=False,
            #     antialias=True)

            # Remove channel dimension [batch, heigh, width, channel]
            cropped_psf = tf.squeeze(cropped_psf, axis=-1)

        # Normalize the PSF
        norm_psf = self.normalize_psf(cropped_psf)

        return norm_psf


class TFBuildPhase(tf.Module):
    """Build a complex phase map from an Optical Path Difference (OPD) map.

    This class takes an OPD map and converts it into a complex phase map. It applies
    necessary obscurations (such as apertures or masks) and zero-padding to match the
    required size for diffraction simulations. The resulting phase map is essential for
    further optical modeling, such as diffraction simulations or other optical system analysis.

    Attributes
    ----------
    phase_N : int
        The desired size of the phase map (e.g., pixel count for height and width).
    lambda_obs : float
        The observed wavelength used for phase calculations, typically in meters.
    obscurations : tf.Tensor
        A tensor representing the obscurations (e.g., apertures or masks) to be applied to the phase.
    """

    def __init__(
        self,
        phase_N: int,
        lambda_obs: float,
        obscurations: tf.Tensor,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the TFBuildPhase class.

        Parameters
        ----------
        phase_N : int
            The size of the phase map (e.g., pixel count).
        lambda_obs : float
            The observed wavelength used for phase calculations.
        obscurations : tf.Tensor
            A tensor representing the obscurations (e.g., apertures or masks) to be applied to the phase.
        name : str, optional
            The name for the TensorFlow module.
        """
        super().__init__(name=name)

        self.phase_N = phase_N
        self.lambda_obs = lambda_obs
        self.obscurations = obscurations

    def zero_padding_diffraction(self, no_pad_phase):
        """Pad the phase map with zeros based on the required size.

        This method adds zero-padding to the input phase map to match the required
        size for diffraction calculations. The padding is computed based on the
        `phase_N` attribute and the input phase map size.

        Parameters
        ----------
        no_pad_phase : tf.Tensor
            The phase map that needs to be padded. Expected shape is [batch_size, height, width].

        Returns
        -------
        padded_phase : tf.Tensor
            The padded phase map with shape [batch_size, phase_N, phase_N].
        """
        phase_shape = tf.shape(no_pad_phase)
        start = tf.math.floordiv(
            tf.cast(self.phase_N, dtype=tf.int32), tf.cast(2, dtype=tf.int32)
        )
        stop = tf.math.floordiv(
            tf.cast(phase_shape[1], dtype=tf.int32), tf.cast(2, dtype=tf.int32)
        )
        pad_num = tf.math.subtract(start, stop)  # start - stop

        padding = [(0, 0), (pad_num, pad_num), (pad_num, pad_num)]

        padded_phase = tf.pad(no_pad_phase, padding)

        return padded_phase

    def apply_obscurations(self, phase: tf.Tensor) -> tf.Tensor:
        """Apply obscurations to the phase map.

        This method multiplies the phase map element-wise with the obscurations
        tensor. The obscurations tensor can represent apertures or masks that block
        or modify portions of the phase map.

        Parameters
        ----------
        phase : tf.Tensor
            The phase map to which obscurations will be applied. Expected shape is [batch_size, height, width].

        Returns
        -------
        tf.Tensor
            The phase map after applying the obscurations.
        """
        return tf.math.multiply(phase, tf.cast(self.obscurations, phase.dtype))

    def opd_to_phase(self, opd: tf.Tensor) -> tf.Tensor:
        """Convert an OPD map to a complex phase map.

        This method takes an optical path difference (OPD) map and converts it into a complex phase map using the formula: phase = exp(i * (2 * pi / lambda_obs) * opd).

        Parameters
        ----------
        opd : tf.Tensor
            The optical path difference map. Expected shape is [batch_size, height, width].

        Returns
        -------
        tf.Tensor
            The complex phase map resulting from the OPD.
        """
        pre_phase = tf.math.multiply(
            tf.cast((2 * np.pi) / self.lambda_obs, opd.dtype), opd
        )
        phase = tf.math.exp(tf.dtypes.complex(tf.cast(0, pre_phase.dtype), pre_phase))
        # return tf.cast(phase, dtype=tf.complex64)
        return phase

    def __call__(self, opd):
        """Convert an OPD map to a padded and obscured phase map.

        This method performs the full pipeline: converting an OPD map to a complex
        phase map, applying obscurations, and adding zero-padding to match the required
        size for diffraction simulations.

        Parameters
        ----------
        opd : tf.Tensor
            The optical path difference map. Expected shape is [batch_size, height, width].

        Returns
        -------
        tf.Tensor
            The final padded phase map after obscurations are applied.
        """
        phase = self.opd_to_phase(opd)
        obsc_phase = self.apply_obscurations(phase)
        padded_phase = self.zero_padding_diffraction(obsc_phase)

        return padded_phase


class TFZernikeOPD(tf.Module):
    """Convert Zernike coefficients into an Optical Path Difference (OPD).

    This class performs the weighted sum of Zernike coefficients and Zernike maps
    to compute the OPD. The Zernike maps and the corresponding Zernike coefficients
    are required to perform the calculation.

    Parameters
    ----------
    zernike_maps : tf.Tensor
        A tensor containing the Zernike maps. The shape should be
        (num_coeffs, x_dim, y_dim), where `num_coeffs` is the number of Zernike coefficients
        and `x_dim`, `y_dim` are the dimensions of each map.

    name : str, optional
        The name of the module. Default is `None`.

    Returns
    -------
    tf.Tensor
        A tensor representing the OPD, with shape (num_star, x_dim, y_dim),
        where `num_star` corresponds to the number of stars and `x_dim`, `y_dim` are
        the dimensions of the OPD map.
    """

    def __init__(self, zernike_maps: tf.Tensor, name: Optional[str] = None) -> None:
        """
        Initialize the TFZernikeOPD class.

        Parameters
        ----------
        zernike_maps : tf.Tensor
            A tensor containing the Zernike maps. Shape should be (num_coeffs, x_dim, y_dim).
        name : str, optional
            The name of the module. Default is `None`.
        """
        super().__init__(name=name)

        self.zernike_maps = zernike_maps

    def __call__(self, z_coeffs: tf.Tensor) -> tf.Tensor:
        """Compute the OPD from Zernike coefficients and maps.

        This method calculates the OPD by performing the weighted sum of Zernike
        coefficients and corresponding Zernike maps. The result is a tensor representing
        the computed OPD for the given coefficients.

        Parameters
        ----------
        z_coeffs : tf.Tensor
            A tensor containing the Zernike coefficients. The shape should be
            (num_star, num_coeffs, 1, 1), where `num_star` is the number of stars and
            `num_coeffs` is the number of Zernike coefficients.

        Returns
        -------
        tf.Tensor
            The resulting OPD tensor, with shape (num_star, x_dim, y_dim).
        """
        opd = tf.math.reduce_sum(tf.math.multiply(self.zernike_maps, z_coeffs), axis=1)
        return opd


class TFZernikeMonochromaticPSF(tf.Module):
    """Build a monochromatic Point Spread Function (PSF) from Zernike coefficients.

    This class computes the monochromatic PSF by following the Zernike model. It
    involves multiple stages, including the calculation of the OPD (Optical Path
    Difference), the phase from the OPD, and diffraction via FFT-based operations.
    The Zernike coefficients are used to generate the PSF.

    Parameters
    ----------
    phase_N : int
        The size of the phase grid, typically a square matrix dimension.

    lambda_obs : float
        The wavelength of the observed light.

    obscurations : tf.Tensor
        A tensor representing the obscurations in the system, which will be applied
        to the phase.

    zernike_maps : tf.Tensor
        A tensor containing the Zernike maps, with the shape (num_coeffs, x_dim, y_dim),
        where `num_coeffs` is the number of Zernike coefficients and `x_dim`, `y_dim` are
        the dimensions of the Zernike maps.

    output_dim : int, optional, default=64
        The output dimension of the PSF, i.e., the size of the resulting image.

    name : str, optional
        The name of the module. Default is `None`.

    Attributes
    ----------
    tf_build_opd_zernike : TFZernikeOPD
        A module used to generate the OPD from the Zernike coefficients.

    tf_build_phase : TFBuildPhase
        A module used to compute the phase from the OPD.

    tf_fft_diffract : TFFftDiffract
        A module that performs the diffraction calculation using FFT-based methods.
    """

    def __init__(
        self,
        phase_N: int,
        lambda_obs: float,
        obscurations: tf.Tensor,
        zernike_maps: tf.Tensor,
        output_dim: int = 64,
        name: Optional[str] = None,
    ):
        """
        Initialize the TFZernikeMonochromaticPSF class.

        Parameters
        ----------
        phase_N : int
            The size of the phase grid (dimension of the square grid).
        lambda_obs : float
            The wavelength of the observed light.
        obscurations : tf.Tensor
            A tensor representing the obscurations that will be applied to the phase.
        zernike_maps : tf.Tensor
            A tensor containing the Zernike maps. Shape should be (num_coeffs, x_dim, y_dim).
        output_dim : int, optional, default=64
            The output dimension of the PSF.
        name : str, optional
            The name of the module.
        """
        super().__init__(name=name)

        self.tf_build_opd_zernike = TFZernikeOPD(zernike_maps)
        self.tf_build_phase = TFBuildPhase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TFFftDiffract(output_dim)

    def __call__(self, z_coeffs):
        """Compute the monochromatic PSF from Zernike coefficients.

        This method computes the PSF by following the steps:
        1. Generate the OPD using the Zernike coefficients and Zernike maps.
        2. Compute the phase from the OPD.
        3. Perform diffraction using FFT-based methods to obtain the PSF.

        Parameters
        ----------
        z_coeffs : tf.Tensor
            A tensor containing the Zernike coefficients. The shape should be
            (num_star, num_coeffs, 1, 1), where `num_star` is the number of stars
            and `num_coeffs` is the number of Zernike coefficients.

        Returns
        -------
        tf.Tensor
            A tensor representing the computed PSF, with shape
            (num_star, output_dim, output_dim), where `output_dim` is the size of
            the resulting PSF image.
        """
        # Generate OPD from Zernike coefficients
        opd = self.tf_build_opd_zernike.__call__(z_coeffs)

        # Compute phase from OPD
        phase = self.tf_build_phase.__call__(opd)

        # Perform diffraction using FFT to compute the PSF
        psf = self.tf_fft_diffract.__call__(phase)

        return psf


class TFMonochromaticPSF(tf.Module):
    """Calculate a monochromatic Point Spread Function (PSF) from an OPD map.

    This class computes the monochromatic Point Spread Function (PSF) by first
    converting the Optical Path Difference (OPD) map into a phase map. Then, it
    applies diffraction using Fast Fourier Transform (FFT) techniques to simulate
    the PSF, which is essential in optical system simulations.

    Attributes
    ----------
    output_Q : int
        The output quality factor used for diffraction simulations.
    tf_build_phase : TFBuildPhase
        A module that builds the phase map from the OPD map, applying necessary
        zero-padding and obscurations.
    tf_fft_diffract : TFFftDiffract
        A module that performs the diffraction simulation using FFT.

    Parameters
    ----------
    phase_N : int
        The size of the phase map (e.g., pixel count for the height and width).
    lambda_obs : float
        The observed wavelength used for phase calculations.
    obscurations : tf.Tensor
        A tensor representing the obscurations (e.g., apertures or masks) to be
        applied to the phase.
    output_Q : int
        The output quality factor used for diffraction simulations.
    output_dim : int, optional
        The output dimension for the PSF, by default 64.
    name : str, optional
        The name for the TensorFlow module, by default None.
    """

    def __init__(
        self, phase_N, lambda_obs, obscurations, output_Q, output_dim=64, name=None
    ):
        """Initialize the TFMonochromaticPSF class.

        Parameters
        ----------
        phase_N : int
            The size of the phase map (e.g., pixel count for the height and width).
        lambda_obs : float
            The observed wavelength used for phase calculations.
        obscurations : tf.Tensor
            A tensor representing the obscurations (e.g., apertures or masks) to be
            applied to the phase.
        output_Q : int
            The output quality factor used for diffraction simulations.
        output_dim : int, optional
            The output dimension for the PSF, by default 64.
        name : str, optional
            The name for the TensorFlow module, by default None.
        """
        super().__init__(name=name)

        self.output_Q = output_Q
        self.tf_build_phase = TFBuildPhase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TFFftDiffract(output_dim, output_Q=self.output_Q)

    def __call__(self, opd):
        """Compute the PSF from an OPD map.

        This method converts the given OPD map into a phase map and performs a diffraction
        simulation using Fast Fourier Transform (FFT) to calculate the monochromatic PSF.

        Parameters
        ----------
        opd : tf.Tensor
            The Optical Path Difference (OPD) map with shape [batch_size, height, width].

        Returns
        -------
        tf.Tensor
            The resulting monochromatic PSF, cast to the same dtype as the input `opd`.
        """
        phase = self.tf_build_phase.__call__(opd)
        psf = self.tf_fft_diffract.__call__(phase)

        return tf.cast(psf, dtype=opd.dtype)
