import numpy as np
import tensorflow as tf


class TF_fft_diffract(tf.Module):
    """Diffract the wavefront into a monochromatic PSF.

    Parameters
    ----------
    output_dim: int
        Dimension of the output square postage stamp
    output_Q: int
        Downsampling factor. Must be integer.
    """

    def __init__(self, output_dim=64, output_Q=2, name=None):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.output_Q = int(output_Q)

        self.downsample_layer = tf.keras.layers.AveragePooling2D(
            pool_size=(self.output_Q, self.output_Q),
            strides=None,
            padding="valid",
            data_format="channels_last",
        )

    def crop_img(self, image):
        # Crop the image
        start = int(image.shape[0] // 2 - self.output_dim // 2)
        stop = int(image.shape[0] // 2 + self.output_dim // 2)

        return image[start:stop, start:stop]

    def tf_crop_img(self, image, output_crop_dim):
        """Crop images with tf methods.

        It handles a batch of 2D images: [batch, width, height]
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
        # Sum over all the dimensions
        norm_factor = tf.math.reduce_sum(psf, axis=[1, 2], keepdims=True)

        return psf / norm_factor

    def __call__(self, input_phase):
        """Calculate the normalized PSF from the padded phase array."""
        # Perform the FFT-based diffraction operation
        # fft_phase = tf.signal.fftshift(tf.signal.fft2d(input_phase))
        fft_phase = tf.signal.fftshift(
            tf.signal.fft2d(input_phase[:, ...]), axes=[1, 2]
        )
        psf = tf.math.pow(tf.cast(tf.math.abs(fft_phase), dtype=tf.float64), 2)

        # Crop the image
        # We crop to output_dim*Q
        cropped_psf = self.tf_crop_img(
            psf, output_crop_dim=int(self.output_dim * self.output_Q)
        )

        # Downsample image
        # We downsample by a factor Q to get output_dim
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


class TF_build_phase(tf.Module):
    """Build complex phase map from OPD map."""

    def __init__(self, phase_N, lambda_obs, obscurations, name=None):
        super().__init__(name=name)

        self.phase_N = phase_N
        self.lambda_obs = lambda_obs
        self.obscurations = obscurations

    def zero_padding_diffraction(self, no_pad_phase):
        """Pad with zeros corresponding to the required lambda.

        Important: To check the original size of the ``no_pad_phase`` variable
        we have to look in the [1] dimension not the [0] as it is the batch.
        """
        # pad_num = int(self.phase_N//2 - no_pad_phase.shape[0]//2)
        phase_shape = tf.shape(no_pad_phase)
        # pure tensorflow
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
        # return tf.pad(no_pad_phase, padding)

    def apply_obscurations(self, phase):
        """Multiply element-wise with the obscurations."""
        return tf.math.multiply(phase, tf.cast(self.obscurations, phase.dtype))

    def opd_to_phase(self, opd):
        """Convert from opd to phase."""
        pre_phase = tf.math.multiply(
            tf.cast((2 * np.pi) / self.lambda_obs, opd.dtype), opd
        )
        phase = tf.math.exp(tf.dtypes.complex(tf.cast(0, pre_phase.dtype), pre_phase))
        # return tf.cast(phase, dtype=tf.complex64)
        return phase

    def __call__(self, opd):
        """Build the phase from the opd."""
        phase = self.opd_to_phase(opd)
        obsc_phase = self.apply_obscurations(phase)
        padded_phase = self.zero_padding_diffraction(obsc_phase)

        return padded_phase


class TF_zernike_OPD(tf.Module):
    """Turn zernike coefficients into an OPD.

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

    def __init__(
        self, phase_N, lambda_obs, obscurations, zernike_maps, output_dim=64, name=None
    ):
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
    """Calculate a monochromatic PSF from an OPD map."""

    def __init__(
        self, phase_N, lambda_obs, obscurations, output_Q, output_dim=64, name=None
    ):
        super().__init__(name=name)

        self.output_Q = output_Q
        self.tf_build_phase = TF_build_phase(phase_N, lambda_obs, obscurations)
        self.tf_fft_diffract = TF_fft_diffract(output_dim, output_Q=self.output_Q)

    def __call__(self, opd):
        phase = self.tf_build_phase.__call__(opd)
        psf = self.tf_fft_diffract.__call__(phase)

        return tf.cast(psf, dtype=opd.dtype)


# class TF_poly_PSF(tf.Module):
#     """Calculate a polychromatic PSF from an OPD and stored SED values.

#     The calculation of the packed values with the respective SED is done
#     with the SimPSFToolkit class but outside the TF class.

#     packed_elems: Tuple of tensors
#         Contains three 1D tensors with the parameters needed for
#         the calculation of each monochromatic PSF.

#         packed_elems[0]: phase_N
#         packed_elems[1]: lambda_obs
#         packed_elems[2]: SED_norm_val
#     """
#     def __init__(self, obscurations, packed_elems, output_dim=64, zernike_maps=None, name=None):
#         super().__init__(name=name)

#         self.obscurations = obscurations
#         self.output_dim = output_dim
#         self.packed_elems = packed_elems
#         self.zernike_maps = zernike_maps

#         self.opd = None

#     def set_packed_elems(self, new_packed_elems):
#         """Set packed elements."""
#         self.packed_elems = new_packed_elems

#     def set_zernike_maps(self, zernike_maps):
#         """Set Zernike maps."""
#         self.zernike_maps = zernike_maps

#     def calculate_from_zernikes(self, z_coeffs):
#         """Calculate polychromatic PSFs from zernike coefficients.

#         Zernike maps required.
#         """
#         tf_zernike_opd_gen = TF_zernike_OPD(self.zernike_maps)
#         # For readability
#         # opd = tf_zernike_opd_gen.__call__(z_coeffs)
#         # poly_psf = self.__call__(opd)
#         # return poly_psf

#         return self.__call__(tf_zernike_opd_gen.__call__(z_coeffs))

#     def calculate_mono_PSF(self, packed_elems):
#         """Calculate monochromatic PSF from packed elements.

#         packed_elems[0]: phase_N
#         packed_elems[1]: lambda_obs
#         packed_elems[2]: SED_norm_val
#         """
#         # Unpack elements
#         phase_N = packed_elems[0]
#         lambda_obs = packed_elems[1]
#         SED_norm_val = packed_elems[2]

#         # Build the monochromatic PSF generator
#         tf_mono_psf_gen = TF_mono_PSF(phase_N, lambda_obs, self.obscurations, output_dim=self.output_dim)

#         # Calculate the PSF
#         mono_psf = tf_mono_psf_gen.__call__(self.opd)

#         # Multiply with the respective normalized SED and return
#         return tf.math.scalar_mul(SED_norm_val, mono_psf)

#     def __call__(self, opd):

#         # Save the OPD that will be shared by all the monochromatic PSFs
#         self.opd = opd

#         # Use tf.function for parallelization over GPU
#         # Not allowed since the dynamic padding for the diffraction does not
#         # work in the @tf.function context
#         # @tf.function
#         def calculate_poly_PSF(elems_to_unpack):
#             return tf.map_fn(self.calculate_mono_PSF,
#                              elems_to_unpack,
#                              parallel_iterations=10,
#                              fn_output_signature=tf.float32)

#         stacked_psfs = calculate_poly_PSF(packed_elems)
#         poly_psf = tf.math.reduce_sum(stacked_psfs, axis=0)

#         return poly_psf
