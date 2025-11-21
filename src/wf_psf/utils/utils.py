"""utility functions for the PSF simulation and modeling.

:Authors: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import PIL
import zernike as zk

try:
    from cv2 import resize, INTER_AREA
except ModuleNotFoundError:
    print("Problem importing opencv..")
import sys


def generalised_sigmoid(x, max_val=1, power_k=1):
    """
    Apply a generalized sigmoid function to the input.

    This function computes a smooth, S-shaped curve that generalizes the standard
    sigmoid function. It's useful for scaling values while maintaining a bounded output.

    Parameters
    ----------
    x : array_like
        Input value(s) to which the generalized sigmoid is applied.
    max_val : float, optional
        Maximum output value. Default is 1.
    power_k : float, optional
        Power parameter controlling the steepness of the curve.
        Default is 1. Higher values create steeper transitions.

    Returns
    -------
    array_like
        Output value(s) scaled by the generalized sigmoid function,
        bounded between -max_val and max_val.

    Notes
    -----
    When power_k=1, this reduces to a standard rational sigmoid function.
    The function is odd, meaning generalised_sigmoid(-x) = -generalised_sigmoid(x).
    """
    return max_val * x / np.power(1 + np.power(np.abs(x), power_k), 1 / power_k)


def single_mask_generator(shape):
    """Generate a single mask with random 2D cosine waves.

    Note: These masks simulate the effect of cosmic rays on the observations.

    Parameters
    ----------
    shape: tuple
        Shape of the mask to be generated.

    Returns
    -------
    cosine_wave: np.ndarray
        A 2D mask with random 2D cosine waves.
    """
    # 2D meshgrid between 0.5 and 1
    x, y = np.meshgrid(np.linspace(0.7, 1.2, shape[1]), np.linspace(0.6, 1.1, shape[0]))
    # random pair of 2D frequencies, xy shifts and flip flag
    fxy_list = [np.random.random(5) * 6 for _ in range(100)]
    # 2D cosine waves
    cosine_wave_list = [
        np.cos(2 * np.pi * (fxy[0] * (x - fxy[2] / 50) + fxy[1] * (y - fxy[3] / 50)))
        for fxy in fxy_list
    ]
    # Sum of all cosine waves with random orientation
    cosine_wave_tot = np.zeros_like(cosine_wave_list[0])
    for cosine_wave, fxy in zip(cosine_wave_list, fxy_list):
        if fxy[4] < 3:
            cosine_wave = np.flipud(cosine_wave)
        cosine_wave_tot += cosine_wave
    # normalize
    cosine_wave = cosine_wave_tot / np.max(cosine_wave_tot)

    # detect values less than 0.6
    return cosine_wave < 0.6


def generate_n_mask(shape, n_masks=1):
    """Generate n masks with random 2D cosine waves.

    A wrapper around single_mask_generator to generate multiple masks.

    Parameters
    ----------
    shape: tuple
        Shape of the masks to be generated.
    n_masks: int
        Number of masks to be generated.

    Returns
    -------
    np.ndarray
        Array of shape (n_masks, shape[0], shape[1]) containing the generated masks.
    """
    return np.array([single_mask_generator(shape) for _ in range(n_masks)])


def generate_SED_elems(SED, psf_simulator, n_bins=20):
    """Generate SED elements for PSF modeling.

    Computes feasible Zernike mode numbers, wavelength values, and normalized
    SED for a given spectral energy distribution (SED) sampled across specified
    wavelength bins. These elements are required for PSF simulation and modeling
    with the TensorFlow-based PSF classes.

    Parameters
    ----------
    SED : np.ndarray
        The unfiltered SED with shape (n_wavelengths, 2). The first column contains
        wavelength positions (in wavelength units), and the second column contains
        the corresponding SED flux values.
    psf_simulator : PSFSimulator
        An instance of the PSFSimulator class initialized with the correct
        optical and instrumental parameters.
    n_bins : int, optional
        Number of wavelength bins to sample the SED. Default is 20.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray or float)
        - feasible_N : np.ndarray, shape (n_bins,)
            Feasible Zernike mode numbers at each wavelength bin.
        - feasible_wv : np.ndarray, shape (n_bins,)
            Sampled wavelength values across the SED.
        - SED_norm : np.ndarray or float
            Normalized SED values corresponding to feasible wavelengths.

    See Also
    --------
    generate_SED_elems_in_tensorflow : TensorFlow version of this function.
    generate_packed_elems : Wrapper that converts output to TensorFlow tensors.
    """
    feasible_wv, SED_norm = psf_simulator.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([psf_simulator.feasible_N(_wv) for _wv in feasible_wv])

    return feasible_N, feasible_wv, SED_norm


def generate_SED_elems_in_tensorflow(
    SED, psf_simulator, n_bins=20, tf_dtype=tf.float64
):
    """Generate SED Elements in TensorFlow Units.

    A function to generate the SED elements needed for using the
    TensorFlow class: TF_poly_PSF.

    Parameters
    ----------
    SED : np.ndarray
        The unfiltered SED. The first column contains the wavelength positions. The second column contains the SED value at each wavelength.
    psf_simulator : PSFSimulator object
        An instance of the PSFSimulator class with the correct initialization values.
    n_bins : int
        Number of wavelength bins
    tf_dtype : tf.DType
        The Tensor Flow dtype to cast each element to (for example `tf.float32`,
        `tf.int32`, etc.).

    Returns
    -------
    list of tf.Tensor
        [feasible_N, feasible_wv, SED_norm]:
        - feasible_N : tf.Tensor, shape (n_bins,), dtype tf_dtype
        - feasible_wv : tf.Tensor, shape (n_bins,), dtype tf_dtype
        - SED_norm : tf.Tensor, scalar or array, dtype tf_dtype
    """
    feasible_wv, SED_norm = psf_simulator.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([psf_simulator.feasible_N(_wv) for _wv in feasible_wv])

    return convert_to_tf([feasible_N, feasible_wv, SED_norm], tf_dtype)


def convert_to_tf(data, tf_dtype):
    """
    Convert a sequence of array-like objects to TensorFlow tensors with a specified dtype.

    Parameters
    ----------
    data : Iterable
        An iterable (e.g., list, tuple) of array-like objects (numpy arrays, Python
        lists/tuples, tf.Tensor, etc.) to be converted to TensorFlow tensors.
    tf_dtype : tf.DType
        The TensorFlow dtype to cast each element to (for example `tf.float32`,
        `tf.int32`, etc.).

    Returns
    -------
    list of tf.Tensor
        A list where each element is the result of calling
        `tf.convert_to_tensor` on the corresponding item from `data`, cast to
        `tf_dtype`.

    Raises
    ------
    TypeError
        If `data` is not an iterable. A `TypeError` may also be raised by
        `tf.convert_to_tensor` for individual elements that cannot be converted.

    Notes
    -----
    - The function preserves the top-level sequence structure by returning a
      list regardless of the input sequence type.
    - Element-wise conversion uses TensorFlow's conversion semantics; shape
      inference and broadcasting follow TensorFlow rules.

    """
    return [tf.convert_to_tensor(x, dtype=tf_dtype) for x in data]


def generate_packed_elems(SED, psf_simulator, n_bins=20):
    """
    Generate packed SED elements as TensorFlow tensors.

    Wrapper around generate_SED_elems(...) that converts the returned NumPy
    arrays into TensorFlow tensors with dtype=tf.float64.

    Parameters
    ----------
    SED : numpy.ndarray
        The unfiltered SED with shape (n_wavelengths, 2). The first column contains the wavelength
        positions (in wavelength units), and the second column contains the corresponding SED flux values.
    psf_simulator : PSFSimulator object
        An instance of the PSF simulator providing calc_SED_wave_values and feasible_N.
    n_bins : int, optional
        Number of wavelength bins used to sample the SED (default 20).

    Returns
    -------
    list of tf.Tensor
        [feasible_N, feasible_wv, SED_norm]:
        - feasible_N : tf.Tensor, shape (n_bins,), dtype tf.float64
        - feasible_wv : tf.Tensor, shape (n_bins,), dtype tf.float64
        - SED_norm : tf.Tensor, scalar or array, dtype tf.float64
    """
    feasible_N, feasible_wv, SED_norm = generate_SED_elems(
        SED, psf_simulator, n_bins=n_bins
    )

    feasible_N = tf.convert_to_tensor(feasible_N, dtype=tf.float64)
    feasible_wv = tf.convert_to_tensor(feasible_wv, dtype=tf.float64)
    SED_norm = tf.convert_to_tensor(SED_norm, dtype=tf.float64)

    # returns the packed tensors
    return [feasible_N, feasible_wv, SED_norm]


def calc_poly_position_mat(pos, x_lims, y_lims, d_max):
    r"""Calculate a matrix with position polynomials.

    Scale positions to the square:
    [self.x_lims[0], self.x_lims[1]] x [self.y_lims[0], self.y_lims[1]]
    to the square [-1,1] x [-1,1]
    """
    # Scale positions
    scaled_pos_x = (pos[:, 0] - x_lims[0]) / (x_lims[1] - x_lims[0])
    scaled_pos_x = (scaled_pos_x - 0.5) * 2
    scaled_pos_y = (pos[:, 1] - y_lims[0]) / (y_lims[1] - y_lims[0])
    scaled_pos_y = (scaled_pos_y - 0.5) * 2

    poly_list = []

    for d in range(d_max + 1):
        # row_idx = d * (d + 1) // 2
        for p in range(d + 1):
            poly_list.append(scaled_pos_x ** (d - p) * scaled_pos_y**p)

    return tf.convert_to_tensor(poly_list, dtype=tf.float32)


def decimate_im(input_im, decim_f):
    r"""Decimate image.

    Decimated by a factor of decim_f.
    Based on the PIL library using the default interpolator.
    Default: PIL.Image.BICUBIC.
    """
    pil_im = PIL.Image.fromarray(input_im)
    (width, height) = (pil_im.width // decim_f, pil_im.height // decim_f)
    im_resized = pil_im.resize((width, height))

    return np.array(im_resized)


def downsample_im(input_im, output_dim):
    r"""Downsample image.

    Based on opencv function resize.
    [doc](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20resize(InputArray%20src,%20OutputArray%20dst,%20Size%20dsize,%20double%20fx,%20double%20fy,%20int%20interpolation))
    The input image is downsampled to the dimensions specified in `output_dim`.
    The downsampling method is based on the `INTER_AREA` method.
    See [tensorflow_doc](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resize-area)
    Each output pixel is computed by first transforming the pixel's footprint
    into the input tensor and then averaging the pixels that intersect the
    footprint. An input pixel's contribution to the average is weighted by the
    fraction of its area that intersects the footprint.
    This is the same as OpenCV's INTER_AREA.
    An explanation of the INTER_AREA method can be found in the next
    [link](https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3).

    This version should be consistent with the tensorflow one.

    Parameters
    ----------
    input_im: np.ndarray (dim_x, dim_y)
        input image
    output_dim: int
        Contains the dimension of the square output image.
    """
    return resize(input_im, (output_dim, output_dim), interpolation=INTER_AREA)


def zernike_generator(n_zernikes, wfe_dim):
    r"""
    Generate Zernike maps.

    Based on the zernike github repository.
    https://github.com/jacopoantonello/zernike

    Parameters
    ----------
    n_zernikes: int
        Number of Zernike modes desired.
    wfe_dim: int
        Dimension of the Zernike map [wfe_dim x wfe_dim].

    Returns
    -------
    zernikes: list of np.ndarray
        List containing the Zernike modes.
        The values outside the unit circle are filled with NaNs.
    """
    # Calculate which n (from the (n,m) Zernike convention) we need
    # so that we have the desired total number of Zernike coefficients
    min_n = (-3 + np.sqrt(1 + 8 * n_zernikes)) / 2
    n = int(np.ceil(min_n))

    # Initialize the zernike generator
    cart = zk.RZern(n)
    # Create a [-1,1] mesh
    ddx = np.linspace(-1.0, 1.0, wfe_dim)
    ddy = np.linspace(-1.0, 1.0, wfe_dim)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)

    c = np.zeros(cart.nk)
    zernikes = []

    # Extract each Zernike map one by one
    for i in range(n_zernikes):
        c *= 0.0
        c[i] = 1.0
        zernikes.append(cart.eval_grid(c, matrix=True))

    return zernikes


def add_noise(image, desired_SNR):
    """Add noise to an image to obtain a desired SNR."""
    sigma_noise = np.sqrt(
        (np.sum(image**2)) / (desired_SNR * image.shape[0] * image.shape[1])
    )
    noisy_image = image + np.random.standard_normal(image.shape) * sigma_noise
    return noisy_image


class NoiseEstimator:
    """
    A class for estimating noise levels in an image.

    Parameters
    ----------
    img_dim : tuple of int
        The dimensions of the image as (height, width).
    win_rad : int
        The radius of the exclusion window (in pixels).
    """

    def __init__(self, img_dim: tuple[int, int], win_rad: int) -> None:
        """
        Initialize a NoiseEstimator instance.

        This constructor sets up the noise estimator by storing the image dimensions
        and exclusion window radius, then initializes the exclusion window mask.

        Parameters
        ----------
        img_dim : tuple of int
            The dimensions of the image as (height, width).
        win_rad : int
            The radius of the exclusion window in pixels. Pixels within this radius
            of the image center are excluded from noise estimation.

        Notes
        -----
        The exclusion window is initialized automatically via _init_window(), creating
        a boolean mask where pixels inside the exclusion radius are marked False
        (excluded) and pixels outside are marked True (included).
        """
        self.img_dim: tuple[int, int] = img_dim
        self.win_rad: int = win_rad

        self._init_window()  # Initialize self.window

    def _init_window(self):
        """
        Initialize the exclusion window mask stored in self.window.

        The mask is a boolean array of shape `self.img_dim` (rows, cols). Pixels
        whose Euclidean distance from the image center is less than or equal to
        `self.win_rad` are marked False (excluded); all other pixels are True
        (included). The mask dtype is `bool`.

        Notes
        -----
        - The image center is computed as (rows / 2, cols / 2). This yields a
          floating-point center so the distance is computed with sub-pixel
          precision; for even dimensions the center lies between pixels.
        - The comparison uses "<=" so pixels exactly at distance `win_rad` are
          excluded. Change to "<" if you prefer a strict interior exclusion.
        - Time complexity is O(rows * cols) for mask construction.
        - No return value; the constructed mask is assigned to `self.window`.
        """
        self.window = np.ones(self.img_dim, dtype=bool)

        mid_x = self.img_dim[0] / 2
        mid_y = self.img_dim[1] / 2

        for _x in range(self.img_dim[0]):
            for _y in range(self.img_dim[1]):
                # If pixel is within the exclusion radius, set it to False
                if np.sqrt((_x - mid_x) ** 2 + (_y - mid_y) ** 2) <= self.win_rad:
                    self.window[_x, _y] = False

    def apply_mask(self, mask: np.ndarray = None) -> np.ndarray:
        """
        Apply a given mask to the exclusion window.

        Parameters
        ----------
        mask : np.ndarray, optional
            A boolean mask to apply to the exclusion window. If None, the exclusion
            window is returned without any modification.

        Returns
        -------
        np.ndarray
            The resulting boolean array after applying the mask to the exclusion window.
        """
        if mask is None:
            return self.window  # Return just the window if no mask is provided
        return self.window & mask  # Otherwise, apply the mask as usual

    @staticmethod
    def sigma_mad(x):
        """
        Robustly estimate the standard deviation using the Median Absolute Deviation (MAD).

        Computes MAD = ``median(|x - median(x)|)`` and scales it by 1.4826 to make the
        estimator consistent with the standard deviation for a Gaussian distribution:

            sigma ≈ 1.4826 * MAD

        Parameters
        ----------
        x : array-like
            Input data. The values are flattened before computation. NaNs are not
            specially handled and will propagate; remove or mask them prior to
            calling if needed.

        Returns
        -------
        float
            Robust estimate of the standard deviation of the input data.

        Notes
        -----
        - The MAD-based estimator is much less sensitive to outliers than the
          sample standard deviation, making it appropriate for noisy data with
          occasional large deviations.
        - The constant 1.4826 is the scaling factor for consistency with the
          standard deviation of a normal distribution.
        """
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    def estimate_noise(self, image: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Estimates the noise level of an image using the MAD estimator.

        Parameters
        ----------
        image : np.ndarray
            The input image for noise estimation.
        mask : np.ndarray, optional
            A boolean mask specifying which pixels to include in the noise estimation.
            If None, the default exclusion window is used. The mask should have the same shape as `image`.

        Returns
        -------
        float
            The estimated noise standard deviation (MAD of the image pixels within the window or mask).
        """
        if mask is not None:
            return self.sigma_mad(image[self.apply_mask(mask)])

        # Use the default window if no mask is provided
        return self.sigma_mad(image[self.window])


class ZernikeInterpolation:
    """Interpolate Zernike coefficients using K-nearest RBF splines.

    This class provides utilities to interpolate Zernike-coefficient vectors
    defined at a set of source positions to arbitrary query positions using
    a local RBF spline fitted to the K nearest source samples.

    The interpolation pipeline:
    - For a given query position, compute Euclidean distances to all source
      positions and select the K nearest neighbors.
    - Use tfa.image.interpolate_spline (RBF / spline interpolation) on the
      selected neighbor positions and their Zernike coefficient vectors to
      compute the interpolated coefficients at the query location.

    Parameters
    ----------
    tf_pos : tf.Tensor, shape (n_sources, 2)
        Source/sample positions (x, y). Expected dtype float32 or convertible.
    tf_zks : tf.Tensor, shape (n_sources, n_zernikes)
        Zernike coefficient vectors at the source positions.
    k : int, default 50
        Number of nearest neighbors to use for the local interpolation. If
        larger than the number of sources, all sources are used.
    order : int, default 2
        Spline order passed to tfa.image.interpolate_spline (e.g. 2 for thin
        plate style interpolation).

    Attributes
    ----------
    tf_pos, tf_zks, k, order
        Stored copies of the constructor inputs.

    Notes
    -----
    - This class relies on TensorFlow Addons' interpolate_spline, which requires
      inputs to include a leading batch dimension; the implementation handles
      that automatically.
    - For best numerical stability and compatibility with TFA, use float32
      tensors for inputs when possible.
    - Two main methods are provided:
        - interpolate_zk(single_pos): interpolate a single position -> 1D vector.
        - interpolate_zks(interp_positions): vectorized interpolation for many
          query positions (uses tf.map_fn under the hood).

    """

    def __init__(self, tf_pos, tf_zks, k=50, order=2):
        self.tf_pos = tf_pos
        self.tf_zks = tf_zks
        self.k = k
        self.order = order

    def interpolate_zk(self, single_pos):
        """Interpolate Zernike coefficients at a single query position using K-nearest neighbors.

        Finds the K nearest training positions to the query position and uses RBF
        spline interpolation to estimate Zernike coefficients at that location.

        Parameters
        ----------
        single_pos : tf.Tensor, shape (2,)
            Query position coordinates as (x, y).

        Returns
        -------
        tf.Tensor, shape (n_zernikes,)
            Interpolated Zernike coefficient vector at the query position.
        """
        # Compute distance
        dist = tf.math.reduce_euclidean_norm(self.tf_pos - single_pos, axis=1) * -1.0
        # Get top K elements
        result = tf.math.top_k(dist, k=self.k)
        # Gather useful elements from the array
        rec_pos = tf.gather(
            self.tf_pos,
            result.indices,
            validate_indices=None,
            axis=0,
            batch_dims=0,
        )
        rec_zks = tf.gather(
            self.tf_zks,
            result.indices,
            validate_indices=None,
            axis=0,
            batch_dims=0,
        )
        # Interpolate
        interp_zk = tfa.image.interpolate_spline(
            train_points=tf.expand_dims(rec_pos, axis=0),
            train_values=tf.expand_dims(rec_zks, axis=0),
            query_points=tf.expand_dims(single_pos[tf.newaxis, :], axis=0),
            order=self.order,
            regularization_weight=0.0,
        )
        # Remove extra dimension required by tfa's interpolate_spline
        interp_zk = tf.squeeze(interp_zk, axis=0)

        return interp_zk

    def interpolate_zks(self, interp_positions):
        """Interpolate Zernike coefficient vectors at multiple query positions.

        Vectorized wrapper that applies self.interpolate_zk to each row of
        interp_positions using tf.map_fn.

        Parameters
        ----------
        interp_positions : tf.Tensor, shape (n_targets, 2)
            Query positions where Zernike coefficients should be interpolated.
            Each row is an (x, y) coordinate.

        Returns
        -------
        tf.Tensor, shape (n_targets, n_zernikes), dtype=tf.float32
            Interpolated Zernike coefficient vectors for each query position.
            tf.map_fn may introduce an extra singleton dimension; this is removed
            by tf.squeeze before returning.

        Notes
        -----
        - self.interpolate_zk expects a 1-D tensor of shape (2,) and returns a
          1-D tensor of length n_zernikes.
        - This function uses tf.map_fn with fn_output_signature=tf.float32 and
          swap_memory=True for efficient batching.
        """
        interp_zks = tf.map_fn(
            self.interpolate_zk,
            interp_positions,
            parallel_iterations=10,
            fn_output_signature=tf.float32,
            swap_memory=True,
        )
        return tf.squeeze(interp_zks, axis=1)


class IndependentZernikeInterpolation:
    """Interpolate each Zernike polynomial independently.

    The interpolation is done independently for each Zernike polynomial.

    Parameters
    ----------
    tf_pos: Tensor (n_sources, 2)
        Positions
    tf_zks: Tensor (n_sources, n_zernikes)
        Zernike coefficients for each position
    order: int
        Order of the RBF interpolation.
        Default is 2, corresponds to thin plate interp (r^2*log(r))

    """

    def __init__(self, tf_pos, tf_zks, order=2):
        self.tf_pos = tf_pos
        self.tf_zks = tf_zks
        self.order = order

        self.target_pos = None

    def interp_one_zk(self, zk_prior):
        """
        Interpolate a single Zernike polynomial across target positions.

        Each Zernike coefficient in `zk_prior` is interpolated independently
        using a spline.

        Parameters
        ----------
        zk_prior : tf.Tensor of shape (n_sources,)
            Zernike coefficients for a single Zernike polynomial, defined at
            the source positions `self.tf_pos`.

        Returns
        -------
        tf.Tensor of shape (n_targets,)
            Interpolated Zernike coefficients at the target positions
            `self.target_pos`.

        Notes
        -----
        This function uses `tfa.image.interpolate_spline`, which requires the
        input to have a batch dimension. The extra dimension is removed before
        returning the result.
        """
        interp_zk = tfa.image.interpolate_spline(
            train_points=tf.expand_dims(self.tf_pos, axis=0),
            train_values=tf.expand_dims(zk_prior[:, tf.newaxis], axis=0),
            query_points=tf.expand_dims(self.target_pos, axis=0),
            order=self.order,
            regularization_weight=0.0,
        )

        # Remove extra dimension required by tfa's interpolate_spline
        return tf.squeeze(interp_zk, axis=0)

    def interpolate_zks(self, target_pos):
        """Vectorize to interpolate to each Zernike.

        Each zernike is computed indepently from the others.

        Parameters
        ----------
        target_pos: Tensor (n_targets, 2)
            Positions to interpolate to.

        Returns
        -------
        Tensor (n_targets, n_zernikes)
        """
        self.target_pos = target_pos

        interp_zks = tf.map_fn(
            self.interp_one_zk,
            tf.transpose(self.tf_zks, perm=[1, 0]),
            parallel_iterations=10,
            fn_output_signature=tf.float32,
            swap_memory=True,
        )

        # Remove null dimension and transpose back to have batch at input
        return tf.transpose(tf.squeeze(interp_zks, axis=2), perm=[1, 0])


def load_multi_cycle_params_click(args):
    """
    Load multiple cycle training parameters.

    For backwards compatibility, the training parameters are received as a string,
    separated and stored in the args dictionary.

    Parameters
    ----------
    args: dictionary
        Comand line arguments dictionary loaded with the click package.

    Returns
    -------
    args: dictionary
        The input dictionary with all multi-cycle training parameters correctly loaded.
    """
    if args["l_rate_param"] is None:
        args["l_rate_param"] = list(
            map(float, args["l_rate_param_multi_cycle"].split(" "))
        )
    if len(args["l_rate_param"]) == 1:
        args["l_rate_param"] = args["l_rate_param"] * args["total_cycles"]
    elif len(args["l_rate_param"]) != args["total_cycles"]:
        print(
            "Invalid argument: --l_rate_param. Expected 1 or {} values but {} were given.".format(
                args["total_cycles"], len(args["l_rate_param"])
            )
        )
        sys.exit()

    if args["l_rate_non_param"] is None:
        args["l_rate_non_param"] = list(
            map(float, args["l_rate_non_param_multi_cycle"].split(" "))
        )
    if len(args["l_rate_non_param"]) == 1:
        args["l_rate_non_param"] = args["l_rate_non_param"] * args["total_cycles"]
    elif len(args["l_rate_non_param"]) != args["total_cycles"]:
        print(
            "Invalid argument: --l_rate_non_param. Expected 1 or {} values but {} were given.".format(
                args["total_cycles"], len(args["l_rate_non_param"])
            )
        )
        sys.exit()

    if args["n_epochs_param"] is None:
        args["n_epochs_param"] = list(
            map(int, args["n_epochs_param_multi_cycle"].split(" "))
        )
    if len(args["n_epochs_param"]) == 1:
        args["n_epochs_param"] = args["n_epochs_param"] * args["total_cycles"]
    elif len(args["n_epochs_param"]) != args["total_cycles"]:
        print(
            "Invalid argument: --n_epochs_param. Expected 1 or {} values but {} were given.".format(
                args["total_cycles"], len(args["n_epochs_param"])
            )
        )
        sys.exit()

    if args["n_epochs_non_param"] is None:
        args["n_epochs_non_param"] = list(
            map(int, args["n_epochs_non_param_multi_cycle"].split(" "))
        )
    if len(args["n_epochs_non_param"]) == 1:
        args["n_epochs_non_param"] = args["n_epochs_non_param"] * args["total_cycles"]
    elif len(args["n_epochs_non_param"]) != args["total_cycles"]:
        print(
            "Invalid argument: --n_epochs_non_param. Expected 1 or {} values but {} were given.".format(
                args["total_cycles"], len(args["n_epochs_non_param"])
            )
        )
        sys.exit()

    return args


def compute_unobscured_zernike_projection(tf_z1, tf_z2, norm_factor=None):
    """Compute a zernike projection for unobscured wavefronts (OPDs).

    Compute internal product between zernikes and OPDs.

    Defined such that Zernikes are orthonormal to each other.

    First one should compute: norm_factor =  unobscured_zernike_projection(tf_zernike,tf_zernike)
    for futur calls: unobscured_zernike_projection(OPD,tf_zernike_k, norm_factor)

    If the OPD has obscurations, or is not an unobscured circular aperture,
    the Zernike polynomials are no longer orthonormal. Therefore, you should consider
    using the function `tf_decompose_obscured_opd_basis` that takes into account the
    obscurations in the projection.
    """
    if norm_factor is None:
        norm_factor = 1
    return np.sum((tf.math.multiply(tf_z1, tf_z2)).numpy()) / (norm_factor)


def decompose_tf_obscured_opd_basis(
    tf_opd, tf_obscurations, tf_zk_basis, n_zernike, iters=20
):
    """Decompose obscured OPD into a basis using an iterative algorithm.

    Tensorflow implementation.

    Parameters
    ----------
    tf_opd : tf.Tensor
        Input OPD that requires to be decomposed on `tf_zk_basis`. The tensor shape is (opd_dim, opd_dim).
    tf_obscurations : tf.Tensor
        Tensor with the obscuration map.  The tensor shape is (opd_dim, opd_dim).
    tf_zk_basis : tf.Tensor
        Zernike polynomial maps. The tensor shape is (n_batch, opd_dim, opd_dim)
    n_zernike : int
        Number of Zernike polynomials to project on.
    iters : int
        Number of iterations of the algorithm.

    Returns
    -------
    obsc_coeffs : np.ndarray
        Array of size `n_zernike` with projected Zernike coefficients

    Raises
    ------
    ValueError
        If `n_zernike` is bigger than tf_zk_basis.shape[0].

    """
    if n_zernike > tf_zk_basis.shape[0]:
        raise ValueError(
            "Number of Zernike polynomials to project (n_zernike) exceeds the available Zernike elements in the provided basis (tf_zk_basis). Please ensure that n_zernike is less than or equal to the number of Zernike elements in tf_zk_basis."
        )
    # Clone input OPD
    input_tf_opd = tf.identity(tf_opd)
    # Clone obscurations and project
    input_tf_obscurations = tf.math.real(tf.identity(tf_obscurations))
    # Compute normalisation factor
    ngood = tf.math.reduce_sum(input_tf_obscurations, axis=None, keepdims=False).numpy()

    obsc_coeffs = np.zeros(n_zernike)
    new_coeffs = np.zeros(n_zernike)

    for count in range(iters):
        for i, b in enumerate(tf_zk_basis):
            this_coeff = (
                tf.math.reduce_sum(
                    tf.math.multiply(input_tf_opd, b), axis=None, keepdims=False
                ).numpy()
                / ngood
            )
            new_coeffs[i] = this_coeff

        for i, b in enumerate(tf_zk_basis):
            input_tf_opd = input_tf_opd - tf.math.multiply(
                new_coeffs[i] * b, input_tf_obscurations
            )

        obsc_coeffs += new_coeffs
        new_coeffs = np.zeros(n_zernike)

    return obsc_coeffs
