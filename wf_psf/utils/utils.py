import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import PIL
import zernike as zk

try:
    from cv2 import resize, INTER_AREA
except:
    print("Problem importing opencv..")
import sys


def scale_to_range(input_array, old_range, new_range):
    # Scale to [0,1]
    input_array = (input_array - old_range[0]) / (old_range[1] - old_range[0])
    # Scale to new_range
    input_array = input_array * (new_range[1] - new_range[0]) + new_range[0]
    return input_array


def calc_wfe(zernike_basis, zks):
    wfe = np.einsum("ijk,ijk->jk", zernike_basis, zks.reshape(-1, 1, 1))
    return wfe


def calc_wfe_rms(zernike_basis, zks, pupil_mask):
    wfe = calc_wfe(zernike_basis, zks)
    wfe_rms = np.sqrt(np.mean((wfe[pupil_mask] - np.mean(wfe[pupil_mask])) ** 2))
    return wfe_rms


def generate_SED_elems(SED, sim_psf_toolkit, n_bins=20):
    """Generate SED Elements.

    A function to generate the SED elements needed for using the
    Tensor Flow class: TF_poly_PSF.

    Parameters
    ----------
    SED:
    sim_psf_toolkit:
        An instance of the SimPSFToolkit class with the correct
    initialization values.
    n_bins: int
        Number of wavelength bins
    """

    feasible_wv, SED_norm = sim_psf_toolkit.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([sim_psf_toolkit.feasible_N(_wv) for _wv in feasible_wv])

    return feasible_N, feasible_wv, SED_norm


def generate_SED_elems_in_tensorflow(
    SED, sim_psf_toolkit, n_bins=20, tf_dtype=tf.float64
):
    """Generate SED Elements in Tensor Flow Units.

    A function to generate the SED elements needed for using the
    Tensor Flow class: TF_poly_PSF.

    Parameters
    ----------
    SED:
    sim_psf_toolkit:
        An instance of the SimPSFToolkit class with the correct
    initialization values.
    n_bins: int
        Number of wavelength bins
    tf_dtype: tf.
        Tensor Flow data type
    """

    feasible_wv, SED_norm = sim_psf_toolkit.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([sim_psf_toolkit.feasible_N(_wv) for _wv in feasible_wv])

    return convert_to_tf([feasible_N, feasible_wv, SED_norm], tf_dtype)


def convert_to_tf(data, tf_dtype):
    return [tf.convert_to_tensor(x, dtype=tf_dtype) for x in data]


def generate_packed_elems(SED, sim_psf_toolkit, n_bins=20):
    """Generate Packed Elements.
    This name is too generic. may make obsolete

    A function to store the packed values for using the TF_poly_PSF.

    Parameters
    ----------
    SED:
    sim_psf_toolkit:
    n_bins: int
        Number of wavelength bins

    Returns
    -------
    list
        List of tensors
    """
    feasible_N, feasible_wv, SED_norm = generate_SED_elems(
        SED, sim_psf_toolkit, n_bins=n_bins
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
        row_idx = d * (d + 1) // 2
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


class NoiseEstimator(object):
    """Noise estimator.

    Parameters
    ----------
    img_dim: tuple of int
        Image size
    win_rad: int
        window radius in pixels

    """

    def __init__(self, img_dim, win_rad):
        self.img_dim = img_dim
        self.win_rad = win_rad
        self.window = None

        self._init_window()

    def _init_window(self):
        # Calculate window function for estimating the noise
        # We couldn't use Galsim to estimate the moments, so we chose to work
        # with the real center of the image (25.5,25.5)
        # instead of using the real centroid. Also, we use 13 instead of
        # 5 * obs_sigma, so that we are sure to cut all the flux from the star
        self.window = np.ones(self.img_dim, dtype=bool)

        mid_x = self.img_dim[0] / 2
        mid_y = self.img_dim[1] / 2

        for _x in range(self.img_dim[0]):
            for _y in range(self.img_dim[1]):
                if np.sqrt((_x - mid_x) ** 2 + (_y - mid_y) ** 2) <= self.win_rad:
                    self.window[_x, _y] = False

    @staticmethod
    def sigma_mad(x):
        r"""Compute an estimation of the standard deviation
        of a Gaussian distribution using the robust
        MAD (Median Absolute Deviation) estimator."""
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    def estimate_noise(self, image):
        r"""Estimate the noise level of the image."""

        # Calculate noise std dev
        return self.sigma_mad(image[self.window])


class ZernikeInterpolation(object):
    """Interpolate zernikes

    This class helps to interpolate zernikes using only the closest K elements
    in a given dataset using a RBF interpolation.

    Parameters
    ----------
    tf_pos: Tensor (n_sources, 2)
        Positions
    tf_zks: Tensor (n_sources, n_zernikes)
        Zernike coefficients for each position
    k: int
        Number of elements to use for the interpolation.
        Default is 50
    order: int
        Order of the RBF interpolation.
        Default is 2, corresponds to thin plate interp (r^2*log(r))

    """

    def __init__(self, tf_pos, tf_zks, k=50, order=2):
        self.tf_pos = tf_pos
        self.tf_zks = tf_zks
        self.k = k
        self.order = order

    def interpolate_zk(self, single_pos):
        """Interpolate a single position"""
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
        """Vectorize to interpolate to each position"""
        interp_zks = tf.map_fn(
            self.interpolate_zk,
            interp_positions,
            parallel_iterations=10,
            fn_output_signature=tf.float32,
            swap_memory=True,
        )
        return tf.squeeze(interp_zks, axis=1)


class IndependentZernikeInterpolation(object):
    """Interpolate each Zernike polynomial independently

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
        """Interpolate each Zerkine polynomial independently"""
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
        """Vectorize to interpolate to each Zernike!

        Each zernike is computed indepently from the others.
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


def PI_zernikes(tf_z1, tf_z2, norm_factor=None):
    """Compute internal product between zernikes and OPDs

    Defined such that Zernikes are orthonormal to each other

    First one should compute: norm_factor =  PI_zernikes(tf_zernike,tf_zernike)
    for futur calls: PI_zernikes(OPD,tf_zernike_k, norm_factor)
    """
    if norm_factor is None:
        norm_factor = 1
    return np.sum((tf.math.multiply(tf_z1, tf_z2)).numpy()) / (norm_factor)
