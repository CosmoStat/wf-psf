import numpy as np
import tensorflow as tf
import PIL
from cv2 import resize, INTER_AREA
import zernike as zk
from wf_psf import SimPSFToolkit as SimPSFToolkit
from wf_psf import tf_psf_field as psf_field

def generate_SED_elems(SED, sim_psf_toolkit, n_bins=20):
    r"""Generate the SED elements needed for using the TF_poly_PSF.

    sim_psf_toolkit: An instance of the SimPSFToolkit class with the correct
    initialization values.
    """

    feasible_wv, SED_norm = sim_psf_toolkit.calc_SED_wave_values(SED, n_bins)
    feasible_N = np.array([sim_psf_toolkit.feasible_N(_wv)  for _wv in feasible_wv])

    return feasible_N, feasible_wv, SED_norm


def generate_packed_elems(SED, sim_psf_toolkit, n_bins=20):
    r"""Generate the packed values for using the TF_poly_PSF."""
    feasible_N, feasible_wv, SED_norm = generate_SED_elems(SED, sim_psf_toolkit, n_bins=n_bins)

    tf_feasible_N = tf.convert_to_tensor(feasible_N, dtype=tf.float64)
    tf_feasible_wv = tf.convert_to_tensor(feasible_wv, dtype=tf.float64)
    tf_SED_norm = tf.convert_to_tensor(SED_norm, dtype=tf.float64)

    # returnes the packed tensors
    return [tf_feasible_N, tf_feasible_wv, tf_SED_norm]


def calc_poly_position_mat(pos, x_lims, y_lims, d_max):
    r""" Calculate a matrix with position polynomials.

    Scale positions to the square:
    [self.x_lims[0], self.x_lims[1]] x [self.y_lims[0], self.y_lims[1]]
    to the square [-1,1] x [-1,1]
    """
    # Scale positions
    scaled_pos_x = (pos[:,0] - x_lims[0]) / (x_lims[1] - x_lims[0])
    scaled_pos_x = (scaled_pos_x - 0.5) * 2
    scaled_pos_y = (pos[:,1] - y_lims[0]) / (y_lims[1] - y_lims[0])
    scaled_pos_y = (scaled_pos_y - 0.5) * 2

    poly_list = []

    for d in range(d_max + 1):
        row_idx = d * (d + 1) // 2
        for p in range(d + 1):
            poly_list.append(scaled_pos_x ** (d - p) * scaled_pos_y ** p)

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
    min_n = (-3 + np.sqrt(1+8*n_zernikes) )/2
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
    """ Add noise to an image to obtain a desired SNR. """
    sigma_noise = np.sqrt((np.sum(image**2))/(desired_SNR * image.shape[0] * image.shape[1]))
    noisy_image = image + np.random.standard_normal(image.shape) * sigma_noise
    return noisy_image


def gen_GT_wf_model(test_wf_file_path, pred_output_Q=1, pred_output_dim=64):
    r""" Generate the ground truth model and output test PSF ar required resolution. 

    If `pred_output_Q=1` the resolution will be 3 times the one of Euclid.
    """
    # Load dataset
    wf_test_dataset = np.load(test_wf_file_path, allow_pickle=True)[()]
    
    # Extract parameters from the wf test dataset
    wf_test_params = wf_test_dataset['parameters']
    wf_test_C_poly = wf_test_dataset['C_poly']
    wf_test_pos = wf_test_dataset['positions']
    tf_test_pos = tf.convert_to_tensor(wf_test_pos, dtype=tf.float32)
    wf_test_SEDs = wf_test_dataset['SEDs']

    # Generate GT model
    batch_size = 16

    # Generate Zernike maps
    zernikes = zernike_generator(
        n_zernikes=wf_test_params['max_order'],
        wfe_dim=wf_test_params['pupil_diameter']
    )

    ## Generate initializations
    # Prepare np input
    simPSF_np = SimPSFToolkit(
        zernikes,
        max_order=wf_test_params['max_order'],
        pupil_diameter=wf_test_params['pupil_diameter'],
        output_dim=wf_test_params['output_dim'],
        oversampling_rate=wf_test_params['oversampling_rate'],
        output_Q=wf_test_params['output_Q']
    )
    simPSF_np.gen_random_Z_coeffs(max_order=wf_test_params['max_order'])
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(
        N_pix=wf_test_params['pupil_diameter'],
        N_filter=wf_test_params['LP_filter_length']
    )
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)


    ## Prepare ground truth model
    # Now Zernike's as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Initialize the model
    GT_tf_semiparam_field = psf_field.TF_SemiParam_field(
        zernike_maps=tf_zernike_cube,
        obscurations=tf_obscurations,
        batch_size=batch_size,
        output_Q=wf_test_params['output_Q'],
        d_max_nonparam=2,
        output_dim=wf_test_params['output_dim'],
        n_zernikes=wf_test_params['max_order'],
        d_max=wf_test_params['d_max'],
        x_lims=wf_test_params['x_lims'],
        y_lims=wf_test_params['y_lims']
    )

    # For the Ground truth model
    GT_tf_semiparam_field.tf_poly_Z_field.assign_coeff_matrix(wf_test_C_poly)
    _ = GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat.assign(
        tf.zeros_like(GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat)
    )

    # Set required output_Q

    GT_tf_semiparam_field.set_output_Q(output_Q=pred_output_Q, output_dim=pred_output_dim)

    GT_tf_semiparam_field = psf_field.build_PSF_model(GT_tf_semiparam_field)

    packed_SED_data = [
        generate_packed_elems(
            _sed,
            simPSF_np,
            n_bins=wf_test_params['n_bins']
        )
        for _sed in wf_test_SEDs
    ]

    # Prepare inputs
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_test_pos , tf_packed_SED_data]

    # Ground Truth model
    GT_predictions = GT_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    return GT_predictions, wf_test_pos


class NoiseEstimator(object):
    """ Noise estimator.

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
                if np.sqrt((_x - mid_x)**2 + (_y - mid_y)**2) <= self.win_rad:
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

