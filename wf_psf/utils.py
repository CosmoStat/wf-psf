import numpy as np
import tensorflow as tf
import PIL

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

    tf_feasible_N = tf.convert_to_tensor(feasible_N, dtype=tf.float64)
    tf_feasible_wv = tf.convert_to_tensor(feasible_wv, dtype=tf.float64)
    tf_SED_norm = tf.convert_to_tensor(SED_norm, dtype=tf.float64)

    # returnes the packed tensors
    return [tf_feasible_N, tf_feasible_wv, tf_SED_norm]

def decimate_im(input_im, decim_f):
    """Decimate image.

    Decimated by a factor of decim_f.
    Based on the PIL library using the default interpolator.
    Default: PIL.Image.BICUBIC.
    """
    pil_im = PIL.Image.fromarray(input_im)
    (width, height) = (pil_im.width // decim_f, pil_im.height // decim_f)
    im_resized = pil_im.resize((width, height))

    return np.array(im_resized)
