"""Centroids.

A module with utils to handle PSF centroids.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import scipy.signal as scisig
from wf_psf.utils.preprocessing import shift_x_y_to_zk1_2_wavediff


def get_zk1_2_for_observed_psf(
    obs_psf,
    mask, 
    pixel_sampling=12e-6,
    reference_shifts=[-1 / 3, -1 / 3],
    sigma_init=2.5,
    n_iter=20,
):
    """Get Zk1 and Zk2 corrections required for an observed PSF.

    The Zk1 and Zk2 should be used with Wavediff so the PSF centroids match.

    Note: The default `reference_shifts` value is for observations at Euclid conditions,
    i.e., pixel sampling and telescope parameters.

    Parameters
    ----------
    obs_psf : np.ndarray
        Observed PSF at Euclid resolution.
    mask : numpy.ndarray
        A mask to apply, which **can contain float values in [0,1]**. 
        - `0` means the pixel is ignored.
        - `1` means the pixel is fully considered.
        - Values in `(0,1]` act as weights for partial consideration.
    pixel_sampling : float
        Pixel sampling in [m]
    reference_shifts : list
        Reference shifts for WaveDiff at Euclid nominal conditions, in [pixel]
    sigma_init : float
        Initial size for the centroid calculator.
    n_iter : int
        Iteration number for the centroid calculation.

    Returns
    -------
    z1_wavediff, z2_wavediff : tuple
        Tip and tilt Zernike coefficients in wavediff convention
    """

    # Build centroid estimator
    centroid_calc = CentroidEstimator(obs_psf, mask=None, sigma_init=sigma_init, n_iter=n_iter)

    current_shifts = centroid_calc.return_shifts()  # In [pixel]

    dx = reference_shifts[1] - current_shifts[1]  # In [pixel]
    dy = reference_shifts[0] - current_shifts[0]  # In [pixel]

    z1_wavediff = shift_x_y_to_zk1_2_wavediff(dx * pixel_sampling)  # Input in [m]
    z2_wavediff = shift_x_y_to_zk1_2_wavediff(dy * pixel_sampling)  # Input in [m]

    return (
        z1_wavediff,
        z2_wavediff,
    )  # Output in Zernike coefficients in wavediff convention


def compute_centroid(poly_psf, mask=None, sigma_init=5.5, n_iter=10):
    """Compute PSF centroid."""
    ref_centroid_calc = CentroidEstimator(
        poly_psf, sigma_init=sigma_init, n_iter=n_iter
    )

    return ref_centroid_calc.get_centroids()


class CentroidEstimator(object):
    r"""Estimate intra-pixel shifts.

    It calculates the centroid of the image and compare it with the stamp
    centroid and returns the proper shift.
    The star centroid is calculated following an iterative procedure where a
    matched elliptical gaussian is used to calculate the moments.

    Parameters
    ----------
    im : numpy.ndarray
        Star image stamp.
    mask : numpy.ndarray
        A mask to apply, which **can contain float values in [0,1]**. 
        - `0` means the pixel is ignored.
        - `1` means the pixel is fully considered.
        - Values in `(0,1]` act as weights for partial consideration.
    sigma_init : float
        Estimated shape of the star in sigma.
        Default is 7.5.
    n_iter : int
        Max iteration number for the iterative estimation procedure.
        Default is 5.
    auto_run : bool
        Auto run the intra-pixel shif calculation in the initialization
        of the class.
        Default is True.
    xc : float
        First guess of the ``x`` component of the star centroid. (optional)
        Default is None.
    yc : float
        First guess of the ``y`` component of the star centroid. (optional)
        Default is None.
    """

    def __init__(self, im, mask=None, sigma_init=7.5, n_iter=5, auto_run=True, xc=None, yc=None):
        r"""Initialize class attributes."""
        self.im = im
        self.mask = mask
        if self.mask is not None:
            self.im = self.im * (1 - self.mask)
        self.stamp_size = im.shape
        self.ranges = np.array([np.arange(i) for i in self.stamp_size])
        self.sigma_init = sigma_init
        self.n_iter = n_iter
        self.xc0, self.yc0 = (
            float(self.stamp_size[0]) / 2,
            float(self.stamp_size[1]) / 2,
        )

        self.window = None
        self.xx = None
        self.yy = None

        if xc is None or yc is None:
            self.xc = self.xc0
            self.yc = self.yc0
        else:
            self.xc = xc
            self.yc = yc
        if auto_run:
            self.estimate()

    def UpdateGrid(self):
        r"""Update the grid where the star stamp is defined."""
        self.xx = np.outer(self.ranges[0] - self.xc, np.ones(self.stamp_size[1]))
        self.yy = np.outer(np.ones(self.stamp_size[0]), self.ranges[1] - self.yc)

    def EllipticalGaussian(self, e1=0, e2=0):
        r"""Compute an elliptical 2D gaussian with arbitrary centroid."""
        # Shear it
        gxx = (1 - e1) * self.xx - e2 * self.yy
        gyy = (1 + e1) * self.yy - e2 * self.xx
        # compute elliptical gaussian
        return np.exp(-(gxx**2 + gyy**2) / (2 * self.sigma_init**2))

    def ComputeMoments(self):
        r"""Compute the star moments.

        Compute the star image normalized first order moments with
        the current window function.
        """
        if self.mask is not None:
            masked_im_window = self.im * self.window * (self.mask == 0)
        else:
            masked_im_window = self.im * self.window

        Q0 = np.sum(masked_im_window)
        Q1 = np.array(
            [
                np.sum(np.sum(masked_im_window, axis=1 - i) * self.ranges[i])
                for i in range(2)
            ]
        )
        # Q2 = np.array([np.sum(
        #     self.im*self.window * self.xx**(2-i) * self.yy**i)
        #     for i in range(3)])
        self.xc = Q1[0] / Q0
        self.yc = Q1[1] / Q0

    def estimate(self):
        r"""Estimate the star image centroid iteratively."""
        for _ in range(self.n_iter):
            self.UpdateGrid()
            self.window = self.EllipticalGaussian()
            # Calculate weighted moments.
            self.ComputeMoments()
        return self.xc, self.yc

    def get_centroids(self):
        r"""Get centroids."""
        return np.array([self.xc, self.yc])

    def return_shifts(self):
        r"""Return intra-pixel shifts.

        Intra-pixel shifts are the difference between
        the estimated centroid and the center of the stamp (or pixel grid).
        """
        return [self.xc - self.xc0, self.yc - self.yc0]


def shift_ker_stack(shifts, upfact, lanc_rad=8):
    r"""Generate shifting kernels and rotated shifting kernels."""
    # lanc_rad = np.ceil(np.max(3*sigmas)).astype(int)
    shap = shifts.shape
    var_shift_ker_stack = np.zeros((2 * lanc_rad + 1, 2 * lanc_rad + 1, shap[0]))
    var_shift_ker_stack_adj = np.zeros((2 * lanc_rad + 1, 2 * lanc_rad + 1, shap[0]))

    for i in range(0, shap[0]):
        uin = shifts[i, :].reshape((1, 2)) * upfact
        var_shift_ker_stack[:, :, i] = lanczos(uin, n=lanc_rad)
        var_shift_ker_stack_adj[:, :, i] = np.rot90(var_shift_ker_stack[:, :, i], 2)

    return var_shift_ker_stack, var_shift_ker_stack_adj


def lanczos(U, n=10, n2=None):
    r"""Generate Lanczos kernel for a given shift."""
    if n2 is None:
        n2 = n
    siz = np.size(U)

    if siz == 2:
        U_in = np.copy(U)
        if len(U.shape) == 1:
            U_in = np.zeros((1, 2))
            U_in[0, 0] = U[0]
            U_in[0, 1] = U[1]
        H = np.zeros((2 * n + 1, 2 * n2 + 1))
        if (U_in[0, 0] == 0) and (U_in[0, 1] == 0):
            H[n, n2] = 1
        else:
            i = 0
            j = 0
            for i in range(0, 2 * n + 1):
                for j in range(0, 2 * n2 + 1):
                    H[i, j] = (
                        np.sinc(U_in[0, 0] - (i - n))
                        * np.sinc((U_in[0, 0] - (i - n)) / n)
                        * np.sinc(U_in[0, 1] - (j - n))
                        * np.sinc((U_in[0, 1] - (j - n)) / n)
                    )

    else:
        H = np.zeros((2 * n + 1,))
        for i in range(0, 2 * n):
            H[i] = np.sinc(np.pi * (U - (i - n))) * np.sinc(np.pi * (U - (i - n)) / n)
    return H


def degradation_op(X, shift_ker, D):
    r"""Shift and decimate fine-grid image."""
    return decim(scisig.fftconvolve(X, shift_ker, mode="same"), D, av_en=0)


def decim(im, d, av_en=1, fft=1):
    r"""Decimate image to lower resolution."""
    im_filt = np.copy(im)
    im_d = np.copy(im)
    if d > 1:
        if av_en == 1:
            siz = d + 1 - (d % 2)
            mask = np.ones((siz, siz)) / siz**2
            if fft == 1:
                im_filt = scisig.fftconvolve(im, mask, mode="same")
            else:
                im_filt = scisig.convolve(im, mask, mode="same")
        n1 = int(np.floor(im.shape[0] / d))
        n2 = int(np.floor(im.shape[1] / d))
        im_d = np.zeros((n1, n2))
        i, j = 0, 0
        for i in range(0, n1):
            for j in range(0, n2):
                im_d[i, j] = im[i * d, j * d]
    if av_en == 1:
        return im_filt, im_d
    else:
        return im_d
