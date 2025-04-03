"""Centroids.

A module with utils to handle PSF centroids.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import scipy.signal as scisig
from wf_psf.utils.preprocessing import shift_x_y_to_zk1_2_wavediff
from typing import Optional


def compute_zernike_tip_tilt(
    star_images: np.ndarray,
    star_masks: Optional[np.ndarray] = None,
    pixel_sampling: float = 12e-6,
    reference_shifts: list[float] = [-1/3, -1/3],
    sigma_init: float = 2.5,
    n_iter: int = 20,
) -> np.ndarray:
    """
    Compute Zernike tip-tilt corrections for a batch of PSF images.

    This function estimates the centroid shifts of multiple PSFs and computes 
    the corresponding Zernike tip-tilt corrections to align them with a reference.

    Parameters
    ----------
    star_images : np.ndarray
        A batch of PSF images (3D array of shape `(num_images, height, width)`).
    star_masks : np.ndarray, optional
        A batch of masks (same shape as `star_postage_stamps`). Each mask can have:
        - `0` to ignore the pixel.
        - `1` to fully consider the pixel.
        - Values in `(0,1]` as weights for partial consideration.
        Defaults to None.
    pixel_sampling : float, optional
        The pixel size in meters. Defaults to `12e-6 m` (12 microns).
    reference_shifts : list[float], optional
        The target centroid shifts in pixels, specified as `[dy, dx]`.  
        Defaults to `[-1/3, -1/3]` (nominal Euclid conditions).
    sigma_init : float, optional
        Initial standard deviation for centroid estimation. Default is `2.5`.
    n_iter : int, optional
        Number of iterations for centroid refinement. Default is `20`.

    Returns
    -------
    np.ndarray
        An array of shape `(num_images, 2)`, where:
        - Column 0 contains `Zk1` (tip) values.
        - Column 1 contains `Zk2` (tilt) values.
    
    Notes
    -----
    - This function processes all images at once using vectorized operations.
    - The Zernike coefficients are computed in the WaveDiff convention.
    """

    # Vectorize the centroid computation
    centroid_estimator = CentroidEstimator(
                            im=star_images,
                            mask=star_masks, 
                            sigma_init=sigma_init,
                            n_iter=n_iter
                            )
    shifts = centroid_estimator.get_intra_pixel_shifts()

    # Ensure reference_shifts is a NumPy array (if it's not already)
    reference_shifts = np.array(reference_shifts)

    # Reshape to ensure it's a column vector (1, 2)
    reference_shifts = reference_shifts[None,:]
  
    # Broadcast reference_shifts to match the shape of shifts
    reference_shifts = np.broadcast_to(reference_shifts, shifts.shape)  
    
    # Compute displacements
    displacements = (reference_shifts - shifts) # 
    
    # Ensure the correct axis order for displacements (x-axis, then y-axis)
    displacements_swapped = displacements[:, [1, 0]] # Adjust axis order if necessary

    # Call shift_x_y_to_zk1_2_wavediff directly on the vector of displacements
    zk1_2_array = shift_x_y_to_zk1_2_wavediff(displacements_swapped.flatten() * pixel_sampling )  # vectorized call
    
    # Reshape the result back to the original shape of displacements
    zk1_2_array = zk1_2_array.reshape(displacements.shape)
  
    return zk1_2_array



class CentroidEstimator:
    """
    Calculate centroids and estimate intra-pixel shifts for a batch of star images.

    This class estimates the centroid of each star in a batch of images using an 
    iterative process that fits an elliptical Gaussian model to the star images. 
    The estimated centroids are returned along with the intra-pixel shifts, which 
    represent the difference between the estimated centroid and the center of the 
    image grid (or pixel grid).

    The process is vectorized, allowing multiple star images to be processed in 
    parallel, which significantly improves performance when working with large batches.

    Parameters
    ----------
    im : numpy.ndarray
        A 3D numpy array of star image stamps. The shape of the array should be 
        (n_images, height, width), where n_images is the number of stars, and 
        height and width are the dimensions of each star's image.
    
    mask : numpy.ndarray, optional
        A 3D numpy array of the same shape as `im`, representing the mask for each star image. 
        A mask value of `1` means the corresponding pixel is fully considered, 
        a value of `0` means the pixel is ignored, and values between `0` and `1` 
        act as weights for partial consideration. If not provided, no mask is applied.

    sigma_init : float, optional
        The initial guess for the standard deviation (sigma) of the elliptical Gaussian 
        that models the star. Default is 7.5.

    n_iter : int, optional
        The number of iterations for the iterative centroid estimation procedure. 
        Default is 5.

    auto_run : bool, optional
        If True, the centroid estimation procedure will be automatically run upon 
        initialization. Default is True.

    xc : float, optional
        The initial guess for the x-component of the centroid. If None, it is set 
        to the center of the image. Default is None.

    yc : float, optional
        The initial guess for the y-component of the centroid. If None, it is set 
        to the center of the image. Default is None.

    Attributes
    ----------
    xc : numpy.ndarray
        The x-components of the estimated centroids for each image. Shape is (n_images,).
    
    yc : numpy.ndarray
        The y-components of the estimated centroids for each image. Shape is (n_images,).

    Methods
    -------
    update_grid()
        Updates the grid of pixel positions based on the current centroid estimates.

    elliptical_gaussian(e1=0, e2=0)
        Computes an elliptical 2D Gaussian with the specified shear parameters.
    
    compute_moments()
        Computes the first-order moments of the star images and updates the centroid estimates.
    
    estimate()
        Runs the iterative centroid estimation procedure for all images.

    get_centroids()
        Returns the estimated centroids for all images as a 2D numpy array.

    get_intra_pixel_shifts()
        Gets the intra-pixel shifts for all images as a list of x and y displacements.

    Notes
    -----
    The iterative centroid estimation procedure fits an elliptical Gaussian to each
    star image and computes the centroid by calculating the weighted moments. The
    `estimate()` method performs the centroid calculation for a batch of images using 
    the iterative approach defined by the `n_iter` parameter. This class is designed 
    to be efficient and scalable when processing large batches of star images.
    """

    def __init__(self, im, mask=None, sigma_init=7.5, n_iter=5, auto_run=True, xc=None, yc=None):
        """Initialize class attributes."""
        self.im = im
        self.mask = mask
        if self.mask is not None:
            self.im = self.im * (1 - self.mask)
        self.stamp_size = im.shape[1:]
        self.sigma_init = sigma_init
        self.n_iter = n_iter
        self.xc0, self.yc0 = (
            float(self.stamp_size[0]) / 2,
            float(self.stamp_size[1]) / 2,
        )

        self.xc = np.full((self.im.shape[0],), self.xc0)
        self.yc = np.full((self.im.shape[0],), self.yc0)

        if auto_run:
            self.estimate()
            

    def update_grid(self):
        """Vectorized update of the grid coordinates for multiple star stamps."""
        num_images, Nx, Ny = self.im.shape  # Extract dimensions

        x_range = np.arange(Nx)
        y_range = np.arange(Ny)

        # Correct subtraction without mixing axes
        self.xx = (x_range - self.xc[:, None])
        self.yy = (y_range - self.yc[:, None])
        
        # Now, expand to the correct shape (num_images, Nx, Ny)
        # Add the extra dimension for the number of stars
        self.xx = self.xx[:, :, None]  # Shape: (num_images, Nx, 1)
        self.yy = self.yy[:, None, :]  # Shape: (num_images, 1, Ny)

        self.xx = np.broadcast_to(self.xx, (num_images, Nx, Ny))
        self.yy = np.broadcast_to(self.yy, (num_images, Nx, Ny))

    def elliptical_gaussian(self, e1=0, e2=0):
        """Compute an elliptical 2D Gaussian with arbitrary centroid."""
        # Shear the grid coordinates
        gxx = (1 - e1) * self.xx - e2 * self.yy
        gyy = (1 + e1) * self.yy - e2 * self.xx
        
        # Compute elliptical Gaussian
        return np.exp(-(gxx**2 + gyy**2) / (2 * self.sigma_init**2))

    def compute_moments(self):
        """Compute the moments for multiple PSFs at once."""
        
        if self.mask is not None:
            masked_im_window = self.im * self.window * (self.mask == 0)
        else:
            masked_im_window = self.im * self.window

        Q0 = np.sum(masked_im_window, axis=(1, 2))  # Sum over images and their pixels
        Q1 = np.array(
            [
                np.sum(np.sum(masked_im_window, axis=2 - i) * np.arange(self.stamp_size[i]), axis=1)
                for i in range(2)
            ]
        )
        self.xc = Q1[0] / Q0
        self.yc = Q1[1] / Q0

    def estimate(self):
        """Estimate centroids for all images."""
        for _ in range(self.n_iter):
            self.update_grid()
            self.window = self.elliptical_gaussian()
            # Calculate weighted moments.
            self.compute_moments()
        return self.xc, self.yc

    def get_centroids(self):
        """Return centroids for all images."""
        return np.array([self.xc, self.yc])

    def get_intra_pixel_shifts(self):
        """Get intra-pixel shifts for all images.
        
        Intra-pixel shifts are the differences between the estimated centroid and the center of the image stamp (or pixel grid). These shifts are calculated for all images in the batch.

        Returns
        -------
        np.array
            A 2D array of shape (num_of_images, 2), where each row corresponds to the x and y shifts for each image.
        """
        shifts = np.array([self.xc - self.xc0, self.yc - self.yc0]) 
    
        return shifts


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
