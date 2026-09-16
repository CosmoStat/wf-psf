"""Noise estimation utilities for image data.

:Authors: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np


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

    def estimate_noise_batch(
        self, images: np.ndarray, masks: np.ndarray = None
    ) -> np.ndarray:
        """
        Estimate the noise standard deviation for a batch of images.

        Parameters
        ----------
        images : np.ndarray
            A batch of images with shape ``(batch_size, height, width)``.
        masks : np.ndarray, optional
            A batch of boolean masks with the same shape as `images`, specifying
            which pixels to include in the noise estimation for each image. If
            None, only the exclusion window is used for every image.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(batch_size,)`` containing the estimated noise
            standard deviation for each image.
        """
        if masks is None:
            return np.array([self.estimate_noise(_im) for _im in images])

        return np.array(
            [
                self.estimate_noise(_im, _mask)
                for _im, _mask in zip(images, masks)
            ]
        )
