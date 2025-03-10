import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging

logger = logging.getLogger(__name__)


class MeshHelper:
    """Mesh Helper.

    A utility class for generating mesh grids.

    """

    @staticmethod
    def build_mesh(x_lims, y_lims, grid_points=None, grid_size=None, endpoint=True):
        """Build Mesh.

        A method to build a mesh.

        Parameters
        ----------
        x_lims: list
            A list representing the lower and upper limits along the x-axis.
        y_lims: list
            A list representing the lower and upper limits along the y-axis.
        grid_points: list or None, optional
            List defining the size of each axis grid for constructing the mesh grid.
            If provided and `grid_size` is also provided, `grid_size` will
            override this parameter. (default is None)
        grid_size: int or None, optional
            Number of points to generate for each axis of the grid. If None and `grid_points`
            is not provided, the default grid size is used. (default is None)
        endpoint: bool, optional
            If True, `stop` is the last sample. Otherwise, it is not included. (default is True).

        Returns
        -------
        tuple
            A tuple containing two 2-dimensional arrays for x- and y-coordinate axes.

        """
        if grid_size is None:
            if grid_points is None:
                raise ValueError(
                    "At least one of 'grid_points' or 'grid_size' must be provided."
                )
            num_x, num_y = grid_points
        else:
            num_x = grid_size
            num_y = grid_size

        # Choose the anchor points on a regular grid
        x = np.linspace(x_lims[0], x_lims[1], num=num_x, endpoint=endpoint)
        y = np.linspace(y_lims[0], y_lims[1], num=num_y, endpoint=endpoint)

        # Build mesh
        return np.meshgrid(x, y)


class CoordinateHelper:
    """Coordinate Helper.

    A utility class for handling coordinate operations.

    """

    @staticmethod
    def scale_positions(x, y, x_lims, y_lims):
        """Scale Positions.

        A method to scale x- and y- positions.

        Parameters
        ----------
        x: numpy.ndarray
            A 1-dimensional numpy ndarray denoting a vector of x positions.
        y: numpy.ndarray
            A 1-dimensional numpy ndarray denoting a vector of y positions.
        x_lims: list
            A list representing the lower and upper limits along the x-axis.
        y_lims: list
            A list representing the lower and upper limits along the y-axis.

        Returns
        -------
        scaled_x: numpy.ndarray
            Scaled x positions.

        scaled_y: numpy.ndarray
            Scaled y positions.


        """
        # Scale positions to the square [-1,1] x [-1,1]
        scaled_x = (x - x_lims[0]) / (x_lims[1] - x_lims[0])
        scaled_x = (scaled_x - 0.5) * 2
        scaled_y = (y - y_lims[0]) / (y_lims[1] - y_lims[0])
        scaled_y = (scaled_y - 0.5) * 2

        return scaled_x, scaled_y

    @staticmethod
    def calculate_shift(x_lims, y_lims, grid_points):
        """Calculate Shift.

        A method to calcuate the step size for shifting positions
        based on the specified coordinate limits and grid points.

        Parameters
        ----------
         x_lims: list
            A list representing the lower and upper limits along the x-axis.
        y_lims: list
            A list representing the lower and upper limits along the y-axis.
        grid_points: list
            List defining the size of each axis grid.

        Returns
        -------
        xstep: int
        ystep: int
        """
        x_step = (x_lims[1] - x_lims[0]) / grid_points[0]
        y_step = (y_lims[1] - y_lims[0]) / grid_points[1]

        return x_step, y_step

    @staticmethod
    def add_random_shift_to_positions(
        xv_grid, yv_grid, grid_points, x_lims, y_lims, seed=None
    ):
        """Add Random Shift to Positions

        Add random shifts to positions within each grid cell,
        ensuring no overlap between neighboring cells.

        Parameters
        ----------
        xv_grid: Numpy array
            Grid of x-coordinates.
        yv_grid: Numpy array
            Grid of y-coordinates.
        grid_points : list
            A list defining the size of each axis grid
        x_lims: list
            Lower and upper limits along the x-axis.
        y_lims: list
            Lower and upper limits along the y-axis.
        seed: int
            Seed for random number generation.

        Returns
        -------
        xv_s: numpy.ndarray
            Positions with added random shifts along the x-axis.
        yv_s: numpy.ndarray
            Positions with added random shifts along the y-axis.


        """
        ## Random position shift
        # It is done as a random shift defined in a
        # square centred in each grid position so that there is
        # no overlapping between the squares.
        np.random.seed(seed)
        xv_rand = np.random.rand(grid_points[0], grid_points[1])
        yv_rand = np.random.rand(grid_points[0], grid_points[1])
        # Calculate the shift length
        x_step, y_step = CoordinateHelper.calculate_shift(x_lims, y_lims, grid_points)
        # Center and scale shifts
        xv_rand = (xv_rand - 0.5) * x_step
        yv_rand = (yv_rand - 0.5) * y_step
        # Add the shift to the grid values
        xv = xv_grid + xv_rand.T
        yv = yv_grid + yv_rand.T

        xv_s, yv_s = CoordinateHelper.check_and_adjust_coordinate_limits(
            xv.flatten(), yv.flatten(), x_lims, y_lims
        )

        return xv_s, yv_s

    @staticmethod
    def check_and_adjust_coordinate_limits(x, y, x_lims, y_lims):
        """Check and adjust coordinate limits.

        A method to check and adjust coordinate limits to within
        the range of x_lims and y_lims, respectively.

        Parameters
        ----------
        x: numpy.ndarray
            A 1-dimensional numpy-ndarray containing positions along x-axis.
        y: numpy.ndarray
            A 1-dimensional numpy-ndarray containing positions along y-axis.
        x_lims: list
            Lower and upper limits along the x-axis.
        y_lims: list
            Lower and upper limits along the y-axis.

        Returns
        -------
        x: numpy.ndarray
            A numpy.ndarraycontaining adjusted positions along the x-axis within the specified limits.
        y: numpy.ndarray
            A numpy.ndarraycontaining adjusted positions along the x-axis within the specified limits.

        """
        x[x > x_lims[1]] = x_lims[1]
        x[x < x_lims[0]] = x_lims[0]
        y[y > y_lims[1]] = y_lims[1]
        y[y < y_lims[0]] = y_lims[0]

        return x, y

    @staticmethod
    def check_position_coordinate_limits(xv, yv, x_lims, y_lims, verbose):
        """Check Position Coordinate Limits.

        This function checks if the given position coordinates (xv, yv) are within the specified limits
        (x_lims, y_lims). It raises a warning if any coordinate is outside the limits.

        Parameters
        ----------
        xv: numpy.ndarray
            The x coordinates to be checked.
        yv: numpy.ndarray
            The y coordinates to be checked.
        x_lims: tuple
            A tuple (min, max) specifying the lower and upper limits for x coordinates.
        y_lims: tuple
            A tuple (min, max) specifying the lower and upper limits for y coordinates.
        verbose: bool
            If True, print warning messages when coordinates are outside the limits.

        Returns
        -------
        None

        """

        x_check = np.sum(xv >= x_lims[1] * 1.1) + np.sum(xv <= x_lims[0] * 1.1)
        y_check = np.sum(yv >= y_lims[1] * 1.1) + np.sum(yv <= y_lims[0] * 1.1)

        if verbose and x_check > 0:
            logger.warning(
                "WARNING! x value is outside the limits [%f, %f]"
                % (x_lims[0], x_lims[1])
            )

        if verbose and y_check > 0:
            logger.warning(
                "WARNING! y value is outside the limits [%f, %f]"
                % (y_lims[0], y_lims[1])
            )


class PolynomialMatrixHelper:
    """PolynomialMatrixHelper.

    Helper class with methods for generating polynomial matrices of positions.

    """

    @staticmethod
    def generate_polynomial_matrix(x, y, x_lims, y_lims, d_max):
        """Generate polynomial matrix of positions.

        This method constructs a polynomial matrix representing spatial variations
        in a two-dimensional field. The polynomial matrix is generated based on the
        given x and y positions, considering a maximum polynomial degree specified
        by d_max.

        Parameters
        ----------
        x: numpy.ndarray
            A 1-dimensional numpy ndarray denoting a vector of x positions.
        y: numpy.ndarray
            A 1-dimensional numpy ndarray denoting a vector of y positions.
        x_lims: list
            Lower and upper limits along the x-axis.
        y_lims: list
            Lower and upper limits along the y-axis.
        d_max: int
            The maximum polynomial degree for the spatial variation of the field.

        Returns
        -------
        Pi: numpy.ndarray
            A 2-dimensional polynomial matrix representing the spatial variations.
        """
        n_mono = (d_max + 1) * (d_max + 2) // 2
        if np.isscalar(x):
            Pi = np.zeros((n_mono, 1))
        else:
            Pi = np.zeros((n_mono, x.shape[0]))

        # Scale positions to the square [-1,1] x [-1,1]
        scaled_x, scaled_y = CoordinateHelper.scale_positions(x, y, x_lims, y_lims)

        for d in range(d_max + 1):
            row_idx = d * (d + 1) // 2
            for p in range(d + 1):
                Pi[row_idx + p, :] = scaled_x ** (d - p) * scaled_y**p

        return Pi


class ZernikeHelper:
    """ZernikeHelper.

    Helper class for generating Zernike Polynomials.


    """

    @staticmethod
    def initialize_Z_matrix(max_order, size, seed=None):
        """Initialize Zernike Matrix.

        This method initializes a Zernike matrix with a specified size determined by
        the maximum order of Zernike polynomials and the length of the position vector
        along the x-coordinate axis. The matrix is populated with random values sampled
        from a standard normal distribution.

        Parameters
        ----------
        max_order: int
            The maximum order of Zernike polynomials to be used in the simulation.
        size: int
            An integer representing the size of the position vector.
        seed: int
            Seed for random number generation.

        Returns
        -------
        numpy.ndarray
            An array of shape (max_order, size) containing randomly generated values
        from a standard normal distribution to initialize the Zernike matrix.


        """
        np.random.seed(seed)
        return np.random.randn(max_order, size)

    @staticmethod
    def normalize_Z_matrix(Z, lim_max_wfe_rms):
        """Normalize Zernike Matrix.

        This method performs normalization on the Zernike matrix. It calculates
        normalization weights as the square root of the sum of squares of the
        Zernike matrix along the second axis. Each row of the matrix is then
        divided by its corresponding normalization weight, scaled by the maximum
        allowed Wave Front Error (WFE) Root-Mean-Square (RMS) error.

        Parameters
        ----------
        Z: numpy.ndarray
            A numpy ndarray representing the Zernike matrix.
        lim_max_wfe_rms: int
            The upper maximum value limit for the Wave Front Error (WFE) Root-Mean-Square (RMS) error.

        Returns
        -------
        Z: numpy.ndarray
             The normalized Zernike matrix after applying the normalization process.

        """
        norm_weights = np.sqrt(np.sum(Z**2, axis=1))
        Z /= norm_weights.reshape((-1, 1)) / lim_max_wfe_rms
        return Z

    @staticmethod
    def initialize_normalized_zernike_matrix(
        max_order, size, lim_max_wfe_rms, seed=None
    ):
        """Initialize Normalized Zernike Matrix.

        This method initializes a normalized Zernike matrix.

        Parameters
        ----------
        max_order: int
            The maximum order of Zernike polynomials to be used in the simulation.
        size: int
            An integer representing the size of the position vector.
        lim_max_wfe_rms: int
            The upper maximum value limit for the Wave Front Error (WFE) Root-Mean-Square (RMS) error.
        seed: int
            Seed for random number generation.

        Returns
        -------
        numpy.ndarray
            A normalized Zernike matrix.

        """
        return ZernikeHelper.normalize_Z_matrix(
            ZernikeHelper.initialize_Z_matrix(max_order, size, seed), lim_max_wfe_rms
        )

    @staticmethod
    def generate_zernike_polynomials(xv, yv, x_lims, y_lims, d_max, polynomial_coeffs):
        """Generate Zernike Polynomials.

        [old name: zernike_poly_gen] This method calculates Zernike polynomials based on the given x and y positions,
        considering a maximum polynomial degree specified by d_max and a set of polynomial
        coefficients.

        Parameters
        ----------
        xv: np.ndarray (dim,)
            x positions.
        yv: np.ndarray (dim,)
            y positions.
        x_lims: list
            Lower and upper limits along the x-axis.
        y_lims: list
            Lower and upper limits along the y-axis.
        d_max: int
            The maximum polynomial degree for the spatial variation of the field.s
        polynomial_coeffs: numpy.ndarray
             An array containing the polynomial coefficients.

        Returns
        -------
        numpy.ndarray
            A 1-dimensional numpy ndarray representing the spatial polynomials generated
        from the given positions and polynomial coefficients.

        """
        # Generate the polynomial matrix
        Pi_samples = PolynomialMatrixHelper.generate_polynomial_matrix(
            xv, yv, x_lims, y_lims, d_max
        )

        return polynomial_coeffs @ Pi_samples

    @staticmethod
    def calculate_zernike(
        xv, yv, x_lims, y_lims, d_max, polynomial_coeffs, verbose=False
    ):
        """Calculate Zernikes for a specific position.

        This method computes Zernike polynomials for given positions (xv, yv).
        The positions (xv, yv) should lie within the specified limits:
        [x_lims[0], x_lims[1]] along the x-axis and [y_lims[0], y_lims[1]] along the y-axis.
        Additionally, the positions should be normalized to the range [-1, +1] along both axes.

        Parameters
        ----------
        xv: numpy.ndarray
            Array containing positions along the x-axis.
        yv: numpy.ndarray
            Array containing positions along the y-axis.
        x_lims: list
            Lower and upper limits along the x-axis.
        y_lims: list
            Lower and upper limits along the y-axis.
        verbose: bool
            Flag to indicate whether to print warning messages when positions are outside the specified limits.

        Returns
        -------
        numpy.ndarray
            Array containing the computed Zernike polynomials for the given positions.

        """
        # Check limits
        CoordinateHelper.check_position_coordinate_limits(
            xv, yv, x_lims, y_lims, verbose
        )

        # Return Zernikes
        # The position scaling is done inside generate_zernike_polynomials
        return ZernikeHelper.generate_zernike_polynomials(
            xv, yv, x_lims, y_lims, d_max, polynomial_coeffs
        )


class SpatialVaryingPSF(object):
    """Spatial Varying PSF.

    Generate PSF field with polynomial variations of Zernike coefficients.

    Parameters
    ----------
    psf_simulator: PSFSimulator object
        Class instance of the PSFSimulator
    d_max: int
        Integer representing the maximum polynomial degree for the FOV spatial variation of WFE.
    grid_points: list
        List defining the size of each axis grid for constructing the (constrained random realisation) polynomial coefficient matrix.
    grid_size: int or None, optional
        Number of points to generate for the grid. If None, the value from
        grid_points attribute will be used. (default is None)
    max_order: int
        The maximum order of Zernike polynomials to be used in the simulation.
    x_lims: list
        A list representing the lower and upper limits along the x-axis.
    y_lims: list
        A list representing the lower and upper limits along the y-axis.
    n_bins: int
        An integer representing the number of equidistant bins to partition the passband to compute polychromatic PSFs.
    lim_max_wfe_rms: float
        The upper maximum value limit for the Wave Front Error (WFE) Root-Mean-Square (RMS) error.
    verbose: bool
        A flag to determine whether to print warning messages.

    """

    def __init__(
        self,
        psf_simulator,
        d_max=2,
        grid_points=[4, 4],
        grid_size=None,
        max_order=45,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        n_bins=35,
        lim_max_wfe_rms=None,
        verbose=False,
        seed=None,
    ):
        # Input attributes
        self.psf_simulator = psf_simulator
        self.max_order = max_order
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.grid_points = grid_points
        self.grid_size = grid_size
        self.n_bins = n_bins
        self.verbose = verbose
        self.seed = seed
        self._lim_max_wfe_rms = lim_max_wfe_rms

        # Class attributes
        self.polynomial_coeffs = None
        self.WFE_RMS = None

        # Build coefficient polynomial matrix
        self.build_polynomial_coeffs()

    @property
    def lim_max_wfe_rms(self):
        """Get the upper limit for Wave Front Error (WFE) Root-Mean-Square (RMS).

        If the custom upper limit `lim_max_wfe_rms` is not set, this property returns
        the maximum WFE RMS value from the PSF simulator. Otherwise, it returns the
        custom upper limit.

        Returns
        -------
        float
            The upper limit for Wave Front Error (WFE) Root-Mean-Square (RMS).
        """
        if self._lim_max_wfe_rms is None:
            return self.psf_simulator.max_wfe_rms
        else:
            return self._lim_max_wfe_rms

    @lim_max_wfe_rms.setter
    def lim_max_wfe_rms(self, value):
        """Set the upper limit for the Wave Front Error (WFE) Root-Mean-Square (RMS).

        This setter method allows you to specify a custom upper limit for the
        Wave Front Error (WFE) Root-Mean-Square (RMS). Once set, this custom limit
        will be used instead of the default limit from the PSF simulator.

        Parameters
        ----------
        value: float
            The new upper limit value to be set.
        """
        self._lim_max_wfe_rms = value

    def estimate_polynomial_coeffs(self, xv, yv, Z):
        """Estimate polynomial coefficients using least squares.

        This method estimates the polynomial coefficients using the least squares
        method based on the provided positions along the x and y axes.

        Parameters
        ----------
        xv: numpy.ndarray
            A 1-dimensional numpy ndarray representing positions along the x-axis.
        yv: numpy.ndarray
            A 1-dimensional numpy ndarray representing positions along the y-axis.

        Z: numpy.ndarray
           A something numpy ndarray representing the Zernike coefficients.

        """
        Pi = PolynomialMatrixHelper.generate_polynomial_matrix(
            xv, yv, self.x_lims, self.y_lims, self.d_max
        )

        return Z @ np.linalg.pinv(Pi)

    def calculate_wfe_rms(self, xv, yv, polynomial_coeffs):
        """Calculate the Wave Front Error (WFE) Root-Mean-Square (RMS).

        This method calculates the WFE RMS for a specific position using the provided
        x and y coordinates and polynomial coefficients.

        Parameters
        ----------
        xv: numpy.ndarray
            A 1-dimensional numpy ndarray representing positions along the x-axis.
        yv: numpy.ndarray
            A 1-dimensional numpy ndarray representing positions along the y-axis.
        polynomial_coeffs: numpy.ndarray
            A numpy ndarray containing the polynomial coefficients.

        Returns
        -------
        numpy.ndarray
            An array containing the WFE RMS values for the provided positions.
        """

        Z = ZernikeHelper.generate_zernike_polynomials(
            xv, yv, self.x_lims, self.y_lims, self.d_max, polynomial_coeffs
        )
        return np.sqrt(np.sum(Z**2, axis=0))

    def build_polynomial_coeffs(self):
        """Build polynomial coefficients for spatial variation.

        This method constructs polynomial coefficients for spatial variation by following these steps:
        1. Build a mesh based on the provided limits and grid points.
        2. Apply random position shifts to the mesh coordinates.
        3. Estimate polynomial coefficients using the shifted positions.
        4. Choose anchor points on a regular grid and calculate the Wave Front Error (WFE) Root-Mean-Square (RMS)
           on this new grid.
        5. Scale the polynomial coefficients to ensure that the mean WFE RMS over the field of view is 80% of the
           maximum allowed WFE RMS per position.
        6. Recalculate the Zernike coefficients using the scaled polynomial coefficients.
        7. Calculate and save the WFE RMS map of the polynomial coefficient values.

        Returns
        -------
        None
        """

        # Build mesh
        xv_grid, yv_grid = MeshHelper.build_mesh(
            self.x_lims, self.y_lims, self.grid_points
        )

        # Apply random position shifts
        xv, yv = CoordinateHelper.add_random_shift_to_positions(
            xv_grid, yv_grid, self.grid_points, self.x_lims, self.y_lims, self.seed
        )

        # Generate normalized Z matrix
        Z = ZernikeHelper.initialize_normalized_zernike_matrix(
            self.max_order, len(xv), self.lim_max_wfe_rms, self.seed
        )

        # Generate Polynomial coefficients for each Zernike
        self.polynomial_coeffs = self.estimate_polynomial_coeffs(xv, yv, Z)

        ## Sampling the space
        # Choose the anchor points on a regular grid
        xv_grid, yv_grid = MeshHelper.build_mesh(
            self.x_lims, self.y_lims, self.grid_points, self.grid_size, endpoint=True
        )

        ## Renormalize and check that the WFE RMS has a max value near the expected one
        # Calculate the WFE_RMS on the new grid
        xv = xv_grid.flatten()
        yv = yv_grid.flatten()

        calc_wfe = self.calculate_wfe_rms(xv, yv, self.polynomial_coeffs)

        # Due to the polynomial behaviour, set the mean WFE_RMS over the field of view to be 80% of
        # the maximum allowed WFE_RMS per position.
        scale_factor = (0.8 * self.lim_max_wfe_rms) / np.mean(calc_wfe)
        self.polynomial_coeffs *= scale_factor

        # Scale the Z coefficients
        ZernikeHelper.generate_zernike_polynomials(
            xv, yv, self.x_lims, self.y_lims, self.d_max, self.polynomial_coeffs
        )

        # Calculate and save the WFE_RMS map of the polynomial coefficients values.
        self.WFE_RMS = self.calculate_wfe_rms(xv, yv, self.polynomial_coeffs).reshape(
            xv_grid.shape
        )

    def plot_WFE_RMS(self, save_img=False, save_name="WFE_field_meshdim"):
        """Plot the Wave Front Error (WFE) Root-Mean-Square (RMS) map.

        This method generates a plot of the WFE RMS map for the Point Spread Function (PSF) field. The plot
        visualizes the distribution of WFE RMS values across the field of view.

        Parameters
        ----------
        save_img: bool, optional
            Flag indicating whether to save the plot as an image file. Default is False.
        save_name: str, optional
            Name of the image file to save. Default is 'WFE_field_meshdim'.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(self.WFE_RMS, interpolation="None")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")
        ax1.set_title("PSF field WFE RMS [um]")
        ax1.set_xlabel("x axis")
        ax1.set_ylabel("y axis")
        if save_img:
            plt.savefig("./" + save_name + ".pdf", bbox_inches="tight")
        plt.show()

    def get_monochromatic_PSF(self, xv, yv, lambda_obs=0.725):
        """Calculate the monochromatic Point Spread Function (PSF) at a specific position and wavelength.

        This method calculates the monochromatic PSF for a given position and wavelength. It utilizes the
        Zernike coefficients of the specific field to generate the PSF using the PSF toolkit generator.

        Parameters
        ----------
        xv: numpy.ndarray
            1-dimensional numpy array containing the x positions.
        yv: numpy.ndarray
            1-dimensional numpy array containing the y positions.
        lambda_obs: float, optional
            Wavelength of observation for which the PSF is calculated. Default is 0.725 micrometers.

        Returns
        -------
        numpy.ndarray
            The generated monochromatic PSF.

        Notes
        -----
        The PSF generator's Zernike coefficients are set based on the provided positions before generating the PSF.

        """
        # Calculate the specific field's zernike coeffs
        zernikes = ZernikeHelper.calculate_zernike(
            xv, yv, self.x_lims, self.y_lims, self.d_max, self.polynomial_coeffs
        )

        # Set the Z coefficients to the PSF toolkit generator
        self.psf_simulator.set_z_coeffs(zernikes)
        # Generate the monochromatic psf
        self.psf_simulator.generate_mono_PSF(lambda_obs=lambda_obs, regen_sample=False)
        # Return the generated PSF
        return self.psf_simulator.get_psf()

    def get_polychromatic_PSF(self, xv, yv, SED):
        """Calculate the polychromatic Point Spread Function (PSF) for a specific position and Spectral Energy Distribution (SED).

        This method calculates the polychromatic PSF for a given position and SED. It utilizes the Zernike coefficients
        of the specific field to generate the PSF using the PSF Simulator generator.

        Parameters
        ----------
        xv: numpy.ndarray
            1-dimensional numpy array containing the x positions.
        yv: numpy.ndarray
            1-dimensional numpy array containing the y positions.
        SED: array_like
            Spectral Energy Distribution (SED) describing the relative intensity of light at different wavelengths.

        Returns
        -------
        tuple
            A tuple containing:
            - polychromatic_psf : numpy.ndarray
                The generated polychromatic PSF.
            - zernikes : numpy.ndarray
                The Zernike coefficients corresponding to the specific field.
            - opd : numpy.ndarray
                The Optical Path Difference (OPD) corresponding to the generated PSF.

        Notes
        -----
        The PSF generator's Zernike coefficients are set based on the provided positions before generating the PSF.
        The SED is used to compute the polychromatic PSF by integrating the monochromatic PSFs over the spectral range.

        """
        # Calculate the specific field's zernike coeffs
        zernikes = ZernikeHelper.calculate_zernike(
            xv, yv, self.x_lims, self.y_lims, self.d_max, self.polynomial_coeffs
        )

        # Set the Z coefficients to the PSF Simulator generator
        self.psf_simulator.set_z_coeffs(zernikes)
        polychromatic_psf = self.psf_simulator.generate_poly_PSF(
            SED, n_bins=self.n_bins
        )
        opd = self.psf_simulator.opd

        return polychromatic_psf, zernikes, opd
