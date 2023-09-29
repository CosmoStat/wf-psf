import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GenPolyFieldPSF(object):
    """Generate PSF field with polynomial variations of Zernike coefficients."""

    def __init__(
        self,
        sim_psf_toolkit,
        d_max=2,
        grid_points=[4, 4],
        max_order=45,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        n_bins=35,
        lim_max_wfe_rms=None,
        auto_init=True,
        verbose=False,
    ):
        # Input attributes
        self.sim_psf_toolkit = sim_psf_toolkit
        self.max_order = max_order
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.grid_points = grid_points
        self.n_bins = n_bins
        self.verbose = verbose
        if lim_max_wfe_rms is None:
            self.lim_max_wfe_rms = sim_psf_toolkit.max_wfe_rms
        else:
            self.lim_max_wfe_rms = lim_max_wfe_rms

        self.auto_init = auto_init

        # Class attributes
        self.C_poly = None
        self.WFE_RMS = None

        # Build coefficient polynomial matric
        if self.auto_init:
            self.build_poly_coefficients()

    def scale_positions(self, xv_flat, yv_flat):
        # Scale positions to the square [-1,1] x [-1,1]
        scaled_x = (xv_flat - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0])
        scaled_x = (scaled_x - 0.5) * 2
        scaled_y = (yv_flat - self.y_lims[0]) / (self.y_lims[1] - self.y_lims[0])
        scaled_y = (scaled_y - 0.5) * 2

        return scaled_x, scaled_y

    def poly_mat_gen(self, xv_flat, yv_flat):
        """Generate polynomial matrix of positions.

        Parameters
        ----------
        xv_flat: np.ndarray (dim,)
            x positions.
        yv_flat: np.ndarray (dim,)
            y positions.
        """
        n_mono = (self.d_max + 1) * (self.d_max + 2) // 2
        if np.isscalar(xv_flat):
            Pi = np.zeros((n_mono, 1))
        else:
            Pi = np.zeros((n_mono, xv_flat.shape[0]))

        # Scale positions to the square [-1,1] x [-1,1]
        scaled_x, scaled_y = self.scale_positions(xv_flat, yv_flat)

        for d in range(self.d_max + 1):
            row_idx = d * (d + 1) // 2
            for p in range(d + 1):
                Pi[row_idx + p, :] = scaled_x ** (d - p) * scaled_y**p

        return Pi

    def zernike_poly_gen(self, xv_flat, yv_flat):
        """Generate zernike values from positions.

        Parameters
        ----------
        xv_flat: np.ndarray (dim,)
            x positions.
        yv_flat: np.ndarray (dim,)
            y positions.
        """
        # Generate the polynomial matrix
        Pi_samples = self.poly_mat_gen(xv_flat, yv_flat)

        return self.C_poly @ Pi_samples

    def set_C_poly(self, C_poly):
        """Set the polynomial coefficients.

        Parameters
        ----------
        C_poly: np.ndarray
            Polynomial coefficients.
        """
        self.C_poly = C_poly

    def set_WFE_RMS(self, WFE_RMS):
        """Set the WFE RMS map.

        Parameters
        ----------
        WFE_RMS: np.ndarray
            WFE_RMS map of the C_poly values.
        """
        self.WFE_RMS = WFE_RMS

    def build_poly_coefficients(self):
        """Build a polynomial coefficient matrix."""
        ## Choose the anchor points on a regular grid
        x = np.linspace(
            self.x_lims[0], self.x_lims[1], num=self.grid_points[0], endpoint=True
        )
        y = np.linspace(
            self.y_lims[0], self.y_lims[1], num=self.grid_points[1], endpoint=True
        )
        # Build mesh
        xv_grid, yv_grid = np.meshgrid(x, y)

        ## Random position shift
        # It is done as a random shift defined in a
        # square centred in each grid position so that there is
        # no overlaping between the squares.
        xv_rand = np.random.rand(self.grid_points[0], self.grid_points[1])
        yv_rand = np.random.rand(self.grid_points[0], self.grid_points[1])
        # Calculate the shift length
        x_step = (self.x_lims[1] - self.x_lims[0]) / self.grid_points[0]
        y_step = (self.y_lims[1] - self.y_lims[0]) / self.grid_points[1]
        # Center and scale shifts
        xv_rand = (xv_rand - 0.5) * x_step
        yv_rand = (yv_rand - 0.5) * y_step
        # Add the shift to the grid values
        xv = xv_grid + xv_rand
        yv = yv_grid + yv_rand
        # Flatten
        xv_flat = xv.flatten()
        yv_flat = yv.flatten()
        # Check the limits
        xv_flat[xv_flat > self.x_lims[1]] = self.x_lims[1]
        xv_flat[xv_flat < self.x_lims[0]] = self.x_lims[0]
        yv_flat[yv_flat > self.y_lims[1]] = self.y_lims[1]
        yv_flat[yv_flat < self.y_lims[0]] = self.y_lims[0]

        ##  Select a random vector of size `max_order` for each position
        # When concatenated into the Z matrix we have:
        Z = np.random.randn(self.max_order, len(xv_flat))
        # Normalize so that each position has the lim_max_wfe_rms
        norm_weights = np.sqrt(np.sum(Z**2, axis=1))
        Z /= norm_weights.reshape((-1, 1)) / self.lim_max_wfe_rms

        ## Generate position polynomial matrix
        Pi = self.poly_mat_gen(xv_flat, yv_flat)

        ## Estimate by least-squares the C matrix
        self.C_poly = Z @ np.linalg.pinv(Pi)
        # Re-estimate the Z matrix with the estimated C
        Z_hat = self.C_poly @ Pi

        ## Sampling the space
        # Choose the anchor points on a regular grid
        x = np.linspace(self.x_lims[0], self.x_lims[1], num=100, endpoint=True)
        y = np.linspace(self.y_lims[0], self.y_lims[1], num=100, endpoint=True)
        xv_grid, yv_grid = np.meshgrid(x, y)
        # Recalculate the Zernike coefficients with the new C_poly matrix
        Z_est = self.zernike_poly_gen(xv_grid.flatten(), yv_grid.flatten())

        ## We need to renormalize and check that the WFE RMS has a max value near the expected one
        # Calculate the WFE_RMS on the new grid
        calc_wfe = np.sqrt(np.sum(Z_est**2, axis=0))
        # Due to the polynomnial behaviour we will set the mean WFE_RMS over the field of view to be 80% of
        # the maximum allowed WFE_RMS per position.
        scale_factor = (0.8 * self.lim_max_wfe_rms) / np.mean(calc_wfe)
        self.C_poly *= scale_factor

        # Recalculate the Z coefficients
        scaled_Z_est = self.zernike_poly_gen(xv_grid.flatten(), yv_grid.flatten())
        # Calculate and save the WFE_RMS map of the C_poly values.
        self.WFE_RMS = np.sqrt(np.sum(scaled_Z_est**2, axis=0)).reshape(xv_grid.shape)

    def show_WFE_RMS(self, save_img=False, save_name="WFE_field_meshdim"):
        """Plot the WFE RMS map."""
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

    def calc_zernike(self, xv_flat, yv_flat):
        """Calculate Zernikes for a specific position.

        Normalize (x,y) inputs
        (x,y) need to be in [self.x_lims[0],self.x_lims[1]] x [self.y_lims[0],self.y_lims[1]]
        (x_norm,y_norm) need to be in [-1, +1] x [-1, +1]
        """
        # Check limits
        x_check = np.sum(xv_flat >= self.x_lims[1] * 1.1) + np.sum(
            xv_flat <= self.x_lims[0] * 1.1
        )
        y_check = np.sum(yv_flat >= self.y_lims[1] * 1.1) + np.sum(
            yv_flat <= self.y_lims[0] * 1.1
        )

        if self.verbose and x_check > 0:
            print(
                "WARNING! x value is outside the limits [%f, %f]"
                % (self.x_lims[0], self.x_lims[1])
            )
            print(xv_flat)
            print(x_check)
        if self.verbose and y_check > 0:
            print(
                "WARNING! y value is outside the limits [%f, %f]"
                % (self.y_lims[0], self.y_lims[1])
            )
            print(yv_flat)
            print(y_check)

        # Return Zernikes
        # The position scaling is done inside zernike_poly_gen
        return self.zernike_poly_gen(xv_flat, yv_flat)

    def get_mono_PSF(self, xv_flat, yv_flat, lambda_obs=0.725):
        """Calculate monochromatic PSF at a specific position and wavelength."""
        # Calculate the specific field's zernike coeffs
        zernikes = self.calc_zernike(xv_flat, yv_flat)
        # Set the Z coefficients to the PSF toolkit generator
        self.sim_psf_toolkit.set_z_coeffs(zernikes)
        # Generate the monochromatic psf
        self.sim_psf_toolkit.generate_mono_PSF(
            lambda_obs=lambda_obs, regen_sample=False
        )
        # Return the generated PSF
        return self.sim_psf_toolkit.get_psf()

    def get_poly_PSF(self, xv_flat, yv_flat, SED):
        """Calculate polychromatic PSF for a specific position and SED."""
        # Calculate the specific field's zernike coeffs
        zernikes = self.calc_zernike(xv_flat, yv_flat)
        # Set the Z coefficients to the PSF toolkit generator
        self.sim_psf_toolkit.set_z_coeffs(zernikes)
        poly_psf = self.sim_psf_toolkit.generate_poly_PSF(SED, n_bins=self.n_bins)
        opd = self.sim_psf_toolkit.opd

        return poly_psf, zernikes, opd
