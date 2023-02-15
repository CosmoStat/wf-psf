import numpy as np
import scipy as sp
import scipy.interpolate as sinterp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GenFieldPSF(object):
    def __init__(
        self,
        sim_psf_toolkit,
        max_order=45,
        xlim=1e3,
        ylim=1e3,
        n_bins=35,
        lim_max_wfe_rms=None,
        verbose=0,
    ):
        # Input attributes
        self.sim_psf_toolkit = sim_psf_toolkit
        self.max_order = max_order
        self.xlim = xlim
        self.ylim = ylim
        self.n_bins = n_bins
        self.verbose = verbose
        if lim_max_wfe_rms is None:
            self.lim_max_wfe_rms = sim_psf_toolkit.max_wfe_rms

        # Class attributes
        self.z_coeffs = []
        self.new_z_coef = None

    def gen_z_coeffs(self, random_gen=False):
        if len(self.z_coeffs) == 0:
            # x-variable
            if random_gen:
                rand_seed = None
            else:
                rand_seed = 1
            # Generate Z coeffs and normalize them
            # Normalize coefficients
            self.sim_psf_toolkit.gen_random_Z_coeffs(
                max_order=self.max_order, rand_seed=rand_seed
            )
            z_coef_1 = self.sim_psf_toolkit.normalize_zernikes(
                self.sim_psf_toolkit.get_z_coeffs(), self.lim_max_wfe_rms
            )
            self.z_coeffs.append(np.array(z_coef_1))

            # y-variable
            if random_gen:
                rand_seed = None
            else:
                rand_seed = 4
            self.sim_psf_toolkit.gen_random_Z_coeffs(
                max_order=self.max_order, rand_seed=rand_seed
            )
            z_coef_4 = self.sim_psf_toolkit.normalize_zernikes(
                self.sim_psf_toolkit.get_z_coeffs(), self.lim_max_wfe_rms
            )
            self.z_coeffs.append(np.array(z_coef_4))

            # global-variable
            if random_gen:
                rand_seed = None
            else:
                rand_seed = 5
            self.sim_psf_toolkit.gen_random_Z_coeffs(
                max_order=self.max_order, rand_seed=rand_seed
            )
            z_coef_5 = self.sim_psf_toolkit.normalize_zernikes(
                self.sim_psf_toolkit.get_z_coeffs(), self.lim_max_wfe_rms
            )
            self.z_coeffs.append(np.array(z_coef_5) / 5)

            # Normalize PSF field
            self.normalize_field()

    def normalize_field(self):
        """Normalize the zernike coefficients for extreme positions.

        At this moment I only check the extreme positions.
        This should check the WFE_RMS throughout all the field.

        """
        max_wfe_corner_1 = self.compute_wfe_rms(x=0, y=0)
        max_wfe_corner_2 = self.compute_wfe_rms(x=self.xlim, y=self.ylim)
        max_wfe_corner_3 = self.compute_wfe_rms(x=0, y=self.ylim)
        max_wfe_corner_4 = self.compute_wfe_rms(x=self.xlim, y=0)

        max_wfe_rms = np.max(
            [max_wfe_corner_1, max_wfe_corner_2, max_wfe_corner_3, max_wfe_corner_4]
        )

        if max_wfe_rms > self.lim_max_wfe_rms:
            if self.verbose:
                print(
                    "WFE_RMS %.4f [um] is exceeding from max value: %.4f [um]. Normalizing."
                    % (max_wfe_rms, self.lim_max_wfe_rms)
                )

            wfe_rms_ratio = self.lim_max_wfe_rms / max_wfe_rms
            for it in range(len(self.z_coeffs)):
                self.z_coeffs[it] = self.z_coeffs[it] * wfe_rms_ratio

    def erase_z_coeffs(self):
        self.z_coeffs = []

    def zernike_coeff_map(self, x, y):
        """Calculate Zernikes for a specific position

        Normalize (x,y) inputs
        (x,y) need to be in [0,self.xlim] x [0,self.ylim]
        (x_norm,y_norm) need to be in [-1, +1] x [-1, +1]
        """
        if x >= self.xlim and x <= 0:
            print(
                "WARNING! x value: %f is not between the limits [0, %f]"
                % (x, self.xlim)
            )
        if y >= self.ylim and y <= 0:
            print(
                "WARNING! y value: %f is not between the limits [0, %f]"
                % (y, self.ylim)
            )

        x_norm = (x - self.xlim / 2) / (self.xlim / 2)
        y_norm = (y - self.ylim / 2) / (self.ylim / 2)

        self.new_z_coef = (
            self.z_coeffs[0] * x_norm + self.z_coeffs[1] * y_norm + self.z_coeffs[2]
        )

        dif_wfe_rms = self.sim_psf_toolkit.check_wfe_rms(z_coeffs=self.new_z_coef)

        if dif_wfe_rms < 0:
            print(
                "WARNING: Position (%d,%d) posses an WFE_RMS of %f.\n"
                % (x, y, self.sim_psf_toolkit.max_wfe_rms - dif_wfe_rms)
            )
            print(
                "It exceeds the maximum allowed error (max WFE_RMS=%.4f [um])"
                % (self.sim_psf_toolkit.max_wfe_rms)
            )

        if self.verbose > 0:
            print("Info for position: (%d, %d)" % (x, y))
            print(
                "WFE_RMS: %.4f [um]" % (self.sim_psf_toolkit.max_wfe_rms - dif_wfe_rms)
            )
            print("MAX_WFE_RMS: %.4f [um]" % (self.sim_psf_toolkit.max_wfe_rms))

    def compute_wfe_rms(self, x, y):
        """Compute the WFE RMS for a specific field position."""
        x_norm = (x - self.xlim / 2) / (self.xlim / 2)
        y_norm = (y - self.ylim / 2) / (self.ylim / 2)

        self.new_z_coef = (
            self.z_coeffs[0] * x_norm + self.z_coeffs[1] * y_norm + self.z_coeffs[2]
        )

        dif_wfe_rms = self.sim_psf_toolkit.check_wfe_rms(z_coeffs=self.new_z_coef)

        return self.sim_psf_toolkit.max_wfe_rms - dif_wfe_rms

    def get_zernike_coeff_map(self):
        if self.new_z_coef is not None:
            return self.new_z_coef
        else:
            print("Coeff map has not been calculated yet.")

    def get_mono_PSF(self, x, y, lambda_obs=0.725):
        # Calculate the specific field's zernike coeffs
        self.zernike_coeff_map(x, y)
        # Set the Z coefficients to the PSF toolkit generator
        self.sim_psf_toolkit.set_z_coeffs(self.get_zernike_coeff_map())
        # Generate the monochromatic psf
        self.sim_psf_toolkit.generate_mono_PSF(
            lambda_obs=lambda_obs, regen_sample=False
        )
        # Return the generated PSF

        return self.sim_psf_toolkit.get_psf()

    def inspect_opd_map(self, cmap="viridis", save_img=False):
        """Plot the last saved OPD map."""
        self.sim_psf_toolkit.plot_opd_phase(cmap, save_img)

    def inspect_field_wfe_rms(self, mesh_bins=20, save_img=False):
        """Plot a chart of WFE_RMS as a function of position throughout the field."""
        x_coord = np.linspace(0, self.xlim, mesh_bins)
        y_coord = np.linspace(0, self.ylim, mesh_bins)

        x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

        wfe_rms_field = np.array(
            [
                self.compute_wfe_rms(_x, _y)
                for _x, _y in zip(x_mesh.flatten(), y_mesh.flatten())
            ]
        )

        wfe_rms_field_mesh = wfe_rms_field.reshape((mesh_bins, mesh_bins))

        # Plot the field
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(wfe_rms_field_mesh, interpolation="None")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")
        ax1.set_title("PSF field WFE RMS [um]")
        ax1.set_xlabel("x axis")
        ax1.set_ylabel("y axis")

        if save_img:
            plt.savefig("./WFE_field_meshdim_%d.pdf" % mesh_bins, bbox_inches="tight")

        plt.show()

    def get_poly_PSF(self, x, y, SED):
        # Calculate the specific field's zernike coeffs
        self.zernike_coeff_map(x, y)
        # Set the Z coefficients to the PSF toolkit generator
        self.sim_psf_toolkit.set_z_coeffs(self.get_zernike_coeff_map())

        return self.sim_psf_toolkit.generate_poly_PSF(SED, n_bins=self.n_bins)
