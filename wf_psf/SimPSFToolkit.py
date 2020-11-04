import numpy as np
import scipy as sp
import scipy.interpolate as sinterp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL

# Pre-defined colormap
top = mpl.cm.get_cmap('Oranges_r', 128)
bottom = mpl.cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')



class SimPSFToolkit(object):
    """Simulate PSFs.

    In the future the zernike maps could be created with galsim.

    Parameters
    ----------
    zernike_maps
    decim_f=16
    pad_factor=2
    max_order=45
    max_wfe: float
        Maximum allowed WFE at ``lambda_norm``. Used for
        normalization. Units in [\mu m].
        Default is ``0.1``.
    lambda_norm: float
        Corresponds to the wavelength at which the normalization
        of the WFE is taking place. Units in [\mu m].
        Default is ``0.550``.
    rand_seed=None
    plot_opt=True

    """

    def __init__(self, zernike_maps, decim_f=16, pad_factor=2, max_order=45, max_wfe=0.1,
                 lambda_norm=0.550, rand_seed=None, plot_opt=False):
        # Input attributes
        self.pad_factor = pad_factor
        self.max_order = max_order
        self.rand_seed = rand_seed
        self.plot_opt = plot_opt
        self.zernike_maps = zernike_maps
        self.decim_f = decim_f
        self.max_wfe = max_wfe
        self.lambda_norm = lambda_norm

        # Class attributes
        self.rand_coeffs = None
        self.psf = None
        self.wf = None
        self.phase = None
        self.pupil_mask = None
        self.lambda_obs = None


    @staticmethod
    def fft_diffraction_op(wf, pupil_mask, pad_factor=2, match_shapes=True):
        """ Perform a fft-based diffraction.

        Parameters
        ----------
        wf: np.ndarray
            A complex 2D array that corresponds to the wavefront function.
        pupil_mask: np.ndarray of bools
            A 2D boolean mask that corresponds to the pupil function.


        Returns
        -------
        psf: np.ndarray
            A real 2D array corresponding to the PSF.

        """
        start = (wf.shape[0]*pad_factor)//2 - wf.shape[0]//2
        stop = (wf.shape[0]*pad_factor)//2 + wf.shape[0]//2

        padded_wf = np.zeros((wf.shape[0]*pad_factor, wf.shape[1]*pad_factor), dtype=np.complex128)

        padded_wf[start:stop, start:stop][pupil_mask] = wf[pupil_mask]

        fft_wf = np.fft.fftshift(np.fft.fft2(padded_wf))
        psf = np.abs(fft_wf)**2

        if match_shapes:
            # Return the psf with its original shape without the padding factor
            x_dif = int((psf.shape[0]/pad_factor)//2)
            y_dif = int((psf.shape[1]/pad_factor)//2)

            return psf[x_dif :psf.shape[0]-x_dif, y_dif :psf.shape[1]-y_dif]
        else:
            return psf

    @staticmethod
    def decimate_im(input_im, decim_f):
        """Decimate image.

        Decimated by a factor of decim_f.
        Based on the PIL library using the default interpolator.

        """

        pil_im = PIL.Image.fromarray(input_im)
        (width, height) = (pil_im.width // decim_f, pil_im.height // decim_f)
        im_resized = pil_im.resize((width, height))

        return np.array(im_resized)

    @staticmethod
    def get_radial_idx(max_order=45):
        it=1
        radial_idxs = []

        while(len(radial_idxs)<=max_order):
            for _it in range(it):
                radial_idxs.append(it-1)

            it+=1

        return np.array(radial_idxs)

    @staticmethod
    def psf_plotter(psf, lambda_obs=0.000, cmap='gist_stern'):
        fig = plt.figure(figsize=(16,10))

        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(psf, cmap=cmap, interpolation='None')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        ax1.set_xticks([]);ax1.set_yticks([])
        ax1.set_title('PSF (lambda=%.3f [um])'%(lambda_obs))

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(np.sqrt(abs(psf)), cmap=cmap, interpolation='None')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation='vertical')
        ax2.set_title('sqrt PSF (lambda=%.3f [um])'%(lambda_obs))
        ax2.set_xticks([]);ax2.set_yticks([])

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(np.log(abs(psf)), cmap=cmap, interpolation='None')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax3, orientation='vertical')
        ax3.set_title('log PSF (lambda=%.3f [um])'%(lambda_obs))
        ax3.set_xticks([]);ax3.set_yticks([])

        plt.show()

    @staticmethod
    def wf_phase_plotter(pupil_mask, wf, phase, lambda_obs, cmap='viridis'):
        fig = plt.figure(figsize=(16,10))

        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(pupil_mask, interpolation='None')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        ax1.set_title('Pupil mask')
        ax1.set_xticks([]);ax1.set_yticks([])

        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(wf, cmap=cmap, interpolation='None')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation='vertical')
        ax2.set_title('WF map [um]')
        ax2.set_xticks([]);ax2.set_yticks([])

        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(np.angle(phase), cmap=cmap, interpolation='None')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax3, orientation='vertical')
        ax3.set_title('Phase map [rad] (lambda=%.3f [um])'%(lambda_obs))
        ax3.set_xticks([]);ax3.set_yticks([])

        plt.show()

    def get_psf(self):
        if self.psf is not None:
            return self.psf
        else:
            print('No PSF has been computed yet.')

    def plot_psf(self, cmap='gist_stern'):
        if self.psf is not None:
            self.psf_plotter(self.psf,self.lambda_obs, cmap)
        else:
            print('No PSF has been computed yet.')

    def plot_wf_phase(self, cmap='viridis'):
        if self.wf is not None:
            self.wf_phase_plotter(self.pupil_mask, self.wf, self.phase, self.lambda_obs, cmap)
        else:
            print('No WF has been computed yet.')

    def gen_random_Z_coeffs(self, max_order=45, rand_seed=None):
        """ Generate a random set of Zernike coefficients.

        The coefficients are generated following a uniform law U~[-1,1]
        divided by their radial zernike index.
        Ex: u_i / r(i) (u_i is a realization of U)

        Parameters
        ----------
        max_order: int
            Maximum order of Zernike polynomials.
        rand_seed: int
            Seed for the random initialization.

        Returns
        -------
        rand_coeffs: list of floats
            List containing the random coefficients.

        """
        if rand_seed is not None:
            np.random.seed(rand_seed)

        rad_idx = self.get_radial_idx(max_order)
        rad_idx[0] = 1

        rand_coeffs = []

        for it in range(max_order):
            rand_coeffs.append((np.random.rand()-0.5)*2./rad_idx[it])

        self.rand_coeffs = rand_coeffs

    def plot_rand_coeffs(self):
        """Plot random Zernike coefficients."""
        if self.rand_coeffs is not None:
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(111)
            im1 = ax1.bar(np.arange(len(self.rand_coeffs)), np.array(self.rand_coeffs))
            ax1.set_title('Phase map')
            ax1.set_xlabel('Zernike coeeficients')
            ax1.set_ylabel('Magnitude [rad]')
            plt.show()
        else:
            print('Random coeffs not generated.')

    def get_rand_coeffs(self):
        """Get random coefficients"""
        if self.rand_coeffs is not None:
            return self.rand_coeffs
        else:
            print('Random coeffs not generated.')


    def generate_mono_PSF(self, lambda_obs=0.725, regen_sample=False):
        """Generate monochromatic PSF."""
        self.lambda_obs = lambda_obs

        if self.rand_coeffs is None or regen_sample is True:
            self.gen_random_Z_coeffs(self.max_order, self.rand_seed)
            if self.plot_opt:
                self.plot_rand_coeffs()
            rand_coeffs = self.get_rand_coeffs()
        else:
            rand_coeffs = self.get_rand_coeffs()


        # Create the phase with the Zernike basis
        wf = 0
        for it in range(self.max_order):
            wf += self.zernike_maps[it]*rand_coeffs[it]

        # Decimate image
        wf = self.decimate_im(wf, self.decim_f)


        # Generate pupil mask
        self.pupil_mask = ~np.isnan(wf)

        # Normalize wfe map
        wfe = np.sqrt(np.mean((wf[self.pupil_mask] -np.mean(wf[self.pupil_mask]))**2))
        mult_factor = (self.max_wfe * self.lambda_norm) / wfe
        wf[self.pupil_mask] *= mult_factor

        # Save the wavefront
        self.wf = wf


        # Generate the full phase
        self.phase = np.zeros(wf.shape, dtype=np.complex128)
        self.phase[self.pupil_mask] = np.exp(2j*np.pi*wf[self.pupil_mask]/self.lambda_obs)

        # Apply the diffraction operator
        self.psf = self.fft_diffraction_op(self.phase, self.pupil_mask, self.pad_factor)

        # Normalize psf
        self.psf /= np.sum(self.psf)

    @staticmethod
    def gen_SED_interp(SED, n_bins=35):
        """Generate SED interpolator.

        Returns the interpolator and the wavelengths in [nm].
        """
        wv_max = 900
        wv_min = 550
        wvlength = np.arange(wv_min, wv_max, int((wv_max-wv_min)/n_bins))

        SED_interp = sinterp.interp1d(
            SED[:,0], SED[:,1], kind='quadratic')

        return wvlength, SED_interp

    def generate_poly_PSF(self, SED, n_bins=35, regen_sample=False):
        """Generate polychromatic PSF with a specific SED.

        The wavelength space will be the Euclid VIS instrument band:
        [550,900]nm and will be sample in ``n_bins``.

        """
        if self.rand_coeffs is None or regen_sample is True:
            self.gen_random_Z_coeffs(self.max_order, self.rand_seed)
            if self.plot_opt:
                self.plot_rand_coeffs()
            rand_coeffs = self.get_rand_coeffs()
        else:
            rand_coeffs = self.get_rand_coeffs()

        # Generate SED interpolator and wavelengtyh array
        wvlength, SED_interp = self.gen_SED_interp(SED, n_bins)

        # Interpolate and normalize SED
        SED_norm = SED_interp(wvlength)
        SED_norm /= np.sum(SED_norm)

        stacked_psf = 0

        for it in range(wvlength.shape[0]):

            wvl_um = wvlength[it]/1e3
            SED_T = SED_norm[it]

            self.generate_mono_PSF(lambda_obs=wvl_um)
            stacked_psf += self.get_psf()*SED_T

#         stacked_psf /= wvlength.shape[0]

        return stacked_psf
