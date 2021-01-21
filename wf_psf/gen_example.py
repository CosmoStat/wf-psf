import numpy as np
import scipy as sp
import scipy.interpolate as sinterp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL

from wf_psf.SimPSFToolkit import SimPSFToolkit as simPSF

# Pre-defined colormap
top = mpl.cm.get_cmap('Oranges_r', 128)
bottom = mpl.cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')



Zcube_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/PA-zernike-cubes/Zernike45.mat'
Zcube = sio.loadmat(Zcube_path)
zernikes = []
for it in range(45):
    zernikes.append(Zcube['Zpols'][0,it][5])


# All parameters in default value
sim_PSF_toolkit = SimPSFToolkit(zernikes, max_order=45, max_wfe_rms=0.1,
                                output_dim=64, rand_seed=None, plot_opt=True, oversampling_rate=2,
                                pix_sampling=12, tel_diameter=1.2, tel_focal_length=24.5,
                                pupil_diameter=1024, euclid_obsc=True, LP_filter_length=3, verbose=0)

# Generate a random sample of coefficients
sim_PSF_toolkit.gen_random_Z_coeffs(max_order=45)
# Normalize coefficients
z_coeffs = sim_PSF_toolkit.normalize_zernikes(sim_PSF_toolkit.get_z_coeffs(), sim_PSF_toolkit.max_wfe_rms)
# Save coefficients
sim_PSF_toolkit.set_z_coeffs(z_coeffs)
# Plot Z coefficients
sim_PSF_toolkit.plot_z_coeffs()

# Generate monochromatic PSFs at different wavelengths
sim_PSF_toolkit.generate_mono_PSF(lambda_obs=0.9, regen_sample=False)
sim_PSF_toolkit.plot_psf()
sim_PSF_toolkit.generate_mono_PSF(lambda_obs=0.8, regen_sample=False)
sim_PSF_toolkit.plot_psf()
sim_PSF_toolkit.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
sim_PSF_toolkit.plot_psf()
sim_PSF_toolkit.generate_mono_PSF(lambda_obs=0.6, regen_sample=False)
sim_PSF_toolkit.plot_psf()
sim_PSF_toolkit.generate_mono_PSF(lambda_obs=0.55, regen_sample=False)
sim_PSF_toolkit.plot_psf()

### Inspect the wavefront
sim_PSF_toolkit.plot_opd_phase(newcmp)


### Load an example SED
# Load the SED
SED_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/templatesCOSMOS/SB1_A_0_UV.sed'
PA_SED_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/PA-zernike-cubes/example_TSED.txt'

my_data = np.genfromtxt(SED_path,dtype=np.dtype('float64'))
euclid_band = np.copy(my_data)
euclid_band = euclid_band[euclid_band[:,0]>=500,:]
euclid_band = euclid_band[euclid_band[:,0]<=950,:]

PA_SED = np.genfromtxt(PA_SED_path,dtype=np.dtype('float64'))
PA_wvlength = np.arange(351)+550
PA_SED_wv = np.concatenate((PA_wvlength.reshape(-1,1), PA_SED.reshape(-1,1)), axis=1)

### Polychromatic PSF with 35 bins
# Turn the plot option on
sim_PSF_toolkit.plot_opt = True
poly_psf_35 = sim_PSF_toolkit.generate_poly_PSF(PA_SED_wv, n_bins=35)
sim_PSF_toolkit.psf_plotter(poly_psf_35, lambda_obs=0.000, cmap='gist_stern')

### Polychromatic PSF with 100 bins
poly_psf_100 = sim_PSF_toolkit.generate_poly_PSF(PA_SED_wv, n_bins=100)
sim_PSF_toolkit.psf_plotter(poly_psf_100, lambda_obs=0.000, cmap='gist_stern')

## Residual between both PSFs
sim_PSF_toolkit.psf_plotter(poly_psf_100 - poly_psf_35, lambda_obs=0.000, cmap='gist_stern')
