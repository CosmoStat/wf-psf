import numpy as np
import scipy as sp
import scipy.interpolate as sinterp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL

from wf_psf import SimPSFToolkit as simPSF

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


sim_psf_toolkit = simPSF.SimPSFToolkit(zernike_maps=zernikes ,plot_opt=True)

# First example
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.725)
psf_725 = sim_psf_toolkit.get_psf()
sim_psf_toolkit.plot_psf(cmap=newcmp)
sim_psf_toolkit.plot_wf_phase(cmap=newcmp)

# Generate PSFs at different wavelengths
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.550)
sim_psf_toolkit.plot_psf()
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.650)
sim_psf_toolkit.plot_psf()
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.750)
sim_psf_toolkit.plot_psf()
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.850)
sim_psf_toolkit.plot_psf()
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.950)
sim_psf_toolkit.plot_psf()


# Generate a polychromatic PSF
SED_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/templatesCOSMOS/SB1_A_0_UV.sed'
PA_SED_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/PA-zernike-cubes/example_TSED.txt'

my_data = np.genfromtxt(SED_path,dtype=np.dtype('float64'))
euclid_band = np.copy(my_data)
euclid_band = euclid_band[euclid_band[:,0]>=500,:]
euclid_band = euclid_band[euclid_band[:,0]<=950,:]

PA_SED = np.genfromtxt(PA_SED_path,dtype=np.dtype('float64'))
PA_wvlength = np.arange(351)+550
PA_SED_wv = np.concatenate((PA_wvlength.reshape(-1,1), PA_SED.reshape(-1,1)), axis=1)


sim_psf_toolkit = simPSF.SimPSFToolkit(zernike_maps=zernikes ,plot_opt=True)

poly_psf = sim_psf_toolkit.generate_poly_PSF(PA_SED_wv, n_bins=35)
sim_psf_toolkit.psf_plotter(poly_psf, lambda_obs=0.000, cmap='gist_stern')

sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.550)
sim_psf_toolkit.plot_psf()
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.725)
sim_psf_toolkit.plot_psf()
sim_psf_toolkit.generate_mono_PSF(lambda_obs=0.900)
sim_psf_toolkit.plot_psf()
