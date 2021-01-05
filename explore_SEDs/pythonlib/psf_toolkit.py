import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
import copy as cp
import matplotlib.pyplot as plt
from sf_tools.image.shape import Ellipticity
from scipy.interpolate import Rbf
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from galsim import hsm, Image

# Sam's custom cmap
colors = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]  # B -> R -> Y
Samcmap = LinearSegmentedColormap.from_list('my_colormap', colors)
Samcmap.set_under('k')

def create_common_wind(ims):
    wind = np.min(np.array(ims)), np.max(np.array(ims))
    return wind

def plot_func(im, wind=False, cmap='gist_stern', norm=None, cutoff=5e-4,
                title='', ticks=False, axislabels='', tight=False):
    if cmap in ['sam','Sam']:
        cmap = Samcmap
        boundaries = np.arange(cutoff, np.max(im), 0.0001)
        norm = BoundaryNorm(boundaries, plt.cm.get_cmap(name=cmap).N)
    if len(im.shape) == 2:
        if not wind:
            plt.imshow(im, cmap=cmap, norm=norm,
                       interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im, cmap=cmap, norm=norm,
                       interpolation='Nearest', vmin=vmin, vmax=vmax)
    else:
        sqrtN = int(np.sqrt(im.shape[0]))
        if not wind:
            plt.imshow(im.reshape(sqrtN,sqrtN), cmap=cmap, norm=norm,
                       interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im.reshape(sqrtN,sqrtN), cmap=cmap, norm=norm, 
                       interpolation='Nearest', vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.colorbar()
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if axislabels:
        plt.xlabel(axislabels)
        plt.ylabel(axislabels)
    if tight:
        plt.tight_layout()
    plt.show()
    
def hsm_shapes(im):
    galim = Image(im)
    moms = hsm.FindAdaptiveMom(galim)
    return moms.observed_shape.g1, moms.observed_shape.g2, moms.moments_sigma

def paulin(gal_size, trupsf_shape, estpsf_shape):
    """Computes Paulin predicted bias values.
    
    Assumes last two inputs are length-3 arrays containing (e1,e2,R^2).
    
    Returns:
    m, c
    m the multiplicative term and c the additive one."""
    deltapsf = estpsf_shape - trupsf_shape
    m = 1. + deltapsf[2] / gal_size
    c = -(trupsf_shape[2]/gal_size*deltapsf[:2] + 
           deltapsf[2]/gal_size*trupsf_shape[:2])
    return m, c

def get_euclid_psf(nb_psf, wvlgth=600, rand_en=True, seed=None,
                   data_path='../../Data/centralfov_psfs/'): 
    psf_names = os.listdir(data_path)
    htest = fits.open(data_path+psf_names[1])
    psf_test = htest[1].data
    nb_wav = len(htest)-1
    
    wvlgth_min = htest[1].header['WLGTH0']*1000
    wvlgth_max = htest[nb_wav].header['WLGTH0']*1000
    wvl_mean = (wvlgth_min+wvlgth_max)/2
    if (wvlgth > wvlgth_max) or (wvlgth < wvlgth_min):
        print('The specfied wavelength should be taken within [',wvlgth_min,',',wvlgth_max,']. Wavelength set to ',wvl_mean)
        wvlgth = wvl_mean
    wvl_id = int((nb_wav-1)*(wvlgth - wvlgth_min)/(wvlgth_max - wvlgth_min)+1)
    nb_max = len(psf_names)-1
    if nb_psf > nb_max:
        print('Too many PSF requested. Returning the max.')
        nb_psf = nb_max
    psf_ind = range(1,nb_psf+1)
    if rand_en and nb_psf<nb_max:
        if seed is not None:
            np.random.seed(seed)
        psf_ind = np.random.choice(nb_max, nb_psf, replace=False) + 2
    print("PSF ind: ",psf_ind)

    shap = psf_test.shape
    htest.close()
    psf_cube = np.zeros((nb_psf,shap[0],shap[1]))
    field_pos_vect = np.zeros((nb_psf,2))
    nb_corrupted = 0
    for i in range(0,nb_psf):
        hdu = fits.open(data_path+psf_names[psf_ind[i]])
        if len(hdu)-1 < nb_wav:
            print("Warning: PSF {} did not have enough wavelengths entries - ignoring it".format(psf_ind[i]))
            nb_corrupted += 1
        else:
            print("Processing PSF {}; Read Wavelength: {}".format(psf_ind[i], hdu[wvl_id].header['WLGTH0']*1000))
            print("PSF aka: ", psf_names[psf_ind[i]])
            psf_cube[i-nb_corrupted,:,:] = hdu[wvl_id].data
            field_pos_vect[i-nb_corrupted,0] = hdu[wvl_id].header['XFIELD']
            field_pos_vect[i-nb_corrupted,1] = hdu[wvl_id].header['YFIELD']
        hdu.close()
    if nb_corrupted:
        return psf_cube[:-nb_corrupted], field_pos_vect[:-nb_corrupted], psf_names[2:]
    else:
        return psf_cube, field_pos_vect, psf_names
        
def get_euclid_psfs_poly(nb_psf, wvlgths=[600], rand_en=True, start_idx=0, seed=None,
                         data_path='../../Data/centralfov_psfs/'): 
    psf_names = os.listdir(data_path)
    psf_names = [name for name in psf_names if name[:3]=='PSF']
    htest = fits.open(data_path+psf_names[1])
    psf_test = htest[1].data
    nb_wav = len(htest)-1
    
    wvlgth_min = htest[1].header['WLGTH0']*1000
    wvlgth_max = htest[nb_wav].header['WLGTH0']*1000
    wvl_mean = (wvlgth_min+wvlgth_max)/2
    wvl_ids = []
    for wvlgth in wvlgths:
        if (wvlgth > wvlgth_max) or (wvlgth < wvlgth_min):
            print('The specfied wavelength should be taken within [',wvlgth_min,',',wvlgth_max,']. Wavelength set to ',wvl_mean)
            wvlgth = wvl_mean
        wvl_ids += [int((nb_wav-1)*(wvlgth - wvlgth_min)/(wvlgth_max - wvlgth_min)+1)]
    nb_max = len(psf_names)-1
    if nb_psf > nb_max:
        print('Too many PSF requested. Returning the max.')
        nb_psf = nb_max
    psf_ind = range(start_idx+1,start_idx+nb_psf+1) # +1 is for .DS_Store I think?
    if rand_en and nb_psf<nb_max:
        if seed is not None:
            np.random.seed(seed)
        psf_ind = np.random.choice(nb_max, nb_psf, replace=False) #+ 2
    print("PSF ind: ",psf_ind)

    shap = psf_test.shape
    htest.close()
    psf_cube = np.zeros((len(wvl_ids),nb_psf,shap[0],shap[1]))
    field_pos_vect = np.zeros((nb_psf,2))
    nb_corrupted = 0
    read_wvls = []
    for i in range(0,nb_psf):
        hdu = fits.open(data_path+psf_names[psf_ind[i]])
        if len(hdu)-1 < nb_wav:
            print("Warning: PSF {} did not have enough wavelengths entries - ignoring it".format(psf_ind[i]))
            nb_corrupted += 1
        else:
            print("Processing PSF {}".format(psf_ind[i]))
            print("PSF aka: ", psf_names[psf_ind[i]])
            
            for j,wvl_id in enumerate(wvl_ids):
                read_wvls += [hdu[wvl_id].header['WLGTH0']*1000]
                print("Read Wavelength: {}".format(read_wvls[-1]))
                psf_cube[j,i-nb_corrupted,:,:] = hdu[wvl_id].data
            field_pos_vect[i-nb_corrupted,0] = hdu[wvl_id].header['XFIELD']
            field_pos_vect[i-nb_corrupted,1] = hdu[wvl_id].header['YFIELD']
        hdu.close()
    if nb_corrupted:
        return psf_cube[:,:-nb_corrupted,:,:], field_pos_vect[:-nb_corrupted], read_wvls, psf_names[2:]
    else:
        return psf_cube, field_pos_vect, read_wvls, psf_names
    
def anti_aliasing(sub_im, downsamp=2):
    shap = sub_im.shape
    new_sub_im = np.ones((downsamp, downsamp))
    new_sub_im[:shap[0],:shap[1]] *= sub_im
    if shap[0] != downsamp:
        if shap[1] != downsamp:
            # handle bottom right corner
            new_sub_im[shap[0]:downsamp,:shap[1]] *= sub_im[-1,:]
            #if downsamp-shap[1]==1:
            new_sub_im[:shap[0],shap[1]:downsamp] *= sub_im[:,-1,np.newaxis]
            #else:
            #    new_sub_im[:shap[0],shap[1]:downsamp] *= sub_im[:,-1]
            new_sub_im[new_sub_im==1] = sub_im[-1,-1]
        else:
            # handle border (line)
            new_sub_im[shap[0]:downsamp,:] *= sub_im[-1,:]
    elif shap[1] != downsamp:
        # handle border (column)
        #if downsamp-shap[1]==1:
        new_sub_im[:,shap[1]:downsamp] *= sub_im[:,-1,np.newaxis]
        #else:
        #    new_sub_im[:,shap[1]:downsamp] *= sub_im[:,-1]
    return np.sum(sub_im)

def decimate(im, downsamp=2):
    lowres = np.zeros(im[::downsamp,::downsamp].shape)
    for i in range(lowres.shape[0]):
        for j in range(lowres.shape[1]):
            lowres[i,j] = anti_aliasing(im[downsamp*i:downsamp*(i+1),
                                downsamp*j:downsamp*(j+1)], downsamp)
    return lowres        
    
def preprocess_PSF(PSFs, fov, fovwind=None, mean_filter_size=None, downsamp_factor=6,
                  final_size=None, bulge_prop = 0, normalize=True, savefolder='./', filename=''):
    if fovwind is not None:
        idx = np.where((fov[:,0] < fovwind[0]) & (fov[:,1] < fovwind[1]))
        fov = fov[idx,:]
        PSFs = PSFs[idx,:]
    nb_PSF, PSF_size, _ = PSFs.shape
    # define mean filter
    if mean_filter_size is None:
        mean_filter_size = downsamp_factor#downsamp_factor+1 - downsamp_factor%2    
    if mean_filter_size:
        mean_kernel = np.full((mean_filter_size,mean_filter_size), 1./mean_filter_size**2)
        mean_filter = lambda psf: convolve(psf, mean_kernel, mode='nearest')
        
    if PSF_size%downsamp_factor:
        downsamp_size = PSF_size/downsamp_factor+1
        downsamp_PSFs = np.empty((nb_PSF, downsamp_size, downsamp_size))
    else:
        downsamp_size = PSF_size/downsamp_factor
        downsamp_PSFs = np.empty((nb_PSF, downsamp_size, downsamp_size))
    if final_size is None:
        final_size = downsamp_size
    final_PSFs = np.empty((nb_PSF, final_size**2))
    for i, psf in enumerate(PSFs):
        # apply mean filter and downsample
        if mean_filter_size:
            downsamp_PSFs[i,:,:] = mean_filter(psf)[::downsamp_factor, ::downsamp_factor]
        else:
            downsamp_PSFs[i,:,:] = psf[::downsamp_factor, ::downsamp_factor]
        # keep only central final_size pixels and flatten
        start = (downsamp_size - final_size)/2
        final_PSFs[i,:] = downsamp_PSFs[i, start:start+final_size, start:start+final_size].flatten()
    # remove bulge if requested
    if bulge_prop:
        bulge_remover = np.min(final_PSFs, axis=0)
        final_PSFs -= bulge_prop * bulge_remover
        # save bulge_remover for reconstruction
        np.save(savefolder+'bulge_remover.npy', bulge_remover)
    # normalize 
    if normalize:
        final_PSFs = (final_PSFs.T / np.sum(final_PSFs, axis=1)).T
    if filename:
        np.save(savefolder+filename, final_PSFs)
    else:
        return final_PSFs        
    
def perf_criteria(PSF):
    shap = PSF.shape
    if len(shap)==1:
        sqpsf = np.reshape(PSF, (int(np.sqrt(len(PSF))), int(np.sqrt(len(PSF)))))
    else:
        sqpsf = cp.copy(PSF)
    Ell = Ellipticity(sqpsf)
    ell = Ell.e
    centroid = Ell.centroid
    size = 0
    for i in range(sqpsf.shape[0]):
        for j in range(sqpsf.shape[0]):
            size += ((i-centroid[0])**2 + (j-centroid[1])**2)*sqpsf[i,j]
    size = np.sqrt(size/np.sum(PSF))
    return ell[0], ell[1], size
    
def return_neighbors(val, obs, obsfov, n_neighbors):
    nbs = np.empty((n_neighbors, obs.shape[1]))
    pos = np.empty((n_neighbors, len(val)))
    distances = np.linalg.norm(obsfov-val, axis=1)
    for j in range(n_neighbors):
        nbs[j,:] = obs[np.argmin(distances),:]
        pos[j,:] = obsfov[np.argmin(distances),:]
        distances[np.argmin(distances)] += 1e6
    return nbs, pos
    
def NMSE(true, rec):
    D = true.shape[0]
    return np.sum(np.linalg.norm(true-rec,axis=1)**2 / (D*np.linalg.norm(true, axis=1)**2))
    
def rca_format(cube):
    return cube.swapaxes(0,1).swapaxes(1,2)

def reg_format(rca_cube):
    return rca_cube.swapaxes(2,1).swapaxes(1,0)
    
def random_shifts(im_stack,amax=0.5,n=10): # random shifts in [-amax,amax]
    output_stack = cp.copy(im_stack)
    shap = im_stack.shape
    shifts = np.zeros((shap[2],2))
   
    for i in range(0,shap[2]):
        u = np.zeros([1,2])
        u[0,0] = amax*(np.random.rand(1)-0.5)
        u[0,1] = amax*(np.random.rand(1)-0.5)
        shifts[i,:] = u.reshape((2,))
        output_stack[:,:,i] = fftconvolve(im_stack[:,:,i],lanczos(u,n=n),mode='same') # here
    return output_stack,shifts

def given_shift(im,shift,n=10):
    lanc_kernel = lanczos(shift,n=n)
    return fftconvolve(im,lanc_kernel,mode='same')

def lanczos(U,n=10,n2=None):
    if n2 is None:
        n2 = n
    siz = np.size(U)
    H = None
    if (siz == 2):
        U_in = cp.copy(U)
        if len(U.shape)==1:
            U_in = np.zeros((1,2))
            U_in[0,0]=U[0]
            U_in[0,1]=U[1]
        H = np.zeros((2*n+1,2*n2+1))
        if (U_in[0,0] == 0) and (U_in[0,1] == 0):
            H[n,n2] = 1
        else:
            i=0
            j=0
            for i in range(0,2*n+1):
                for j in range(0,2*n2+1):
                    H[i,j] = np.sinc(U_in[0,0]-(i-n))*np.sinc((U_in[0,0]-(i-n))/n
                    )*np.sinc(U_in[0,1]-(j-n))*np.sinc((U_in[0,1]-(j-n))/n)
    else :
        H = np.zeros((2*n+1,))
        for i in range(0,2*n):
            H[i] = np.sinc(np.pi*(U-(i-n)))*np.sinc(np.pi*(U-(i-n))/n)
    return H

# RBF interpolator
def rbf_components(rep_train, pos_train, pos_test, n_neighbors=15):
    ntrain, n_components = rep_train.shape
    ntest = pos_test.shape[0]
    rep_test = np.empty((ntest, n_components))
    for i, pos in enumerate(pos_test):
        # determine neighbors
        nbs, pos_nbs = return_neighbors(pos, rep_train, pos_train, n_neighbors)
        # train RBF and interpolate for each component
        for j in range(n_components):
            rbfi = Rbf(pos_nbs[:,0], pos_nbs[:,1], nbs[:,j], function='thin_plate')
            rep_test[i,j] = rbfi(pos[0], pos[1])
    return rep_test

def e_to_chi(e):
    """ Convert from one to another definition of ellipticity;
    See eg. Bartelmann & Schneider 2001, eq (4.11)."""
    sqmod = np.sum(e**2)
    return 2.*e / (1+sqmod)

def chi_to_e(chi):
    """ And the other way around;
    See eg. Bartelmann & Schneider 2001, eq (4.11)."""
    sqmod = np.sum(chi**2)
    return chi / (1+(1-sqmod)**.5)



