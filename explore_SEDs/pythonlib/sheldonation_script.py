# -*- coding: utf-8 -*-

""" This is the old version of the "interpolation_script.py" file in the
ShapePipe pipeline, where you used Sheldon & Rykoff's psfex module instead
of yer own PySFEx. Here for testing purposes.
"""

"""INTERPOLATION SCRIPT

This module computes the PSFs from a PSFEx model at several galaxy positions,
using Erin Sheldon & Eli Rykoff's psfex module, available on GitHub at:
https://github.com/esheldon/psfex

:Author: Morgan Schmitz

:Version: 1.1.0

:Date: 06/02/2018

"""

import numpy as np
import psfex
import scatalog as sc
import re
from astropy.io import fits
import galsim.hsm as hsm
from galsim import Image

class PSFExInterpolator(object):
    """Interpolator class.

    This class uses a PSFEx output file to compute the PSF at desired positions.

    """

    def __init__(self, dotpsf_path, galcat_path, output_path, pos_params=None, get_shapes=True):
        """Class initialiser

        Parameters
        ----------
        dotpsf_path : str
            Path to PSFEx output file.
        galcat_path : str
            Path to SExtractor-like galaxy catalog.
        output_path : str
            Path to folder where output PSFs should be written.
        pos_params : list, optional
            Desired position parameters. If provided, there should be exactly two, and they must also be present in the
            galaxy catalog. Otherwise, they are read directly from the .psf file.

        """

        self._dotpsf_path = dotpsf_path # Path to PSFEx output file
        self._galcat_path = galcat_path # Path to catalog containing galaxy positions
        self._output_path = output_path+'galaxy_psf'   # Path to output file to be written
        if pos_params: # handle provided, but empty pos_params (for use within CosmoStat's ShapePipe)
            if not len(pos_params)==2:
                raise ValueError('{} position parameters were passed on; there should be exactly two.'.format(len(pos_params)))
            self._pos_params = pos_params
        else:
            self._pos_params = None
        self.gal_pos = None
        self.interp_PSFs = None

        # get number naming convention for this particular run
        s=re.split("\-([0-9]*)\-([0-9]+)\.",self._galcat_path)
        self._img_number='-{0}-{1}'.format(s[1],s[2])

        # if required, compute and save shapes
        if get_shapes:
            self._get_psfshapes()
            self._has_shapes = True

    def _get_position_parameters(self):
        """ Read position parameters from .psf file.

        """

        dotpsf = sc.FITSCatalog(self._dotpsf_path)
        dotpsf.open()
        self._pos_params = [dotpsf.get_header()['POLNAME1'], dotpsf.get_header()['POLNAME2']]
        dotpsf.close()

    def _get_galaxy_positions(self):
        """ Extract galaxy positions from galaxy catalog.

        """
        if self._pos_params is None:
            self._get_position_parameters()

        galcat = sc.FITSCatalog(self._galcat_path, SEx_catalog=True)
        galcat.open()
        try:
            self.gal_pos = np.array([[x,y] for x,y in zip(galcat.get_data()[self._pos_params[0]],
                                     galcat.get_data()[self._pos_params[1]])])
        except KeyError as detail:
            # extract erroneous position parameter from original exception
            err_pos_param = detail.args[0][4:-15]
            pos_param_err = 'Required position parameter '+err_pos_param+\
            'was not found in galaxy catalog. Leave pos_params (or EXTRA_CODE_OPTION) blank to read them from .psf file.'
            raise KeyError(pos_param_err)
        galcat.close()

    def _interpolate(self):
        """ Run Sheldon & Rykoff's PSFEx interpolator method at desired positions.

        """
        if self.gal_pos is None:
            self._get_galaxy_positions()

        pex = psfex.PSFEx(self._dotpsf_path)
        self.interp_PSFs = np.array([pex.get_rec(x,y) for x,y in zip(self.gal_pos[:,0],
                                     self.gal_pos[:,1])])

    def _get_psfshapes(self):
        """ Compute shapes of PSF at galaxy positions using HSM.

        """
        if self.interp_PSFs is None:
            self._interpolate()
        psf_moms = [hsm.FindAdaptiveMom(Image(psf), strict=False) for psf in self.interp_PSFs]

        self.psf_shapes = np.array([[moms.observed_shape.g1, moms.observed_shape.g2,
                        moms.moments_sigma] for moms in psf_moms])
        self.hsm_flags = np.array([bool(mom.error_message) for mom in psf_moms]).astype(int)

    def write_output(self):
        """ Save computed PSFs to fits file.

        """
        if self.interp_PSFs is None:
            self._interpolate()
        output = sc.FITSCatalog(self._output_path+self._img_number+'.fits',
                                open_mode=sc.BaseCatalog.OpenMode.ReadWrite,
                                SEx_catalog=True)
        if self._has_shapes:
            data = {'VIGNET': self.interp_PSFs,
                'E1_PSF_HSM': self.psf_shapes[:,0], 'E2_PSF_HSM': self.psf_shapes[:,1],
                'SIGMA_PSF_HSM': self.psf_shapes[:,2], 'HSM_FLAG': self.hsm_flags}
        else:
            data = {'VIGNET': self.interp_PSFs}
        output.save_as_fits(data, sex_cat_path=self._galcat_path)
