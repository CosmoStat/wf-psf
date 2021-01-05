""" This script contains functions to compute the $\rho_i$ statistics,
for i from 2 to 5, in "DES fashion", that is, using the shapes measured
from stars rather than PSF models. For the latter, just use the default
sys_tests in Stile - which also are what these function are very largely
based on.
"""


import numpy
import stile
import stile.stile_utils
from stile.sys_tests import BaseCorrelationFunctionSysTest
try:
    import treecorr
    from treecorr.corr2 import corr2_valid_params
    has_treecorr = True
except ImportError:
    has_treecorr = False
    import warnings
    warnings.warn("treecorr package cannot be imported. You may "+
                  "wish to install it if you would like to use the correlation functions within "+
                  "Stile.")

try:
    import matplotlib
    # We should decide which backend to use (this line allows running matplotlib even on sessions
    # without properly defined displays, eg through PBS)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

class DESRho2SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of PSF shapes with residual star shapes (star shapes - psf shapes).
    """
    short_name = 'rho2des'
    long_name = 'Rho2 statistics (as defined in DES shape catalogue papers)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'psf_g1', 'psf_g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'], data['g1'],
                                         data['g2'], data['w']],
                                         names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is None:
            data2 = data
        new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'], data2['g1']-data2['psf_g1'],
                                          data2['g2']-data2['psf_g2'], data2['w']],
                                          names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'], random['g1'],
                                               random['g2'], random['w']],
                                               names = ['ra', 'dec', 'g1', 'g2', 'w'])

        else:
            new_random = random
        if random2 is None:
            random2 = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                                                data2['g1']-data2['psf_g1'],
                                                data2['g2']-data2['psf_g2'], data2['w']],
                                                names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

class DESRho3SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of star shapes weighted by the residual size.
    """
    short_name = 'rho3'
    long_name = 'Rho3 statistics (Auto-correlation of star shapes weighted by the residual size)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'sigma',
                            'g1', 'g2', 'psf_sigma', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'],
                                 data['g1']*(data['sigma']-data['psf_sigma'])/data['sigma'],
                                 data['g2']*(data['sigma']-data['psf_sigma'])/data['sigma'],
                                 data['w']],
                                 names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is not None:
            new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                             data2['g1']*(data2['sigma']-data2['psf_sigma'])/data2['sigma'],
                             data2['g2']*(data2['sigma']-data2['psf_sigma'])/data2['sigma'],
                             data2['w']],
                             names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_data2 = data2
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'],
                         random['g1']*(random['sigma']-random['psf_sigma'])/random['sigma'],
                         random['g2']*(random['sigma']-random['psf_sigma'])/random['sigma'],
                         random['w']],
                         names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random = random

        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                    random2['g1']*(random2['sigma']-random2['psf_sigma'])/random2['sigma'],
                    random2['g2']*(random2['sigma']-random2['psf_sigma'])/random2['sigma'],
                    random2['w']],
                    names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

class DESRho4SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of star shapes weighted by the residual size.
    """
    short_name = 'rho4'
    long_name = 'Rho4 statistics (Correlation of residual star shapes weighted by residual size)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'sigma',
                            'psf_g1', 'psf_g2', 'psf_sigma', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'], data['g1'] - data['psf_g1'],
                                         data['g2']-data['psf_g2'], data['w']],
                                        names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is None:
            data2 = data
        new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                             data2['g1']*(data2['sigma']-data2['psf_sigma'])/data2['sigma'],
                             data2['g2']*(data2['sigma']-data2['psf_sigma'])/data2['sigma'],
                             data2['w']],
                             names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'],
                                               random['g1']-random['psf_g1'],
                                               random['g2']-random['psf_g2'], random['w']],
                                              names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random = random
        if random2 is None:
            random2 = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                    random2['g1']*(random2['sigma']-random2['psf_sigma'])/random2['sigma'],
                    random2['g2']*(random2['sigma']-random2['psf_sigma'])/random2['sigma'],
                    random2['w']],
                   names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

class DESRho5SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of star shapes weighted by the residual size.
    """
    short_name = 'rho5'
    long_name = 'Rho5 statistics (Correlation of star and PSF shapes weighted by residual size)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'sigma',
                            'g1', 'g2', 'psf_sigma', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'],data['g1'],
                                         data['g2'], data['w']],
                                        names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is None:
            data2 = data
        new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                             data2['g1']*(data2['sigma']-data2['psf_sigma'])/data2['sigma'],
                             data2['g2']*(data2['sigma']-data2['psf_sigma'])/data2['sigma'],
                             data2['w']],
                             names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'],
                                               random['g1'], random['g2'], random['w']],
                                              names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random = random
        if random2 is None:
            random2 = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                    random2['g1']*(random2['sigma']-random2['psf_sigma'])/random2['sigma'],
                    random2['g2']*(random2['sigma']-random2['psf_sigma'])/random2['sigma'],
                    random2['w']],
                   names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

