"""Preprocessing.

A module with utils to preprocess data.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np


def shift_x_y_to_zk1_2_wavediff(dxy, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 1(2) for a given shifts in x(y) in WaveDifff conventions.

    All inputs should be in [m].
    A displacement of, for example, 0.5 pixels should be scaled with the corresponding pixel scale,
    e.g. 12[um], to get a displacement in [m], which would be `dxy=0.5*12e-6`.

    The output zernike coefficient is in [um] units as expected by wavediff.

    To apply match the centroid with a `dx` that has a corresponding `zk1`,
    the new PSF should be generated with `-zk1`.

    The same applies to `dy` and `zk2`.

    Parameters
    ----------
    dxy : float
        Centroid shift in [m]. It can be on the x-axis or the y-axis.
    tel_focal_length : float
        Telescope focal length in [m].
    tel_diameter : float
        Telescope aperture diameter in [m].
    """
    reference_pix_sampling = 12e-6
    zernike_norm_factor = 2.

    # return zernike_norm_factor * (dx/reference_pix_sampling) / (tel_focal_length * tel_diameter / 2)
    return zernike_norm_factor * (tel_diameter/2) * np.sin(np.arctan((dxy/reference_pix_sampling)/tel_focal_length)) * 3.


def defocus_to_zk4_zemax(dz, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 4 value for a given defocus in zemax conventions.

    All inputs should be in [m].

    Parameters
    ----------
    dz : float
        Shift in the z-axis, perpendicular to the focal plane. Units in [m].
    tel_focal_length : float
        Telescope focal length in [m].
    tel_diameter : float
        Telescope aperture diameter in [m].
    """
    # Base calculation 
    zk4 = dz / (8. * (tel_focal_length/tel_diameter)**2)
    # Apply Z4 normalisation 
    # This step depends on the normalisation of the Zernike basis used 
    zk4 /= np.sqrt(3)
    # Convert to waves with a reference of 800nm
    zk4 /= 800e-9
    # Remove the peak to valley value
    zk4 /= 2.

    return  zk4


def defocus_to_zk4_wavediff(dz, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 4 value for a given defocus in WaveDifff conventions.

    All inputs should be in [m].

    The output zernike coefficient is in [um] units as expected by wavediff.

    Parameters
    ----------
    dz : float
        Shift in the z-axis, perpendicular to the focal plane. Units in [m].
    tel_focal_length : float
        Telescope focal length in [m].
    tel_diameter : float
        Telescope aperture diameter in [m].
    """
    # Base calculation 
    zk4 = dz / (8. * (tel_focal_length/tel_diameter)**2)
    # Apply Z4 normalisation 
    # This step depends on the normalisation of the Zernike basis used 
    zk4 /= np.sqrt(3)

    # Remove the peak to valley value
    zk4 /= 2.

    # Change units to [um] as Wavediff uses
    zk4 *= 1e6

    return  zk4


