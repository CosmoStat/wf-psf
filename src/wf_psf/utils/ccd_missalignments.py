"""CCD missalignments.

A module with utils to handle CCD missalignments.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np


class CCDMissalignmentCalculator(object):
    """CCD missalignment calculator.

    The `tiles_data` matrix is a datacube; each slice is a 4x3 matrix which gives the 
    4 corners of a tile: first and second column -> x/y in mm and the third -> z, in µm.
    """
    def __init__(
        self,
        tiles_path,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
    ):

        self.tiles_path = tiles_path
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.tiles_data = np.load(self.tiles_path, allow_pickle=True)[()]['tile']

        assert self.tiles_data.shape == (4, 3, 36)

        self.tiles_x_lims = None
        self.tiles_y_lims = None
        self.tiles_z_lims = None
        self.tiles_z_average = None

        self.preprocess_tile_data()


    def preprocess_tile_data(self):
        """Preprocess tile data.
        """
        self.tiles_x_lims = np.array([
            np.min(self.tiles_data[:,0,:]),
            np.max(self.tiles_data[:,0,:]),
        ])
        self.tiles_y_lims = np.array([
            np.min(self.tiles_data[:,1,:]),
            np.max(self.tiles_data[:,1,:]),
        ])
        self.tiles_z_lims = np.array([
            np.min(self.tiles_data[:,2,:]),
            np.max(self.tiles_data[:,2,:]),
        ])
        self.tiles_z_average = np.mean(self.tiles_z_lims)


    def scale_position_to_tile_reference(self, pos):
        """Scale input position into tiles coordinate system.

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position
        """

        a=1

    def get_ccd_from_position(self, pos):
        """Get CCD ID from the position.

        The ID correponds to the orden in the input `self.tiles_data`

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position
        
        """
        a=1
        pass
    
    def get_dz_from_position(self, pos):
        """Get z-axis displacement for a focal plane position.

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position
        """

        a=1

    def get_zk4_from_position(self, pos):
        """Get defocus Zernike contribution from focal plane position.

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position
        """

        a=1