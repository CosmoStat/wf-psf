"""CCD missalignments.

A module with utils to handle CCD missalignments.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import matplotlib.path as mpltPath
from scipy.spatial import KDTree
from wf_psf.utils.preprocessing import defocus_to_zk4_wavediff


class CCDMissalignmentCalculator(object):
    """CCD missalignment calculator.

    The `tiles_data` matrix is a datacube; each slice is a 4x3 matrix which gives the
    4 corners of a tile: first and second column -> x/y in mm and the third -> z, in µm.


    Parameters
    ----------
    tiles_path : str
        Path to tiles stored tiles data.
    x_lims : list or np.ndarray
        x coordinate limits in WaveDiff coordinate system (focal plane). Shape: (2,)
    y_lims : list or np.ndarray
        y coordinate limits in WaveDiff coordinate system (focal plane). Shape: (2,)
    tel_focal_length : float
        Telescope focal length in [m]
    tel_diameter : float
        Telescope aperture diameter in [m]
    """

    def __init__(
        self,
        tiles_path,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        tel_focal_length=24.5,
        tel_diameter=1.2,
    ):
        self.tiles_path = tiles_path
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.tel_focal_length = tel_focal_length
        self.tel_diameter = tel_diameter
        self.tiles_data = np.load(self.tiles_path, allow_pickle=True)[()]["tile"]

        # assert self.tiles_data.shape == (4, 3, 36)
        assert self.tiles_data.shape[1] == 3  # Corresponds to x,y,z coordinates

        self.tiles_x_lims = None
        self.tiles_y_lims = None
        self.tiles_z_lims = None
        self.tiles_z_average = None

        self.ccd_polygons = None
        self.scaled_data = None
        self.n_points_per_ccd = None
        self.kdtree = None

        self.normal_list = None
        self.d_list = None

        self.preprocess_tile_data()
        self.initialise_polygons()
        self.initialise_kdtree()
        self.precompute_CCD_planes()

    def preprocess_tile_data(self):
        """Preprocess tile data."""
        self.tiles_x_lims = np.array(
            [
                np.min(self.tiles_data[:, 0, :]),
                np.max(self.tiles_data[:, 0, :]),
            ]
        )
        self.tiles_y_lims = np.array(
            [
                np.min(self.tiles_data[:, 1, :]),
                np.max(self.tiles_data[:, 1, :]),
            ]
        )
        self.tiles_z_lims = np.array(
            [
                np.min(self.tiles_data[:, 2, :]),
                np.max(self.tiles_data[:, 2, :]),
            ]
        )
        self.tiles_z_average = np.mean(self.tiles_z_lims)

    def initialise_polygons(self):
        """Initialise polygons to look for CCD IDs"""

        # Build polygon list corresponding to each CCD
        self.ccd_polygons = []

        self.scaled_data = np.copy(self.tiles_data)

        for it in range(self.tiles_data.shape[2]):
            # Scale positions to wavediff reference
            for jj in range(self.scaled_data.shape[0]):
                self.scaled_data[jj, 0:2, it] = (
                    self.scale_position_to_wavediff_reference(
                        self.scaled_data[jj, 0:2, it]
                    )
                )
            # Build polygons point list
            curr_polygon = [
                [_x, _y]
                for _x, _y in zip(
                    self.scaled_data[:, 0, it], self.scaled_data[:, 1, it]
                )
            ]
            # Build and add polygons to list
            self.ccd_polygons.append(mpltPath.Path(curr_polygon))

    def initialise_kdtree(self):
        flattened_points = np.zeros(
            (int(self.scaled_data.shape[0] * self.scaled_data.shape[2]), 2)
        )

        self.n_points_per_ccd = self.scaled_data.shape[0]
        for it_p in range(self.scaled_data.shape[2]):
            idx_start = int(it_p * self.n_points_per_ccd)
            idx_end = int((it_p + 1) * self.n_points_per_ccd)
            flattened_points[idx_start:idx_end, :] = self.scaled_data[:, 0:2, it_p]

        self.kdtree = KDTree(flattened_points)

    def precompute_CCD_planes(self):
        self.normal_list = []
        self.d_list = []

        for it in range(self.scaled_data.shape[2]):
            x0 = self.scaled_data[0, 0, it]
            x1 = self.scaled_data[1, 0, it]
            x2 = self.scaled_data[2, 0, it]

            y0 = self.scaled_data[0, 1, it]
            y1 = self.scaled_data[1, 1, it]
            y2 = self.scaled_data[2, 1, it]

            z0 = self.scaled_data[0, 2, it]
            z1 = self.scaled_data[1, 2, it]
            z2 = self.scaled_data[2, 2, it]

            ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
            vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

            normal = np.array(
                [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
            )  # u_cross_v
            point = np.array(self.scaled_data[0, :, it])

            d = -point.dot(normal)

            self.normal_list.append(normal)
            self.d_list.append(d)

    @staticmethod
    def compute_z_from_plane_data(pos, normal, d):
        """Compute z value from plane data.

        Plane equation:
        normal . pos + d = 0

        If
        normal = (a,b,c),
        and,
        a*x + b*y + c*z + d = 0,
        then,
        z = (-a*x -b*y -d) / c

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position in wavediff coordinate system
            respecting `self.x_lims` and `self.y_lims`. Shape: (2,)
        normal : np.ndarray
            Plane normal vector. Shape: (3,)
        d : np.ndarray
            `d` value from the plane ecuation. Shape (3,)
        """

        z = (-normal[0] * pos[0] - normal[1] * pos[1] - d) * 1.0 / normal[2]

        return z

    def scale_position_to_tile_reference(self, pos):
        """Scale input position into tiles coordinate system.

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position in wavediff coordinate system
            respecting `self.x_lims` and `self.y_lims`. Shape: (2,)
        """

        self.check_position_wavediff_limits(pos)

        pos_x = pos[0]
        pos_y = pos[1]

        scaled_x = (pos_x - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0])
        scaled_x = (
            scaled_x * (self.tiles_x_lims[1] - self.tiles_x_lims[0])
            + self.tiles_x_lims[0]
        )

        scaled_y = (pos_y - self.y_lims[0]) / (self.y_lims[1] - self.y_lims[0])
        scaled_y = (
            scaled_y * (self.tiles_y_lims[1] - self.tiles_y_lims[0])
            + self.tiles_y_lims[0]
        )

        return np.array([scaled_x, scaled_y])

    def scale_position_to_wavediff_reference(self, pos):
        """Scale input position into wavediff coordinate system.

        Parameters
        ----------
        pos : np.ndarray
            Tile position in input tile coordinate system. Shape: (2,)
        """

        self.check_position_tile_limits(pos)

        pos_x = pos[0]
        pos_y = pos[1]

        scaled_x = (pos_x - self.tiles_x_lims[0]) / (
            self.tiles_x_lims[1] - self.tiles_x_lims[0]
        )
        scaled_x = scaled_x * (self.x_lims[1] - self.x_lims[0]) + self.x_lims[0]

        scaled_y = (pos_y - self.tiles_y_lims[0]) / (
            self.tiles_y_lims[1] - self.tiles_y_lims[0]
        )
        scaled_y = scaled_y * (self.y_lims[1] - self.y_lims[0]) + self.y_lims[0]

        return np.array([scaled_x, scaled_y])

    def check_position_wavediff_limits(self, pos):
        """Check if position is within wavediff limits."""

        if (pos[0] < self.x_lims[0] or pos[0] > self.x_lims[1]) or (
            pos[1] < self.y_lims[0] or pos[1] > self.y_lims[1]
        ):
            raise ValueError(
                "Input position is not within the WaveDiff focal plane limits."
            )

    def check_position_tile_limits(self, pos):
        """Check if position is within tile limits."""

        if (pos[0] < self.tiles_x_lims[0] or pos[0] > self.tiles_x_lims[1]) or (
            pos[1] < self.tiles_y_lims[0] or pos[1] > self.tiles_y_lims[1]
        ):
            raise ValueError(
                "Input position is not within the tile focal plane limits."
            )

    @staticmethod
    def check_position_format(pos):
        if type(pos) is list:
            pos = np.array(pos)

        if len(pos.shape) == 1:
            pos = pos.reshape(1, -1)

        return pos

    def get_ccd_from_position(self, pos):
        """Get CCD ID from the position.

        The ID correponds to the orden in the input `self.tiles_data`

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position respecting `self.x_lims` and `self.y_lims`. Shape: (2,)

        """
        # Check if position is inside the focal plane limits, if not raise Error
        self.check_position_wavediff_limits(pos)

        pos = self.check_position_format(pos)

        # Test for each CCD if the position is inside
        ccds_results = np.array(
            [ccd_polygon.contains_points(pos)[0] for ccd_polygon in self.ccd_polygons]
        )

        # See inside how many CCD areas it falls
        non_zero_occurrence = np.count_nonzero(ccds_results)

        if non_zero_occurrence == 1:
            # Extract value if the ccd was identified
            ccd_id = np.nonzero(ccds_results)[0][0]

        elif non_zero_occurrence == 0:
            # Handle the case where the position is in a gap
            # Look for closest point in the flattened list
            _, flat_index = self.kdtree.query(pos)
            # Get the corresponding CCD ID
            ccd_id = int(flat_index[0] // self.n_points_per_ccd)

        elif non_zero_occurrence >= 2:
            # This should not occure unless something strange is going on
            raise ValueError("Input position gives more than one CCD ID.")

        return ccd_id

    def get_dz_from_position(self, pos):
        """Get z-axis displacement for a focal plane position.

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position respecting `self.x_lims` and `self.y_lims`. Shape: (2,)

        Returns
        -------
        dz : float
            The delta in z-axis (perpendicular to the focal plane) in [m].
        """
        self.check_position_wavediff_limits(pos)

        ccd_id = self.get_ccd_from_position(pos)

        z = self.compute_z_from_plane_data(
            pos=pos,
            normal=self.normal_list[ccd_id],
            d=self.d_list[ccd_id],
        )

        # Compute the dz with respect to the mean, and change unit from [um] to [m]
        dz = (z - self.tiles_z_average) * 1e-6

        return dz

    def get_zk4_from_position(self, pos):
        """Get defocus Zernike contribution from focal plane position.

        Parameters
        ----------
        pos : np.ndarray
            Focal plane position respecting `self.x_lims` and `self.y_lims`. Shape: (2,)

        Returns
        -------
        float
            Zernike 4 value in wavediff convention corresponding to
            the delta z of the given input position `pos`.
        """

        dz = self.get_dz_from_position(pos)

        return defocus_to_zk4_wavediff(dz, self.tel_focal_length, self.tel_diameter)
