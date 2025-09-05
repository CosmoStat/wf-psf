"""CCD misalignments.

A module with utilities to handle CCD missalignments.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

from typing import Union
import numpy as np
import matplotlib.path as mpltPath
from scipy.spatial import KDTree
from wf_psf.data.data_handler import get_np_obs_positions


def compute_ccd_misalignment(model_params, positions: np.ndarray) -> np.ndarray:
    """Compute CCD misalignment.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
    positions : np.ndarray
        Numpy array containing the positions of the stars in the focal plane.
        Shape: (n_stars, 2), where n_stars is the number of stars and 2 corresponds to x and y coordinates.

    Returns
    -------
    zernike_ccd_misalignment_array : np.ndarray
        Numpy array containing the Zernike contributions to model the CCD chip misalignments.
    """
    obs_positions = positions

    ccd_misalignment_calculator = CCDMisalignmentCalculator(
        tiles_path=model_params.ccd_misalignments_input_path,
        x_lims=model_params.x_lims,
        y_lims=model_params.y_lims,
        tel_focal_length=model_params.tel_focal_length,
        tel_diameter=model_params.tel_diameter,
    )
    # Compute required zernike 4 for each position
    zk4_values = np.array(
        [
            ccd_misalignment_calculator.get_zk4_from_position(single_pos)
            for single_pos in obs_positions
        ]
    ).reshape(-1, 1)

    # Zero pad array to get shape (n_stars, n_zernike=4)
    zernike_ccd_misalignment_array = np.pad(
        zk4_values, pad_width=[(0, 0), (3, 0)], mode="constant", constant_values=0
    )

    return zernike_ccd_misalignment_array


class CCDMisalignmentCalculator:
    """CCD Misalignment Calculator.

    This class processes and analyzes CCD misalignment data using tile position information.

    The `tiles_data` array is a data cube where each slice is a 4×3 matrix representing
    the four corners of a tile. The first two columns correspond to x/y coordinates (in mm),
    and the third column represents z displacement (in µm).

    Parameters
    ----------
    tiles_path : str
        Path to the stored tiles data file.
    x_lims : Union[list[float], np.ndarray] = [0, 1e3], optional
        x-coordinate limits in the WaveDiff coordinate system (focal plane). Shape: (2,).
        Defaults to [0, 1e3].
    y_lims : Union[list[float], np.ndarray] = [0, 1e3], optional
        y-coordinate limits in the WaveDiff coordinate system (focal plane). Shape: (2,).
        Defaults to [0, 1e3].
    tel_focal_length : float, optional
        Telescope focal length in meters. Defaults to 24.5.
    tel_diameter : float, optional
        Telescope aperture diameter in meters. Defaults to 1.2.

    Attributes
    ----------
    tiles_data : np.ndarray
        Loaded tile data from the specified file.
    tiles_x_lims : np.ndarray
        Minimum and maximum x-coordinate values from `tiles_data`.
    tiles_y_lims : np.ndarray
        Minimum and maximum y-coordinate values from `tiles_data`.
    tiles_z_lims : np.ndarray
        Minimum and maximum z-coordinate values from `tiles_data`.
    tiles_z_average : float
        Average z-coordinate value across all tiles.
    ccd_polygons : list[mpltPath.Path]
        List of CCD boundary polygons.
    scaled_data : np.ndarray
        Scaled tile data.
    n_points_per_ccd : int
        Number of points per CCD.
    kdtree : KDTree
        KDTree structure for spatial queries.
    normal_list : np.ndarray
        List of normal vectors for CCD planes.
    d_list : np.ndarray
        List of plane offset values for CCD planes.
    """

    def __init__(
        self,
        tiles_path: str,
        x_lims: Union[list[float], np.ndarray] = [0, 1e3],
        y_lims: Union[list[float], np.ndarray] = [0, 1e3],
        tel_focal_length: float = 24.5,
        tel_diameter: float = 1.2,
    ) -> None:
        self.tiles_path = tiles_path
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.tel_focal_length = tel_focal_length
        self.tel_diameter = tel_diameter

        self.tiles_data = np.load(self.tiles_path, allow_pickle=True)[()]["tile"]

        if self.tiles_data.shape[1] != 3:
            raise ValueError("Tile data must have three coordinate columns (x, y, z).")

        # Initialize attributes
        self.tiles_x_lims, self.tiles_y_lims, self.tiles_z_lims = (
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
        )
        self.tiles_z_average: float = 0.0

        self.ccd_polygons: list[mpltPath.Path] = []
        self.scaled_data: np.ndarray = np.array([])
        self.n_points_per_ccd: int = 0
        self.kdtree: Union[KDTree, None] = None

        self.normal_list, self.d_list = np.empty(0), np.empty(0)

        self._initialize()

    def _initialize(self) -> None:
        """Run all required initialization steps."""
        self._preprocess_tile_data()
        self._initialize_polygons()
        self._initialize_kdtree()
        self._precompute_CCD_planes()

    def _preprocess_tile_data(self) -> None:
        """Preprocess tile data by computing spatial limits and averages."""
        self.tiles_x_lims = np.array(
            [np.min(self.tiles_data[:, 0, :]), np.max(self.tiles_data[:, 0, :])]
        )
        self.tiles_y_lims = np.array(
            [np.min(self.tiles_data[:, 1, :]), np.max(self.tiles_data[:, 1, :])]
        )
        self.tiles_z_lims = np.array(
            [np.min(self.tiles_data[:, 2, :]), np.max(self.tiles_data[:, 2, :])]
        )

        self.tiles_z_average = np.mean(self.tiles_z_lims)

    def _initialize_polygons(self):
        """Initialize polygons to look for CCD IDs"""

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

    def _initialize_kdtree(self):
        flattened_points = np.zeros(
            (int(self.scaled_data.shape[0] * self.scaled_data.shape[2]), 2)
        )

        self.n_points_per_ccd = self.scaled_data.shape[0]
        for it_p in range(self.scaled_data.shape[2]):
            idx_start = int(it_p * self.n_points_per_ccd)
            idx_end = int((it_p + 1) * self.n_points_per_ccd)
            flattened_points[idx_start:idx_end, :] = self.scaled_data[:, 0:2, it_p]

        self.kdtree = KDTree(flattened_points)

    def _precompute_CCD_planes(self):
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

            ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
            vx, vy, vz = x2 - x0, y2 - y0, z2 - z0

            normal = np.array(
                [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
            )  # u_cross_v
            point = np.array(self.scaled_data[0, :, it])

            d = -point.dot(normal)

            self.normal_list.append(normal)
            self.d_list.append(d)

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
        from wf_psf.data.data_zernike_utils import defocus_to_zk4_wavediff

        dz = self.get_dz_from_position(pos)

        return defocus_to_zk4_wavediff(dz, self.tel_focal_length, self.tel_diameter)

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

    @staticmethod
    def check_position_format(pos):
        if type(pos) is list:
            pos = np.array(pos)

        if len(pos.shape) == 1:
            pos = pos.reshape(1, -1)

        return pos
