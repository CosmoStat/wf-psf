"""UNIT TESTS FOR PACKAGE MODULE: Sims.

This module contains unit tests for the wf_psf.sims.spatialvaryingpsf module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import pytest
import numpy as np
from wf_psf.sims import SpatialVaryingPSF
import os


def test_correctness_of_grid_generation(example_limits_and_grid):
    x_lims, y_lims, grid_points = example_limits_and_grid
    xv_grid, yv_grid = SpatialVaryingPSF.MeshHelper.build_mesh(
        x_lims, y_lims, grid_points
    )
    assert xv_grid.shape == (10, 5)
    assert yv_grid.shape == (10, 5)
    # Add more assertions to check spacing, etc.


def test_bounds_of_scaled_positions(example_limits_bounds):
    """Test Bounds of Scaled Positions.

    This unit test checks whether the elements of each array
    for x and y are within the range [-1, 1].

    """
    x, y, x_lims, y_lims = example_limits_bounds
    x_scale, y_scale = SpatialVaryingPSF.CoordinateHelper.scale_positions(
        x, y, x_lims, y_lims
    )
    assert np.logical_and(x_scale >= -1, x_scale <= 1).all()
    assert np.logical_and(y_scale >= -1, y_scale <= 1).all()


def test_correctness_of_shift(example_limits_and_grid):
    x_lims, y_lims, grid_points = example_limits_and_grid

    x_step, y_step = SpatialVaryingPSF.CoordinateHelper.calculate_shift(
        x_lims, y_lims, grid_points
    )
    assert x_step == 4
    assert y_step == 0.6


def test_add_random_shift_to_positions(example_limits_and_grid, xv_and_yv_grid):
    x_lims, y_lims, grid_points = example_limits_and_grid
    xv_grid, yv_grid = xv_and_yv_grid
    seed = 3838284

    (
        shifted_x,
        shifted_y,
    ) = SpatialVaryingPSF.CoordinateHelper.add_random_shift_to_positions(
        xv_grid, yv_grid, grid_points, x_lims, y_lims, seed
    )
    assert np.logical_and(shifted_x >= x_lims[0], shifted_x <= x_lims[1]).all()
    assert np.logical_and(shifted_y >= y_lims[0], shifted_y <= y_lims[1]).all()


def test_check_and_adjust_coordinate_limits(example_limits_bounds):
    x, y, x_lims, y_lims = example_limits_bounds
    (
        adjusted_x,
        adjusted_y,
    ) = SpatialVaryingPSF.CoordinateHelper.check_and_adjust_coordinate_limits(
        x, y, x_lims, y_lims
    )

    assert np.logical_and(adjusted_x >= x_lims[0], adjusted_x <= x_lims[1]).all()
    assert np.logical_and(adjusted_y >= y_lims[0], adjusted_y <= y_lims[1]).all()


def test_bounds_polynomial_matrix_coefficients(example_limits_bounds):
    x, y, x_lims, y_lims = example_limits_bounds
    d_max = 2
    Pi = SpatialVaryingPSF.PolynomialMatrixHelper.generate_polynomial_matrix(
        x, y, x_lims, y_lims, d_max
    )
    assert np.logical_and(Pi >= -1, Pi <= 1).all()
