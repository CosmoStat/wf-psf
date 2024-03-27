"""UNIT TESTS FOR PACKAGE MODULE: Sims.

This module contains unit tests for the wf_psf.sims.spatial_varying_psf module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import pytest
import numpy as np
from wf_psf.sims.spatial_varying_psf import (
    MeshHelper,
    CoordinateHelper,
    ZernikeHelper,
    PolynomialMatrixHelper,
    SpatialVaryingPSF,
)
import os
import logging


@pytest.fixture
def mock_x_lims():
    return [0, 1]


@pytest.fixture
def mock_y_lims():
    return [0, 1]


@pytest.fixture
def mock_grid_points():
    return [10, 20]


@pytest.fixture
def mock_grid_size():
    return 15


def test_build_mesh_with_grid_points(mock_x_lims, mock_y_lims, mock_grid_points):
    x_grid, y_grid = MeshHelper.build_mesh(
        mock_x_lims, mock_y_lims, grid_points=mock_grid_points
    )
    assert x_grid.shape == (mock_grid_points[1], mock_grid_points[0])
    assert y_grid.shape == (mock_grid_points[1], mock_grid_points[0])


def test_build_mesh_with_grid_size(mock_x_lims, mock_y_lims, mock_grid_size):
    x_grid, y_grid = MeshHelper.build_mesh(
        mock_x_lims, mock_y_lims, grid_size=mock_grid_size
    )
    assert x_grid.shape == (mock_grid_size, mock_grid_size)
    assert y_grid.shape == (mock_grid_size, mock_grid_size)


def test_build_mesh_with_both_params(
    mock_x_lims, mock_y_lims, mock_grid_points, mock_grid_size
):
    x_grid, y_grid = MeshHelper.build_mesh(
        mock_x_lims,
        mock_y_lims,
        grid_points=mock_grid_points,
        grid_size=mock_grid_size,
    )
    assert x_grid.shape == (mock_grid_size, mock_grid_size)
    assert y_grid.shape == (mock_grid_size, mock_grid_size)


def test_build_mesh_with_no_params(mock_x_lims, mock_y_lims):
    with pytest.raises(ValueError):
        MeshHelper.build_mesh(mock_x_lims, mock_y_lims)


def test_build_mesh_default_params():
    # Test case: Default parameters
    x_lims = [0, 1]
    y_lims = [0, 1]
    grid_points = [3, 3]  # 3x3 grid
    mesh_x, mesh_y = MeshHelper.build_mesh(x_lims, y_lims, grid_points)
    assert mesh_x.shape == (3, 3), "Mesh grid shape should be (3, 3)"
    assert mesh_y.shape == (3, 3), "Mesh grid shape should be (3, 3)"
    assert np.allclose(
        mesh_x, np.array([[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]])
    ), "Mesh x coordinates are incorrect"
    assert np.allclose(
        mesh_y, np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    ), "Mesh y coordinates are incorrect"


def test_build_mesh_custom_params():
    # Test case: Custom parameters
    x_lims = [0, 2]
    y_lims = [1, 3]
    grid_points = [2, 4]  # 2x4 grid
    mesh_x, mesh_y = MeshHelper.build_mesh(x_lims, y_lims, grid_points)
    assert mesh_x.shape == (4, 2), "Mesh grid shape should be (4, 2)"
    assert mesh_y.shape == (4, 2), "Mesh grid shape should be (4, 2)"
    assert np.allclose(
        mesh_x, np.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    ), "Mesh x coordinates are incorrect"
    assert np.allclose(
        mesh_y,
        np.array(
            [[1.0, 1.0], [1.66666667, 1.66666667], [2.33333333, 2.33333333], [3.0, 3.0]]
        ),
    ), "Mesh y coordinates are incorrect"


def test_build_mesh_grid_size_parameter():
    # Test case: Custom number of points
    x_lims = [0, 1]
    y_lims = [0, 1]
    grid_points = [3, 3]  # 3x3 grid
    grid_size = 2  # 2 points in each direction
    mesh_x, mesh_y = MeshHelper.build_mesh(
        x_lims, y_lims, grid_points, grid_size=grid_size
    )
    assert mesh_x.shape == (2, 2), "Mesh grid shape should be (2, 2)"
    assert mesh_y.shape == (2, 2), "Mesh grid shape should be (2, 2)"
    assert np.allclose(
        mesh_x, np.array([[0, 1], [0, 1]])
    ), "Mesh x coordinates are incorrect"
    assert np.allclose(
        mesh_y, np.array([[0, 0], [1, 1]])
    ), "Mesh y coordinates are incorrect"


def test_bounds_of_scaled_positions(example_limits_bounds):
    """Test Bounds of Scaled Positions.

    This unit test checks whether the elements of each array
    for x and y are within the range [-1, 1].

    """
    x, y, x_lims, y_lims = example_limits_bounds
    x_scale, y_scale = CoordinateHelper.scale_positions(x, y, x_lims, y_lims)
    assert np.logical_and(x_scale >= -1, x_scale <= 1).all()
    assert np.logical_and(y_scale >= -1, y_scale <= 1).all()


def test_correctness_of_shift(example_limits_and_grid):
    x_lims, y_lims, grid_points = example_limits_and_grid

    x_step, y_step = CoordinateHelper.calculate_shift(x_lims, y_lims, grid_points)
    assert x_step == 4
    assert y_step == 0.6


def test_add_random_shift_to_positions(example_limits_and_grid, xv_and_yv_grid):
    x_lims, y_lims, grid_points = example_limits_and_grid
    xv_grid, yv_grid = xv_and_yv_grid
    seed = 3838284

    (
        shifted_x,
        shifted_y,
    ) = CoordinateHelper.add_random_shift_to_positions(
        xv_grid, yv_grid, grid_points, x_lims, y_lims, seed
    )
    assert np.logical_and(shifted_x >= x_lims[0], shifted_x <= x_lims[1]).all()
    assert np.logical_and(shifted_y >= y_lims[0], shifted_y <= y_lims[1]).all()


def test_check_and_adjust_coordinate_limits(example_limits_bounds):
    x, y, x_lims, y_lims = example_limits_bounds
    (
        adjusted_x,
        adjusted_y,
    ) = CoordinateHelper.check_and_adjust_coordinate_limits(x, y, x_lims, y_lims)

    assert np.logical_and(adjusted_x >= x_lims[0], adjusted_x <= x_lims[1]).all()
    assert np.logical_and(adjusted_y >= y_lims[0], adjusted_y <= y_lims[1]).all()


def test_check_position_coordinate_limits_within_limits(caplog):
    # Test case: xv and yv within the limits
    xv = np.array([0.5, 1.5, 2.5])
    yv = np.array([1.0, 2.0, 3.0])
    x_lims = [0, 3]
    y_lims = [0, 4]
    with caplog.at_level(logging.DEBUG):
        CoordinateHelper.check_position_coordinate_limits(
            xv, yv, x_lims, y_lims, verbose=True
        )
    # Check if log messages are captured
    assert not any(
        record.levelname == "INFO" and "WARNING!" in record.message
        for record in caplog.records
    ), "No warning message should be logged for coordinates within limits"


def test_check_position_coordinate_limits_outside_limits(caplog):
    # Test case: xv and yv outside the limits
    xv = np.array([-0.5, 3.5, 2.5])
    yv = np.array([1.0, 4.0, 3.0])
    x_lims = [0, 3]
    y_lims = [0, 3]
    with caplog.at_level(logging.DEBUG):
        CoordinateHelper.check_position_coordinate_limits(
            xv, yv, x_lims, y_lims, verbose=True
        )

    # Check if log messages are captured
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "x value" in caplog.text


def test_check_position_coordinate_limits_no_verbose(caplog):
    # Test case: No verbose output
    xv = np.array([-0.5, 3.5, 2.5])
    yv = np.array([1.0, 4.0, 3.0])
    x_lims = [0, 3]
    y_lims = [0, 3]
    with caplog.at_level(logging.DEBUG):
        CoordinateHelper.check_position_coordinate_limits(
            xv, yv, x_lims, y_lims, verbose=False
        )
    # Check if log messages are captured
    assert not any(
        record.levelname == "INFO" and "WARNING!" in record.message
        for record in caplog.records
    ), "No warning message should be logged for coordinates within limits"


def test_bounds_polynomial_matrix_coefficients(example_limits_bounds):
    x, y, x_lims, y_lims = example_limits_bounds
    d_max = 2
    Pi = PolynomialMatrixHelper.generate_polynomial_matrix(x, y, x_lims, y_lims, d_max)
    assert np.logical_and(Pi >= -1, Pi <= 1).all()


def test_shape_initialize_Z_matrix():
    max_order = 5
    length = 10
    Z = ZernikeHelper.initialize_Z_matrix(max_order, length)

    assert np.shape(Z) == (max_order, length)


def test_empty_input_vector_initialize_Z_matrix():
    z_matrix = ZernikeHelper.initialize_Z_matrix(10, len(np.array([])))
    assert z_matrix.shape == (10, 0)


def test_zero_max_order_initialize_Z_matrix():
    z_matrix = ZernikeHelper.initialize_Z_matrix(0, 3)
    assert z_matrix.shape == (0, 3)


def test_large_max_order_initialize_Z_matrix():
    z_matrix = ZernikeHelper.initialize_Z_matrix(1000, 3)

    assert z_matrix.shape == (1000, 3)


def test_basic_functionality_initialize_Z_matrix():
    Z_matrix = ZernikeHelper.initialize_Z_matrix(2, 4, 930293)
    expected_Z = np.array(
        [
            [-1.07491132, 0.55969984, 1.4472938, 1.72761226],
            [-0.69694839, -0.14165003, -1.97485039, -0.20909905],
        ]
    )
    np.testing.assert_allclose(Z_matrix, expected_Z)


def test_basic_functionality_normalize_Z_matrix():
    Z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    normalized_Z = ZernikeHelper.normalize_Z_matrix(Z, 10)
    expected_normalized_Z = np.array(
        [[2.67261242, 5.34522484, 8.01783726], [4.55842306, 5.69802882, 6.83763459]]
    )
    np.testing.assert_allclose(normalized_Z, expected_normalized_Z)


@pytest.mark.parametrize(
    "Z, max_limit",
    [
        (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 8),
        (np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]), 10),
        (np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]), 100),
        # Add more Z array values and max_limit values as needed
    ],
)
def test_bounds_normalize_Z_matrix(Z, max_limit):
    normalized_Z = ZernikeHelper.normalize_Z_matrix(Z, max_limit)

    assert np.logical_and(normalized_Z >= -max_limit, normalized_Z <= max_limit).all()


def test_basic_functionality_initialize_normalized_zernike_matrix():
    Z = ZernikeHelper.initialize_normalized_zernike_matrix(2, 4, 2, 930293)
    expected_initialize_normalized_Z = np.array(
        [
            [-0.84013338, 0.43745238, 1.13118153, 1.35027393],
            [-0.66080324, -0.13430377, -1.87243067, -0.19825475],
        ]
    )


def test_generate_zernike_polynomials():
    # Define input parameters
    xv = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    yv = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    x_lims = [0.0, 1.0]
    y_lims = [0.0, 1.0]
    d_max = 2
    polynomial_coeffs = np.array([np.arange(6)] * 10)

    # Call the function
    result = ZernikeHelper.generate_zernike_polynomials(
        xv, yv, x_lims, y_lims, d_max, polynomial_coeffs
    )

    # Define expected output shape
    expected_shape = (10, 5)

    # Assert the shape of the result
    assert result.shape == expected_shape


def test_WFE_RMS_build_polynomial_coeffs(spatial_varying_psf):
    expected_WFE_RMS = np.array(
        [
            [1.59860658, 0.98645338, 0.37585926, 0.24511776],
            [1.72689883, 1.23068974, 0.89311116, 0.91177326],
            [2.1686159, 1.83035052, 1.65774055, 1.70195683],
            [2.77807291, 2.54555448, 2.44798666, 2.50121219],
        ]
    )

    np.testing.assert_allclose(spatial_varying_psf.WFE_RMS, expected_WFE_RMS)


def test_polynomial_coeffs_build_polynomial_coeffs(spatial_varying_psf):
    expected_polynomial_coeffs = np.array(
        [[0.85217187, 0.42910555, 1.12535278], [-1.05870064, 0.81260699, -0.43522383]]
    )
    np.testing.assert_allclose(
        spatial_varying_psf.polynomial_coeffs, expected_polynomial_coeffs
    )


def test_calculate_zernikes(spatial_varying_psf):
    xv = np.array([0.1599547, 2.0, 0.0, 2.0])
    yv = np.array([0.0, 0.0, 1.56821882, 1.8144926])
    expected_zernikes = np.array(
        [
            [-0.63364901, 0.15592463, 1.06251295, 2.19786893],
            [-1.3061035, 0.18913017, -2.11861001, -0.60058024],
        ]
    )

    zernikes = ZernikeHelper.calculate_zernike(
        xv,
        yv,
        spatial_varying_psf.x_lims,
        spatial_varying_psf.y_lims,
        spatial_varying_psf.d_max,
        spatial_varying_psf.polynomial_coeffs,
    )

    np.testing.assert_allclose(zernikes, expected_zernikes)
