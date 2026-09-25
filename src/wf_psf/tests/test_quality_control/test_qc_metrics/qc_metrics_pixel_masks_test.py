"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Metrics Pixel Masks.

This module contains unit tests for the quality control metrics pixel masks module.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import pytest

import numpy as np
from typing import Any
from wf_psf.quality_control.context import QualityControlContext
from wf_psf.quality_control.metrics.pixel_masks import PixelMaskMetric


@pytest.fixture()
def params_factory():
    def params(aperture_params: dict[str, Any]):
        return {
            "aperture": {
                "type": aperture_params["type"],
                "centre": aperture_params["centre"],
                "radius": aperture_params["radius"],
                "unit": aperture_params["unit"],
            },
        }

    return params


class PSFDataset:
    """Simple dataset of numpy arrays."""

    def __init__(self, n_src: int = 10, n_pix: int = 32):
        self.n_src = n_src
        self.n_pix = n_pix
        self.positions = np.arange(n_src * 2).reshape(n_src, 2)
        self.masks = np.zeros((n_src, n_pix, n_pix), dtype=bool)


@pytest.fixture
def dataset_factory():
    def factory(n_src=10, n_pix=32):
        return PSFDataset(n_src=n_src, n_pix=n_pix)

    return factory


@pytest.fixture
def context():
    return QualityControlContext(dataset=PSFDataset())


@pytest.mark.parametrize(
    "centre, expected",
    [
        (
            "stamp_centre",
            np.ones((PSFDataset().n_src, 2)) * (PSFDataset().n_pix - 1) / 2,
        ),
        ("centroid", PSFDataset().positions),
    ],
)
def test_get_centres_stamp_valid(centre, expected, context, params_factory):
    aperture_params = {
        "type": "circular",
        "centre": centre,
        "radius": 5,
        "unit": "pixel",
    }

    _, Nx, Ny = np.shape(context.dataset.masks)
    context.dataset.masks[:, int(Nx / 2 - 1), int(Ny / 2 - 1)] = True
    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))
    centres = pixel_mask._get_centres(context)
    assert np.array_equal(centres, expected)


def test_get_centres_stamp_invalid(context, params_factory):
    aperture_params = {
        "type": "circular",
        "centre": "foobar",
        "radius": 5,
        "unit": "pixel",
    }

    _, Nx, Ny = np.shape(context.dataset.masks)
    context.dataset.masks[:, int(Nx / 2 - 1), int(Ny / 2 - 1)] = True
    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))

    with pytest.raises(
        ValueError, match=f"Unknown aperture centre: '{aperture_params['centre']}'."
    ):
        _ = pixel_mask._get_centres(context)


def test_get_aperture_radius_valid(params_factory):
    aperture_params = {
        "type": "circular",
        "centre": "stamp_centre",
        "radius": 5,
        "unit": "pixel",
    }

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))

    radius = pixel_mask._get_aperture_radius()
    assert radius == aperture_params["radius"]


def test_get_aperture_radius_invalid_unit(params_factory):
    aperture_params = {
        "type": "circular",
        "centre": "stamp_centre",
        "radius": 5,
        "unit": "sigma",
    }

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))

    with pytest.raises(
        ValueError, match=f"Unknown aperture unit: '{aperture_params['unit']}'."
    ):
        _ = pixel_mask._get_aperture_radius()


@pytest.mark.parametrize("radius", [(0), (np.inf)])
def test_get_aperture_radius_invalid(radius, params_factory):
    aperture_params = {
        "type": "circular",
        "centre": "stamp_centre",
        "radius": radius,
        "unit": "pixel",
    }

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))

    with pytest.raises(
        ValueError,
        match=f"Aperture radius must be finite and positive, got '{radius}'.",
    ):
        _ = pixel_mask._get_aperture_radius()


@pytest.mark.parametrize(
    "centre, radius, positions, expected_mask",
    [
        (
            "stamp_centre",
            1,
            None,
            np.array(
                [
                    [False, False, False, False, False],
                    [False, False, True, False, False],
                    [False, True, True, True, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                ]
            ),
        ),
        (
            "stamp_centre",
            np.sqrt(2),
            None,
            np.array(
                [
                    [False, False, False, False, False],
                    [False, True, True, True, False],
                    [False, True, True, True, False],
                    [False, True, True, True, False],
                    [False, False, False, False, False],
                ]
            ),
        ),
        ("stamp_centre", 10, None, np.ones((5, 5), dtype=bool)),
        (
            "centroid",
            1,
            np.array([[1, 2]]),
            np.array(
                [
                    [False, False, False, False, False],
                    [False, True, False, False, False],
                    [True, True, True, False, False],
                    [False, True, False, False, False],
                    [False, False, False, False, False],
                ]
            ),
        ),
        (
            "centroid",
            1,
            np.array([[1, 3]]),
            np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, True, False, False, False],
                    [True, True, True, False, False],
                    [False, True, False, False, False],
                ]
            ),
        ),
    ],
)
def test_get_aperture_mask_valid(
    centre, radius, positions, expected_mask, params_factory, dataset_factory
):
    aperture_params = {
        "type": "circular",
        "centre": centre,
        "radius": radius,
        "unit": "pixel",
    }
    dataset = dataset_factory(n_src=1, n_pix=5)
    context = QualityControlContext(dataset=dataset)

    if positions is not None:
        context.dataset.positions = positions

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))
    aperture = pixel_mask._get_aperture_mask(context)

    np.testing.assert_array_equal(aperture[0], expected_mask)


@pytest.mark.parametrize(
    "centre, radius, positions, masks, expected",
    [
        (
            "stamp_centre",
            1,
            None,
            np.array(
                [
                    [
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                    ]
                ]
            ),
            (1, 0.2),
        ),
        (
            "stamp_centre",
            np.sqrt(2),
            None,
            np.array(
                [
                    [
                        [True, False, False, False, True],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, False, False, False],
                        [True, False, False, False, True],
                    ]
                ]
            ),
            (0, 0),
        ),
        (
            "stamp_centre",
            2,
            None,
            np.array(
                [
                    [
                        [True, False, False, False, True],
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, False],
                        [True, False, False, False, True],
                    ]
                ]
            ),
            (1, 1 / 13),
        ),
        (
            "stamp_centre",
            10,
            None,
            np.array(
                [
                    [
                        [True, False, False, False, True],
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, False],
                        [True, False, False, False, True],
                    ]
                ]
            ),
            (5, 0.2),
        ),
        (
            "centroid",
            1,
            np.array([[1, 2]]),
            np.array(
                [
                    [
                        [True, False, False, False, True],
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, False],
                        [True, False, False, False, True],
                    ]
                ]
            ),
            (1, 0.2),
        ),
        (
            "centroid",
            2,
            np.array([[1, 3]]),
            np.array(
                [
                    [
                        [True, False, False, False, True],
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, False],
                        [True, False, False, False, True],
                    ]
                ]
            ),
            (2, 2 / 11),
        ),
    ],
)
def test_compute_aperture_masked_pixel_stats_valid(
    centre, radius, positions, masks, expected, params_factory, dataset_factory
):
    aperture_params = {
        "type": "circular",
        "centre": centre,
        "radius": radius,
        "unit": "pixel",
    }
    dataset = dataset_factory(n_src=1, n_pix=5)
    context = QualityControlContext(dataset=dataset)

    if positions is not None:
        context.dataset.positions = positions

    if masks is not None:
        context.dataset.masks = masks

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))
    aperture_stats = pixel_mask._compute_aperture_masked_pixel_stats(context)
    np.testing.assert_array_equal(aperture_stats[0], expected[0])
    np.testing.assert_allclose(aperture_stats[1], expected[1])


def test_compute_aperture_masked_pixel_stats_invalid(params_factory, dataset_factory):
    aperture_params = {
        "type": "circular",
        "centre": "stamp_centre",
        "radius": 0.7,
        "unit": "pixel",
    }
    dataset = dataset_factory(n_src=1, n_pix=32)
    context = QualityControlContext(dataset=dataset)

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))

    with pytest.raises(
        ValueError, match="Aperture contains no pixels for one or more samples."
    ):
        _ = pixel_mask._compute_aperture_masked_pixel_stats(context)


def test_compute(params_factory, dataset_factory):
    aperture_params = {
        "type": "circular",
        "centre": "stamp_centre",
        "radius": 1,
        "unit": "pixel",
    }

    dataset = dataset_factory(n_src=1, n_pix=5)
    dataset.masks = np.array(
        [
            [
                [True, False, False, False, True],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [True, False, False, False, False],
            ]
        ]
    )
    context = QualityControlContext(dataset=dataset)

    pixel_mask = PixelMaskMetric(params=params_factory(aperture_params))

    metrics = pixel_mask.compute(context)

    expected = {
        "total_masked_pixels": np.array([4]),
        "total_masked_fraction": np.array([4 / 25]),
        "aperture_masked_pixels": np.array([1]),
        "aperture_masked_fraction": np.array([1 / 5]),
    }

    assert set(metrics) == set(expected)
    np.testing.assert_array_equal(
        metrics["total_masked_pixels"], expected["total_masked_pixels"]
    )
    np.testing.assert_allclose(
        metrics["total_masked_fraction"], expected["total_masked_fraction"]
    )
    np.testing.assert_array_equal(
        metrics["aperture_masked_pixels"], expected["aperture_masked_pixels"]
    )
    np.testing.assert_allclose(
        metrics["aperture_masked_fraction"],
        expected["aperture_masked_fraction"],
    )
