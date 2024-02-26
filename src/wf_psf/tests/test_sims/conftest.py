"""FIXTURES FOR GENERATING TESTS FOR WF-PSF MODULES: CONFTEST.

This module contains fixtures to use in unit tests for 
various wf_psf packages.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.sims.SpatialVaryingPSF import SpatialVaryingPSF
from wf_psf.sims import SpatialVaryingPSF
import numpy as np


class PSF_Simulator:
    max_wfe_rms = 55


psf_params = RecursiveNamespace(
    grid_points=[4, 4],
    max_order=45,
    x_lims=[0, 1e3],
    y_lims=[0, 1e3],
    psf_simulator=PSF_Simulator(),
    d_max=2,
    n_bins=35,
    lim_max_wfe_rms=55,
    auto_init=True,
    verbose=False,
    seed=832848,
)


@pytest.fixture(scope="module", params=[psf_params])
def spatial_varying_psf():
    return SpatialVaryingPSF(psf_params.psf_simulator, psf_params.seed)


@pytest.fixture
def example_limits_and_grid():
    x_lims = [-5, 15]
    y_lims = [4, 10]
    grid_points = [5, 10]
    return x_lims, y_lims, grid_points


@pytest.fixture
def example_limits_bounds():
    x_lims = [0, 10]
    y_lims = [0, 10]
    x = np.random.rand(5) * max(x_lims)
    y = np.random.rand(5) * max(y_lims)

    return x, y, x_lims, y_lims


@pytest.fixture
def xv_and_yv_grid(example_limits_and_grid):
    x_lims, y_lims, grid_points = example_limits_and_grid
    xv_grid, yv_grid = SpatialVaryingPSF.MeshHelper.build_mesh(
        x_lims, y_lims, grid_points
    )
    return xv_grid, yv_grid
