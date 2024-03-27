"""FIXTURES FOR GENERATING TESTS FOR WF-PSF MODULES: CONFTEST.

This module contains fixtures to use in unit tests for wf_psf.sims package.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""
import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.sims.spatial_varying_psf import SpatialVaryingPSF, MeshHelper
import numpy as np


class PSF_Simulator:
    max_wfe_rms = 55


psf_params = RecursiveNamespace(
    grid_points=[2, 2],
    grid_size=4,
    max_order=2,
    x_lims=[0, 2],
    y_lims=[0, 2],
    psf_simulator=PSF_Simulator(),
    d_max=1,
    n_bins=2,
    lim_max_wfe_rms=2,
    verbose=False,
    seed=930293,
)


@pytest.fixture(scope="module", params=[psf_params])
def spatial_varying_psf():
    return SpatialVaryingPSF(
        psf_params.psf_simulator,
        psf_params.d_max,
        psf_params.grid_points,
        psf_params.grid_size,
        psf_params.max_order,
        psf_params.x_lims,
        psf_params.y_lims,
        psf_params.n_bins,
        psf_params.lim_max_wfe_rms,
        psf_params.verbose,
        psf_params.seed,
    )


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
    xv_grid, yv_grid = MeshHelper.build_mesh(x_lims, y_lims, grid_points)
    return xv_grid, yv_grid
