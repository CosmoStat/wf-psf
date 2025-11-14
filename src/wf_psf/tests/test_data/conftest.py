"""FIXTURES FOR GENERATING TESTS FOR WF-PSF MODULES: CONFTEST.

This module contains fixtures to use in unit tests for
various wf_psf packages.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
import numpy as np
import tensorflow as tf
from types import SimpleNamespace

from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.psf_models import psf_models
from wf_psf.tests.test_data.test_data_utils import MockData

training_config = RecursiveNamespace(
    id_name="-coherent_euclid_200stars",
    data_config="data_config.yaml",
    load_data_on_init=True,
    metrics_config="metrics_config.yaml",
    model_params=RecursiveNamespace(
        model_name="poly",
        n_bins_lda=8,
        output_Q=3,
        oversampling_rate=3,
        output_dim=32,
        pupil_diameter=256,
        use_prior=True,
        correct_centroids=False,
        sigma_centroid_window=2.5,
        add_ccd_misalignments=False,
        ccd_misalignments_input_path="/Users/tl255879/Documents/research/Euclid/real_data/CCD_missalignments/tiles.npy",
        use_sample_weights=True,
        sample_weights_sigmoid=RecursiveNamespace(
            apply_sigmoid=True,
            sigmoid_max_val=5.0,
            sigmoid_power_k=1.0,
        ),
        interpolation_type=None,
        sed_interp_pts_per_bin=0,
        sed_extrapolate=True,
        sed_interp_kind="linear",
        sed_sigma=0,
        x_lims=[0.0, 1000.0],
        y_lims=[0.0, 1000.0],
        pix_sampling=12,
        tel_diameter=1.2,
        tel_focal_length=24.5,
        euclid_obsc=True,
        LP_filter_length=3,
        param_hparams=RecursiveNamespace(
            random_seed=3877572,
            l2_param=0.0,
            n_zernikes=15,
            d_max=2,
            save_optim_history_param=True,
        ),
        nonparam_hparams=RecursiveNamespace(
            d_max_nonparam=5,
            num_graph_features=10,
            l1_rate=1e-08,
            project_dd_features=False,
            reset_dd_features=False,
            save_optim_history_nonparam=True,
        ),
    ),
    training_hparams=RecursiveNamespace(
        n_epochs_params=[2, 2],
        n_epochs_non_params=[2, 2],
        batch_size=32,
        loss="mse",
        multi_cycle_params=RecursiveNamespace(
            total_cycles=2,
            cycle_def="complete",
            save_all_cycles=True,
            saved_cycle="cycle2",
            learning_rate_params=[1.0e-2, 1.0e-2],
            learning_rate_non_params=[1.0e-1, 1.0e-1],
            n_epochs_params=[2, 2],
            n_epochs_non_params=[2, 2],
        ),
    ),
)

data = RecursiveNamespace(
    training=RecursiveNamespace(
        data_dir="data",
        file="coherent_euclid_dataset/train_Euclid_res_200_TrainStars_id_001.npy",
    ),
    test=RecursiveNamespace(
        data_dir="data",
        file="coherent_euclid_dataset/test_Euclid_res_id_001.npy",
    ),
)


@pytest.fixture
def mock_data(scope="module"):
    """Fixture to provide mock data for testing."""
    # Mock positions and Zernike priors
    training_positions = tf.constant([[1, 2], [3, 4]])
    test_positions = tf.constant([[5, 6], [7, 8]])
    training_zernike_priors = tf.constant([[0.1, 0.2], [0.3, 0.4]])
    test_zernike_priors = tf.constant([[0.5, 0.6], [0.7, 0.8]])

    # Define dummy 5x5 image patches for stars (mock star images)
    # Define varied values for 5x5 star images
    noisy_stars = tf.constant(
        [np.arange(25).reshape(5, 5), np.arange(25, 50).reshape(5, 5)], dtype=tf.float32
    )

    noisy_masks = tf.constant([np.eye(5), np.ones((5, 5))], dtype=tf.float32)

    stars = tf.constant([np.full((5, 5), 100), np.full((5, 5), 200)], dtype=tf.float32)

    masks = tf.constant([np.zeros((5, 5)), np.tri(5)], dtype=tf.float32)

    return MockData(
        training_positions,
        test_positions,
        training_zernike_priors,
        test_zernike_priors,
        noisy_stars,
        noisy_masks,
        stars,
        masks,
    )


@pytest.fixture
def mock_data_inference():
    """Flat dataset for inference path only."""
    return SimpleNamespace(
        dataset={
            "positions": np.array([[9, 9], [10, 10]]),
            "zernike_prior": np.array([[0.9, 0.9]]),
            # no "missing_key" → used to trigger allow_missing behavior
        }
    )


@pytest.fixture
def simple_image(scope="module"):
    """Fixture for a simple star image."""
    num_images = 1  # Change this to test with multiple images
    image = np.zeros((num_images, 5, 5))  # Create a 3D array
    image[:, 2, 2] = 1  # Place the star at the center for each image
    return image


@pytest.fixture
def identity_mask(scope="module"):
    """Creates a mask where all pixels are fully considered."""
    return np.ones((5, 5))


@pytest.fixture
def multiple_images(scope="module"):
    """Fixture for a batch of images with stars at different positions."""
    images = np.zeros((3, 5, 5))  # 3 images, each of size 5x5
    images[0, 2, 2] = 1  # Star at center of image 0
    images[1, 1, 3] = 1  # Star at (1, 3) in image 1
    images[2, 3, 1] = 1  # Star at (3, 1) in image 2
    return images


@pytest.fixture(scope="module", params=[data])
def data_params():
    return data


@pytest.fixture(scope="module", params=[training_config])
def simPSF():
    return psf_models.simPSF(training_config.model_params)


@pytest.fixture(scope="module", params=[training_config])
def model_params():
    return training_config.model_params
