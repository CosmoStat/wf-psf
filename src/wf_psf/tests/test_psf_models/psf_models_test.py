"""UNIT TESTS FOR PACKAGE MODULE: PSF MODELS.

This module contains unit tests for the wf_psf.psf_models psf_models module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>, Tobias Liaudat <tobias.liaudat@cea.fr>


"""

from wf_psf.psf_models import (
    psf_models,
    psf_model_semiparametric,
    psf_model_physical_polychromatic,
)
import tensorflow as tf
import numpy as np


def test_get_psf_model_weights_filepath():
    weights_filepath = "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint*_poly*_sample_w_bis1_2k_cycle2*"

    ans = psf_models.get_psf_model_weights_filepath(weights_filepath)
    assert (
        ans
        == "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint_callback_poly_sample_w_bis1_2k_cycle2"
    )


@psf_models.register_psfclass
class RegisterPSFFactoryClass(psf_models.PSFModelBaseFactory):
    ids = ("psf_factory",)


def test_register_psffactoryclass():
    assert psf_models.PSF_FACTORY["psf_factory"] == RegisterPSFFactoryClass
    assert (
        psf_models.PSF_FACTORY["poly"] == psf_model_semiparametric.SemiParamFieldFactory
    )
    assert (
        psf_models.PSF_FACTORY["physical_poly"]
        == psf_model_physical_polychromatic.PhysicalPolychromaticFieldFactory
    )


def test_set_psf_model():
    psf_class = psf_models.set_psf_model("psf_factory")
    assert psf_class == RegisterPSFFactoryClass

    psf_class = psf_models.set_psf_model("poly")
    assert psf_class == psf_model_semiparametric.SemiParamFieldFactory

    psf_class = psf_models.set_psf_model("physical_poly")
    assert (
        psf_class == psf_model_physical_polychromatic.PhysicalPolychromaticFieldFactory
    )


def test_generate_zernike_maps_3d():
    # Define test parameters
    n_zernikes = 5
    pupil_diam = 10

    # Call the function to generate Zernike maps
    zernike_maps = psf_models.generate_zernike_maps_3d(n_zernikes, pupil_diam)

    # Assertions to verify properties of the returned tensor
    assert isinstance(
        zernike_maps, tf.Tensor
    )  # Check if the returned value is a TensorFlow tensor
    assert (
        zernike_maps.dtype == tf.float32
    )  # Check if the data type of the tensor is float32

    # Expected shape of the tensor based on the input parameters
    expected_shape = (n_zernikes, pupil_diam, pupil_diam)
    assert zernike_maps.shape == expected_shape


def test_tf_obscurations():
    # Define standard input parameters
    pupil_diam = 128
    N_filter = 3
    rotation_angle = 90

    # Call the function to generate rotated obscurations
    rotated_obscurations = psf_models.tf_obscurations(
        pupil_diam=pupil_diam, N_filter=N_filter, rotation_angle=rotation_angle
    )

    # Generate non-rotated obscurations
    non_rotated_obscurations = psf_models.tf_obscurations(
        pupil_diam=pupil_diam, N_filter=N_filter, rotation_angle=0
    )

    # Assertions to verify properties of the returned tensor
    assert isinstance(
        rotated_obscurations, tf.Tensor
    )  # Check if the returned value is a TensorFlow tensor
    assert (
        rotated_obscurations.dtype == tf.complex64
    )  # Check if the data type of the tensor is float32

    # Expected shape of the tensor based on the input parameters
    expected_shape = (pupil_diam, pupil_diam)
    assert rotated_obscurations.shape == expected_shape

    k = int((rotation_angle // 90) % 4)
    manually_rotated_obscurations = np.rot90(non_rotated_obscurations.numpy(), k=k)

    # Check if the rotated obscurations are equal to manually rotated obscurations
    np.testing.assert_array_almost_equal(
        rotated_obscurations.numpy(), manually_rotated_obscurations
    )
