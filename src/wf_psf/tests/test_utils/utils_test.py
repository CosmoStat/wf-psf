"""UNIT TESTS FOR PACKAGE MODULE: UTILS.

This module contains unit tests for the wf_psf.utils utils module.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>


"""

import pytest
import tensorflow as tf
import numpy as np
from wf_psf.utils.utils import zernike_generator
from wf_psf.sims.SimPSFToolkit import SimPSFToolkit
from wf_psf.psf_models.tf_layers import TF_zernike_OPD


def test_unobscured_zernike_projection():
    from wf_psf.utils.utils import unobscured_zernike_projection

    n_zernikes = 20
    wfe_dim = 256
    tol = 1e-1

    # Create zernike basis
    zernikes = zernike_generator(n_zernikes=n_zernikes, wfe_dim=wfe_dim)
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )
    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Create random zernike coefficient array
    zk_array = np.random.randn(1, n_zernikes, 1, 1)
    tf_zk_array = tf.convert_to_tensor(zk_array, dtype=tf.float32)

    # Generate layer
    tf_zernike_opd = TF_zernike_OPD(tf_zernike_cube)
    # Compute OPD
    tf_unobscured_opd = tf_zernike_opd(tf_zk_array)

    # Compute normalisation factor
    norm_factor = unobscured_zernike_projection(tf_zernike_cube[0, :, :], tf_zernike_cube[0, :, :])

    # Compute projections for each zernike
    estimated_zk_array = np.array(
        [
            unobscured_zernike_projection(
                tf_unobscured_opd, tf_zernike_cube[j, :, :], norm_factor=norm_factor
            )
            for j in range(n_zernikes)
        ]
    )

    rmse_error = np.linalg.norm(estimated_zk_array - zk_array[0, :, 0, 0])

    assert rmse_error < tol


def test_tf_decompose_obscured_opd_basis():
    from wf_psf.utils.utils import tf_decompose_obscured_opd_basis

    n_zernikes = 20
    wfe_dim = 256
    tol = 1e-5

    # Create zernike basis
    zernikes = zernike_generator(n_zernikes=n_zernikes, wfe_dim=wfe_dim)
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )
    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Create obscurations
    obscurations = SimPSFToolkit.generate_pupil_obscurations(N_pix=wfe_dim, N_filter=2)
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.float32)

    # Create random zernike coefficient array
    zk_array = np.random.randn(1, n_zernikes, 1, 1)
    tf_zk_array = tf.convert_to_tensor(zk_array, dtype=tf.float32)

    # Generate layer
    tf_zernike_opd = TF_zernike_OPD(tf_zernike_cube)
    # Compute OPD
    tf_unobscured_opd = tf_zernike_opd(tf_zk_array)
    # Obscure the OPD
    tf_obscured_opd = tf.math.multiply(
        tf_unobscured_opd, tf.expand_dims(tf_obscurations, axis=0)
    )

    # Compute zernike array from OPD
    obsc_coeffs = tf_decompose_obscured_opd_basis(
        tf_opd=tf_obscured_opd,
        tf_obscurations=tf_obscurations,
        tf_zk_basis=tf_zernike_cube,
        n_zernike=n_zernikes,
        iters=100,
    )

    rmse_error = np.linalg.norm(obsc_coeffs - zk_array[0, :, 0, 0])

    assert rmse_error < tol
