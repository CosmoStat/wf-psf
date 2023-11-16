"""UNIT TESTS FOR PACKAGE MODULE: Train.

This module contains unit tests for the wf_psf.train module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.training import train
from wf_psf.psf_models import psf_models
import tensorflow as tf
import numpy as np
import os
from re import search


cwd = os.getcwd()

validation_dir = "src/wf_psf/tests/data/validation/main_random_seed"


@pytest.fixture(scope="module")
def tmp_checkpoint_dir(tmp_path_factory):
    tmp_chkp_dir = tmp_path_factory.mktemp("checkpoint")
    return str(tmp_chkp_dir)


@pytest.fixture(scope="module")
def tmp_optimizer_dir(tmp_path_factory):
    tmp_optim_hist_dir = tmp_path_factory.mktemp("optim-hist")
    return str(tmp_optim_hist_dir)


@pytest.fixture(scope="module")
def tmp_psf_model_dir(tmp_path_factory):
    tmp_psf_model_dir = tmp_path_factory.mktemp("psf_model_dir")
    return str(tmp_psf_model_dir)


@pytest.fixture(scope="module")
def checkpoint_dir():
    return os.path.join(
        cwd,
        validation_dir,
        "checkpoint",
    )


@pytest.fixture(scope="module")
def optimizer_dir():
    return os.path.join(
        cwd,
        validation_dir,
        "optim-hist",
    )


@pytest.fixture(scope="module")
def psf_model_dir():
    return os.path.join(
        cwd,
        validation_dir,
        "psf_model",
    )


@pytest.mark.skip(reason="Requires gpu")
def test_train(
    training_params,
    training_data,
    test_data,
    checkpoint_dir,
    optimizer_dir,
    psf_model_dir,
    tmp_checkpoint_dir,
    tmp_optimizer_dir,
    tmp_psf_model_dir,
    psf_model,
):
    train.train(
        training_params,
        training_data,
        test_data,
        tmp_checkpoint_dir,
        tmp_optimizer_dir,
        tmp_psf_model_dir,
    )

    weights_type_dict = {
        checkpoint_dir: "checkpoint_callback_",
        psf_model_dir: "psf_model_",
    }

    # Evaluate the weights for each checkpoint callback and the final psf models wrt baseline
    weights_basename = (
        training_params.model_params.model_name + training_params.id_name + "_cycle"
    )

    tmp_psf_model = psf_models.get_psf_model(
        training_params.model_params, training_params.training_hparams
    )

    for weights_dir, tmp_weights_dir in zip(
        [checkpoint_dir, psf_model_dir], [tmp_checkpoint_dir, tmp_psf_model_dir]
    ):
        first_cycle = 1

        if search("psf_model", weights_dir):
            if not training_params.training_hparams.multi_cycle_params.save_all_cycles:
                first_cycle = (
                    training_params.training_hparams.multi_cycle_params.total_cycles
                )

        for cycle in range(
            first_cycle,
            training_params.training_hparams.multi_cycle_params.total_cycles,
        ):
            basename_cycle = (
                weights_dir
                + "/"
                + weights_type_dict[weights_dir]
                + weights_basename
                + str(cycle)
            )

            tmp_basename_cycle = (
                tmp_weights_dir
                + "/"
                + weights_type_dict[weights_dir]
                + weights_basename
                + str(cycle)
            )

            psf_model.load_weights(basename_cycle)
            saved_model_weights = psf_model.get_weights()

            tmp_psf_model.load_weights(tmp_basename_cycle)
            tmp_saved_model_weights = tmp_psf_model.get_weights()

            diff = abs(
                np.array(saved_model_weights) - np.array(tmp_saved_model_weights)
            )
            for arr in diff:
                assert np.mean(arr) < 1.0e-9
