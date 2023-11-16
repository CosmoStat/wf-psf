"""Metrics Interface.

A module which defines the classes and methods
to manage metrics evaluation of the trained psf model.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
import wf_psf.data.training_preprocessing as training_preprocessing
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.psf_models import psf_models
from wf_psf.metrics import metrics as wf_metrics
import os
import logging
import wf_psf.utils.io as io

logger = logging.getLogger(__name__)


def ground_truth_psf_model(metrics_params, coeff_matrix):
    psf_model = psf_models.get_psf_model(
        metrics_params.ground_truth_model.model_params,
        metrics_params.metrics_hparams,
    )
    psf_model.tf_poly_Z_field.assign_coeff_matrix(coeff_matrix)

    psf_model.tf_np_poly_opd.alpha_mat.assign(
        np.zeros_like(psf_model.tf_np_poly_opd.alpha_mat)  # type: ignore
    )

    return psf_model


class MetricsParamsHandler:
    """Metrics Parameters Handler.

    A class to handle training parameters accessed:

    Parameters
    ----------
    metrics_params: Recursive Namespace object
        Recursive Namespace object containing metrics input parameters
    trained_model: Recursive Namespace object
        Recursive Namespace object containing trained model input parameters


    """

    def __init__(self, metrics_params, trained_model):
        self.metrics_params = metrics_params
        self.trained_model = trained_model

    def evaluate_metrics_polychromatic_lowres(self, psf_model, simPSF, dataset):
        """Evaluate Polychromatic PSF Low-Res Metrics.

        A function to evaluate metrics for Low-Res Polychromatic PSF.

        Inputs
        ------
        psf_model: object
            PSF model class instance of the psf model selected for metrics evaluation.
        simPSF: object
            SimPSFToolkit instance
        test_dataset: dict
            Test dataset dictionary

        Returns
        -------
        poly_metric: dict
            Dictionary containing RMSE, Relative RMSE values, and
            corresponding Standard Deviation values for Low-Res Polychromatic PSF metrics.

        """
        logger.info("Computing polychromatic metrics at low resolution.")

        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=ground_truth_psf_model(
                self.metrics_params, dataset["C_poly"]
            ),
            simPSF_np=simPSF,
            tf_pos=dataset["positions"],
            tf_SEDs=dataset["SEDs"],
            n_bins_lda=self.trained_model.model_params.n_bins_lda,
            n_bins_gt=self.metrics_params.ground_truth_model.model_params.n_bins_lda,
            batch_size=self.metrics_params.metrics_hparams.batch_size,
            dataset_dict=dataset,
        )

        poly_metric = {
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "std_rmse": std_rmse,
            "std_rel_rmse": std_rel_rmse,
        }
        return poly_metric

    def evaluate_metrics_mono_rmse(self, psf_model, simPSF, dataset):
        """Evaluate Monochromatic PSF RMSE Metrics.

        A function to evaluate metrics for Monochromatic PSF.

        Inputs
        ------
        psf_model: object
            PSF model class instance of the psf model selected for metrics evaluation.
        simPSF: object
            SimPSFToolkit instance
        test_dataset: dict
            Test dataset dictionary

        Returns
        -------
        mono_metric: dict
            Dictionary containing RMSE, Relative RMSE values, and
            corresponding Standard Deviation values for Monochromatic PSF metrics.

        """
        logger.info("Computing monochromatic metrics.")
        lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation
        (
            rmse_lda,
            rel_rmse_lda,
            std_rmse_lda,
            std_rel_rmse_lda,
        ) = wf_metrics.compute_mono_metric(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=ground_truth_psf_model(
                self.metrics_params, dataset["C_poly"]
            ),
            simPSF_np=simPSF,
            tf_pos=dataset["positions"],
            lambda_list=lambda_list,
        )

        mono_metric = {
            "rmse_lda": rmse_lda,
            "rel_rmse_lda": rel_rmse_lda,
            "std_rmse_lda": std_rmse_lda,
            "std_rel_rmse_lda": std_rel_rmse_lda,
        }
        return mono_metric

    def evaluate_metrics_opd(self, psf_model, simPSF, dataset):
        """Evaluate OPD Metrics.

        A function to evaluate metrics for Optical Path Differences.

        Inputs
        ------
        psf_model: object
            PSF model class instance of the psf model selected for metrics evaluation.
        simPSF: object
            SimPSFToolkit instance
        test_dataset: dict
            Test dataset dictionary

        Returns
        -------
        opd_metric: dict
            Dictionary containing RMSE, Relative RMSE values, and
            corresponding Standard Deviation values for OPD metrics.

        """
        logger.info("Computing OPD metrics.")
        (
            rmse_opd,
            rel_rmse_opd,
            rmse_std_opd,
            rel_rmse_std_opd,
        ) = wf_metrics.compute_opd_metrics(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=ground_truth_psf_model(
                self.metrics_params, dataset["C_poly"]
            ),
            pos=dataset["positions"],
            batch_size=self.metrics_params.metrics_hparams.batch_size,
        )

        opd_metric = {
            "rmse_opd": rmse_opd,
            "rel_rmse_opd": rel_rmse_opd,
            "rmse_std_opd": rmse_std_opd,
            "rel_rmse_std_opd": rel_rmse_std_opd,
        }
        return opd_metric

    def evaluate_metrics_shape(self, psf_model, simPSF, dataset):
        """Evaluate PSF Shape Metrics.

        A function to evaluate metrics for PSF shape.

        Inputs
        ------
        psf_model: object
            PSF model class instance of the psf model selected for metrics evaluation.
        simPSF: object
            SimPSFToolkit instance
        dataset: dict
            Test dataset dictionary

        Returns
        shape_results: dict
            Dictionary containing RMSE, Relative RMSE values, and
            corresponding Standard Deviation values for PSF Shape metrics.

        """
        logger.info("Computing Shape metrics.")

        shape_results = wf_metrics.compute_shape_metrics(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=ground_truth_psf_model(
                self.metrics_params, dataset["C_poly"]
            ),
            simPSF_np=simPSF,
            SEDs=dataset["SEDs"],
            tf_pos=dataset["positions"],
            n_bins_lda=self.trained_model.model_params.n_bins_lda,
            n_bins_gt=self.metrics_params.ground_truth_model.model_params.n_bins_lda,
            batch_size=self.metrics_params.metrics_hparams.batch_size,
            output_Q=self.metrics_params.metrics_hparams.output_Q,
            output_dim=self.metrics_params.metrics_hparams.output_dim,
            opt_stars_rel_pix_rmse=self.metrics_params.metrics_hparams.opt_stars_rel_pix_rmse,
            dataset_dict=dataset,
        )
        return shape_results


def evaluate_model(
    metrics_params,
    trained_model_params,
    training_data,
    test_data,
    psf_model,
    weights_path,
    metrics_output,
):
    r"""Evaluate the trained model.

    For parameters check the training script click help.

    Inputs
    ------
    metrics_params: Recursive Namespace object
        Recursive Namespace object containing metrics input parameters
    trained_model_params: Recursive Namespace object
        Recursive Namespace object containing trained model input parameters
    training_data: object
        TrainingDataHandler object
    test_data: object
        TestDataHandler object
    psf_model: object
        PSF model object
    weights_path: str
        Directory location of model weights
    metrics_output: str
        Directory location of metrics output

    """
    # Start measuring elapsed time
    starting_time = time.time()

    try:
        ## Load datasets
        # -----------------------------------------------------
        # Get training data
        logger.info(f"Fetching and preprocessing training and test data...")

        # Initialize metrics_handler
        metrics_handler = MetricsParamsHandler(metrics_params, trained_model_params)

        ## Prepare models
        # Prepare np input
        simPSF_np = training_data.simPSF

        ## Load the model's weights
        try:
            logger.info("Loading PSF model weights from {}".format(weights_path))
            psf_model.load_weights(weights_path)
        except:
            logger.exception("An error occurred with the weights_path file.")
            exit()

        ## Metric evaluation on the test dataset
        logger.info("\n***\nMetric evaluation on the test dataset\n***\n")

        # Polychromatic star reconstructions
        poly_metric = metrics_handler.evaluate_metrics_polychromatic_lowres(
            psf_model, simPSF_np, test_data.test_dataset
        )

        # Monochromatic star reconstructions
        if metrics_params.eval_mono_metric_rmse:
            mono_metric = metrics_handler.evaluate_metrics_mono_rmse(
                psf_model, simPSF_np, test_data.test_dataset
            )
        else:
            mono_metric = None

        # OPD metrics
        if metrics_params.eval_opd_metric_rmse:
            opd_metric = metrics_handler.evaluate_metrics_opd(
                psf_model, simPSF_np, test_data.test_dataset
            )
        else:
            opd_metric = None

        # Shape metrics
        logger.info(
            "Computing polychromatic high-resolution metrics and shape metrics."
        )
        shape_results_dict = metrics_handler.evaluate_metrics_shape(
            psf_model, simPSF_np, test_data.test_dataset
        )
        # Save metrics
        test_metrics = {
            "poly_metric": poly_metric,
            "mono_metric": mono_metric,
            "opd_metric": opd_metric,
            "shape_results_dict": shape_results_dict,
        }

        ## Metric evaluation on the train dataset
        logger.info("\n***\nMetric evaluation on the train dataset\n***\n")

        # Polychromatic star reconstructions
        logger.info("Computing polychromatic metrics at low resolution.")

        train_poly_metric = metrics_handler.evaluate_metrics_polychromatic_lowres(
            psf_model, simPSF_np, training_data.train_dataset
        )

        # Monochromatic star reconstructions turn into a class
        if metrics_params.eval_mono_metric_rmse:
            train_mono_metric = metrics_handler.evaluate_metrics_mono_rmse(
                psf_model, simPSF_np, training_data.train_dataset
            )
        else:
            train_mono_metric = None

        # OPD metrics turn into a class
        if metrics_params.eval_opd_metric_rmse:
            train_opd_metric = metrics_handler.evaluate_metrics_opd(
                psf_model, simPSF_np, training_data.train_dataset
            )
        else:
            train_opd_metric = None

        # Shape metrics  turn into a class
        if metrics_params.eval_train_shape_sr_metric_rmse:
            train_shape_results_dict = metrics_handler.evaluate_metrics_shape(
                psf_model, simPSF_np, training_data.train_dataset
            )
        else:
            train_shape_results_dict = None

        # Save metrics into dictionary
        train_metrics = {
            "poly_metric": train_poly_metric,
            "mono_metric": train_mono_metric,
            "opd_metric": train_opd_metric,
            "shape_results_dict": train_shape_results_dict,
        }

        ## Save results
        metrics = {"test_metrics": test_metrics, "train_metrics": train_metrics}
        run_id_name = (
            trained_model_params.model_params.model_name + trained_model_params.id_name
        )
        output_path = metrics_output + "/" + "metrics-" + run_id_name
        np.save(output_path, metrics, allow_pickle=True)

        ## Print final time
        final_time = time.time()
        logger.info("\nTotal elapsed time: %f" % (final_time - starting_time))

        ## Close log file
        logger.info("\n Good bye..")

        return metrics
    except Exception as e:
        logger.info("Error: %s" % e)
        raise
