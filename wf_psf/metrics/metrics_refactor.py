"""Metrics.

A module which defines the classes and methods
to manage metrics evaluation of the trained psf model.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
from wf_psf.utils.read_config import read_conf
import wf_psf.data.training_preprocessing as training_preprocessing
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.psf_models import psf_models
from wf_psf.metrics import metrics as wf_metrics
import os
import logging
import wf_psf.utils.io as io

logger = logging.getLogger(__name__)


class MetricsParamsHandler:
    """Metrics Parameters Handler.

    A class to handle training parameters accessed:

    Parameters
    ----------
    metrics_params: Recursive Namespace object
        Recursive Namespace object containing metrics input parameters


    """

    def __init__(self, metrics_params):
        self.metrics_params = metrics_params

    @property
    def ground_truth_psf_model(self):
        psf_model = psf_models.get_psf_model(
            self.metrics_params.ground_truth_model.model_params,
            self.metrics_params.metrics_hparams.batch_size,
        )
        psf_model.set_zero_nonparam()
        return psf_model

    def evaluate_metrics_polychromatic_lowres(self, psf_model, simPSF, test_dataset):
        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=self.ground_truth_psf_model,
            simPSF_np=simPSF,
            tf_pos=test_dataset["positions"],
            tf_SEDs=test_dataset["SEDs"],
            n_bins_lda=self.metrics_params.model_params.n_bins_lda,
            n_bins_gt=self.metrics_params.ground_truth_model.model_params.n_bins_lda,
            batch_size=self.metrics_params.metrics_hparams.batch_size,
            dataset_dict=test_dataset,
        )

        poly_metric = {
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "std_rmse": std_rmse,
            "std_rel_rmse": std_rel_rmse,
        }
        return poly_metric

    def evaluate_metrics_mono_rmse(self, psf_model, simPSF, test_dataset):
        logger.info("Computing monochromatic metrics.")
        lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation
        (
            rmse_lda,
            rel_rmse_lda,
            std_rmse_lda,
            std_rel_rmse_lda,
        ) = wf_metrics.compute_mono_metric(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=self.ground_truth_psf_model,
            simPSF_np=simPSF,
            tf_pos=test_dataset["positions"],
            lambda_list=lambda_list,
        )

        mono_metric = {
            "rmse_lda": rmse_lda,
            "rel_rmse_lda": rel_rmse_lda,
            "std_rmse_lda": std_rmse_lda,
            "std_rel_rmse_lda": std_rel_rmse_lda,
        }
        return mono_metric

    def evaluate_metrics_opd(self, psf_model, simPSF, test_dataset):
        logger.info("Computing OPD metrics.")
        (
            rmse_opd,
            rel_rmse_opd,
            rmse_std_opd,
            rel_rmse_std_opd,
        ) = wf_metrics.compute_opd_metrics(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=self.ground_truth_psf_model,
            pos=test_dataset["positions"],
            batch_size=self.metrics_params.metrics_hparams.batch_size,
        )

        opd_metric = {
            "rmse_opd": rmse_opd,
            "rel_rmse_opd": rel_rmse_opd,
            "rmse_std_opd": rmse_std_opd,
            "rel_rmse_std_opd": rel_rmse_std_opd,
        }


def evaluate_model(metrics_params, training_data, test_data, psf_model, weights_path):
    r"""Evaluate the trained model.

    For parameters check the training script click help.
    """
    # Start measuring elapsed time
    starting_time = time.time()

    try:
        ## Load datasets
        # -----------------------------------------------------
        # Get training data
        logger.info(f"Fetching and preprocessing training and test data...")

        metrics_handler = MetricsParamsHandler(metrics_params)

        ## Prepare models
        # Prepare np input
        simPSF_np = training_data.simPSF

        ## Load the model's weights
        psf_model.load_weights(weights_path)

        ## Metric evaluation on the test dataset
        logger.info("\n***\nMetric evaluation on the test dataset\n***\n")

        # Polychromatic star reconstructions
        logger.info("Computing polychromatic metrics at low resolution.")

        poly_metric = metrics_handler.eval_metrics_polychromatic_lowres(
            psf_model, simPSF_np, test_data.test_dataset
        )
      
        # Monochromatic star reconstructions
        # if metrics_params.metrics_hparams.eval_mono_metric_rmse:
        mono_metric = metrics_handler.evaluate_metrics_mono_rmse(
             psf_model, simPSF_np, test_data.test_dataset
         )
        # else:
        #     mono_metric = None

        # OPD metrics
        # if metrics_params.metrics_hparams.eval_opd_metric_rmse:
        #     print("Computing OPD metrics.")
        #     (
        #         rmse_opd,
        #         rel_rmse_opd,
        #         rmse_std_opd,
        #         rel_rmse_std_opd,
        #     ) = wf_metrics.compute_opd_metrics(
        #         tf_semiparam_field=psf_model,
        #         GT_tf_semiparam_field=GT_tf_semiparam_field,
        #         pos=tf_test_pos,
        #         batch_size=args["eval_batch_size"],
        #     )

        #     opd_metric = {
        #         "rmse_opd": rmse_opd,
        #         "rel_rmse_opd": rel_rmse_opd,
        #         "rmse_std_opd": rmse_std_opd,
        #         "rel_rmse_std_opd": rel_rmse_std_opd,
        #     }
        # else:
        #     opd_metric = None

        # Check if all stars SR pixel RMSE are needed
        if "opt_stars_rel_pix_rmse" not in args:
            args["opt_stars_rel_pix_rmse"] = False

        # Shape metrics
        print("Computing polychromatic high-resolution metrics and shape metrics.")
        shape_results_dict = wf_metrics.compute_shape_metrics(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=GT_tf_semiparam_field,
            simPSF_np=simPSF_np,
            SEDs=test_SEDs,
            tf_pos=tf_test_pos,
            n_bins_lda=args["n_bins_lda"],
            n_bins_gt=args["n_bins_gt"],
            output_Q=1,
            output_dim=64,
            batch_size=args["eval_batch_size"],
            opt_stars_rel_pix_rmse=args["opt_stars_rel_pix_rmse"],
            dataset_dict=test_dataset,
        )

        # Save metrics
        test_metrics = {
            "poly_metric": poly_metric,
            "mono_metric": mono_metric,
            "opd_metric": opd_metric,
            "shape_results_dict": shape_results_dict,
        }

        ## Metric evaluation on the train dataset
        print("\n***\nMetric evaluation on the train dataset\n***\n")

        # Polychromatic star reconstructions
        print("Computing polychromatic metrics at low resolution.")
        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            tf_semiparam_field=psf_model,
            GT_tf_semiparam_field=GT_tf_semiparam_field,
            simPSF_np=simPSF_np,
            tf_pos=tf_train_pos,
            tf_SEDs=train_SEDs,
            n_bins_lda=args["n_bins_lda"],
            n_bins_gt=args["n_bins_gt"],
            batch_size=args["eval_batch_size"],
            dataset_dict=train_dataset,
        )

        train_poly_metric = {
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "std_rmse": std_rmse,
            "std_rel_rmse": std_rel_rmse,
        }

        # Monochromatic star reconstructions turn into a class
        if args["eval_mono_metric_rmse"] is True:
            print("Computing monochromatic metrics.")
            lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation
            (
                rmse_lda,
                rel_rmse_lda,
                std_rmse_lda,
                std_rel_rmse_lda,
            ) = wf_metrics.compute_mono_metric(
                tf_semiparam_field=psf_model,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                simPSF_np=simPSF_np,
                tf_pos=tf_train_pos,
                lambda_list=lambda_list,
            )

            train_mono_metric = {
                "rmse_lda": rmse_lda,
                "rel_rmse_lda": rel_rmse_lda,
                "std_rmse_lda": std_rmse_lda,
                "std_rel_rmse_lda": std_rel_rmse_lda,
            }
        else:
            train_mono_metric = None

        # OPD metrics turn into a class
        if args["eval_opd_metric_rmse"] is True:
            print("Computing OPD metrics.")
            (
                rmse_opd,
                rel_rmse_opd,
                rmse_std_opd,
                rel_rmse_std_opd,
            ) = wf_metrics.compute_opd_metrics(
                tf_semiparam_field=psf_model,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                pos=tf_train_pos,
                batch_size=args["eval_batch_size"],
            )

            train_opd_metric = {
                "rmse_opd": rmse_opd,
                "rel_rmse_opd": rel_rmse_opd,
                "rmse_std_opd": rmse_std_opd,
                "rel_rmse_std_opd": rel_rmse_std_opd,
            }
        else:
            train_opd_metric = None

        # Shape metrics  turn into a class
        if (
            args["eval_train_shape_sr_metric_rmse"] is True
            or "eval_train_shape_sr_metric_rmse" not in args
        ):
            print("Computing polychromatic high-resolution metrics and shape metrics.")
            train_shape_results_dict = wf_metrics.compute_shape_metrics(
                tf_semiparam_field=psf_model,
                GT_tf_semiparam_field=GT_tf_semiparam_field,
                simPSF_np=simPSF_np,
                SEDs=train_SEDs,
                tf_pos=tf_train_pos,
                n_bins_lda=args["n_bins_lda"],
                n_bins_gt=args["n_bins_gt"],
                output_Q=1,
                output_dim=64,
                batch_size=args["eval_batch_size"],
                dataset_dict=train_dataset,
            )

            # Save metrics into dictionary
            train_metrics = {
                "poly_metric": train_poly_metric,
                "mono_metric": train_mono_metric,
                "opd_metric": train_opd_metric,
                "shape_results_dict": train_shape_results_dict,
            }
        else:
            train_metrics = None

        ## Save results
        metrics = {"test_metrics": test_metrics, "train_metrics": train_metrics}
        output_path = args["metric_base_path"] + "metrics-" + run_id_name
        np.save(output_path, metrics, allow_pickle=True)

        ## Print final time
        final_time = time.time()
        print("\nTotal elapsed time: %f" % (final_time - starting_time))

        ## Close log file
        print("\n Good bye..")

    except Exception as e:
        print("Error: %s" % e)
        raise
