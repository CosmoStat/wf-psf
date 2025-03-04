"""Metrics Interface.

A module which defines the classes and methods
to manage metrics evaluation of the trained psf model.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
from typing import Dict, Any
import time
from wf_psf.psf_models import psf_models
from wf_psf.metrics import metrics as wf_metrics
import logging

logger = logging.getLogger(__name__)


class MetricsParamsHandler:
    """Metrics Parameters Handler.

    A class to handle metrics-related parameters.

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

    def evaluate_metrics_polychromatic_lowres(self, 
                                              psf_model: Any, 
                                              simPSF: Any, 
                                              data: Any, 
                                              dataset: Dict[str, Any]
                                             ) -> Dict[str, float]:
        """Evaluate RMSE metrics for low-resolution polychromatic PSF.

        This method computes Root Mean Square Error (RMSE) metrics for a 
        low-resolution polychromatic Point Spread Function (PSF) model.

        Parameters
        ----------
        psf_model : object
            An instance of the PSF model selected for metrics evaluation.
        simPSF : object
            An instance of the PSFSimulator.
        data : object
            A DataConfigHandler object containing training and test datasets.
        dataset : dict
            Dictionary containing dataset details, including:
            - ``SEDs`` (Spectral Energy Distributions)
            - ``positions`` (Star positions)
            - ``C_poly``  Tensor or None, optional
                The Zernike coefficient matrix used in generating simulations of the PSF model. This
                matrix defines the Zernike polynomials up to a given order used to simulate the PSF
                field. It may be present in some datasets or only required for some classes. 
                If not present or required, the model will proceed without it.


        Returns
        -------
        dict
            A dictionary containing the RMSE, relative RMSE, and their 
            corresponding standard deviation values.

            - ``rmse`` : float
                Root Mean Square Error (RMSE).
            - ``rel_rmse`` : float
                Relative RMSE.
            - ``std_rmse`` : float
                Standard deviation of RMSE.
            - ``std_rel_rmse`` : float
                Standard deviation of relative RMSE.

        """
        logger.info("Computing polychromatic metrics at low resolution.")

        # Check if testing predictions should be masked
        mask = self.trained_model.training_hparams.loss == "mask_mse"

        rmse, rel_rmse, std_rmse, std_rel_rmse = wf_metrics.compute_poly_metric(
            tf_semiparam_field=psf_model,
            gt_tf_semiparam_field=psf_models.get_psf_model(
                self.metrics_params.ground_truth_model.model_params,
                self.metrics_params.metrics_hparams,
                data,
                dataset.get("C_poly", None),  # Extract C_poly or default to None
            ),
            simPSF_np=simPSF,
            tf_pos=dataset["positions"],
            tf_SEDs=dataset["SEDs"],
            n_bins_lda=self.trained_model.model_params.n_bins_lda,
            n_bins_gt=self.metrics_params.ground_truth_model.model_params.n_bins_lda,
            batch_size=self.metrics_params.metrics_hparams.batch_size,
            dataset_dict=dataset,
            mask=mask,
        )

        return {
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "std_rmse": std_rmse,
            "std_rel_rmse": std_rel_rmse,
        }


    def evaluate_metrics_mono_rmse(self, 
                                   psf_model: Any, 
                                   simPSF: Any, 
                                   data: Any, 
                                   dataset: Dict[str, Any]
                                  ) -> Dict[str, float]:
        """Evaluate RMSE metrics for Monochromatic PSF.

        This method computes Root Mean Square Error (RMSE) metrics for a 
        monochromatic Point Spread Function (PSF) model across a range of 
        wavelengths.

        Parameters
        ----------
        psf_model : object
            An instance of the PSF model selected for metrics evaluation.
        simPSF : object
            An instance of the PSFSimulator.
        data : object
            A DataConfigHandler object containing training and test datasets.
        dataset : dict
            Dictionary containing dataset details, including:
            - ``positions`` (Star positions)
            - ``C_poly``  (Tensor or None, optional)
                The Zernike coefficient matrix used in generating simulations of the PSF model. This
                matrix defines the Zernike polynomials up to a given order used to simulate the PSF
                field. It may be present in some datasets or only required for some classes. 
                If not present or required, the model will proceed without it.

        Returns
        -------
        dict
            A dictionary containing RMSE, relative RMSE, and their corresponding 
            standard deviation values computed over a wavelength range.

            - ``rmse_lda`` : float
                Root Mean Square Error (RMSE) over wavelengths.
            - ``rel_rmse_lda`` : float
                Relative RMSE over wavelengths.
            - ``std_rmse_lda`` : float
                Standard deviation of RMSE over wavelengths.
            - ``std_rel_rmse_lda`` : float
                Standard deviation of relative RMSE over wavelengths.
        """
        logger.info("Computing monochromatic metrics.")

        # Define the wavelength range (550nm to 900nm with 10nm intervals)
        lambda_list = np.arange(0.55, 0.9, 0.01)  # 10nm separation

        (
            rmse_lda,
            rel_rmse_lda,
            std_rmse_lda,
            std_rel_rmse_lda,
        ) = wf_metrics.compute_mono_metric(
            tf_semiparam_field=psf_model,
            gt_tf_semiparam_field=psf_models.get_psf_model(
                self.metrics_params.ground_truth_model.model_params,
                self.metrics_params.metrics_hparams,
                data,
                dataset.get("C_poly", None),
            ),
            simPSF_np=simPSF,
            tf_pos=dataset["positions"],
            lambda_list=lambda_list,
        )

        return {
            "rmse_lda": rmse_lda,
            "rel_rmse_lda": rel_rmse_lda,
            "std_rmse_lda": std_rmse_lda,
            "std_rel_rmse_lda": std_rel_rmse_lda,
        }
   

    def evaluate_metrics_opd(self, 
                             psf_model: Any, 
                             simPSF: Any, 
                             data: Any, 
                             dataset: Dict[str, Any]
                            ) -> Dict[str, float]:
        """Evaluate Optical Path Difference (OPD) metrics.
                
        This method computes Root Mean Square Error (RMSE) and relative RMSE 
        for Optical Path Differences (OPD), along with their standard deviations.

        Parameters
        ----------
        psf_model: object
            An instance of the PSF model selected for metrics evaluation.
        simPSF: object
            An instance of the PSFSimulator.
        data : object
            A DataConfigHandler object containing training and test datasets.
        dataset : dict
            Dictionary containing dataset details, including:
            - ``positions`` (Star positions)
            - ``C_poly`` (Tensor or None, optional)
                The Zernike coefficient matrix used in generating simulations of the PSF model. This
                matrix defines the Zernike polynomials up to a given order used to simulate the PSF
                field. It may be present in some datasets or only required for some classes. 
                If not present or required, the model will proceed without it.

        Returns
        -------
        dict
            A dictionary containing RMSE, relative RMSE, and their corresponding 
            standard deviation values for OPD metrics.

            - ``rmse_opd`` : float
                Root Mean Square Error (RMSE) for OPD.
            - ``rel_rmse_opd`` : float
                Relative RMSE for OPD.
            - ``rmse_std_opd`` : float
                Standard deviation of RMSE for OPD.
            - ``rel_rmse_std_opd`` : float
                Standard deviation of relative RMSE for OPD.

        """
        logger.info("Computing OPD metrics.")

        (
            rmse_opd,
            rel_rmse_opd,
            rmse_std_opd,
            rel_rmse_std_opd,
        ) = wf_metrics.compute_opd_metrics(
            tf_semiparam_field=psf_model,
            gt_tf_semiparam_field=psf_models.get_psf_model(
                self.metrics_params.ground_truth_model.model_params,
                self.metrics_params.metrics_hparams,
                data,
                dataset.get("C_poly", None),  # Extract C_poly if available
            ),
            pos=dataset["positions"],
            batch_size=self.metrics_params.metrics_hparams.batch_size,
        )

        return {
            "rmse_opd": rmse_opd,
            "rel_rmse_opd": rel_rmse_opd,
            "rmse_std_opd": rmse_std_opd,
            "rel_rmse_std_opd": rel_rmse_std_opd,
        }


    def evaluate_metrics_shape(self, 
                               psf_model: Any, 
                               simPSF: Any, 
                               data: Any, 
                               dataset: Dict[str, Any]
                              ) -> Dict[str, float]:
        """Evaluate PSF Shape Metrics.

        Computes shape-related metrics for the PSF model, including RMSE, 
        relative RMSE, and their standard deviations.

        Parameters
        ----------
        psf_model : object
            Instance of the PSF model selected for evaluation.
        simPSF : object
            Instance of the PSFSimulator.
        data : object
            A DataConfigHandler object containing training and test datasets.
        dataset : dict
            Dictionary containing dataset details, including:
            - ``SEDs`` (Spectral Energy Distributions)
            - ``positions`` (Star positions)
            - ``C_poly`` (Tensor or None, optional)
                The Zernike coefficient matrix used in generating simulations of the PSF model. This
                matrix defines the Zernike polynomials up to a given order used to simulate the PSF
                field. It may be present in some datasets or only required for some classes. 
                If not present or required, the model will proceed without it.

        Returns
        -------
        shape_results: dict
            Dictionary containing RMSE, Relative RMSE values, and
            corresponding Standard Deviation values for PSF Shape metrics.

        """
        logger.info("Computing Shape metrics.")

        shape_results = wf_metrics.compute_shape_metrics(
            tf_semiparam_field=psf_model,
            gt_tf_semiparam_field=psf_models.get_psf_model(
                self.metrics_params.ground_truth_model.model_params,
                self.metrics_params.metrics_hparams,
                data,
                dataset.get("C_poly", None),
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
    data,
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
    data: DataHandler object
        DataHandler object containing training and test data
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
        logger.info("Fetching and preprocessing training and test data...")

        # Initialize metrics_handler
        metrics_handler = MetricsParamsHandler(metrics_params, trained_model_params)

        ## Prepare models
        # Prepare np input
        simPSF_np = data.training_data.simPSF

        ## Load the model's weights
        try:
            logger.info("Loading PSF model weights from {}".format(weights_path))
            psf_model.load_weights(weights_path)
        except Exception as e: 
            logger.exception("An error occurred with the weights_path file: %s", e)
            exit()

        # Define datasets
        datasets = {"test": data.test_data.dataset, "train": data.training_data.dataset}

        # Initialise dictionary to store metrics
        all_metrics = {}

        # Define metric names and their corresponding evaluation flags
        metric_evaluation_flags = {
            "poly_metric": {
                "test": True,
                "train": True,
            },
            "mono_metric": {
                "test": metrics_params.eval_mono_metric,
                "train": metrics_params.eval_mono_metric,
            },
            "opd_metric": {
                "test": metrics_params.eval_opd_metric,
                "train": metrics_params.eval_opd_metric,
            },
            "shape_results_dict": {
                "test": metrics_params.eval_train_shape_results_dict,
                "train": metrics_params.eval_train_shape_results_dict,
            },
        }

        # Define the metric evaluation functions
        metric_functions = {
            "poly_metric": metrics_handler.evaluate_metrics_polychromatic_lowres,
            "mono_metric": metrics_handler.evaluate_metrics_mono_rmse,
            "opd_metric": metrics_handler.evaluate_metrics_opd,
            "shape_results_dict": metrics_handler.evaluate_metrics_shape,
        }

        for dataset_type, dataset in datasets.items():
            ## Metric evaluation
            logger.info(
                f"\n***\nMetric evaluation on the {dataset_type} dataset\n***\n"
            )

            # Create dictionary to store metrics for the current dataset
            dataset_metrics = {}

            # Evaluate metrics based on evaluation flags
            for metric_name, metric_function in metric_functions.items():
                # Check if any attribute in the metrics_params contains the
                # substring metric_name
                if metric_evaluation_flags[metric_name][dataset_type]:
                    dataset_metrics[metric_name] = metric_function(
                        psf_model,
                        simPSF_np,
                        data,
                        dataset,
                    )
                else:
                    dataset_metrics[metric_name] = None

            # Store dataset metrics in the overall metrics dictionary
            all_metrics[f"{dataset_type}_metrics"] = dataset_metrics

        run_id_name = (
            trained_model_params.model_params.model_name + trained_model_params.id_name
        )
        output_path = metrics_output + "/" + "metrics-" + run_id_name
        np.save(output_path, all_metrics, allow_pickle=True)

        ## Print final time
        final_time = time.time()
        logger.info("\nTotal elapsed time: %f" % (final_time - starting_time))

        ## Close log file
        logger.info("\n Good bye..")

        return all_metrics
    except Exception as e:
        logger.info("Error: %s" % e)
        raise
