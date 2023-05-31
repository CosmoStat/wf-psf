"""WF_PSF Run.

This module setups the run of the WF_PSF pipeline.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import argparse
from wf_psf.utils.read_config import read_stream, read_conf
from wf_psf.utils.io import FileIOHandler
import os
import logging.config
import logging
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.training import train
from wf_psf.psf_models import psf_models
from wf_psf.metrics.metrics_interface import evaluate_model


def setProgramOptions():
    """Define Program Options.

    Set command-line options for
    this program.

    Returns
    -------
    args: type
        Argument Parser Namespace

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conffile",
        "-c",
        type=str,
        required=True,
        help="a configuration file containing program settings.",
    )

    parser.add_argument(
        "--repodir",
        "-r",
        type=str,
        required=True,
        help="the path of the code repository directory.",
    )

    parser.add_argument(
        "--outputdir",
        "-o",
        type=str,
        required=True,
        help="the path of the output directory.",
    )

    args = parser.parse_args()

    return args


def mainMethod():
    """Main Method.

    The main entry point to wavediff program.


    """
    args = setProgramOptions()

    file_handler = FileIOHandler(args.repodir, args.outputdir)
    file_handler.setup_outputs()

    logger = logging.getLogger("wavediff")

    logger.info("#")
    logger.info("# Entering wavediff mainMethod()")
    logger.info("#")

    configs_path = os.path.dirname(args.conffile)
    configs = read_stream(args.conffile)
    configs_file = os.path.basename(args.conffile)
    file_handler.copy_conffile_to_output_dir(configs_path, configs_file)

    config_types = {
        "data_conf": data_params,
        "training_conf": training_params,
        "metrics_conf": metrics_params,
        "plotting_conf": plotting_params,
    }
    for conf in configs:
        for k, v in conf.items():
            try:
                config_types[k] = read_conf(os.path.join(configs_path, v))
                logger.info(config_types[k])
                file_handler.copy_conffile_to_output_dir(configs_path, v)
            except FileNotFoundError as e:
                logger.exception(e)
                exit()
            except ValueError as e:
                logger.exception(e)
                exit()
            except TypeError as e:
                logger.exception(e)
                if v is not None:
                    exit()

    try:
        logger.info("Performing training...")

        simPSF = psf_models.simPSF(config_types["training_conf"].training.model_params)

        training_data = TrainingDataHandler(
            config_types["data_conf"].data.training,
            simPSF,
            config_types["training_conf"].training.model_params.n_bins_lda,
        )

        test_data = TestDataHandler(
            config_types["data_conf"].data.test,
            simPSF,
            config_types["training_conf"].training.model_params.n_bins_lda,
        )

        psf_model, checkpoint_filepath = train.train(
            config_types["training_conf"].training,
            training_data,
            test_data,
            file_handler.get_checkpoint_dir(file_handler._run_output_dir),
            file_handler.get_optimizer_dir(file_handler._run_output_dir),
        )

        if config_types["metrics_conf"] is not None:
            logger.info("Performing metrics evaluation of trained PSF model...")
            evaluate_model(
                config_types["metrics_conf"].metrics,
                config_types["training_conf"].training.training,
                training_data,
                test_data,
                psf_model,
                checkpoint_filepath,
                file_handler.get_metrics_dir(file_handler._run_output_dir),
            )

    except AttributeError:
        logger.info("Training or Metrics not set in configs.yaml. Skipping...")

    if (
        config_types["training_conf"] is None
        and config_types["metrics_conf"] is not None
    ):
        try:
            logger.info("Performing metrics evaluation only...")
            # Get Config File for Trained PSF Model
            trained_params = read_conf(
                os.path.join(
                    config_types["metrics_conf"].metrics.trained_model_path,
                    config_types["metrics_conf"].metrics.trained_model_config,
                )
            )
            logger.info(trained_params.training)
            simPSF = psf_models.simPSF(trained_params.training.model_params)

            checkpoint_filepath = train.filepath_chkp_callback(
                file_handler.get_checkpoint_dir(
                    config_types["metrics_conf"].metrics.trained_model_path
                ),
                trained_params.training.model_params.model_name,
                trained_params.training.id_name,
                config_types["metrics_conf"].metrics.saved_training_cycle,
            )

            training_data = TrainingDataHandler(
                config_types["data_conf"].data.training,
                simPSF,
                trained_params.training.model_params.n_bins_lda,
            )

            test_data = TestDataHandler(
                config_types["data_conf"].data.test,
                simPSF,
                trained_params.training.model_params.n_bins_lda,
            )

            psf_model = psf_models.get_psf_model(
                trained_params.training.model_params,
                trained_params.training.training_hparams,
            )

            evaluate_model(
                config_types["metrics_conf"].metrics,
                trained_params.training,
                training_data,
                test_data,
                psf_model,
                checkpoint_filepath,
                file_handler.get_metrics_dir(file_handler._run_output_dir),
            )
        except AttributeError:
            logger.exception(
                "Configs are not correctly set in configs.yaml.  Please check your config file."
            )
        except FileNotFoundError as e:
            logger.exception(e)

    try:
        logger.info("Generating plots...")
        # Get Config File for Trained PSF Model
        trained_params = read_conf(
            os.path.join(
                config_types["plotting_conf"].plots.trained_model_path,
                config_types["plotting_conf"].plots.trained_model_config,
            )
        )

        metrics_params = read_conf(
            os.path.join(
                config_types["plotting_conf"].plots.trained_model_path,
                config_types["plotting_conf"].plots.metrics_config,
            )
        )
    except AttributeError:
        logger.exception(
            "Configs are not correctly set in configs.yaml.  Please check your config file."
        )
    except FileNotFoundError as e:
        logger.exception(e)

    logger.info("#")
    logger.info("# Exiting wavediff mainMethod()")
    logger.info("#")


if __name__ == "__main__":
    mainMethod()
