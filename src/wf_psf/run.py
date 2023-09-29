"""WF_PSF Run.

This module setups the run of the WF_PSF pipeline.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import argparse
from wf_psf.utils.read_config import read_stream
from wf_psf.utils.io import FileIOHandler
import os
import logging.config
import logging
from wf_psf.utils.configs_handler import get_run_config


def setProgramOptions():
    """Define Program Options.

    Set command-line options for
    this program.

    Returns
    -------
    args: argparge.Namespace object
        Argument Parser Namespace object

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

    configs_path = os.path.dirname(args.conffile)
    configs = read_stream(args.conffile)
    configs_file = os.path.basename(args.conffile)

    file_handler = FileIOHandler(args.repodir, args.outputdir, configs_path)
    file_handler.setup_outputs()
    file_handler.copy_conffile_to_output_dir(configs_file)

    logger = logging.getLogger("wavediff")

    logger.info("#")
    logger.info("# Entering wavediff mainMethod()")
    logger.info("#")

    try:
        for conf in configs:
            for k, v in conf.items():
                config_class = get_run_config(
                    k, os.path.join(configs_path, v), file_handler
                )
                logger.info(config_class)
                file_handler.copy_conffile_to_output_dir(v)
                config_class.run()

    except Exception as e:
        logger.error(
            "Check your config file {} for errors. Error Msg: {}.".format(
                args.conffile, e
            ),
            exc_info=True,
        )

    logger.info("#")
    logger.info("# Exiting wavediff mainMethod()")
    logger.info("#")


if __name__ == "__main__":
    mainMethod()
