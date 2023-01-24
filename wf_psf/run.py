"""
:file: wf_psf/wavediff.py

:date: 19/01/23
:author: jpollack

"""
import argparse
import logging
import logging.config
from read_config import read_stream, read_conf
import wf_psf.io as io
import os


parser = argparse.ArgumentParser()

parser.add_argument('--conffile', '-c', type=str, required=True,
                    help="a configuration file containing program settings.")

parser.add_argument('--workspace', '-w', type=str, required=True,
                    help="Workspace directory containing configuration files.")

args = parser.parse_args()
config = vars(args)
print(config['conffile'])

logging.config.fileConfig(os.path.join(args.workspace, 'config/logging.conf'))
logging.basicConfig(filename='example.log',
                    encoding='utf-8', level=logging.DEBUG)

f_handler = logging.FileHandler('example.log')

logger = logging.getLogger("wavediff")

logger.info('#')
logger.info('# Entering wavediff mainMethod()')
logger.info('#')
logger.addHandler(f_handler)
configs = read_stream(os.path.join(args.workspace, config['conffile']))

for conf in configs:
    if hasattr(conf, "env_vars"):
        io.set_env_workdir(conf.env_vars.workdir)
        io.set_env_repodir(conf.env_vars.repodir)
        out = io.WFPSF_FileStructure()
        out.make_wfpsf_file_struct()

    if hasattr(conf, "training_conf"):
        # load training_conf
        training_params = read_conf(os.path.join(
            args.workspace, conf.training_conf))
        print(training_params)
