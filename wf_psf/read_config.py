"""
:file: wf_psf/train.py

:date: 18/01/23
:author: jpollack

"""

import yaml
import pprint

from types import SimpleNamespace
import os


class RecursiveNamespace(SimpleNamespace):

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def read_yaml(conf_file="config.yaml"):
    """A function to read YAML file."""
    with open(conf_file) as f:
        config = yaml.safe_load(f)

    return config


def read_conf(conf_file="config.yaml"):
    """A function to read a yaml conf file recursively."""
    with open(conf_file, "r") as f:
        my_conf = yaml.safe_load(f)
    return RecursiveNamespace(**my_conf)


def read_stream(conf_file="configs.yaml"):
    """A function to read multiple docs in a config."""
    stream = open(conf_file, "r")
    docs = yaml.load_all(stream, yaml.FullLoader)
   # with open(conf_file, "r") as f:
    #    docs = yaml.safe_load_all(f)

    for doc in docs:
        conf = RecursiveNamespace(**doc)
        yield conf


if __name__ == "__main__":
    workdir = os.getenv('HOME')

    # read the yaml config
    my_config = read_yaml(os.path.join(
        workdir, "Projects/wf-psf/wf_psf/config/training_config.yaml"))

    # prtty print my_config
    pprint.pprint(my_config)

    # prtty print env config
    wf_conf = read_conf(os.path.join(
        workdir, "Projects/wf-psf/wf_psf/config/config.yaml"))
    pprint.pprint(wf_conf)

    # prtty print multiple docs in config
    wf_confs = read_stream(os.path.join(
        workdir, "Projects/wf-psf/wf_psf/config/configs.yaml"))

    for wf_conf in wf_confs:
        print('hi', wf_conf)
