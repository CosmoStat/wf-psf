#!/bin/python

import numpy as np
from astropy.io import fits
import rca
from joblib import Parallel, delayed, parallel_backend, cpu_count

import click
@click.command()

@click.option(
    "--n_comp",
    default=4,
    type=int,
    help="RCA number of eigenPSFs.")
@click.option(
    "--upfact",
    default=1,
    type=int,
    help="Upsampling factor.")
@click.option(
    "--ksig",
    default=3.,
    type=float,
    help="Denoising parameter K.")
@click.option(
    "--run_id",
    default="rca",
    type=str,
    help="Id used for the saved models and validations.")

def main(**args):
    print(args)
    with parallel_backend("loky", inner_max_num_threads=4):
        results = Parallel(n_jobs=1)(
            delayed(rca_procedure)(**args) for i in range(1)
        )

def rca_procedure(**args):
    # Model parameters
    n_comp = args['n_comp']
    upfact = args['upfact']
    ksig = args['ksig']

    run_id = args['run_id']

    print('n_comp ', n_comp)
    print('upfact ', upfact)
    print('ksig ', ksig)
    print('run_id ', run_id)


if __name__ == "__main__":
  main()
