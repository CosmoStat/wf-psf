[![arXiv:2203.04908](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2203.04908-B31B1B.svg)](http://arxiv.org/abs/2203.04908) [![yapf](https://img.shields.io/badge/code%20style-yapf-blue.svg)](https://www.python.org/dev/peps/pep-0008/) [![License](https://img.shields.io/badge/License-MIT-brigthgreen.svg)](https://github.com/tobias-liaudat/wf-psf/tree/master/LICENSE)

<h1 align='center'>WaveDiff</h1>
<h2 align='center'>A differentiable data-driven wavefront-based PSF modelling framework.</h2>

WaveDiff is a differentiable PSF modelling pipeline constructed with [Tensorflow](https://github.com/tensorflow/tensorflow). It was developed at the [CosmoStat lab](https://www.cosmostat.org) at CEA Paris-Saclay.

See the [documentation](https://cosmostat.github.io/wf-psf/) for details on how to install and run WaveDiff.

This repository includes:
- A differentiable PSF model entirely built in [Tensorflow](https://github.com/tensorflow/tensorflow).
- A [numpy-based PSF simulator](https://github.com/CosmoStat/wf-psf/tree/dummy_main/src/wf_psf/sims).
- All the scripts, jobs and notebooks required to reproduce the results in [arXiv:2203.04908](http://arxiv.org/abs/2203.04908) and [arXiv:2111.12541](https://arxiv.org/abs/2111.12541).
