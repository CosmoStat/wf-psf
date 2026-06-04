# Installation


WaveDiff is a software written in Python and uses the [TensorFlow](https://www.tensorflow.org) framework to train and evaluate the model parameters of the PSF model. Therefore, it is advisable to run the WaveDiff software on machines equipped with a GPU.

Note: You may want to set up a dedicated Python environment for running WaveDiff using e.g. [Conda](https://docs.conda.io/en/latest/).  You can use the minimal installer [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to set up the environment in which we run the command below to install the subset of packages needed for running WaveDiff.

## Recommended Environment

WaveDiff is currently tested with:

- Python 3.10 and 3.11
- TensorFlow 2.15.0
- `tf.keras` (bundled with TensorFlow 2.15)

Python 3.9 has been removed from the CI test matrix due to TensorFlow compatibility constraints and updated internal dependencies.

We recommend creating the provided conda environment before installing WaveDiff:

```bash
conda env create -f environment.yml
conda activate wavediff-env
```

## Installation Steps

Clone the repository:

```bash
git clone https://github.com/CosmoStat/wf-psf.git
```

Create the recommended environment (if you haven't already):

```bash
conda env create -f environment.yml
conda activate wavediff-env
```

Install WaveDiff:

```
$ cd wf-psf
$ pip install .
```

You can now proceed to the next section, where we show you how to run WaveDiff.
