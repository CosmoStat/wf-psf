# Installation


WaveDiff is a software written in Python and uses the [TensorFlow](https://www.tensorflow.org) framework to train and evaluate the model parameters of the PSF model. Therefore, it is advisable to run the WaveDiff software on machines equipped with a GPU.  

Note: You may want to set up a dedicated Python environment for running WaveDiff using e.g. [Conda](https://docs.conda.io/en/latest/).  You can use the minimal installer [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to set up the environment in which we run the command below to install the subset of packages needed for running WaveDiff.

In the [WaveDiff repository](https://github.com/CosmoStat/wf-psf.git), we provide in the file `environment.yml` the specific environment used in the testing of `WaveDiff`. We recommend users to work with this environment for the current release. 

## Installation Steps

Clone the repository:

```
$ git clone https://github.com/CosmoStat/wf-psf.git
```

Next you can enter the `wf-psf` directory and run `pip` to install WaveDiff and its [dependencies](dependencies.md). 

```
$ cd wf-psf
$ pip install .
```

You can now proceed to the next section, where we show you how to run WaveDiff.
