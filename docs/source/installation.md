# Installation


WaveDiff is a software written in the programming language Python and uses the [TensorFlow](https://www.tensorflow.org) framework to train and evaluate the model parameters of the PSF model. Therefore, it is advisable to run the WaveDiff software on machines equipped with a GPU.  

Note: You may want to set up a dedicated Python environment for running WaveDiff using [Conda](https://docs.conda.io/en/latest/).  You can use the minimal installer [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) to install the subset of packages needed for WaveDiff.

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

