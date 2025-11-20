# Basic Execution

The WaveDiff pipeline is launched by the entrypoint script: `src/wf_psf/run.py`, which is referenced by the command `wavediff`.

A list of command-line arguments can be displayed using the `--help` option:

```
> wavediff --help
usage: run.py [-h] --conffile CONFFILE --repodir REPODIR --outputdir OUTPUTDIR

required arguments:
  --conffile CONFFILE, -c CONFFILE
                        a configuration file containing program settings.
  --repodir REPODIR, -r REPODIR
                        the path of the code repository directory.
  --outputdir OUTPUTDIR, -o OUTPUTDIR
                        the path of the output directory.

optional arguments:
  -h, --help            show this help message and exit
```
The first argument: `--confile CONFFILE` specifies the path to the {ref}`master configuration file<master_config_file>` storing the pipeline tasks to be executed at runtime.

The second argument: `--repodir REPODIR` is the path to the `wf-psf` repository.

The third argument: `--outputdir OUTPUTDIR` is used to set the path to the main output directory, which stores the `WaveDiff` results.

To run `WaveDiff`, use the following command:

```
> wavediff -c /path/to/config/file -r /path/to/wf-psf -o /path/to/output/dir
```

## WaveDiff Output Directory Structure

WaveDiff begins with processing the input configuration files and setting up the output file structure. The application proceeds then with creating a set of nested sub-directories in the location specified by  `--outputdir` to store various outputs (e.g. logfiles, model checkpoints, complete model graph, etc. described below). The name of the parent subdirectory is a composition of the string `wf-outputs-` and a timestamp of the current run accurate to the microsecond, i.e. `wf-outputs-20231119151932213823`. Within this subdirectory, further subdirectories are generated to store the corresponding output.

Below is an example of the set of directories generated during each execution of the WaveDiff pipeline.

(wf-outputs)=
```
wf-outputs-20231119151932213823
├── checkpoint
├── config
├── log-files
├── metrics
├── optim-hist
├── plots
└── psf_model
```
A description of each subdirectory is provided in the following table.

| Sub-directory    | Purpose                                       |
|--------------|---------------------------------------------------|
| checkpoint   | Stores the checkpoint weights produced during the training pipeline task. |
| config       | Stores all of the configuration files (see [Configuration](Configuration)) provided as input during the run. |
| log-files    | Stores the log-file of the run.  |
| metrics       | Stores the metrics results generated during the metrics pipeline task.  |
| optim-hist    |  Stores the training history of the model parameters.  |
| plots         | Stores metrics plots generated during the plotting pipeline task.   | 
| psf_models    |  Stores the final trained PSF models for each training cycle. |


The next section covers the organisational structure of the configuration files.



